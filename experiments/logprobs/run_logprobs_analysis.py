#!/usr/bin/env python3
"""
run_logprobs_analysis.py — Format-constraint coupling logprobs experiment.

Uses GPT-5-mini's logprobs API to provide internal-evidence for the
surface-form pattern matching hypothesis. Compares token-level generation
probabilities under matched vs mismatched format-schema conditions.

Three conditions (WHO data, 5 chunks each):
  1. Matched:     SGE serialization + SGE Schema    → expect high logprob
  2. Mismatched:  Raw CSV text      + SGE Schema    → expect low logprob
  3. No-Schema:   SGE serialization + default prompt → expect low logprob

Metrics:
  - Mean logprob of entity-type tokens (e.g., "Country_Code")
  - Mean logprob of delimiter tokens (<|#|>)
  - Generation entropy per response
  - Schema-slot token probability mass
"""

from __future__ import annotations

import os
import sys
import json
import math
import csv
import statistics
from pathlib import Path
from datetime import datetime

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.prompt_injector import (
    generate_system_prompt,
    generate_user_prompt_template,
    render_user_prompt,
    TUPLE_DELIMITER,
    COMPLETION_DELIMITER,
)

# ── Config ──────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL = "gpt-5-mini"
N_CHUNKS = 5  # first 5 WHO chunks

# ── Paths ───────────────────────────────────────────────────────────────
WHO_CHUNKS_DIR = PROJECT_ROOT / "output" / "who_life_expectancy" / "chunks"
WHO_SCHEMA_PATH = PROJECT_ROOT / "output" / "who_life_expectancy" / "extraction_schema.json"
WHO_CSV_PATH = PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# ── LightRAG default prompt (no schema) ────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that extracts entities and relationships from text.
Extract all meaningful entities and relationships from the provided text.
Output format:
entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
relation<|#|>source_entity<|#|>target_entity<|#|>keywords<|#|>description
End with <|COMPLETE|>
"""


def load_sge_chunks(n: int = N_CHUNKS) -> list[str]:
    """Load pre-computed SGE serialized chunks."""
    chunk_files = sorted(WHO_CHUNKS_DIR.glob("chunk_*.txt"))[:n]
    return [f.read_text(encoding="utf-8") for f in chunk_files]


def load_raw_csv_chunks(n: int = N_CHUNKS) -> list[str]:
    """Generate naive CSV text chunks (mimicking LightRAG baseline)."""
    with open(WHO_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip metadata rows (World Bank format)
    header_idx = 0
    for i, row in enumerate(rows):
        if row and row[0] in ("Country Name", "Country Code"):
            header_idx = i
            break
    header = rows[header_idx]
    data_rows = rows[header_idx + 1:]

    # Create naive text chunks (one country per chunk, like LightRAG)
    chunks = []
    for row in data_rows[:n]:
        text_parts = []
        for col, val in zip(header, row):
            if val.strip():
                text_parts.append(f"{col}: {val}")
        chunks.append("\n".join(text_parts))
    return chunks


def call_with_logprobs(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> dict:
    """Call GPT-5-mini with logprobs enabled."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    logprobs_data = choice.logprobs

    return {
        "content": content,
        "logprobs": logprobs_data,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


def analyze_logprobs(result: dict, schema_tokens: list[str]) -> dict:
    """Analyze logprobs for schema-relevant tokens."""
    logprobs_data = result["logprobs"]
    if not logprobs_data or not logprobs_data.content:
        return {"error": "no logprobs returned"}

    all_logprobs = []
    schema_token_logprobs = []
    delimiter_logprobs = []
    entity_line_count = 0
    relation_line_count = 0
    refusal_detected = False

    content = result["content"]
    if any(phrase in content.lower() for phrase in [
        "i cannot", "i can't", "as an ai", "i'm sorry", "i apologize",
        "i'm not able", "unable to"
    ]):
        refusal_detected = True

    for token_info in logprobs_data.content:
        token = token_info.token
        logprob = token_info.logprob
        all_logprobs.append(logprob)

        # Check if token matches schema entity types
        for st in schema_tokens:
            if st.lower() in token.lower():
                schema_token_logprobs.append(logprob)
                break

        # Check delimiter tokens
        if "<|#|>" in token or "<|COMPLETE|>" in token:
            delimiter_logprobs.append(logprob)

    # Count structured output lines
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("entity" + TUPLE_DELIMITER):
            entity_line_count += 1
        elif line.startswith("relation" + TUPLE_DELIMITER):
            relation_line_count += 1

    # Compute entropy (average negative logprob = perplexity proxy)
    mean_logprob = statistics.mean(all_logprobs) if all_logprobs else 0.0
    entropy = -mean_logprob  # higher entropy = more uncertain

    return {
        "n_tokens": len(all_logprobs),
        "mean_logprob": round(mean_logprob, 4),
        "entropy": round(entropy, 4),
        "schema_token_mean_logprob": (
            round(statistics.mean(schema_token_logprobs), 4)
            if schema_token_logprobs else None
        ),
        "schema_token_count": len(schema_token_logprobs),
        "delimiter_mean_logprob": (
            round(statistics.mean(delimiter_logprobs), 4)
            if delimiter_logprobs else None
        ),
        "delimiter_count": len(delimiter_logprobs),
        "entity_lines": entity_line_count,
        "relation_lines": relation_line_count,
        "refusal_detected": refusal_detected,
        "output_length": len(content),
    }


def run_experiment():
    """Run the 3-condition logprobs experiment."""
    print("=" * 70)
    print("FORMAT-CONSTRAINT COUPLING: LOGPROBS ANALYSIS")
    print(f"Model: {MODEL} | Chunks: {N_CHUNKS} | Time: {datetime.now()}")
    print("=" * 70)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load data
    schema = json.loads(WHO_SCHEMA_PATH.read_text(encoding="utf-8"))
    sge_chunks = load_sge_chunks(N_CHUNKS)
    raw_chunks = load_raw_csv_chunks(N_CHUNKS)

    # Schema-relevant tokens to track
    schema_tokens = schema.get("entity_types", []) + schema.get("relation_types", [])
    # Add: Country_Code, HAS_VALUE
    print(f"Schema tokens to track: {schema_tokens}")

    # Generate prompts
    sge_system_prompt = generate_system_prompt(schema, language="English")
    sge_user_template = generate_user_prompt_template(schema)

    conditions = {
        "matched": {
            "description": "SGE serialization + SGE Schema",
            "system_prompt": sge_system_prompt,
            "chunks": sge_chunks,
            "user_template": sge_user_template,
        },
        "mismatched": {
            "description": "Raw CSV text + SGE Schema",
            "system_prompt": sge_system_prompt,
            "chunks": raw_chunks,
            "user_template": sge_user_template,
        },
        "no_schema": {
            "description": "SGE serialization + default prompt (no schema)",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "chunks": sge_chunks,
            "user_template": (
                "Extract entities and relationships from the text below.\n"
                "Output delimiter-based lines. End with <|COMPLETE|>.\n\n"
                "```\n{input_text}\n```\n\n<Output>\n"
            ),
        },
    }

    results = {}

    for cond_name, cond in conditions.items():
        print(f"\n{'─' * 50}")
        print(f"Condition: {cond_name} — {cond['description']}")
        print(f"{'─' * 50}")

        cond_results = []
        for i, chunk in enumerate(cond["chunks"]):
            print(f"  Chunk {i + 1}/{len(cond['chunks'])}...", end=" ", flush=True)

            user_prompt = cond["user_template"].replace("{input_text}", chunk)

            try:
                raw_result = call_with_logprobs(
                    client, cond["system_prompt"], user_prompt
                )
                analysis = analyze_logprobs(raw_result, schema_tokens)
                analysis["chunk_idx"] = i
                analysis["raw_content_preview"] = raw_result["content"][:200]
                analysis["usage"] = raw_result["usage"]
                cond_results.append(analysis)

                if "error" in analysis:
                    print(f"NO_LOGPROBS | output_len={len(raw_result['content'])}")
                else:
                    status = "REFUSAL" if analysis.get("refusal_detected") else "OK"
                    print(
                        f"{status} | entropy={analysis['entropy']:.3f} "
                        f"| entities={analysis['entity_lines']} "
                        f"| relations={analysis['relation_lines']}"
                    )
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
                cond_results.append({"chunk_idx": i, "error": str(e)})

        results[cond_name] = {
            "description": cond["description"],
            "per_chunk": cond_results,
        }

    # ── Aggregate & Report ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    summary = {}
    for cond_name, cond_data in results.items():
        valid = [r for r in cond_data["per_chunk"] if "error" not in r]
        if not valid:
            summary[cond_name] = {"error": "all chunks failed"}
            continue

        agg = {
            "description": cond_data["description"],
            "n_valid": len(valid),
            "mean_entropy": round(statistics.mean([r["entropy"] for r in valid]), 4),
            "mean_entity_lines": round(statistics.mean([r["entity_lines"] for r in valid]), 1),
            "mean_relation_lines": round(statistics.mean([r["relation_lines"] for r in valid]), 1),
            "refusal_rate": round(sum(1 for r in valid if r["refusal_detected"]) / len(valid), 3),
        }

        schema_lps = [r["schema_token_mean_logprob"] for r in valid
                      if r["schema_token_mean_logprob"] is not None]
        if schema_lps:
            agg["mean_schema_token_logprob"] = round(statistics.mean(schema_lps), 4)

        delim_lps = [r["delimiter_mean_logprob"] for r in valid
                     if r["delimiter_mean_logprob"] is not None]
        if delim_lps:
            agg["mean_delimiter_logprob"] = round(statistics.mean(delim_lps), 4)

        summary[cond_name] = agg

        print(f"\n{cond_name}: {cond_data['description']}")
        for k, v in agg.items():
            if k != "description":
                print(f"  {k}: {v}")

    # ── Hypothesis Test ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("HYPOTHESIS EVALUATION")
    print("─" * 70)

    m = summary.get("matched", {})
    mm = summary.get("mismatched", {})
    ns = summary.get("no_schema", {})

    if all(k in s for s in [m, mm] for k in ["mean_entropy"]):
        e_diff = mm["mean_entropy"] - m["mean_entropy"]
        print(f"  Entropy(mismatched) - Entropy(matched) = {e_diff:+.4f}")
        if e_diff > 0:
            print("  → Matched condition has LOWER entropy (more confident)")
        else:
            print("  → Unexpected: mismatched has lower entropy")

    if "mean_schema_token_logprob" in m and "mean_schema_token_logprob" in mm:
        lp_diff = m["mean_schema_token_logprob"] - mm["mean_schema_token_logprob"]
        print(f"  Schema-token logprob(matched) - logprob(mismatched) = {lp_diff:+.4f}")
        if lp_diff > 0:
            print("  → Schema tokens generated with HIGHER probability in matched condition")

    # ── Save ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "logprobs_format_constraint_coupling",
        "model": MODEL,
        "n_chunks": N_CHUNKS,
        "timestamp": datetime.now().isoformat(),
        "schema_tokens_tracked": schema_tokens,
        "summary": summary,
        "per_condition": results,
    }
    out_path = RESULTS_DIR / "logprobs_analysis_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
