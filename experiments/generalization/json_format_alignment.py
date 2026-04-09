#!/usr/bin/env python3
"""
json_format_alignment.py — Format-constraint alignment generalization experiment.

Tests whether the format-constraint alignment phenomenon (Schema designed for
format A fails on format B) generalizes beyond CSV to JSON and Markdown formats.

Experiment matrix (6 conditions on WHO Life Expectancy data):
  json_structured_schema   : JSON records    + WHO Schema  → format-aligned
  json_structured_default  : JSON records    + Default     → no schema
  json_flat_schema         : CSV-in-JSON     + WHO Schema  → format-misaligned
  json_flat_default        : CSV-in-JSON     + Default     → no schema
  markdown_structured_schema: Markdown table + WHO Schema  → format-aligned
  markdown_flat_schema     : Raw CSV text    + WHO Schema  → format-misaligned

Key predictions:
  json_flat_schema FC < json_flat_default FC
      (Schema hurts on misaligned format — mirrors CSV's Schema-only < Baseline)
  json_structured_schema FC > json_structured_default FC
      (Schema helps on aligned format — mirrors CSV's Full SGE > Serial-only)
  Same pattern for markdown conditions

Usage:
    python3 experiments/generalization/json_format_alignment.py
    python3 experiments/generalization/json_format_alignment.py --condition json_structured_schema
    python3 experiments/generalization/json_format_alignment.py --eval-only
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

# ── Paths ─────────────────────────────────────────────────────────────────────
WHO_CSV = PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"
WHO_SGE_OUTPUT = PROJECT_ROOT / "output" / "who_life_expectancy"
GOLD_PATH = PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_v2.jsonl"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "json_format_alignment_results.json"

# Year columns present in the WHO gold standard
YEAR_COLS = ["2000", "2005", "2010", "2015", "2019", "2021"]

ALL_CONDITIONS = [
    "json_structured_schema",
    "json_structured_default",
    "json_flat_schema",
    "json_flat_default",
    "markdown_structured_schema",
    "markdown_structured_default",
    "markdown_flat_schema",
]

# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding (Ollama via urllib3 to bypass macOS proxy) ─────────────────────
import urllib3 as _urllib3
_pool = _urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


def _ollama_embed_sync(texts: list[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        truncated = text[:1000] if len(text) > 1000 else text
        body = json.dumps({"model": "mxbai-embed-large", "prompt": truncated}).encode()
        resp = _pool.urlopen(
            "POST", "/api/embeddings", body=body,
            headers={"Content-Type": "application/json"}, timeout=120.0,
        )
        emb = json.loads(resp.data)["embedding"]
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    loop = asyncio.get_event_loop()
    for attempt in range(3):
        try:
            return await loop.run_in_executor(None, _ollama_embed_sync, texts)
        except Exception as e:
            if attempt < 2:
                print(f"  [warn] Embed attempt {attempt + 1} failed: {e}, retrying...")
                await asyncio.sleep(2)
            else:
                print(f"  [warn] Embed failed 3x, using hash fallback")
                return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)


# ── Data loading ──────────────────────────────────────────────────────────────
def read_who_csv() -> "pd.DataFrame":
    """Read WHO Life Expectancy CSV, returning only data rows (no WB metadata)."""
    import pandas as pd
    df = pd.read_csv(str(WHO_CSV), encoding="utf-8-sig")
    # Drop rows where Country Code is NaN (WHO regional aggregates sometimes lack it)
    df = df.dropna(subset=["Country Code"])
    return df


def get_gold_countries() -> list[str]:
    """Extract unique country codes that appear as subjects in the gold standard."""
    subjects = set()
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            subjects.add(record["triple"]["subject"])
    return sorted(subjects)


def _get_year_value(row: "pd.Series", year: str) -> str | None:
    """Return stringified year value or None if missing/non-numeric."""
    if year not in row.index:
        return None
    val = row[year]
    try:
        fval = float(val)
        if np.isnan(fval):
            return None
        return str(round(fval, 2))
    except (TypeError, ValueError):
        return None


# ── Format constructors ───────────────────────────────────────────────────────
def format_json_structured(df: "pd.DataFrame", countries: list[str]) -> list[str]:
    """One JSON record per country with explicit keys. Returns 25 chunks."""
    chunks = []
    for code in countries:
        rows = df[df["Country Code"] == code]
        if rows.empty:
            print(f"  [warn] Country {code} not found in CSV, skipping")
            continue
        row = rows.iloc[0]
        country_name = str(row.get("Country Name", "")) if not _is_nan(row.get("Country Name")) else code
        measurements = {}
        for year in YEAR_COLS:
            val = _get_year_value(row, year)
            if val is not None:
                measurements[year] = float(val)
        record = {
            "entity": code,
            "entity_type": "Country_Code",
            "country_name": country_name,
            "indicator": "Life_Expectancy_At_Birth",
            "unit": "years",
            "measurements": measurements,
        }
        chunks.append(json.dumps(record, ensure_ascii=False))
    return chunks


def format_json_flat(df: "pd.DataFrame", countries: list[str]) -> list[str]:
    """Raw CSV text wrapped in a JSON string, chunked at ~4000 chars."""
    filtered = df[df["Country Code"].isin(countries)]
    csv_text = filtered.to_csv(index=False)
    lines = csv_text.split("\n")
    raw_chunks = _chunk_lines(lines, max_chars=3500)
    result = []
    for chunk_text in raw_chunks:
        wrapper = {"data_format": "csv", "content": chunk_text}
        result.append(json.dumps(wrapper, ensure_ascii=False))
    return result


def format_markdown_structured(df: "pd.DataFrame", countries: list[str]) -> list[str]:
    """Markdown pipe table, one chunk per country with selected year columns."""
    chunks = []
    header = "| Country | Year | Life Expectancy |\n|---------|------|----------------|"
    for code in countries:
        rows = df[df["Country Code"] == code]
        if rows.empty:
            print(f"  [warn] Country {code} not found in CSV, skipping")
            continue
        row = rows.iloc[0]
        md_rows = [header]
        for year in YEAR_COLS:
            val = _get_year_value(row, year)
            if val is not None:
                md_rows.append(f"| {code}     | {year} | {val:<15} |")
        if len(md_rows) > 2:
            chunks.append("\n".join(md_rows))
    return chunks


def format_markdown_flat(df: "pd.DataFrame", countries: list[str]) -> list[str]:
    """Raw CSV text (plain comma-separated, no markdown formatting)."""
    filtered = df[df["Country Code"].isin(countries)]
    csv_text = filtered.to_csv(index=False)
    lines = csv_text.split("\n")
    return _chunk_lines(lines, max_chars=4000)


def _is_nan(val) -> bool:
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return val is None or str(val).strip() in ("", "nan", "NaN")


def _chunk_lines(lines: list[str], max_chars: int) -> list[str]:
    """Split lines into chunks respecting max_chars, preserving line boundaries."""
    chunks = []
    current: list[str] = []
    current_size = 0
    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── Schema loading ────────────────────────────────────────────────────────────
def load_who_schema() -> tuple[str, dict]:
    """Load WHO SGE system prompt and extraction schema."""
    prompt_path = WHO_SGE_OUTPUT / "prompts" / "system_prompt.txt"
    if not prompt_path.exists():
        prompt_path = WHO_SGE_OUTPUT / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"system_prompt.txt not found under {WHO_SGE_OUTPUT}")

    system_prompt = prompt_path.read_text(encoding="utf-8")

    schema_path = WHO_SGE_OUTPUT / "extraction_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"extraction_schema.json not found at {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return system_prompt, schema


def _escape_prompt(raw: str, entity_types: list[str]) -> tuple[str, dict]:
    """Escape braces for LightRAG template compatibility and build addon_params."""
    escaped = raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    addon_params = {"language": "Chinese", "entity_types": entity_types}
    return escaped, addon_params


# ── LightRAG runner ───────────────────────────────────────────────────────────
async def run_condition(
    name: str,
    chunks: list[str],
    use_schema: bool,
    work_dir: Path,
) -> dict:
    """Insert chunks into a fresh LightRAG instance for one condition."""
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"CONDITION: {name}  ({len(chunks)} chunks, schema={use_schema})")
    print(f"{'=' * 60}")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    addon_params: dict = {}

    if use_schema:
        system_prompt_raw, schema = load_who_schema()
        entity_types = schema.get("entity_types", ["Entity"])
        escaped, addon_params = _escape_prompt(system_prompt_raw, entity_types)
        PROMPTS["entity_extraction_system_prompt"] = escaped
        print(f"  Schema entity types: {entity_types}")

    try:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params=addon_params if use_schema else {},
            llm_model_max_async=5,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()

        for i, chunk in enumerate(chunks, 1):
            if i == 1 or i % 10 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
            await rag.ainsert(chunk)

        await rag.finalize_storages()
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    nodes, edges = 0, 0
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        print(f"  Graph: {nodes} nodes, {edges} edges")
    else:
        print("  WARNING: graph file not found after insertion")

    return {
        "condition": name,
        "chunks": len(chunks),
        "nodes": nodes,
        "edges": edges,
        "timestamp": datetime.now().isoformat(),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_condition(name: str, graph_dir: Path) -> dict:
    """Compute EC/FC for one condition against the WHO gold standard."""
    from evaluation.evaluate_coverage import load_gold, load_graph, check_entity_coverage, check_fact_coverage

    graph_path = graph_dir / "graph_chunk_entity_relation.graphml"

    if not graph_path.exists():
        print(f"  ERROR: graph not found at {graph_path}")
        return {"ec": 0.0, "fc": 0.0, "nodes": 0, "edges": 0,
                "ec_matched": 0, "ec_total": 0, "fc_covered": 0, "fc_total": 0}

    gold_entities, facts = load_gold(str(GOLD_PATH))
    G, graph_nodes, entity_text = load_graph(str(graph_path))

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  [{name}] EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
          f"FC={fc:.4f} ({len(covered)}/{len(facts)})  "
          f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

    return {
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "ec_matched": len(matched_entities),
        "ec_total": len(gold_entities),
        "fc_covered": len(covered),
        "fc_total": len(facts),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format-constraint alignment generalization experiment"
    )
    parser.add_argument(
        "--condition", "-c",
        choices=ALL_CONDITIONS,
        default=None,
        help="Run a single condition (default: all 6)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip LightRAG runs; recompute FC from existing graphs only",
    )
    args = parser.parse_args()

    conditions_to_run = [args.condition] if args.condition else ALL_CONDITIONS

    # Validate paths
    for path, label in [
        (WHO_CSV, "WHO CSV"),
        (WHO_SGE_OUTPUT, "WHO SGE output"),
        (GOLD_PATH, "Gold standard"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Build input data once
    df = read_who_csv()
    countries = get_gold_countries()
    print(f"Gold countries ({len(countries)}): {countries}")

    chunk_map = {
        "json_structured_schema":    (format_json_structured(df, countries), True),
        "json_structured_default":   (format_json_structured(df, countries), False),
        "json_flat_schema":          (format_json_flat(df, countries), True),
        "json_flat_default":         (format_json_flat(df, countries), False),
        "markdown_structured_schema": (format_markdown_structured(df, countries), True),
        "markdown_structured_default": (format_markdown_structured(df, countries), False),
        "markdown_flat_schema":      (format_markdown_flat(df, countries), True),
    }

    condition_meta = {
        "json_structured_schema":    {"format": "JSON records", "schema": True},
        "json_structured_default":   {"format": "JSON records", "schema": False},
        "json_flat_schema":          {"format": "JSON-wrapped CSV", "schema": True},
        "json_flat_default":         {"format": "JSON-wrapped CSV", "schema": False},
        "markdown_structured_schema": {"format": "Markdown table", "schema": True},
        "markdown_structured_default": {"format": "Markdown table", "schema": False},
        "markdown_flat_schema":      {"format": "Raw text", "schema": True},
    }

    # Load existing results (if any) to accumulate across partial runs
    results: dict = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if "conditions" in existing:
            results = existing["conditions"]

    for name in conditions_to_run:
        work_dir = PROJECT_ROOT / "output" / f"json_alignment_{name}" / "lightrag_storage"
        chunks, use_schema = chunk_map[name]
        print(f"\nChunks for [{name}]: {len(chunks)}")

        if not args.eval_only:
            run_stats = await run_condition(name, chunks, use_schema, work_dir)
        else:
            print(f"  [eval-only] Skipping LightRAG run for {name}")
            run_stats = {"condition": name, "chunks": len(chunks)}

        print(f"\n[Evaluate] {name}...")
        eval_result = evaluate_condition(name, work_dir)

        results[name] = {
            **condition_meta[name],
            "chunks": run_stats.get("chunks", len(chunks)),
            "EC": eval_result["ec"],
            "FC": eval_result["fc"],
            "ec_matched": eval_result["ec_matched"],
            "ec_total": eval_result["ec_total"],
            "fc_covered": eval_result["fc_covered"],
            "fc_total": eval_result["fc_total"],
            "nodes": eval_result["nodes"],
            "edges": eval_result["edges"],
        }

    # Save combined results
    output = {
        "experiment": "format_constraint_alignment_generalization",
        "dataset": "who_life_expectancy",
        "principle": "Schema constraints designed for format A fail when applied to format B",
        "conditions": results,
        "validation": {
            "csv_reference": {
                "full_sge_fc": 1.000,
                "schema_only_fc": 0.380,
                "baseline_fc": 0.167,
            }
        },
        "last_updated": datetime.now().isoformat(),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("FORMAT-CONSTRAINT ALIGNMENT GENERALIZATION RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Condition':<35} {'Format':<22} {'Schema':>6} {'Chunks':>7} {'EC':>8} {'FC':>8}")
    print("-" * 80)
    for cname, cdata in results.items():
        schema_flag = "Yes" if cdata.get("schema") else "No"
        print(
            f"{cname:<35} {cdata.get('format', ''):<22} {schema_flag:>6} "
            f"{cdata.get('chunks', 0):>7} {cdata.get('EC', 0):>8.4f} {cdata.get('FC', 0):>8.4f}"
        )

    # Validate predictions
    if all(k in results for k in ["json_flat_schema", "json_flat_default"]):
        pred1 = results["json_flat_schema"]["FC"] < results["json_flat_default"]["FC"]
        print(f"\n[P1] json_flat_schema FC ({results['json_flat_schema']['FC']:.4f}) "
              f"< json_flat_default FC ({results['json_flat_default']['FC']:.4f}): "
              f"{'CONFIRMED' if pred1 else 'NOT confirmed'}")

    if all(k in results for k in ["json_structured_schema", "json_structured_default"]):
        pred2 = results["json_structured_schema"]["FC"] > results["json_structured_default"]["FC"]
        print(f"[P2] json_structured_schema FC ({results['json_structured_schema']['FC']:.4f}) "
              f"> json_structured_default FC ({results['json_structured_default']['FC']:.4f}): "
              f"{'CONFIRMED' if pred2 else 'NOT confirmed'}")

    if all(k in results for k in ["markdown_structured_schema", "markdown_flat_schema"]):
        pred3 = results["markdown_structured_schema"]["FC"] > results["markdown_flat_schema"]["FC"]
        print(f"[P3] markdown_structured_schema FC ({results['markdown_structured_schema']['FC']:.4f}) "
              f"> markdown_flat_schema FC ({results['markdown_flat_schema']['FC']:.4f}): "
              f"{'CONFIRMED' if pred3 else 'NOT confirmed'}")


if __name__ == "__main__":
    asyncio.run(main())
