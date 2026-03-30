#!/usr/bin/env python3
"""
run_decoupled_ablation.py — Decoupled ablation: Schema-Only condition.

Isolates the contribution of "schema prompt injection" from "SGE serialization"
by combining raw CSV text (vanilla processing) with the SGE extraction schema prompt.

Experimental conditions matrix:
  C2 (full SGE):       SGE serialization + schema prompt   → best FC
  C4 (serial only):    SGE serialization + default prompt   → partial improvement
  C5 (vanilla):        raw CSV text      + default prompt   → baseline
  C6 (schema only):    raw CSV text      + schema prompt    → THIS EXPERIMENT

For 3 datasets: WHO Life Expectancy, WB Child Mortality, Inpatient 2023.

Usage:
    python3 experiments/run_decoupled_ablation.py
    python3 experiments/run_decoupled_ablation.py --dataset who
"""

from __future__ import annotations

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

# ── API config (same as run_baseline_only.py) ────────────────────────────────
API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024


# ── Dataset configurations ───────────────────────────────────────────────────
DATASETS = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv": PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv",
        "sge_output": PROJECT_ROOT / "output" / "who_life_expectancy",
        "gold": PROJECT_ROOT / "evaluation" / "gold_who_life_expectancy_v2.jsonl",
        "skiprows": 0,
        "encoding": "utf-8-sig",
        "sep": ",",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv": PROJECT_ROOT / "dataset" / "世界银行数据" / "child_mortality" / "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "sge_output": PROJECT_ROOT / "output" / "wb_child_mortality",
        "gold": PROJECT_ROOT / "evaluation" / "gold_wb_child_mortality_v2.jsonl",
        "skiprows": 4,
        "encoding": "utf-8-sig",
        "sep": ",",
    },
    "inpatient": {
        "label": "Inpatient 2023",
        "csv": PROJECT_ROOT / "dataset" / "住院病人统计" / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "sge_output": PROJECT_ROOT / "output" / "inpatient_2023",
        "gold": PROJECT_ROOT / "evaluation" / "gold_inpatient_2023.jsonl",
        "skiprows": 0,
        "encoding": "utf-8-sig",
        "sep": ",",
    },
}


# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Hash-based embedding (same as baseline) ─────────────────────────────────
def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)


# ── CSV reading ──────────────────────────────────────────────────────────────
def read_csv_as_text(ds_config: dict) -> str:
    """Read raw CSV into a plain text string using pandas, same as baseline would see."""
    import pandas as pd
    from stage1.features import _detect_encoding

    csv_path = str(ds_config["csv"])
    skiprows = ds_config["skiprows"]

    # Detect encoding
    encoding = _detect_encoding(csv_path)
    if "utf-16" in encoding:
        df = pd.read_csv(csv_path, encoding="utf-16", sep="\t", skiprows=skiprows)
    else:
        # Try detected encoding first, then fallbacks
        df = None
        for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc, skiprows=skiprows)
                break
            except (UnicodeDecodeError, Exception):
                continue
        if df is None:
            raise ValueError(f"Cannot read CSV: {csv_path}")

    print(f"  CSV shape: {df.shape}")
    return df.to_string()


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks at line boundaries, respecting max_chars."""
    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_size = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_size + line_len > max_chars and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_size = 0
        current_chunk_lines.append(line)
        current_size += line_len

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


# ── Load SGE schema prompt ───────────────────────────────────────────────────
def load_schema_prompt(sge_output_dir: Path) -> tuple[str, dict]:
    """Load the SGE system prompt and extraction schema from existing output."""
    # Try prompts/ subdirectory first, then root
    prompt_path = sge_output_dir / "prompts" / "system_prompt.txt"
    if not prompt_path.exists():
        prompt_path = sge_output_dir / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found in {sge_output_dir}/prompts/ or {sge_output_dir}/"
        )

    system_prompt = prompt_path.read_text(encoding="utf-8")

    schema_path = sge_output_dir / "extraction_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Extraction schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    return system_prompt, schema


# ── Run schema-only LightRAG ────────────────────────────────────────────────
async def run_schema_only(
    dataset_key: str,
    ds_config: dict,
) -> dict:
    """Run the schema-only condition for one dataset."""
    label = ds_config["label"]
    sge_output_dir = ds_config["sge_output"]
    output_dir = PROJECT_ROOT / "output" / f"ablation_schema_only_{dataset_key}"
    work_dir = output_dir / "lightrag_storage"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"SCHEMA-ONLY ABLATION: {label}")
    print(f"{'=' * 60}")

    # Step 1: Read raw CSV as plain text
    print("\n[Step 1] Reading raw CSV as plain text...")
    csv_text = read_csv_as_text(ds_config)
    chunks = chunk_text(csv_text)
    print(f"  Total text length: {len(csv_text)} chars")
    print(f"  Chunks: {len(chunks)}")

    # Step 2: Load SGE schema prompt
    print("\n[Step 2] Loading SGE schema prompt...")
    system_prompt_raw, schema = load_schema_prompt(sge_output_dir)
    entity_types = schema.get("entity_types", ["Entity"])
    print(f"  Entity types: {entity_types}")
    print(f"  Prompt length: {len(system_prompt_raw)} chars")

    # Step 3: Escape braces for LightRAG template compatibility
    # (same logic as run_lightrag_integration.py lines 273-279)
    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    # Step 4: Override system prompt and run LightRAG
    print(f"\n[Step 3] Running LightRAG with schema prompt + raw text...")
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped

    addon_params = {
        "language": "Chinese",
        "entity_types": entity_types,
    }

    try:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params=addon_params,
            llm_model_max_async=10,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()

        for i, chunk_text_item in enumerate(chunks, 1):
            if i % 10 == 0 or i == len(chunks) or i == 1:
                print(f"  [{i}/{len(chunks)}] ({len(chunk_text_item)} chars)")
            await rag.ainsert(chunk_text_item)

        await rag.finalize_storages()
    finally:
        # Always restore original prompt
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    # Step 5: Collect graph stats
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    stats = {
        "dataset": dataset_key,
        "label": label,
        "condition": "schema_only",
        "chunks": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"\n  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    else:
        print("\n  WARNING: graph file not found after insertion")
        stats["nodes"] = 0
        stats["edges"] = 0

    # Save experiment stats
    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return stats


# ── Evaluate FC ──────────────────────────────────────────────────────────────
def evaluate_fc(dataset_key: str, ds_config: dict) -> dict:
    """Evaluate fact coverage for the schema-only graph."""
    from evaluation.evaluate_coverage import load_gold, load_graph, check_fact_coverage, check_entity_coverage

    graph_path = (
        PROJECT_ROOT / "output" / f"ablation_schema_only_{dataset_key}"
        / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    )
    gold_path = ds_config["gold"]

    if not graph_path.exists():
        print(f"  ERROR: graph not found at {graph_path}")
        return {"ec": 0.0, "fc": 0.0, "nodes": 0, "edges": 0}

    if not gold_path.exists():
        print(f"  ERROR: gold standard not found at {gold_path}")
        return {"ec": 0.0, "fc": 0.0, "nodes": 0, "edges": 0}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph(str(graph_path))

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  Gold: {len(gold_entities)} entities, {len(facts)} facts")
    print(f"  EC: {len(matched_entities)}/{len(gold_entities)} = {ec:.4f}")
    print(f"  FC: {len(covered)}/{len(facts)} = {fc:.4f}")

    if not_covered:
        reasons = {}
        for nc in not_covered:
            r = nc.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"  Uncovered breakdown: {reasons}")

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


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Decoupled ablation: schema-only condition")
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        default=None,
        help="Run only this dataset (default: all 3)",
    )
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())
    all_results = []

    # Validate paths before starting
    for dk in datasets_to_run:
        ds = DATASETS[dk]
        if not ds["csv"].exists():
            print(f"ERROR: CSV not found: {ds['csv']}", file=sys.stderr)
            sys.exit(1)
        if not ds["sge_output"].exists():
            print(f"ERROR: SGE output not found: {ds['sge_output']}", file=sys.stderr)
            sys.exit(1)
        if not ds["gold"].exists():
            print(f"ERROR: Gold standard not found: {ds['gold']}", file=sys.stderr)
            sys.exit(1)

    # Run schema-only condition for each dataset
    for dk in datasets_to_run:
        ds = DATASETS[dk]
        stats = await run_schema_only(dk, ds)

        # Evaluate
        print(f"\n[Evaluate] {ds['label']}...")
        eval_result = evaluate_fc(dk, ds)

        result = {
            "dataset": dk,
            "dataset_label": ds["label"],
            "condition": "schema_only",
            **eval_result,
            "timestamp": stats["timestamp"],
            "chunks_inserted": stats["chunks"],
        }
        all_results.append(result)

    # Save combined results
    results_path = PROJECT_ROOT / "experiments" / "results" / "decoupled_ablation_results.json"
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary table
    print(f"\n{'=' * 70}")
    print("DECOUPLED ABLATION RESULTS — Schema-Only Condition")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<25} {'EC':>8} {'FC':>8} {'Nodes':>8} {'Edges':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset_label']:<25} {r['ec']:>8.4f} {r['fc']:>8.4f} {r['nodes']:>8} {r['edges']:>8}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
