#!/usr/bin/env python3
"""
run_decoupled_ablation_extended.py — Decoupled ablation extended to WB Population and WB Maternal.

Extends the 4-condition orthogonal ablation matrix to 2 additional datasets:
  WB Population (25×150 Type-II)
  WB Maternal Mortality (25×150 Type-II)

Conditions (Table 4 / §4.3):
  Full SGE     : SGE serialization + schema prompt   → output/wb_population, output/wb_maternal
  Serial-only  : SGE serialization + default prompt  → output/ablation_c4_serial_only_wb_pop, ...
  Schema-only  : raw CSV text      + schema prompt   → output/ablation_schema_only_wb_pop, ...
  Baseline     : raw CSV text      + default prompt  → output/baseline_wb_population, ...

Full SGE and Baseline graphs already exist — only Serial-only and Schema-only require new runs.

Usage:
    # Run both conditions for both datasets (default)
    python3 experiments/ablation/run_decoupled_ablation_extended.py

    # Run one condition at a time
    python3 experiments/ablation/run_decoupled_ablation_extended.py --condition serial_only
    python3 experiments/ablation/run_decoupled_ablation_extended.py --condition schema_only

    # Run one dataset only
    python3 experiments/ablation/run_decoupled_ablation_extended.py --dataset wb_pop

    # Evaluate only (skip LightRAG runs, just recompute FC from existing graphs)
    python3 experiments/ablation/run_decoupled_ablation_extended.py --eval-only

Estimated LLM calls:
    Serial-only  wb_pop : 265 chunks × 1 call = ~265 calls
    Serial-only  wb_mat : 242 chunks × 1 call = ~242 calls
    Schema-only  wb_pop : ~265 raw chunks × 1 call = ~265 calls
    Schema-only  wb_mat : ~242 raw chunks × 1 call = ~242 calls
    Total new calls: ~1014 (at llm_model_max_async=5, ~5–10 min per condition)
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

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

# ── Dataset configurations ────────────────────────────────────────────────────
DATASETS = {
    "wb_pop": {
        "label": "WB Population",
        "csv": PROJECT_ROOT / "dataset" / "世界银行数据" / "population" / "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        # Pre-serialized SGE chunks (used by Serial-only condition)
        "chunks_dir": PROJECT_ROOT / "output" / "wb_population" / "chunks",
        # Full SGE output dir (for schema prompt, and for FC evaluation of Full SGE)
        "sge_output": PROJECT_ROOT / "output" / "wb_population",
        # Baseline output dir (for FC evaluation of Baseline)
        "baseline_output": PROJECT_ROOT / "output" / "baseline_wb_population",
        # Gold standard
        "gold": PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_population_v2.jsonl",
        # WB CSVs have 4 metadata rows before the header
        "skiprows": 4,
        "encoding": "utf-8-sig",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "csv": PROJECT_ROOT / "dataset" / "世界银行数据" / "maternal_mortality" / "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "chunks_dir": PROJECT_ROOT / "output" / "wb_maternal" / "chunks",
        "sge_output": PROJECT_ROOT / "output" / "wb_maternal",
        "baseline_output": PROJECT_ROOT / "output" / "baseline_wb_maternal",
        "gold": PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_maternal_v2.jsonl",
        "skiprows": 4,
        "encoding": "utf-8-sig",
    },
}

# ── LLM function ──────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Hash-based embedding (deterministic, no API calls) ───────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_sge_chunks(chunks_dir: Path) -> list[str]:
    """Load pre-serialized SGE chunks from disk."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")
    return [f.read_text(encoding="utf-8").strip() for f in chunk_files if f.read_text(encoding="utf-8").strip()]


def read_csv_as_text(ds_config: dict) -> str:
    """Read raw CSV into a plain text string (same view as baseline LightRAG)."""
    import pandas as pd
    from stage1.features import _detect_encoding

    csv_path = str(ds_config["csv"])
    skiprows = ds_config["skiprows"]

    encoding = _detect_encoding(csv_path)
    df = None
    for enc in [encoding, "utf-8-sig", "utf-8", "gbk"]:
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
    """Split text into chunks at line boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_size = 0
        current_lines.append(line)
        current_size += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def load_schema_prompt(sge_output_dir: Path) -> tuple[str, dict]:
    """Load SGE system prompt and extraction schema from an existing SGE output directory."""
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


def _graph_stats(graph_path: Path) -> dict:
    """Return node/edge counts for a graphml file, or zeros if missing."""
    if not graph_path.exists():
        return {"nodes": 0, "edges": 0}
    import networkx as nx
    G = nx.read_graphml(str(graph_path))
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


def _already_done(work_dir: Path, min_nodes: int = 5) -> bool:
    """Return True if a completed graph already exists in work_dir."""
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        return False
    stats = _graph_stats(graph_path)
    return stats["nodes"] > min_nodes


# ── Condition runners ─────────────────────────────────────────────────────────
async def run_serial_only(dataset_key: str, ds_config: dict) -> dict:
    """C4 condition: SGE chunks + LightRAG default prompt (no schema override)."""
    label = ds_config["label"]
    output_dir = PROJECT_ROOT / "output" / f"ablation_c4_serial_only_{dataset_key}"
    work_dir = output_dir / "lightrag_storage"

    if _already_done(work_dir):
        graph_path = work_dir / "graph_chunk_entity_relation.graphml"
        stats = _graph_stats(graph_path)
        print(f"\n  SKIP {label} serial-only (already done, {stats['nodes']} nodes)")
        return {"dataset": dataset_key, "label": label, "condition": "serial_only",
                "skipped": True, **stats}

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"SERIAL-ONLY (C4): {label}")
    print(f"{'=' * 60}")

    print("\n[Step 1] Loading SGE-serialized chunks...")
    chunks = load_sge_chunks(ds_config["chunks_dir"])
    print(f"  Loaded {len(chunks)} chunks")

    print("\n[Step 2] Running LightRAG with DEFAULT prompt + SGE chunks...")
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        llm_model_max_async=5,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    for i, chunk in enumerate(chunks, 1):
        if i == 1 or i % 50 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    await rag.finalize_storages()

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    stats = _graph_stats(graph_path)
    result = {
        "dataset": dataset_key, "label": label, "condition": "serial_only",
        "chunks": len(chunks), "timestamp": datetime.now().isoformat(), **stats,
    }
    (output_dir / "experiment_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    return result


async def run_schema_only(dataset_key: str, ds_config: dict) -> dict:
    """Schema-only condition: raw CSV text + SGE schema prompt (no SGE serialization)."""
    label = ds_config["label"]
    output_dir = PROJECT_ROOT / "output" / f"ablation_schema_only_{dataset_key}"
    work_dir = output_dir / "lightrag_storage"

    if _already_done(work_dir):
        graph_path = work_dir / "graph_chunk_entity_relation.graphml"
        stats = _graph_stats(graph_path)
        print(f"\n  SKIP {label} schema-only (already done, {stats['nodes']} nodes)")
        return {"dataset": dataset_key, "label": label, "condition": "schema_only",
                "skipped": True, **stats}

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"SCHEMA-ONLY: {label}")
    print(f"{'=' * 60}")

    print("\n[Step 1] Reading raw CSV as plain text...")
    csv_text = read_csv_as_text(ds_config)
    chunks = chunk_text(csv_text)
    print(f"  Total text length: {len(csv_text)} chars, {len(chunks)} chunks")

    print("\n[Step 2] Loading SGE schema prompt...")
    system_prompt_raw, schema = load_schema_prompt(ds_config["sge_output"])
    entity_types = schema.get("entity_types", ["Entity"])
    print(f"  Entity types: {entity_types}")
    print(f"  Prompt length: {len(system_prompt_raw)} chars")

    # Escape braces for LightRAG template compatibility (same logic as run_lightrag_integration.py)
    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    print(f"\n[Step 3] Running LightRAG with schema prompt + raw text...")
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped

    addon_params = {"language": "English", "entity_types": entity_types}

    try:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params=addon_params,
            llm_model_max_async=5,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()

        for i, chunk_item in enumerate(chunks, 1):
            if i == 1 or i % 50 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}] ({len(chunk_item)} chars)")
            await rag.ainsert(chunk_item)

        await rag.finalize_storages()
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    stats = _graph_stats(graph_path)
    result = {
        "dataset": dataset_key, "label": label, "condition": "schema_only",
        "chunks": len(chunks), "timestamp": datetime.now().isoformat(), **stats,
    }
    (output_dir / "experiment_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    return result


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_fc(graph_path: Path, gold_path: Path, label: str) -> dict:
    """Evaluate EC/FC for a given graph against gold standard."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_fact_coverage, check_entity_coverage,
    )

    if not graph_path.exists():
        print(f"  ERROR: graph not found at {graph_path}")
        return {"ec": 0.0, "fc": 0.0, "ec_matched": 0, "ec_total": 0,
                "fc_covered": 0, "fc_total": 0, "nodes": 0, "edges": 0}

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
        reasons: dict[str, int] = {}
        for nc in not_covered:
            r = nc.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"  Uncovered breakdown: {reasons}")

    return {
        "ec": round(ec, 4), "fc": round(fc, 4),
        "ec_matched": len(matched_entities), "ec_total": len(gold_entities),
        "fc_covered": len(covered), "fc_total": len(facts),
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
    }


def evaluate_all_conditions(dataset_key: str, ds_config: dict) -> list[dict]:
    """Evaluate all 4 conditions for one dataset."""
    label = ds_config["label"]
    gold_path = ds_config["gold"]
    timestamp = datetime.now().isoformat()

    condition_graphs = {
        "full_sge": ds_config["sge_output"] / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "serial_only": PROJECT_ROOT / "output" / f"ablation_c4_serial_only_{dataset_key}" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "schema_only": PROJECT_ROOT / "output" / f"ablation_schema_only_{dataset_key}" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": ds_config["baseline_output"] / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    }

    results = []
    for condition, graph_path in condition_graphs.items():
        print(f"\n[Evaluate] {label} — {condition}...")
        metrics = evaluate_fc(graph_path, gold_path, label)
        results.append({
            "dataset": dataset_key,
            "dataset_label": label,
            "condition": condition,
            **metrics,
            "timestamp": timestamp,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decoupled ablation extended to WB Population and WB Maternal",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        default=None,
        help="Run only this dataset (default: both)",
    )
    parser.add_argument(
        "--condition", "-c",
        choices=["serial_only", "schema_only", "both"],
        default="both",
        help="Which new conditions to run (default: both)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip LightRAG runs, only evaluate existing graphs across all 4 conditions",
    )
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    # Path validation
    for dk in datasets_to_run:
        ds = DATASETS[dk]
        for name, path in [
            ("CSV", ds["csv"]),
            ("chunks_dir", ds["chunks_dir"]),
            ("sge_output", ds["sge_output"]),
            ("gold", ds["gold"]),
        ]:
            if not path.exists():
                print(f"ERROR: {name} not found for {dk}: {path}", file=sys.stderr)
                sys.exit(1)

    all_results: list[dict] = []

    if not args.eval_only:
        for dk in datasets_to_run:
            ds = DATASETS[dk]

            if args.condition in ("serial_only", "both"):
                await run_serial_only(dk, ds)

            if args.condition in ("schema_only", "both"):
                await run_schema_only(dk, ds)

    # Evaluate all 4 conditions per dataset
    for dk in datasets_to_run:
        ds = DATASETS[dk]
        results = evaluate_all_conditions(dk, ds)
        all_results.extend(results)

    # Save combined results
    results_path = PROJECT_ROOT / "experiments" / "results" / "decoupled_ablation_extended_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary table
    print(f"\n{'=' * 80}")
    print("DECOUPLED ABLATION — WB Population & WB Maternal")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<25} {'Condition':<15} {'EC':>8} {'FC':>8} {'Nodes':>8} {'Edges':>8}")
    print("-" * 80)
    for r in all_results:
        cond_missing = r.get("fc", 0.0) == 0.0 and r.get("nodes", 0) == 0
        marker = "  (MISSING)" if cond_missing else ""
        print(
            f"{r['dataset_label']:<25} {r['condition']:<15} "
            f"{r['ec']:>8.4f} {r['fc']:>8.4f} "
            f"{r['nodes']:>8} {r['edges']:>8}{marker}"
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
