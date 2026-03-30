#!/usr/bin/env python3
"""
run_c4_serialization_only.py — C4 condition: Serialization-only ablation.

SGE-serialized chunks + LightRAG default prompt (NO schema injection).
Completes the 2x2 orthogonal design for Table 5:
  Full SGE:       SGE serialization + schema prompt
  Schema-only:    raw CSV text      + schema prompt
  Baseline:       raw CSV text      + default prompt
  C4 (this):      SGE serialization + default prompt   ← MISSING CELL

For 3 datasets: WHO Life Expectancy, WB Child Mortality, Inpatient 2023.

Usage:
    python3 experiments/run_c4_serialization_only.py
    python3 experiments/run_c4_serialization_only.py --dataset who
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

API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

DATASETS = {
    "who": {
        "label": "WHO Life Expectancy",
        "chunks_dir": PROJECT_ROOT / "output" / "who_life_expectancy" / "chunks",
        "gold": PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_v2.jsonl",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "chunks_dir": PROJECT_ROOT / "output" / "wb_child_mortality" / "chunks",
        "gold": PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_child_mortality_v2.jsonl",
    },
    "inpatient": {
        "label": "Inpatient 2023",
        "chunks_dir": PROJECT_ROOT / "output" / "inpatient_2023" / "chunks",
        "gold": PROJECT_ROOT / "evaluation" / "gold" / "gold_inpatient_2023.jsonl",
    },
}


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


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


def load_sge_chunks(chunks_dir: Path) -> list[str]:
    """Load pre-serialized SGE chunks from disk."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")
    chunks = []
    for f in chunk_files:
        text = f.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(text)
    return chunks


async def run_c4(dataset_key: str, ds_config: dict) -> dict:
    """Run C4 condition: SGE chunks + default LightRAG prompt."""
    label = ds_config["label"]
    output_dir = PROJECT_ROOT / "output" / f"ablation_c4_serial_only_{dataset_key}"
    work_dir = output_dir / "lightrag_storage"

    # Skip if already completed
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        if G.number_of_nodes() > 5:
            print(f"\n  SKIP {label} (already done, {G.number_of_nodes()} nodes)")
            return {
                "dataset": dataset_key,
                "label": label,
                "condition": "serialization_only",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "skipped": True,
            }

    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"C4 SERIALIZATION-ONLY: {label}")
    print(f"{'=' * 60}")

    # Load SGE-serialized chunks
    print("\n[Step 1] Loading SGE-serialized chunks...")
    chunks = load_sge_chunks(ds_config["chunks_dir"])
    print(f"  Loaded {len(chunks)} chunks")

    # Run LightRAG with DEFAULT prompt (no schema override)
    print(f"\n[Step 2] Running LightRAG with default prompt + SGE chunks...")
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
        if i % 20 == 0 or i == len(chunks) or i == 1:
            print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    await rag.finalize_storages()

    stats = {
        "dataset": dataset_key,
        "label": label,
        "condition": "serialization_only",
        "chunks": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"\n  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def evaluate_fc(dataset_key: str, ds_config: dict) -> dict:
    """Evaluate EC/FC for the C4 graph."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_fact_coverage, check_entity_coverage,
    )

    graph_path = (
        PROJECT_ROOT / "output" / f"ablation_c4_serial_only_{dataset_key}"
        / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    )
    gold_path = ds_config["gold"]

    if not graph_path.exists():
        print(f"  ERROR: graph not found at {graph_path}")
        return {"ec": 0.0, "fc": 0.0}

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


async def main():
    parser = argparse.ArgumentParser(
        description="C4 ablation: SGE serialization + default prompt",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        default=None,
        help="Run only this dataset (default: all 3)",
    )
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())
    all_results = []

    for dk in datasets_to_run:
        ds = DATASETS[dk]
        if not ds["chunks_dir"].exists():
            print(f"ERROR: Chunks dir not found: {ds['chunks_dir']}", file=sys.stderr)
            sys.exit(1)
        if not ds["gold"].exists():
            print(f"ERROR: Gold standard not found: {ds['gold']}", file=sys.stderr)
            sys.exit(1)

    for dk in datasets_to_run:
        ds = DATASETS[dk]
        stats = await run_c4(dk, ds)

        print(f"\n[Evaluate] {ds['label']}...")
        eval_result = evaluate_fc(dk, ds)

        result = {
            "dataset": dk,
            "dataset_label": ds["label"],
            "condition": "serialization_only",
            **eval_result,
            "timestamp": stats.get("timestamp", datetime.now().isoformat()),
            "chunks_inserted": stats.get("chunks", 0),
        }
        all_results.append(result)

    results_path = PROJECT_ROOT / "experiments" / "results" / "c4_serialization_only_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 70}")
    print("C4 SERIALIZATION-ONLY RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<25} {'EC':>8} {'FC':>8} {'Nodes':>8} {'Edges':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset_label']:<25} {r['ec']:>8.4f} {r['fc']:>8.4f} {r['nodes']:>8} {r['edges']:>8}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
