#!/usr/bin/env python3
"""
row_local_baseline.py — Header-Aware Row-Local Prompt Baseline.

Tests whether per-row serialization with full headers (but NO schema
constraints or dynamic schema induction) is sufficient to improve
cell-fact binding fidelity.

Key isolation:
  - Each chunk = one CSV row + all column headers
  - LightRAG DEFAULT system prompt (no schema hint)
  - No Stage 1/2/3 pipeline components
  - Contrast with table_aware_baseline (naive chunks + prompt hint)
    and fixed_stv_baseline (per-row + fixed schema)

Usage:
    python3 evaluation/row_local_baseline.py
    python3 evaluation/row_local_baseline.py --dataset who
    python3 evaluation/row_local_baseline.py --dataset who --fresh
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)
from stage1.preprocessor import preprocess_csv

# ---------------------------------------------------------------------------
# API config (same as other baselines)
# ---------------------------------------------------------------------------

API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "output_dir": "output/row_local_who",
        "language": "English",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv_path": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "output_dir": "output/row_local_wb_cm",
        "language": "English",
    },
    "wb_pop": {
        "label": "WB Population",
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "output_dir": "output/row_local_wb_pop",
        "language": "English",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "csv_path": "dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "output_dir": "output/row_local_wb_mat",
        "language": "English",
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "output_dir": "output/row_local_inpatient",
        "language": "English",
    },
    "fortune500": {
        "label": "Fortune 500 Revenue",
        "csv_path": "dataset/non_gov/fortune500_revenue.csv",
        "gold": "evaluation/gold/gold_fortune500_revenue.jsonl",
        "output_dir": "output/row_local_fortune500",
        "language": "English",
    },
    "the": {
        "label": "THE University Ranking",
        "csv_path": "dataset/non_gov/the_university_ranking.csv",
        "gold": "evaluation/gold/gold_the_university_ranking.jsonl",
        "output_dir": "output/row_local_the",
        "language": "English",
    },
}

# ---------------------------------------------------------------------------
# Row-local chunk generation
# ---------------------------------------------------------------------------

def generate_row_local_chunks(csv_path: str) -> list[str]:
    """
    Generate one text chunk per CSV row, with full headers prepended.

    Each chunk looks like:
        Headers: Country Code, Country Name, Indicator Code, 2000, 2001, ...
        Row 5: AFG, Afghanistan, WHOSIS_000001, 53.82, 55.25, ...

    No schema constraints, no structure interpretation — just raw headers + values.
    Processes ALL rows (gold standard evaluator only checks target entities).
    """
    df, _ = preprocess_csv(str(PROJECT_ROOT / csv_path))

    headers_str = ", ".join(str(c) for c in df.columns)
    chunks = []

    for idx, (_, row) in enumerate(df.iterrows()):
        values = ", ".join(str(v) for v in row.values)
        chunk = (
            f"Headers: {headers_str}\n"
            f"Row {idx + 1}: {values}\n"
            f"\n"
            f"Extract all entities and relationships from the data row above. "
            f"Each column header describes what the corresponding value represents."
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# LLM and embedding functions
# ---------------------------------------------------------------------------

async def llm_model_func(
    prompt,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
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
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=safe_embedding_func,
)


# ---------------------------------------------------------------------------
# LightRAG runner (default prompt, row-local chunks)
# ---------------------------------------------------------------------------

async def run_row_local_lightrag(
    chunks: list[str],
    working_dir: Path,
    language: str,
    fresh: bool,
) -> dict:
    """Run LightRAG with DEFAULT prompt on row-local chunks."""
    working_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        graph_file = working_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            graph_file.unlink()
            print("  Removed existing graph for fresh run.")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": language},
        llm_model_max_async=4,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
        force_llm_summary_on_merge=999,
    )
    await rag.initialize_storages()

    print(f"  Inserting {len(chunks)} row-local chunks...")
    for i, chunk in enumerate(chunks, 1):
        if i <= 3 or i % 25 == 0:
            print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    await rag.finalize_storages()

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats: dict = {
        "working_dir": str(working_dir),
        "chunks_inserted": len(chunks),
        "graph_file_exists": graph_path.exists(),
    }

    if graph_path.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(str(graph_path))
            stats["node_count"] = G.number_of_nodes()
            stats["edge_count"] = G.number_of_edges()
            print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges")
        except Exception as e:
            print(f"  [warn] Could not parse graphml: {e}", file=sys.stderr)

    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    """Run FC/EC evaluation on a graph."""
    if not Path(graph_path).exists():
        print(f"  [warn] Graph not found: {graph_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  [{label}] EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
          f"FC={fc:.4f} ({len(covered)}/{len(facts)})")

    return {
        "label": label,
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "ec_matched": len(matched_entities),
        "ec_total": len(gold_entities),
        "fc_covered": len(covered),
        "fc_total": len(facts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_dataset(dataset_key: str, fresh: bool) -> dict:
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"ROW-LOCAL BASELINE: {cfg['label']}")
    print(f"{'='*60}")

    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    lightrag_storage = output_dir / "lightrag_storage"

    # Generate row-local chunks from CSV
    print(f"\n[Step 1] Generating row-local chunks...")
    chunks = generate_row_local_chunks(cfg["csv_path"])
    print(f"  Generated {len(chunks)} chunks")

    # Run LightRAG with default prompt
    print(f"\n[Step 2] Running LightRAG (default prompt)...")
    stats = await run_row_local_lightrag(
        chunks=chunks,
        working_dir=lightrag_storage,
        language=cfg["language"],
        fresh=fresh,
    )

    # Evaluate
    print(f"\n[Step 3] FC/EC Evaluation...")
    graph_path = str(lightrag_storage / "graph_chunk_entity_relation.graphml")
    result = evaluate_graph_fc(graph_path, gold_path, "Row-Local")

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "baseline_type": "row_local",
        "description": "Per-row serialization with full headers, LightRAG default prompt, no schema",
        "stats": stats,
        "evaluation": result,
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(datasets: list[str], fresh: bool) -> None:
    output_path = PROJECT_ROOT / "evaluation" / "results" / "row_local_baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)

    for ds in datasets:
        result = await run_dataset(ds, fresh)
        all_results[ds] = result

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Dataset':<25s} {'EC':>8s} {'FC':>8s}")
    print(f"{'-'*60}")
    for ds, r in all_results.items():
        ev = r.get("evaluation", {})
        print(f"{r['label']:<25s} {ev.get('ec', 0):.4f}   {ev.get('fc', 0):.4f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-local prompt baseline")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Run single dataset (default: all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing graph before running")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    asyncio.run(main_async(datasets, args.fresh))


if __name__ == "__main__":
    main()
