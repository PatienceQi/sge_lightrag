#!/usr/bin/env python3
"""
fixed_stv_baseline.py — Fixed Subject/Time/Value Relation Set Baseline.

Tests whether SGE's dynamic schema induction (Stage 1+2) adds value
beyond having a fixed, generic schema template.

Design:
  - Uses SGE's row-level serialization (Stage 3 serializer) for per-row chunks
  - Applies a HARDCODED generic schema: entity_types=["Country","StatValue"],
    relation_types=["HAS_VALUE_IN_YEAR"] — no Stage 1/2 pipeline
  - Injects via prompt_injector.py (same injection mechanism as SGE)

This isolates the contribution of DYNAMIC schema induction vs STATIC schema.

Usage:
    python3 evaluation/fixed_stv_baseline.py
    python3 evaluation/fixed_stv_baseline.py --dataset who
    python3 evaluation/fixed_stv_baseline.py --dataset who --fresh
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
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
from lightrag.prompt import PROMPTS

from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)
from stage1.preprocessor import preprocess_csv
from stage3.prompt_injector import generate_system_prompt

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

# ---------------------------------------------------------------------------
# Fixed generic schema (one-size-fits-all, no dynamic induction)
# ---------------------------------------------------------------------------

FIXED_SCHEMA_TYPE_II = {
    "table_type": "Time-Series-Matrix",
    "entity_types": ["Country", "StatValue"],
    "relation_types": ["HAS_VALUE_IN_YEAR"],
    "column_roles": {},  # not used — serialization is manual
    "extraction_constraints": {
        "max_entity_types": 2,
        "one_entity_per_row": True,
        "instructions": [
            "Each row represents ONE subject entity (a country or region).",
            "Year columns contain numeric measurements for that subject.",
            "Create one StatValue per non-empty year-value pair.",
            "Relation: Subject --[HAS_VALUE_IN_YEAR]--> StatValue.",
            "Include the year in the relation keywords.",
        ],
    },
    "prompt_context": (
        "This is a statistical CSV table. Each row is a subject entity, "
        "and year columns contain time-series measurements."
    ),
}

FIXED_SCHEMA_TYPE_III = {
    "table_type": "Hierarchical-Hybrid",
    "entity_types": ["Category", "StatValue"],
    "relation_types": ["HAS_VALUE", "HAS_SUB_ITEM"],
    "column_roles": {},
    "extraction_constraints": {
        "max_entity_types": 2,
        "instructions": [
            "Rows have hierarchical categories (composite key columns).",
            "Year columns contain numeric measurements.",
            "Create Category entities for key columns.",
            "Create StatValue entities for year-value pairs.",
            "Relation: Category --[HAS_VALUE]--> StatValue for values.",
            "Relation: Parent --[HAS_SUB_ITEM]--> Child for hierarchy.",
        ],
    },
    "prompt_context": (
        "This is a hierarchical statistical table with composite keys "
        "and year-based value columns."
    ),
}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "output_dir": "output/fixed_stv_who",
        "schema": FIXED_SCHEMA_TYPE_II,
        "language": "English",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv_path": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "output_dir": "output/fixed_stv_wb_cm",
        "schema": FIXED_SCHEMA_TYPE_II,
        "language": "English",
    },
    "wb_pop": {
        "label": "WB Population",
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "output_dir": "output/fixed_stv_wb_pop",
        "schema": FIXED_SCHEMA_TYPE_II,
        "language": "English",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "csv_path": "dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "output_dir": "output/fixed_stv_wb_mat",
        "schema": FIXED_SCHEMA_TYPE_II,
        "language": "English",
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "output_dir": "output/fixed_stv_inpatient",
        "schema": FIXED_SCHEMA_TYPE_III,
        "language": "English",
    },
    "fortune500": {
        "label": "Fortune 500 Revenue",
        "csv_path_abs": "/Users/qipatience/Desktop/SGE/dataset/non_gov/fortune500_revenue.csv",
        "gold": "evaluation/gold/gold_fortune500_revenue.jsonl",
        "output_dir": "output/fixed_stv_fortune500",
        "schema": FIXED_SCHEMA_TYPE_II,
        "language": "English",
    },
    "the": {
        "label": "THE University Ranking",
        "csv_path_abs": "/Users/qipatience/Desktop/SGE/dataset/non_gov/the_university_ranking.csv",
        "gold": "evaluation/gold/gold_the_university_ranking.jsonl",
        "output_dir": "output/fixed_stv_the",
        "schema": FIXED_SCHEMA_TYPE_III,
        "language": "English",
    },
}


# ---------------------------------------------------------------------------
# Row-level serialization (simple, no Stage 2 column_roles)
# ---------------------------------------------------------------------------

def serialize_rows_simple(csv_path: str, abs_path: Optional[str] = None) -> list[str]:
    """
    Simple per-row serialization: Header: Value pairs, one chunk per row.

    Unlike SGE's serializer.py which uses column_roles for type-aware formatting,
    this just dumps each row as key-value pairs.
    Processes ALL rows (gold standard evaluator only checks target entities).
    """
    resolved = abs_path if abs_path else str(PROJECT_ROOT / csv_path)
    df, _ = preprocess_csv(resolved)

    chunks = []
    for _, row in df.iterrows():
        lines = []
        for col in df.columns:
            val = row[col]
            if str(val).strip() and str(val).lower() not in ("nan", ""):
                lines.append(f"{col}: {val}")
        chunks.append("\n".join(lines))

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
# LightRAG runner with fixed schema injection
# ---------------------------------------------------------------------------

async def run_fixed_stv_lightrag(
    chunks: list[str],
    working_dir: Path,
    schema: dict,
    language: str,
    fresh: bool,
) -> dict:
    """Run LightRAG with fixed schema prompt on per-row chunks."""
    if fresh:
        import shutil
        if working_dir.exists():
            shutil.rmtree(str(working_dir))
            print("  Removed entire storage directory for fresh run.")
    working_dir.mkdir(parents=True, exist_ok=True)

    # Generate system prompt from fixed schema (same mechanism as SGE)
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    fixed_prompt = generate_system_prompt(schema, language=language)
    # Escape literal braces so LightRAG's .format(**context_base) doesn't break
    escaped_prompt = fixed_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped_prompt = escaped_prompt.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped_prompt

    print(f"  Fixed schema prompt length: {len(escaped_prompt)} chars")

    try:
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

        print(f"  Inserting {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            if i <= 3 or i % 25 == 0:
                print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
            await rag.ainsert(chunk)

        await rag.finalize_storages()

    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

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
    print(f"FIXED S/T/V BASELINE: {cfg['label']}")
    print(f"{'='*60}")

    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    lightrag_storage = output_dir / "lightrag_storage"

    # Generate simple per-row chunks
    print(f"\n[Step 1] Row-level serialization (no column_roles)...")
    chunks = serialize_rows_simple(
        csv_path=cfg.get("csv_path", ""),
        abs_path=cfg.get("csv_path_abs"),
    )
    print(f"  Generated {len(chunks)} chunks")

    # Run LightRAG with fixed schema
    print(f"\n[Step 2] Running LightRAG (fixed S/T/V schema)...")
    stats = await run_fixed_stv_lightrag(
        chunks=chunks,
        working_dir=lightrag_storage,
        schema=cfg["schema"],
        language=cfg["language"],
        fresh=fresh,
    )

    # Evaluate
    print(f"\n[Step 3] FC/EC Evaluation...")
    graph_path = str(lightrag_storage / "graph_chunk_entity_relation.graphml")
    result = evaluate_graph_fc(graph_path, gold_path, "Fixed-STV")

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "baseline_type": "fixed_stv",
        "description": (
            "Per-row serialization + fixed generic schema "
            "(Country/StatValue, HAS_VALUE_IN_YEAR), no dynamic induction"
        ),
        "schema_used": cfg["schema"]["table_type"],
        "stats": stats,
        "evaluation": result,
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(datasets: list[str], fresh: bool) -> None:
    output_path = PROJECT_ROOT / "evaluation" / "results" / "fixed_stv_baseline_results.json"
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

    print(f"\n{'='*60}")
    print(f"{'Dataset':<25s} {'EC':>8s} {'FC':>8s}")
    print(f"{'-'*60}")
    for ds, r in all_results.items():
        ev = r.get("evaluation", {})
        print(f"{r['label']:<25s} {ev.get('ec', 0):.4f}   {ev.get('fc', 0):.4f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed S/T/V baseline")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Run single dataset (default: all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing graph before running")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    asyncio.run(main_async(datasets, args.fresh))


if __name__ == "__main__":
    main()
