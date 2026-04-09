#!/usr/bin/env python3
"""
table_aware_baseline.py — Table-Aware Prompt Baseline for SGE-LightRAG.

Answers reviewer question M1: "Why not just tell the LLM about the table
structure via prompts instead of using the three-stage pipeline?"

This baseline:
  1. Feeds the SAME naive-serialized CSV text used by the LightRAG Baseline
  2. Adds a structured prompt hint to the entity_extraction_system_prompt
  3. Runs LightRAG entity extraction with this enhanced prompt
  4. Runs FC evaluation and compares: SGE vs Baseline vs Table-Aware

Key isolation: ONLY the system prompt is modified. No Stage 1 (topology
classification), Stage 2 (schema induction), or Stage 3 (structured
serialization) are used. This isolates "prompt hints" vs "full pipeline."

Usage:
    python3 evaluation/table_aware_baseline.py
    python3 evaluation/table_aware_baseline.py --dataset who
    python3 evaluation/table_aware_baseline.py --dataset who --fresh
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
from lightrag.prompt import PROMPTS

from evaluation.evaluate_coverage import load_gold, load_graph, check_entity_coverage, check_fact_coverage

# ---------------------------------------------------------------------------
# API config (same as other scripts)
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
        "label": "WHO Life Expectancy (EN, Type-II)",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "chunks_dir": "output/who_life_expectancy/chunks",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "output_dir": "output/table_aware_who",
        "table_hint": (
            "IMPORTANT: The text below was extracted from a statistical CSV table.\n"
            "- The table has columns: Country Code, Country Name, Indicator Code, and year columns (2000-2021)\n"
            "- Each row represents one country's time-series data for a health indicator (Life expectancy at birth)\n"
            "- Values in each year column are numeric measurements (years of life expectancy)\n"
            "- When extracting entities, treat each country as a primary entity\n"
            "- Associate numeric values with their corresponding country and year\n"
            "- Preserve the subject-time-value binding (e.g., \"AFG, 2021, 59.13\")\n"
            "- Do NOT create separate year nodes; embed year-value data in the country entity or its edges\n"
        ),
        "language": "English",
    },
    "who_strong": {
        "label": "WHO Life Expectancy — STRONG prompt (EN, Type-II)",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "chunks_dir": "output/who_life_expectancy/chunks",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "output_dir": "output/table_aware_who_strong",
        "table_hint": (
            "---TABLE STRUCTURE---\n"
            "This text comes from WHO Life Expectancy Dataset (Type-II Time-Series Table).\n"
            "\n"
            "COLUMNS (6 total):\n"
            "  1. Country Name (string, metadata) — Human-readable country label\n"
            "  2. Country Code (string, ENTITY KEY) — ISO 3-letter code (AFG, ALB, AGO, ...)\n"
            "  3. Indicator Name (string, metadata) — \"Life expectancy at birth (years)\"\n"
            "  4. Indicator Code (string, metadata) — WHOSIS_000001\n"
            "  5-26. Year columns: 2000, 2001, ..., 2021 (numeric values, MEASUREMENTS)\n"
            "\n"
            "DATA EXAMPLE (first 3 rows):\n"
            "  AFG | Afghanistan | 2000=53.82, 2005=56.79, 2010=62.32, 2015=63.16, 2019=63.52, 2021=59.13\n"
            "  ALB | Albania | 2000=73.68, 2005=75.40, 2010=76.72, 2015=77.62, 2019=78.73, 2021=76.39\n"
            "  AGO | Angola | 2000=49.37, 2005=53.58, 2010=57.65, 2015=60.32, 2019=61.64, 2021=62.13\n"
            "\n"
            "---EXTRACTION SCHEMA---\n"
            "Entity Types:\n"
            "  - Country_Code: ISO 3-letter country code (e.g., AFG, ALB) — ONE per row\n"
            "  - YearValue: Temporal measurement (e.g., \"2000=53.82_years\") — ONE per year with data\n"
            "\n"
            "Relationship Type:\n"
            "  - HAS_MEASUREMENT: Country_Code --[HAS_MEASUREMENT]--> YearValue\n"
            "\n"
            "---CRITICAL RULES---\n"
            "1. Extract EXACTLY ONE Country_Code entity per row (from \"Country Code\" column)\n"
            "2. For EACH year column (2000-2021) with a non-empty numeric value:\n"
            "   - Create ONE YearValue entity with name format: YEAR=VALUE_years\n"
            "   - Example: 2000=53.82_years NOT just 53.82 or 2000\n"
            "   - Create ONE HAS_MEASUREMENT relation linking country to year-value\n"
            "3. Skip empty cells and non-numeric values\n"
            "4. DO NOT create Year entities, Indicator entities, or Country Name entities\n"
            "5. DO NOT create separate nodes for country name — use only Country_Code\n"
            "6. Use exact naming format for YearValue (year=value_unit) to enable precise matching\n"
        ),
        "language": "English",
    },
}

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
    """Deterministic hash-based embedding fallback."""
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
# Prompt construction
# ---------------------------------------------------------------------------

TABLE_AWARE_PROMPT_WRAPPER = """\
{table_hint}
---

{original_prompt}"""


def build_table_aware_prompt(original_prompt: str, table_hint: str) -> str:
    """
    Prepend a table-structure hint to the existing LightRAG system prompt.

    This is the key isolation: we inject structural knowledge about the CSV
    format without running SGE's topology/schema/serialization pipeline.

    Parameters
    ----------
    original_prompt : LightRAG's default entity_extraction_system_prompt
    table_hint      : dataset-specific structural description

    Returns
    -------
    str — the enhanced system prompt with table hint prepended
    """
    return TABLE_AWARE_PROMPT_WRAPPER.format(
        table_hint=table_hint.strip(),
        original_prompt=original_prompt,
    )


# ---------------------------------------------------------------------------
# LightRAG runner
# ---------------------------------------------------------------------------

async def run_table_aware_lightrag(
    chunks: list[str],
    working_dir: Path,
    table_hint: str,
    language: str,
    fresh: bool,
) -> dict:
    """
    Run LightRAG with an enhanced system prompt (table-aware hint).

    Parameters
    ----------
    chunks      : list of text chunks (same naive serialization as baseline)
    working_dir : LightRAG storage directory
    table_hint  : table-structure description to prepend to system prompt
    language    : LLM response language
    fresh       : if True, delete existing graph before running

    Returns
    -------
    dict with graph stats
    """
    working_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        graph_file = working_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            graph_file.unlink()
            print("  Removed existing graph for fresh run.")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    enhanced_prompt = build_table_aware_prompt(original_prompt, table_hint)

    print(f"  Enhanced prompt length: {len(enhanced_prompt)} chars (original: {len(original_prompt)})")

    # Override the global PROMPTS dict — same approach as SGE's integrator.py
    PROMPTS["entity_extraction_system_prompt"] = enhanced_prompt

    try:
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params={"language": language},
            llm_model_max_async=5,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()

        insert_chunks = [f"[TABLE_AWARE]\n{c}" for c in chunks]
        print(f"  Inserting {len(insert_chunks)} chunks...")

        for i, chunk in enumerate(insert_chunks, 1):
            print(f"  [{i}/{len(insert_chunks)}] ({len(chunk)} chars)")
            await rag.ainsert(chunk)

        await rag.finalize_storages()

    finally:
        # Always restore original prompt — same safety pattern as run_lightrag_integration.py
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats: dict = {
        "working_dir": str(working_dir),
        "chunks_inserted": len(insert_chunks),
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
# FC evaluation
# ---------------------------------------------------------------------------

def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    """
    Run FC evaluation on a single graph against a gold standard.

    Parameters
    ----------
    graph_path : path to the .graphml file
    gold_path  : path to the gold .jsonl file
    label      : human-readable label for logging

    Returns
    -------
    dict with ec, fc, matched/total counts
    """
    if not Path(graph_path).exists():
        print(f"  [warn] Graph not found: {graph_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "graph_not_found"}
    if not Path(gold_path).exists():
        print(f"  [warn] Gold not found: {gold_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "gold_not_found"}

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
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_dataset(dataset_key: str, fresh: bool) -> dict:
    """Build the table-aware graph and evaluate FC for one dataset."""
    if dataset_key not in DATASETS:
        print(f"ERROR: unknown dataset '{dataset_key}'. Available: {list(DATASETS.keys())}",
              file=sys.stderr)
        sys.exit(1)

    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"TABLE-AWARE BASELINE: {cfg['label']}")
    print(f"{'='*60}")

    # Resolve paths
    chunks_dir = PROJECT_ROOT / cfg["chunks_dir"]
    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    lightrag_storage = output_dir / "lightrag_storage"

    sge_graph_path = str(PROJECT_ROOT / cfg["sge_graph"])
    baseline_graph_path = str(PROJECT_ROOT / cfg["baseline_graph"])
    table_aware_graph_path = str(lightrag_storage / "graph_chunk_entity_relation.graphml")

    # Load chunks from existing SGE output (same naive-serialized text as baseline)
    if not chunks_dir.exists():
        print(f"ERROR: chunks directory not found: {chunks_dir}", file=sys.stderr)
        print("  Run the SGE pipeline first: python3 scripts/runners/run_lightrag_integration.py",
              file=sys.stderr)
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    if not chunk_files:
        print(f"ERROR: no .txt chunks found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"\nLoaded {len(chunks)} chunks from {chunks_dir}")

    # Build table-aware graph
    print(f"\n[Step 1] Building Table-Aware Graph...")
    print(f"  Output: {lightrag_storage}")
    stats = await run_table_aware_lightrag(
        chunks=chunks,
        working_dir=lightrag_storage,
        table_hint=cfg["table_hint"],
        language=cfg["language"],
        fresh=fresh,
    )

    # Save run stats
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "run_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run FC evaluation on all three systems
    print(f"\n[Step 2] FC Evaluation...")
    table_aware_result = evaluate_graph_fc(table_aware_graph_path, gold_path, "Table-Aware")
    sge_result         = evaluate_graph_fc(sge_graph_path, gold_path, "SGE")
    baseline_result    = evaluate_graph_fc(baseline_graph_path, gold_path, "Baseline")

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "timestamp": datetime.now().isoformat(),
        "table_aware": table_aware_result,
        "sge": sge_result,
        "baseline": baseline_result,
    }


def print_comparison(result: dict) -> None:
    """Print a formatted three-way comparison table."""
    ta  = result["table_aware"]
    sge = result["sge"]
    bl  = result["baseline"]

    print(f"\n{'='*60}")
    print(f"THREE-WAY COMPARISON: {result['label']}")
    print(f"{'='*60}")
    print(f"{'Method':<20} | {'EC':>6} | {'FC':>6} | {'Nodes':>6} | {'Edges':>6}")
    print(f"{'-'*60}")

    for r, name in [(sge, "SGE"), (ta, "Table-Aware"), (bl, "Baseline")]:
        if r.get("error"):
            print(f"  {name:<18} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}  [{r['error']}]")
        else:
            print(f"  {name:<18} | {r['ec']:>6.4f} | {r['fc']:>6.4f} | "
                  f"{r.get('node_count', '?'):>6} | {r.get('edge_count', '?'):>6}")

    print(f"{'='*60}")

    # Interpretation summary
    sge_fc = sge.get("fc", 0.0)
    ta_fc  = ta.get("fc", 0.0)
    bl_fc  = bl.get("fc", 0.0)

    if not ta.get("error"):
        ta_vs_base = ta_fc - bl_fc
        sge_vs_ta  = sge_fc - ta_fc
        print(f"\n[Interpretation]")
        print(f"  Table-Aware vs Baseline: {ta_vs_base:+.4f} "
              f"({'improvement' if ta_vs_base > 0 else 'no improvement'})")
        print(f"  SGE vs Table-Aware:      {sge_vs_ta:+.4f} "
              f"({'SGE adds value' if sge_vs_ta > 0.05 else 'marginal difference'})")

        if sge_vs_ta > 0.05:
            print(f"\n  Conclusion: Prompt hints alone are insufficient (Δ FC={sge_vs_ta:.4f}).")
            print(f"  SGE's three-stage pipeline provides structural gains beyond prompt engineering.")
        elif ta_vs_base > 0.05:
            print(f"\n  Conclusion: Table-Aware prompt helps but SGE achieves parity.")
            print(f"  Both approaches improve over baseline.")
        else:
            print(f"\n  Conclusion: Prompt hints provide no significant improvement over baseline.")


def main():
    parser = argparse.ArgumentParser(
        description="Table-aware prompt baseline for SGE-LightRAG comparison."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="who",
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate (default: who)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing table-aware graph before running (force rebuild)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_dataset(args.dataset, args.fresh))

    print_comparison(result)

    # Save results
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "table_aware_baseline_results.json"

    # Merge with any existing results (preserving other datasets)
    existing: list = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, IOError):
            existing = []

    # Replace entry for this dataset if present, otherwise append
    updated = [r for r in existing if r.get("dataset") != args.dataset]
    updated.append(result)

    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
