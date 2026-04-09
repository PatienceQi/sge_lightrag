#!/usr/bin/env python3
"""
run_mechanism_probes.py — Probing experiment: Token Alignment vs Semantic Understanding.

2x2 factorial design:
  Factor 1: Entity type name alignment (token-aligned vs misaligned)
  Factor 2: Semantic description accuracy (correct vs misleading)

Conditions:
  AX  : Country_Code + correct description  (baseline, FC=1.000 known)
  BX-1: Nation_Identifier + correct description  (near synonym)
  BX-2: GeopoliticalEntity + correct description  (distant synonym)
  BX-3: RowKey_Alpha3 + correct description + functional explanation
  AY-1: Country_Code + "represents measurement year"  (mild misdirection)
  AY-2: Country_Code + "represents disease category"  (strong misdirection)
  AY-3: Country_Code + "extract Indicator Code instead"  (instruction conflict)
  BY  : Nation_Identifier + "represents disease category"  (double mismatch)
  C1  : Country_Code + RECORDS_MEASUREMENT relation rename

All conditions use SGE-serialized WHO chunks + modified Schema → LightRAG → FC.

Usage:
    python3 experiments/ablation/run_mechanism_probes.py
    python3 experiments/ablation/run_mechanism_probes.py --condition BX-1
    python3 experiments/ablation/run_mechanism_probes.py --dry-run
"""

from __future__ import annotations

import asyncio
import copy
import json
import hashlib
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS

from stage3.prompt_injector import generate_system_prompt
from evaluation.config import API_KEY, BASE_URL, MODEL, EMBED_DIM
from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHO_SGE_OUTPUT = PROJECT_ROOT / "output" / "who_life_expectancy"
WHO_GOLD = PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_v2.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Baseline WHO Schema (from extraction_schema.json)
# ---------------------------------------------------------------------------

def _load_baseline_schema() -> dict:
    """Load the original WHO SGE extraction schema."""
    schema_path = WHO_SGE_OUTPUT / "extraction_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _build_schema_variant(
    entity_type: str,
    relation_type: str,
    entity_extraction_template: str,
    relation_extraction_template: str,
    extraction_constraints: list[str],
) -> dict:
    """Build a schema variant preserving column_roles and time headers from baseline."""
    base = _load_baseline_schema()
    return {
        "table_type": base["table_type"],
        "entity_types": [entity_type],
        "relation_types": [relation_type],
        "column_roles": base["column_roles"],
        "entity_extraction_template": entity_extraction_template,
        "relation_extraction_template": relation_extraction_template,
        "extraction_constraints": extraction_constraints,
        "parsed_time_headers": base["parsed_time_headers"],
        "time_dimension": base["time_dimension"],
    }


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

def build_conditions() -> dict[str, dict]:
    """Build all probing conditions. Each returns a Schema dict."""

    conditions = {}

    # --- BX-1: Nation_Identifier (near synonym, semantically correct) ---
    conditions["BX-1"] = _build_schema_variant(
        entity_type="Nation_Identifier",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "Nation_Identifier entity node."
        ),
        relation_extraction_template=(
            "For each time column, create: (Nation_Identifier) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Nation_Identifier entity identified by 'Country Code'.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- BX-2: GeopoliticalEntity (distant synonym, semantically correct) ---
    conditions["BX-2"] = _build_schema_variant(
        entity_type="GeopoliticalEntity",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "GeopoliticalEntity entity node."
        ),
        relation_extraction_template=(
            "For each time column, create: (GeopoliticalEntity) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one GeopoliticalEntity entity identified by 'Country Code'.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- BX-3: RowKey_Alpha3 (alien name, functional description) ---
    conditions["BX-3"] = _build_schema_variant(
        entity_type="RowKey_Alpha3",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "RowKey_Alpha3 entity node. RowKey_Alpha3 represents the primary "
            "subject identifier in each data row (a 3-letter ISO code)."
        ),
        relation_extraction_template=(
            "For each time column, create: (RowKey_Alpha3) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one RowKey_Alpha3 entity identified by 'Country Code'.",
            "RowKey_Alpha3 is the primary row key (3-letter ISO code, e.g. AFG, CHN, USA).",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- AY-1: Country_Code + mild misdirection ("measurement year") ---
    conditions["AY-1"] = _build_schema_variant(
        entity_type="Country_Code",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "Country_Code entity node. Country_Code represents the "
            "measurement year or time period identifier."
        ),
        relation_extraction_template=(
            "For each time column, create: (Country_Code) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Note: Country_Code is the temporal dimension, not the "
            "geographic dimension. Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Country_Code entity identified by 'Country Code'. "
            "Country_Code is the measurement year.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- AY-2: Country_Code + strong misdirection ("disease category") ---
    conditions["AY-2"] = _build_schema_variant(
        entity_type="Country_Code",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "Country_Code entity node. Country_Code represents the "
            "disease classification category (ICD code)."
        ),
        relation_extraction_template=(
            "For each time column, create: (Country_Code) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Country_Code entities are disease categories, not geographic "
            "entities. Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Country_Code entity identified by 'Country Code'. "
            "Country_Code is a disease category.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
            "IMPORTANT: Country_Code values like AFG, CHN, USA are disease "
            "classification codes, not country identifiers.",
        ],
    )

    # --- AY-3: Country_Code + instruction conflict ("extract Indicator Code") ---
    conditions["AY-3"] = _build_schema_variant(
        entity_type="Country_Code",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Indicator Code' "
            "(not 'Country Code') as a Country_Code entity node. "
            "The column labeled 'Country Code' is metadata and should be ignored."
        ),
        relation_extraction_template=(
            "For each time column, create: (Country_Code) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Use the Indicator Code column as the entity identifier. "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Country_Code entity identified by "
            "'Indicator Code' (not 'Country Code').",
            "The 'Country Code' column is metadata and should NOT be used "
            "as entity identifier.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- BY: Nation_Identifier + disease misdirection (double mismatch) ---
    conditions["BY"] = _build_schema_variant(
        entity_type="Nation_Identifier",
        relation_type="HAS_VALUE",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "Nation_Identifier entity node. Nation_Identifier represents "
            "the disease classification category."
        ),
        relation_extraction_template=(
            "For each time column, create: (Nation_Identifier) "
            "-[HAS_VALUE {year: X, status: Y, unit: Z}]-> (value). "
            "Nation_Identifier entities are disease categories. "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Nation_Identifier entity identified by "
            "'Country Code'. Nation_Identifier is a disease category.",
            "Time columns encode fiscal year, status, and unit in compound headers.",
            "Parse compound headers (newline-separated) into year/status/unit attributes.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- C1: Country_Code + RECORDS_MEASUREMENT relation rename ---
    conditions["C1"] = _build_schema_variant(
        entity_type="Country_Code",
        relation_type="RECORDS_MEASUREMENT",
        entity_extraction_template=(
            "For each row, extract the value of 'Country Code' as a "
            "Country_Code entity node."
        ),
        relation_extraction_template=(
            "For each time column, create: (Country_Code) "
            "-[RECORDS_MEASUREMENT {period: X, observation_status: Y, "
            "measurement_unit: Z}]-> (numeric_observation). "
            "Example time columns: {2000}; {2001}; {2002}."
        ),
        extraction_constraints=[
            "Each row represents one Country_Code entity identified by 'Country Code'.",
            "Time columns encode period, observation_status, and measurement_unit "
            "in compound headers.",
            "Parse compound headers (newline-separated) into "
            "period/observation_status/measurement_unit attributes.",
            "Create one RECORDS_MEASUREMENT relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    return conditions


# ---------------------------------------------------------------------------
# LLM + Embedding functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Load SGE serialized chunks
# ---------------------------------------------------------------------------

def load_sge_chunks() -> list[str]:
    """Load SGE-serialized WHO text chunks from existing output."""
    chunks_path = WHO_SGE_OUTPUT / "lightrag_storage" / "kv_store_text_chunks.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [v["content"] for v in chunks_data.values() if "content" in v]


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

async def run_condition(
    condition_name: str,
    schema: dict,
    chunks: list[str],
    dry_run: bool = False,
) -> dict:
    """Run LightRAG with modified Schema on SGE chunks, return FC results."""

    output_dir = PROJECT_ROOT / "output" / f"probe_{condition_name.lower().replace('-', '_')}"
    work_dir = output_dir / "lightrag_storage"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"  Entity types: {schema['entity_types']}")
    print(f"  Relation types: {schema['relation_types']}")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipping LightRAG execution")
        return {
            "condition": condition_name,
            "dry_run": True,
            "schema_entity_types": schema["entity_types"],
            "schema_relation_types": schema["relation_types"],
        }

    # Generate system prompt from modified schema
    system_prompt_raw = generate_system_prompt(schema, language="Chinese")

    # Escape braces for LightRAG template compatibility
    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    # Override and run
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped

    entity_types = schema.get("entity_types", ["Entity"])
    addon_params = {"language": "Chinese", "entity_types": entity_types}

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

        for i, chunk_text in enumerate(chunks, 1):
            if i % 10 == 0 or i == len(chunks) or i == 1:
                print(f"  [{i}/{len(chunks)}] ({len(chunk_text)} chars)")
            await rag.ainsert(chunk_text)

        await rag.finalize_storages()
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    # Evaluate FC
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    result = {
        "condition": condition_name,
        "schema_entity_types": schema["entity_types"],
        "schema_relation_types": schema["relation_types"],
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(WHO_GOLD))
        _, graph_nodes, entity_text = load_graph(str(graph_path))

        matched_entities = check_entity_coverage(gold_entities, graph_nodes)
        ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

        covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
        fc = len(covered) / len(facts) if facts else 0.0

        result["ec"] = round(ec, 4)
        result["fc"] = round(fc, 4)
        result["ec_matched"] = len(matched_entities)
        result["ec_total"] = len(gold_entities)
        result["fc_covered"] = len(covered)
        result["fc_total"] = len(facts)

        print(f"  EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})")
        print(f"  FC={fc:.4f} ({len(covered)}/{len(facts)})")
        print(f"  Nodes={result['nodes']}, Edges={result['edges']}")
    else:
        print("  WARNING: graph file not found")
        result.update({"ec": 0.0, "fc": 0.0, "nodes": 0, "edges": 0})

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    conditions = build_conditions()

    # Filter if specific condition requested
    if args.condition:
        if args.condition not in conditions:
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {', '.join(conditions.keys())}")
            return
        conditions = {args.condition: conditions[args.condition]}

    # Load SGE chunks once
    print("Loading SGE-serialized WHO chunks...")
    chunks = load_sge_chunks()
    print(f"  Loaded {len(chunks)} chunks")

    # Run all conditions
    all_results = []
    for name, schema in conditions.items():
        result = await run_condition(name, schema, chunks, args.dry_run)
        all_results.append(result)

    # Add baseline reference
    baseline_ref = {
        "condition": "AX (baseline)",
        "schema_entity_types": ["Country_Code"],
        "schema_relation_types": ["HAS_VALUE"],
        "ec": 1.000,
        "fc": 1.000,
        "fc_covered": 150,
        "fc_total": 150,
        "nodes": 4508,
        "edges": 4312,
        "note": "Known result from main experiment",
    }

    # Summary
    print(f"\n{'='*70}")
    print("MECHANISM PROBE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'Entity Type':<22} {'Semantics':<12} {'FC':>6} {'EC':>6} {'Nodes':>6}")
    print("-" * 70)
    print(f"{'AX (ref)':<12} {'Country_Code':<22} {'correct':<12} {'1.000':>6} {'1.000':>6} {'4508':>6}")
    for r in all_results:
        if r.get("dry_run"):
            print(f"{r['condition']:<12} {r['schema_entity_types'][0]:<22} {'[dry run]':<12}")
        else:
            fc_str = f"{r.get('fc', 0):.3f}"
            ec_str = f"{r.get('ec', 0):.3f}"
            nodes_str = str(r.get("nodes", 0))
            print(f"{r['condition']:<12} {r['schema_entity_types'][0]:<22} {'—':<12} {fc_str:>6} {ec_str:>6} {nodes_str:>6}")
    print("=" * 70)

    # Interpretation guide
    bx1 = next((r for r in all_results if r["condition"] == "BX-1"), None)
    ay2 = next((r for r in all_results if r["condition"] == "AY-2"), None)
    if bx1 and ay2 and not bx1.get("dry_run"):
        bx1_fc = bx1.get("fc", 0)
        ay2_fc = ay2.get("fc", 0)
        print("\nINTERPRETATION:")
        if bx1_fc >= 0.8 and ay2_fc <= 0.3:
            print("  -> SEMANTIC UNDERSTANDING dominates: LLM reads Schema semantics,")
            print("     token name alignment is not necessary.")
        elif bx1_fc <= 0.3 and ay2_fc >= 0.8:
            print("  -> SURFACE-FORM PATTERN MATCHING dominates: LLM matches tokens,")
            print("     ignores semantic descriptions.")
        elif bx1_fc >= 0.8 and ay2_fc >= 0.8:
            print("  -> STRUCTURAL ROLE MAPPING: Both work — coupling is at column_roles")
            print("     level, not entity type name level.")
        else:
            print(f"  -> MIXED: BX-1 FC={bx1_fc:.3f}, AY-2 FC={ay2_fc:.3f}")
            print("     Both factors contribute. Regression needed to quantify.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Mechanism probing: Token alignment vs Semantic understanding",
        "dataset": "WHO Life Expectancy",
        "gold_facts": 150,
        "baseline_reference": baseline_ref,
        "probe_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = RESULTS_DIR / "mechanism_probe_results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mechanism Probing Experiment")
    parser.add_argument("--condition", type=str, default=None,
                        help="Run single condition (e.g. BX-1, AY-2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print schemas without running LightRAG")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
