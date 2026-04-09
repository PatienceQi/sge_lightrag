#!/usr/bin/env python3
"""
run_wb_cm_probes.py — Probing experiment on WB Child Mortality dataset.

Tests format-constraint coupling by modifying two specific Schema components:
  - Factor A: Entity type name (Country_Code vs GeopoliticalEntity)
  - Factor Y: Column reference in extraction template (correct vs conflicting)

Conditions:
  AX    : Country_Code + correct column reference (baseline, FC=1.000 known)
  BX-2  : GeopoliticalEntity + correct column reference (distant synonym)
  AY-3  : Country_Code + conflicting column reference ('Indicator Code')

All conditions use SGE-serialized WB CM chunks + modified Schema → LightRAG → FC.

Usage:
    python3 experiments/ablation/run_wb_cm_probes.py
    python3 experiments/ablation/run_wb_cm_probes.py --condition BX-2
    python3 experiments/ablation/run_wb_cm_probes.py --dry-run
"""

from __future__ import annotations

import asyncio
import copy
import json
import hashlib
import shutil
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

WB_CM_SGE_OUTPUT = PROJECT_ROOT / "output" / "wb_child_mortality"
WB_CM_GOLD = PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_child_mortality_v2.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Baseline WB CM Schema (from extraction_schema.json)
# ---------------------------------------------------------------------------

def _load_baseline_schema() -> dict:
    """Load the original WB Child Mortality SGE extraction schema."""
    schema_path = WB_CM_SGE_OUTPUT / "extraction_schema.json"
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
    """Build all probing conditions for WB Child Mortality. Each returns a Schema dict."""

    conditions = {}

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
            "-[HAS_VALUE {year: X, unit: Y}]-> (value). "
            "Example time columns: {2000}; {2005}; {2010}."
        ),
        extraction_constraints=[
            "Each row represents one GeopoliticalEntity entity identified by 'Country Code'.",
            "Time columns encode the measurement year as plain year integers.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- AY-3: Country_Code + column reference conflict ("Indicator Code") ---
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
            "-[HAS_VALUE {year: X, unit: Y}]-> (value). "
            "Use the Indicator Code column as the entity identifier. "
            "Example time columns: {2000}; {2005}; {2010}."
        ),
        extraction_constraints=[
            "Each row represents one Country_Code entity identified by "
            "'Indicator Code' (not 'Country Code').",
            "The 'Country Code' column is metadata and should NOT be used "
            "as entity identifier.",
            "Time columns encode the measurement year as plain year integers.",
            "Create one HAS_VALUE relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ],
    )

    # --- AY-2: Country_Code + strong misdirection ("product barcode") ---
    # Deep copy baseline, only add misdirection text to preserve template structure
    base = _load_baseline_schema()
    ay2 = copy.deepcopy(base)
    ay2["entity_extraction_template"] = (
        base["entity_extraction_template"]
        + " Country_Code represents a product barcode classification identifier."
    )
    ay2["relation_extraction_template"] = (
        base["relation_extraction_template"]
        + " Country_Code entities are product barcodes, not geographic entities."
    )
    ay2["extraction_constraints"] = [
        c + (" Country_Code is a product barcode identifier." if i == 0 else "")
        for i, c in enumerate(base["extraction_constraints"])
    ]
    conditions["AY-2"] = ay2

    # --- BY: GeopoliticalEntity + strong misdirection ("product barcode") ---
    # Deep copy BX-2 and add misdirection
    by = copy.deepcopy(conditions["BX-2"])
    by["entity_extraction_template"] = (
        conditions["BX-2"]["entity_extraction_template"]
        + " GeopoliticalEntity represents a product barcode classification identifier."
    )
    by["relation_extraction_template"] = (
        conditions["BX-2"]["relation_extraction_template"]
        + " GeopoliticalEntity entities are product barcodes."
    )
    by["extraction_constraints"] = [
        c + (" GeopoliticalEntity is a product barcode identifier." if i == 0 else "")
        for i, c in enumerate(conditions["BX-2"]["extraction_constraints"])
    ]
    conditions["BY"] = by

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
    """Load SGE-serialized WB Child Mortality text chunks from existing output."""
    chunks_path = (
        WB_CM_SGE_OUTPUT / "lightrag_storage" / "kv_store_text_chunks.json"
    )
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

    safe_name = condition_name.lower().replace("-", "_")
    output_dir = PROJECT_ROOT / "output" / f"probe_wb_cm_{safe_name}"
    work_dir = output_dir / "lightrag_storage"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"  Entity types: {schema['entity_types']}")
    print(f"  Relation types: {schema['relation_types']}")
    print(f"  entity_extraction_template: {schema['entity_extraction_template'][:80]}...")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Schema preview:")
        print(f"    entity_extraction_template: {schema['entity_extraction_template']}")
        print(f"    relation_extraction_template: {schema['relation_extraction_template']}")
        for i, c in enumerate(schema["extraction_constraints"], 1):
            print(f"    constraint[{i}]: {c}")
        return {
            "condition": condition_name,
            "dry_run": True,
            "schema_entity_types": schema["entity_types"],
            "schema_relation_types": schema["relation_types"],
            "entity_extraction_template": schema["entity_extraction_template"],
            "relation_extraction_template": schema["relation_extraction_template"],
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
        "entity_extraction_template": schema["entity_extraction_template"],
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(WB_CM_GOLD))
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
# AX baseline condition (runs LightRAG with unmodified schema)
# ---------------------------------------------------------------------------

async def run_ax_baseline(chunks: list[str], dry_run: bool = False) -> dict:
    """Run AX condition using the original baseline schema unchanged."""
    base_schema = _load_baseline_schema()

    output_dir = PROJECT_ROOT / "output" / "probe_wb_cm_ax"
    work_dir = output_dir / "lightrag_storage"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("CONDITION: AX (baseline — original schema)")
    print(f"  Entity types: {base_schema.get('entity_types', ['Country_Code'])}")
    print(f"  Relation types: {base_schema.get('relation_types', ['HAS_VALUE'])}")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Using original extraction_schema.json unchanged.")
        print(f"    entity_extraction_template: {base_schema['entity_extraction_template']}")
        return {
            "condition": "AX",
            "dry_run": True,
            "schema_entity_types": base_schema.get("entity_types", ["Country_Code"]),
            "schema_relation_types": base_schema.get("relation_types", ["HAS_VALUE"]),
            "note": "Original schema — reference condition",
        }

    system_prompt_raw = generate_system_prompt(base_schema, language="Chinese")
    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped

    entity_types = base_schema.get("entity_types", ["Country_Code"])
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

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    result = {
        "condition": "AX",
        "schema_entity_types": entity_types,
        "schema_relation_types": base_schema.get("relation_types", ["HAS_VALUE"]),
        "note": "Original schema — reference condition",
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(WB_CM_GOLD))
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
    all_conditions = build_conditions()

    # Validate --condition flag
    valid_names = {"AX", "BX-2", "AY-2", "AY-3", "BY"}
    if args.condition:
        if args.condition not in valid_names:
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {', '.join(sorted(valid_names))}")
            return
        run_ax = args.condition == "AX"
        probe_conditions = (
            {args.condition: all_conditions[args.condition]}
            if args.condition != "AX"
            else {}
        )
    else:
        run_ax = True
        probe_conditions = all_conditions

    # Load SGE chunks once
    print("Loading SGE-serialized WB Child Mortality chunks...")
    chunks = load_sge_chunks()
    print(f"  Loaded {len(chunks)} chunks")

    all_results = []

    # Run AX baseline first if selected
    if run_ax:
        ax_result = await run_ax_baseline(chunks, args.dry_run)
        all_results.append(ax_result)

    # Run probe conditions
    for name, schema in probe_conditions.items():
        result = await run_condition(name, schema, chunks, args.dry_run)
        all_results.append(result)

    # Known reference for summary (from main evaluation)
    ax_reference = {
        "condition": "AX (reference)",
        "schema_entity_types": ["Country_Code"],
        "schema_relation_types": ["HAS_VALUE"],
        "ec": 1.000,
        "fc": 1.000,
        "fc_covered": 150,
        "fc_total": 150,
        "nodes": 5218,
        "edges": 5509,
        "note": "Known result from main evaluation (all_results_v2.json)",
    }

    # Summary table
    print(f"\n{'='*72}")
    print("WB CHILD MORTALITY PROBE RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"{'Condition':<12} {'Entity Type':<22} {'Column Ref':<20} {'FC':>6} {'EC':>6} {'Nodes':>6}")
    print("-" * 72)
    print(
        f"{'AX (ref)':<12} {'Country_Code':<22} {'Country Code (correct)':<20}"
        f" {'1.000':>6} {'1.000':>6} {'5218':>6}"
    )
    for r in all_results:
        if r.get("dry_run"):
            et = r["schema_entity_types"][0] if r.get("schema_entity_types") else "—"
            print(f"{r['condition']:<12} {et:<22} {'[dry run]':<20}")
        else:
            et = r["schema_entity_types"][0] if r.get("schema_entity_types") else "—"
            col_ref = "Indicator Code" if r["condition"] == "AY-3" else "Country Code"
            fc_str = f"{r.get('fc', 0):.3f}"
            ec_str = f"{r.get('ec', 0):.3f}"
            nodes_str = str(r.get("nodes", 0))
            print(
                f"{r['condition']:<12} {et:<22} {col_ref:<20}"
                f" {fc_str:>6} {ec_str:>6} {nodes_str:>6}"
            )
    print("=" * 72)

    # Interpretation
    bx2 = next((r for r in all_results if r["condition"] == "BX-2"), None)
    ay3 = next((r for r in all_results if r["condition"] == "AY-3"), None)
    if (bx2 and not bx2.get("dry_run")) or (ay3 and not ay3.get("dry_run")):
        print("\nINTERPRETATION:")
        if bx2 and not bx2.get("dry_run"):
            bx2_fc = bx2.get("fc", 0)
            if bx2_fc >= 0.8:
                print(f"  BX-2 FC={bx2_fc:.3f}: Entity type rename causes only gradual degradation.")
                print("    -> LLM reads column_roles correctly despite name change.")
            else:
                print(f"  BX-2 FC={bx2_fc:.3f}: Distant synonym causes significant degradation.")
                print("    -> Token alignment matters for entity type grounding.")
        if ay3 and not ay3.get("dry_run"):
            ay3_fc = ay3.get("fc", 0)
            if ay3_fc < 0.5:
                print(f"  AY-3 FC={ay3_fc:.3f}: Column reference conflict causes FC collapse.")
                print("    -> Instruction-level column reference is the binding constraint.")
            else:
                print(f"  AY-3 FC={ay3_fc:.3f}: Column reference conflict does not fully disrupt.")
                print("    -> LLM may recover via entity_type token or column_roles.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "WB Child Mortality probing: format-constraint coupling "
            "(entity type name vs column reference)"
        ),
        "dataset": "WB Child Mortality",
        "gold_facts": 150,
        "conditions_tested": ["AX", "BX-2", "AY-3"],
        "ax_reference": ax_reference,
        "probe_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = RESULTS_DIR / "wb_cm_probe_results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WB Child Mortality Probing Experiment (format-constraint coupling)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Run single condition: AX, BX-2, or AY-3",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schemas without running LightRAG",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
