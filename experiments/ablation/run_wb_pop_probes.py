#!/usr/bin/env python3
"""
run_wb_pop_probes.py — Probing experiment: Format-Constraint Coupling on WB Population.

Tests how modifying different parts of the extraction Schema affects graph
construction fidelity (FC) for the World Bank Population dataset.

Conditions:
  AX    : Original Schema unchanged (baseline, expected FC=1.000)
  BX-2  : GeopoliticalEntity (distant synonym for Country entity type)
  AY-3  : Country entity type + instruction to extract 'Country Code' instead
           of 'Country Name' (column reference conflict)

WB Pop schema differences from WHO:
  - entity_type="Country" (not "Country_Code")
  - relation_type="POPULATION" (not "HAS_VALUE")
  - subject column="Country Name" (not "Country Code")
  - Uses "Record:" serialization with country name before parenthesis
  - entity_extraction_template references "before the parenthesis" pattern

Usage:
    python3 experiments/ablation/run_wb_pop_probes.py
    python3 experiments/ablation/run_wb_pop_probes.py --condition BX-2
    python3 experiments/ablation/run_wb_pop_probes.py --dry-run
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

WB_POP_SGE_OUTPUT = PROJECT_ROOT / "output" / "wb_population"
WB_POP_GOLD = PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_population_v2.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Baseline WB Pop Schema (from extraction_schema.json)
# ---------------------------------------------------------------------------

def _load_baseline_schema() -> dict:
    """Load the original WB Population SGE extraction schema."""
    schema_path = WB_POP_SGE_OUTPUT / "extraction_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _build_schema_variant(
    entity_type: str,
    relation_type: str,
    entity_extraction_template: str,
    relation_extraction_template: str,
    extraction_constraints: list[str],
) -> dict:
    """Build a schema variant preserving column_roles and structure from baseline."""
    base = _load_baseline_schema()
    variant = {
        "table_type": base["table_type"],
        "entity_types": [entity_type, "StatValue"],
        "relation_types": [relation_type],
        "column_roles": base["column_roles"],
        "entity_extraction_template": entity_extraction_template,
        "relation_extraction_template": relation_extraction_template,
        "extraction_constraints": extraction_constraints,
        "parsed_time_headers": base.get("parsed_time_headers", []),
    }
    if "time_dimension" in base:
        variant["time_dimension"] = base["time_dimension"]
    return variant


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

def build_conditions() -> dict[str, dict]:
    """Build all probing conditions. Each returns a Schema dict."""

    conditions = {}

    # --- BX-2: GeopoliticalEntity (distant synonym, semantically correct) ---
    # Changes entity_type from "Country" to "GeopoliticalEntity".
    # Updates both templates to use "GeopoliticalEntity" consistently.
    # column_roles still points to "Country Name" as subject.
    # Expected: gradual FC degradation due to token misalignment.
    conditions["BX-2"] = _build_schema_variant(
        entity_type="GeopoliticalEntity",
        relation_type="POPULATION",
        entity_extraction_template=(
            "Each 'Record:' line introduces one GeopoliticalEntity entity. "
            "The geopolitical entity name (before the parenthesis) is the "
            "primary entity identifier. "
            "Example: 'Record: China (CHN), indicator=Total_Population' "
            "-> GeopoliticalEntity entity named 'China'."
        ),
        relation_extraction_template=(
            "For each observation line 'NAME observed population_yearYYYY=VALUE persons.', "
            "create: (GeopoliticalEntity: NAME) -[POPULATION]-> (StatValue) where "
            "StatValue describes the measurement as a phrase like "
            "'population_year2023=1410710000_persons'. "
            "Include the EXACT numeric value in the StatValue entity name."
        ),
        extraction_constraints=[
            "IMPORTANT: Create a POPULATION edge for EVERY observation line in the chunk.",
            "The source entity is the geopolitical entity name (e.g., 'China', 'India', 'United States').",
            "The StatValue entity name MUST include the numeric value — e.g., 'population_year2023=1410710000_persons'.",
            "The edge description MUST mention both the year and the exact numeric value.",
            "Include both geopolitical entity name and code (from the Record header) in the GeopoliticalEntity entity description.",
            "Do NOT skip any observation line — every line generates one POPULATION edge.",
        ],
    )

    # --- AY-3: Country entity type + column reference conflict ---
    # Keeps entity_type as "Country" but instructs the LLM to extract the
    # 'Country Code' column value instead of 'Country Name'.
    # This tests whether changing the column reference causes catastrophic
    # FC collapse, since gold entities are country names (e.g., "Argentina"),
    # not country codes (e.g., "ARG").
    # Expected: FC collapse (<0.5).
    conditions["AY-3"] = _build_schema_variant(
        entity_type="Country",
        relation_type="POPULATION",
        entity_extraction_template=(
            "Each 'Record:' line introduces one Country entity. "
            "Extract the value of 'Country Code' column (the 3-letter code "
            "inside the parenthesis) as the primary entity identifier. "
            "Example: 'Record: China (CHN), indicator=Total_Population' "
            "-> Country entity named 'CHN'. "
            "The country name before the parenthesis is metadata and should be ignored."
        ),
        relation_extraction_template=(
            "For each observation line 'NAME observed population_yearYYYY=VALUE persons.', "
            "create: (Country: CODE) -[POPULATION]-> (StatValue) where CODE is the "
            "3-letter country code from the Record header, and StatValue describes the "
            "measurement as a phrase like 'population_year2023=1410710000_persons'. "
            "Include the EXACT numeric value in the StatValue entity name."
        ),
        extraction_constraints=[
            "IMPORTANT: Create a POPULATION edge for EVERY observation line in the chunk.",
            "The source entity MUST be the 3-letter country code (e.g., 'ARG', 'AUS', 'CHN'), NOT the country name.",
            "The StatValue entity name MUST include the numeric value — e.g., 'population_year2023=1410710000_persons'.",
            "The edge description MUST mention both the year and the exact numeric value.",
            "Use 'Country Code' (the code in parentheses) as the entity identifier, not 'Country Name'.",
            "Do NOT skip any observation line — every line generates one POPULATION edge.",
        ],
    )

    # --- AY-2: Country + strong misdirection ("chemical compound") ---
    # Deep copy baseline, only append misdirection to preserve template structure
    base = _load_baseline_schema()
    ay2 = copy.deepcopy(base)
    ay2["entity_extraction_template"] = (
        base["entity_extraction_template"]
        + " Country represents a chemical compound identifier."
    )
    ay2["relation_extraction_template"] = (
        base["relation_extraction_template"]
        + " Country entities are chemical compound identifiers."
    )
    ay2["extraction_constraints"] = [
        c + (" Country is a chemical compound identifier." if i == 0 else "")
        for i, c in enumerate(base["extraction_constraints"])
    ]
    conditions["AY-2"] = ay2

    # --- BY: GeopoliticalEntity + strong misdirection ("chemical compound") ---
    # Deep copy BX-2 and append misdirection
    by = copy.deepcopy(conditions["BX-2"])
    by["entity_extraction_template"] = (
        conditions["BX-2"]["entity_extraction_template"]
        + " GeopoliticalEntity represents a chemical compound identifier."
    )
    by["relation_extraction_template"] = (
        conditions["BX-2"]["relation_extraction_template"]
        + " GeopoliticalEntity entities are chemical compound identifiers."
    )
    by["extraction_constraints"] = [
        c + (" GeopoliticalEntity is a chemical compound identifier." if i == 0 else "")
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
    """Load SGE-serialized WB Population text chunks from existing output."""
    chunks_path = WB_POP_SGE_OUTPUT / "lightrag_storage" / "kv_store_text_chunks.json"
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
    output_dir = PROJECT_ROOT / "output" / f"probe_wb_pop_{safe_name}"
    work_dir = output_dir / "lightrag_storage"
    # Clean previous run to avoid LightRAG doc dedup rejecting same-content chunks
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"  Entity types: {schema['entity_types']}")
    print(f"  Relation types: {schema['relation_types']}")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}")

    if dry_run:
        print("  Entity extraction template:")
        for line in schema["entity_extraction_template"].split(". "):
            print(f"    {line.strip()}")
        print("  Relation extraction template:")
        for line in schema["relation_extraction_template"].split(". "):
            print(f"    {line.strip()}")
        print("  [DRY RUN] Skipping LightRAG execution")
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
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(WB_POP_GOLD))
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
# Baseline AX: run with original unmodified schema
# ---------------------------------------------------------------------------

async def run_ax_condition(chunks: list[str], dry_run: bool = False) -> dict:
    """Run AX condition using the original WB Pop schema unchanged."""
    base_schema = _load_baseline_schema()

    condition_name = "AX"
    output_dir = PROJECT_ROOT / "output" / "probe_wb_pop_ax"
    work_dir = output_dir / "lightrag_storage"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name} (original schema, baseline reference)")
    print(f"  Entity types: {base_schema.get('entity_types', ['Country', 'StatValue'])}")
    print(f"  Relation types: {base_schema.get('relation_types', ['POPULATION'])}")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}")

    if dry_run:
        print("  Entity extraction template:")
        for line in base_schema["entity_extraction_template"].split(". "):
            print(f"    {line.strip()}")
        print("  [DRY RUN] Skipping LightRAG execution")
        return {
            "condition": condition_name,
            "dry_run": True,
            "schema_entity_types": base_schema.get("entity_types", ["Country", "StatValue"]),
            "schema_relation_types": base_schema.get("relation_types", ["POPULATION"]),
            "note": "Original schema — baseline reference (expected FC=1.000)",
        }

    # Generate system prompt from original schema
    system_prompt_raw = generate_system_prompt(base_schema, language="Chinese")

    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped

    entity_types = base_schema.get("entity_types", ["Country", "StatValue"])
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
        "condition": condition_name,
        "schema_entity_types": entity_types,
        "schema_relation_types": base_schema.get("relation_types", ["POPULATION"]),
        "note": "Original schema — baseline reference (expected FC=1.000)",
        "timestamp": datetime.now().isoformat(),
    }

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(WB_POP_GOLD))
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
    probe_conditions = build_conditions()

    # Determine what to run
    run_ax = True
    run_probes = dict(probe_conditions)

    if args.condition:
        if args.condition == "AX":
            run_probes = {}
        elif args.condition in probe_conditions:
            run_ax = False
            run_probes = {args.condition: probe_conditions[args.condition]}
        else:
            available = ["AX"] + list(probe_conditions.keys())
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {', '.join(available)}")
            return

    # Load SGE chunks once
    print("Loading SGE-serialized WB Population chunks...")
    chunks = load_sge_chunks()
    print(f"  Loaded {len(chunks)} chunks")

    all_results = []

    # Run AX first (baseline)
    if run_ax:
        ax_result = await run_ax_condition(chunks, args.dry_run)
        all_results.append(ax_result)

    # Run probe conditions
    for name, schema in run_probes.items():
        result = await run_condition(name, schema, chunks, args.dry_run)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("WB POPULATION PROBE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'Entity Type':<22} {'Description':<18} {'FC':>6} {'EC':>6} {'Nodes':>6}")
    print("-" * 70)

    descriptions = {
        "AX": "original schema",
        "BX-2": "distant synonym",
        "AY-3": "col ref conflict",
    }

    for r in all_results:
        cond = r["condition"]
        if r.get("dry_run"):
            entity_type = r["schema_entity_types"][0] if r["schema_entity_types"] else "?"
            desc = descriptions.get(cond, "—")
            print(f"{cond:<12} {entity_type:<22} {desc:<18} {'[dry]':>6}")
        else:
            entity_type = r["schema_entity_types"][0] if r["schema_entity_types"] else "?"
            desc = descriptions.get(cond, "—")
            fc_str = f"{r.get('fc', 0):.3f}"
            ec_str = f"{r.get('ec', 0):.3f}"
            nodes_str = str(r.get("nodes", 0))
            print(f"{cond:<12} {entity_type:<22} {desc:<18} {fc_str:>6} {ec_str:>6} {nodes_str:>6}")

    print("=" * 70)

    # Interpretation guide (only when all 3 conditions were run)
    ax_r = next((r for r in all_results if r["condition"] == "AX"), None)
    bx2_r = next((r for r in all_results if r["condition"] == "BX-2"), None)
    ay3_r = next((r for r in all_results if r["condition"] == "AY-3"), None)

    if ax_r and bx2_r and ay3_r and not any(
        r.get("dry_run") for r in [ax_r, bx2_r, ay3_r]
    ):
        ax_fc = ax_r.get("fc", 0)
        bx2_fc = bx2_r.get("fc", 0)
        ay3_fc = ay3_r.get("fc", 0)
        print("\nINTERPRETATION:")
        print(f"  AX  (Country, Country Name col): FC={ax_fc:.3f}  [reference]")
        print(f"  BX-2 (GeopoliticalEntity):       FC={bx2_fc:.3f}  [entity type rename]")
        print(f"  AY-3 (Country, Country Code col): FC={ay3_fc:.3f}  [column ref conflict]")
        if bx2_fc >= 0.8:
            print("  -> BX-2: LLM is robust to entity type label synonym; semantic")
            print("     understanding compensates for token misalignment.")
        elif bx2_fc < 0.5:
            print("  -> BX-2: Entity type name matters; distant synonym causes FC drop.")
        if ay3_fc < 0.5:
            print("  -> AY-3: Column reference is critical; redirecting to Country Code")
            print("     causes entity naming mismatch vs gold (names vs codes).")
        elif ay3_fc >= 0.8:
            print("  -> AY-3: LLM resists column reference conflict; gold entities still found.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "WB Population probing: Format-constraint coupling "
            "(entity type rename + column reference conflict)"
        ),
        "dataset": "WB Population",
        "gold_facts": 150,
        "conditions_tested": ["AX", "BX-2", "AY-3"],
        "condition_descriptions": {
            "AX": "Original schema unchanged (baseline reference, expected FC=1.000)",
            "BX-2": "GeopoliticalEntity replaces Country (distant synonym, token misalignment)",
            "AY-3": "Country entity type + 'Country Code' column ref (conflicts with gold naming)",
        },
        "probe_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = RESULTS_DIR / "wb_pop_probe_results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WB Population Format-Constraint Coupling Probing Experiment"
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Run single condition: AX, BX-2, or AY-3",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print schemas without running LightRAG",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
