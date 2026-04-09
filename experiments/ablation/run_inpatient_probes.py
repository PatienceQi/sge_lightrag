#!/usr/bin/env python3
"""
run_inpatient_probes.py — Probing on Inpatient (Type-III) to validate anchoring hierarchy.

Two conditions:
  BX-1: Disease → Medical_Condition (near synonym rename)
  AY-3: Column reference conflict (extract ICD code instead of disease name)

Usage:
    python3 experiments/ablation/run_inpatient_probes.py
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import shutil
import sys
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

INPATIENT_SGE_OUTPUT = PROJECT_ROOT / "output" / "inpatient_2023"
INPATIENT_GOLD = PROJECT_ROOT / "evaluation" / "gold" / "gold_inpatient_2023.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _load_baseline_schema() -> dict:
    path = INPATIENT_SGE_OUTPUT / "extraction_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunks() -> list[str]:
    path = INPATIENT_SGE_OUTPUT / "lightrag_storage" / "kv_store_text_chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [v["content"] for v in data.values() if "content" in v]


def build_conditions() -> dict[str, dict]:
    base = _load_baseline_schema()
    conditions = {}

    # BX-1: Disease → Medical_Condition (near synonym)
    bx1 = json.loads(json.dumps(base))
    bx1["entity_types"] = ["Medical_Condition"]
    bx1["entity_extraction_template"] = bx1["entity_extraction_template"].replace(
        "Disease", "Medical_Condition"
    )
    bx1["relation_extraction_template"] = bx1["relation_extraction_template"].replace(
        "Disease", "Medical_Condition"
    )
    bx1["extraction_constraints"] = [
        c.replace("Disease", "Medical_Condition") for c in bx1["extraction_constraints"]
    ]
    conditions["BX-1"] = bx1

    # AY-3: Column reference conflict — extract ICD code column instead of disease name
    ay3 = json.loads(json.dumps(base))
    # Replace references to 疾病类别 with ICD code column
    icd_col = "《疾病和有关健康问题的国际统计分类》第十次修订本的详细序号"
    ay3["entity_extraction_template"] = (
        f"Composite key columns: '{icd_col}'. "
        f"Empty cells inherit the last non-empty value from the same column above. "
        f"Top-level key '{icd_col}' → Disease entity. "
        f"The column '疾病类别' is metadata and should NOT be used as entity identifier."
    )
    ay3["extraction_constraints"] = [
        f"Top-level key '{icd_col}' defines the root entity (Disease).",
        f"The '疾病类别' column is metadata only — do NOT use it as entity name.",
        "Numeric columns become HAS_VALUE relations from the leaf entity.",
        "Remarks/notes columns become metadata properties (not separate entities).",
        "Skip rows that are entirely empty or contain only section headers.",
    ]
    conditions["AY-3"] = ay3

    # AY-2: Disease + strong misdirection ("financial transaction category")
    ay2 = json.loads(json.dumps(base))
    ay2["entity_extraction_template"] = (
        "Composite key columns: '疾病类别'. "
        "Empty cells inherit the last non-empty value from the same column above. "
        "Top-level key '疾病类别' → Disease entity. "
        "Disease represents a financial transaction category classification."
    )
    ay2["extraction_constraints"] = [
        "Detect hierarchy by sparse fill (empty cells inherit parent value).",
        "Top-level key '疾病类别' defines root entity (Disease). "
        "Disease is a financial transaction category.",
        "Numeric columns become HAS_VALUE relations from the leaf entity.",
        "Remarks/notes columns become metadata properties (not separate entities).",
        "Skip rows that are entirely empty or contain only section headers.",
    ]
    conditions["AY-2"] = ay2

    # BY: Medical_Condition + strong misdirection ("financial transaction category")
    by_cond = json.loads(json.dumps(base))
    by_cond["entity_types"] = ["Medical_Condition"]
    by_cond["entity_extraction_template"] = (
        "Composite key columns: '疾病类别'. "
        "Empty cells inherit the last non-empty value from the same column above. "
        "Top-level key '疾病类别' → Medical_Condition entity. "
        "Medical_Condition represents a financial transaction category classification."
    )
    by_cond["relation_extraction_template"] = by_cond["relation_extraction_template"].replace(
        "Disease", "Medical_Condition"
    )
    by_cond["extraction_constraints"] = [
        "Detect hierarchy by sparse fill (empty cells inherit parent value).",
        "Top-level key '疾病类别' defines root entity (Medical_Condition). "
        "Medical_Condition is a financial transaction category.",
        "Numeric columns become HAS_VALUE relations from the leaf entity.",
        "Remarks/notes columns become metadata properties (not separate entities).",
        "Skip rows that are entirely empty or contain only section headers.",
    ]
    conditions["BY"] = by_cond

    return conditions


# LLM + Embedding
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


async def run_condition(name: str, schema: dict, chunks: list[str]) -> dict:
    output_dir = PROJECT_ROOT / "output" / f"probe_inpatient_{name.lower().replace('-', '_')}"
    work_dir = output_dir / "lightrag_storage"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"INPATIENT PROBE: {name}")
    print(f"  Entity types: {schema['entity_types']}")
    print(f"  Chunks: {len(chunks)}")
    print(f"{'='*60}", flush=True)

    system_prompt_raw = generate_system_prompt(schema, language="Chinese")
    escaped = system_prompt_raw.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    PROMPTS["entity_extraction_system_prompt"] = escaped
    entity_types = schema.get("entity_types", ["Entity"])

    try:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params={"language": "Chinese", "entity_types": entity_types},
            llm_model_max_async=5,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()
        for i, chunk in enumerate(chunks, 1):
            if i % 50 == 0 or i == len(chunks) or i == 1:
                print(f"  [{i}/{len(chunks)}]", flush=True)
            await rag.ainsert(chunk)
        await rag.finalize_storages()
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    result = {"condition": name, "timestamp": datetime.now().isoformat()}

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        result["nodes"] = G.number_of_nodes()
        result["edges"] = G.number_of_edges()

        gold_entities, facts = load_gold(str(INPATIENT_GOLD))
        _, graph_nodes, entity_text = load_graph(str(graph_path))
        matched = check_entity_coverage(gold_entities, graph_nodes)
        ec = len(matched) / len(gold_entities) if gold_entities else 0.0
        covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
        fc = len(covered) / len(facts) if facts else 0.0
        result.update({"ec": round(ec, 4), "fc": round(fc, 4),
                       "ec_matched": len(matched), "ec_total": len(gold_entities),
                       "fc_covered": len(covered), "fc_total": len(facts)})
        print(f"  EC={ec:.4f}  FC={fc:.4f}  Nodes={result['nodes']}", flush=True)
    else:
        result.update({"ec": 0, "fc": 0, "nodes": 0, "edges": 0})

    return result


async def main_async():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None,
                        help="Run single condition: BX-1, AY-3, AY-2, or BY")
    args = parser.parse_args()

    conditions = build_conditions()
    if args.condition:
        if args.condition not in conditions:
            print(f"Unknown: {args.condition}. Available: {list(conditions.keys())}")
            return
        conditions = {args.condition: conditions[args.condition]}

    chunks = _load_chunks()
    print(f"Loaded {len(chunks)} Inpatient chunks", flush=True)

    results = []
    for name, schema in conditions.items():
        r = await run_condition(name, schema, chunks)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("INPATIENT PROBING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<10} {'Entity Type':<22} {'FC':>6} {'EC':>6}")
    print("-" * 50)
    print(f"{'AX (ref)':<10} {'Disease':<22} {'0.938':>6} {'1.000':>6}")
    for r in results:
        print(f"{r['condition']:<10} {'—':<22} {r.get('fc',0):.3f} {r.get('ec',0):.3f}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"dataset": "Inpatient 2023", "baseline_fc": 0.938,
           "results": results, "timestamp": datetime.now().isoformat()}
    path = RESULTS_DIR / "inpatient_probe_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    asyncio.run(main_async())
