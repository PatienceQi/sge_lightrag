#!/usr/bin/env python3
"""
Run LLM-enhanced SGE pipeline (SGE only, no baseline) on 3 failed OOD datasets.
Reuses existing baseline from output/ood/ for evaluation.
"""
from __future__ import annotations
import sys, json, asyncio, hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2_llm.inductor import induce_schema as induce_schema_llm
from stage2.inducer import induce_schema as induce_schema_rule_core
from stage3.serializer import serialize_csv
from stage3.integrator import patch_lightrag

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM          = 1024

import requests as _requests

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )

def _hash_embed(text):
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec

def _ollama_embed_sync(texts):
    resp = _requests.post(
        f"{OLLAMA_BASE_URL}/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "input": texts}, timeout=60,
    )
    resp.raise_for_status()
    return np.array([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)

async def safe_embedding_func(texts):
    import asyncio as _aio
    loop = _aio.get_event_loop()
    for attempt in range(5):
        try:
            return await loop.run_in_executor(None, _ollama_embed_sync, texts)
        except Exception as e:
            if attempt < 4:
                await _aio.sleep(2 * (attempt + 1))
            else:
                return np.array([_hash_embed(t) for t in texts], dtype=np.float32)

EMBEDDING_FUNC = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func)
_original_extract_entities = _op.extract_entities

async def _sge_extract_entities(chunks, knowledgebase, entity_vdb, relationships_vdb, global_config, pipeline_status=None, llm_response_cache=None):
    return await _original_extract_entities(chunks, knowledgebase, entity_vdb, relationships_vdb, global_config, pipeline_status=pipeline_status, llm_response_cache=llm_response_cache)


OOD_DIR = Path("../dataset/ood_blind_test")
OUTPUT_BASE = Path("output/ood_llm")
GOLD_DIR = Path("evaluation/gold")
RULE_OUTPUT = Path("output/ood")

DATASETS = ["wb_unemployment", "wb_immunization_dpt", "wb_immunization_measles"]


def run_sge_pipeline(csv_path, sge_output_dir):
    features = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Type={table_type}, Cols={len(features.raw_columns)}")

    rule_schema = induce_schema_rule_core(meta_schema, features)
    try:
        llm_schema = induce_schema_llm(str(csv_path))
        extraction_schema = {
            **rule_schema,
            "entity_types": llm_schema["entity_types"],
            "relation_types": llm_schema["relation_types"],
            "prompt_context": llm_schema.get("prompt_context", ""),
        }
        print(f"  LLM Schema: {llm_schema['entity_types']}")
    except Exception as e:
        print(f"  LLM failed ({e}), using rule schema")
        extraction_schema = rule_schema

    chunks = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks: {len(chunks)}")

    sge_output_dir.mkdir(parents=True, exist_ok=True)
    (sge_output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chunks, extraction_schema, payload


async def run_lightrag_sge(chunks, working_dir, addon_params):
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=1,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    for i, chunk in enumerate(chunks, 1):
        print(f"    [{i}/{len(chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats = {"chunks": len(chunks)}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    await rag.finalize_storages()
    return stats


def evaluate_fc(graph_dir, gold_path):
    """Compute EC and FC for a graph against gold standard."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_entity_coverage, check_fact_coverage
    )
    graphml = Path(graph_dir) / "graph_chunk_entity_relation.graphml"
    if not graphml.exists():
        return 0.0, 0.0, {}
    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(str(graphml))
    matched_ents = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_ents) / len(gold_entities) if gold_entities else 0
    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0
    return ec, fc, {"n_nodes": len(graph_nodes)}


async def main():

    results = []
    for name in DATASETS:
        csv_path = OOD_DIR / f"{name}.csv"
        gold_path = GOLD_DIR / f"gold_ood_{name}.jsonl"
        out_dir = OUTPUT_BASE / name
        sge_dir = out_dir / "sge_budget"
        sge_work = sge_dir / "lightrag_storage"

        # Skip if already completed
        if sge_work.exists() and (sge_work / "graph_chunk_entity_relation.graphml").exists():
            import networkx as nx
            G = nx.read_graphml(str(sge_work / "graph_chunk_entity_relation.graphml"))
            if G.number_of_nodes() > 10:
                print(f"\n=== SKIP {name} (already done, {G.number_of_nodes()} nodes) ===")
                # Still evaluate
                if gold_path.exists():
                    ec, fc, details = evaluate_fc(str(sge_work), str(gold_path))
                    row = {"dataset": name, "sge_llm_ec": round(ec, 3), "sge_llm_fc": round(fc, 3), "nodes": G.number_of_nodes()}
                    # Get baseline from rule output
                    rule_base = RULE_OUTPUT / name / "baseline_budget" / "lightrag_storage"
                    if rule_base.exists():
                        bec, bfc, _ = evaluate_fc(str(rule_base), str(gold_path))
                        row["baseline_fc"] = round(bfc, 3)
                    # Get rule SGE
                    rule_sge = RULE_OUTPUT / name / "sge_budget" / "lightrag_storage"
                    if rule_sge.exists():
                        rec, rfc, _ = evaluate_fc(str(rule_sge), str(gold_path))
                        row["sge_rule_fc"] = round(rfc, 3)
                    results.append(row)
                    print(f"  LLM FC={row['sge_llm_fc']}, Rule FC={row.get('sge_rule_fc','?')}, Base FC={row.get('baseline_fc','?')}")
                continue

        print(f"\n=== {name} ===")
        chunks, schema, payload = run_sge_pipeline(csv_path, sge_dir)

        original_prompt = PROMPTS["entity_extraction_system_prompt"]
        raw_prompt = payload["system_prompt"]
        escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped
        _op.extract_entities = _sge_extract_entities

        try:
            stats = await run_lightrag_sge(chunks, sge_work, payload["addon_params"])
        finally:
            PROMPTS["entity_extraction_system_prompt"] = original_prompt
            _op.extract_entities = _original_extract_entities

        row = {"dataset": name, "nodes": stats.get("nodes", 0)}

        if gold_path.exists():
            ec, fc, details = evaluate_fc(str(sge_work), str(gold_path))
            row["sge_llm_ec"] = round(ec, 3)
            row["sge_llm_fc"] = round(fc, 3)
            print(f"  LLM SGE: EC={ec:.3f} FC={fc:.3f}")

            rule_base = RULE_OUTPUT / name / "baseline_budget" / "lightrag_storage"
            if rule_base.exists():
                bec, bfc, _ = evaluate_fc(str(rule_base), str(gold_path))
                row["baseline_fc"] = round(bfc, 3)

            rule_sge = RULE_OUTPUT / name / "sge_budget" / "lightrag_storage"
            if rule_sge.exists():
                rec, rfc, _ = evaluate_fc(str(rule_sge), str(gold_path))
                row["sge_rule_fc"] = round(rfc, 3)

            print(f"  Rule FC={row.get('sge_rule_fc','?')}, Base FC={row.get('baseline_fc','?')}")

        results.append(row)

    print("\n" + "=" * 70)
    print(f"{'Dataset':<25} {'LLM FC':>8} {'Rule FC':>8} {'Base FC':>8} {'Nodes':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['dataset']:<25} {r.get('sge_llm_fc','?'):>8} {r.get('sge_rule_fc','?'):>8} {r.get('baseline_fc','?'):>8} {r.get('nodes','?'):>6}")

    out_path = Path("evaluation/results/ood_llm_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
