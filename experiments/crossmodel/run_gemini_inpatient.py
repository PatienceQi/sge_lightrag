#!/usr/bin/env python3
"""
run_gemini_inpatient.py — Gemini 2.5 Flash on Inpatient 2023 dataset.

Runs the full SGE pipeline (Stage 1→2→3) on the HK Inpatient 2023 dataset
using Gemini 2.5 Flash as the LightRAG extraction backend, then evaluates FC
against the gold standard.

Usage:
    python3 experiments/crossmodel/run_gemini_inpatient.py
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import asyncio
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2_llm.inductor import induce_schema as induce_schema_llm
from stage2.inducer import induce_schema as induce_schema_rule
from stage3.serializer import serialize_csv
from stage3.integrator import patch_lightrag

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL    = "gemini-2.5-flash"
EMBED_DIM = 1024

DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "crossmodel_gemini_inpatient"
GOLD_PATH    = PROJECT_ROOT / "evaluation" / "gold" / "gold_inpatient_2023.jsonl"
CSV_PATH     = (
    DATASET_ROOT / "住院病人统计"
    / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths"
      " in Hong Kong by Disease 2023 (SC).csv"
)
EVAL_SCRIPT  = PROJECT_ROOT / "evaluation" / "evaluate_coverage.py"
RESULTS_DIR  = PROJECT_ROOT / "experiments" / "results"


# ── LLM function ──────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    kwargs.setdefault("timeout", 120)
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding: Ollama via urllib3 (bypass macOS proxy) ───────────────────────
import urllib3 as _urllib3
_pool = _urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)


def _ollama_embed_sync(texts: list[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        if len(text) > 1000:
            text = text[:1000]
        body = json.dumps({"model": "mxbai-embed-large", "prompt": text}).encode()
        resp = _pool.urlopen(
            "POST", "/api/embeddings", body=body,
            headers={"Content-Type": "application/json"}, timeout=120.0,
        )
        emb = json.loads(resp.data)["embedding"]
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    import asyncio as _aio
    loop = _aio.get_event_loop()
    for attempt in range(3):
        try:
            return await loop.run_in_executor(None, _ollama_embed_sync, texts)
        except Exception as e:
            if attempt < 2:
                print(f"  [warn] Embed attempt {attempt+1} failed: {e}, retrying...")
                await _aio.sleep(2)
            else:
                print(f"  [warn] Embed failed 3x, using hash fallback")
                return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)


# ── SGE pipeline ──────────────────────────────────────────────────────────────
def run_sge_pipeline(csv_path: Path, output_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"SGE PIPELINE: {csv_path.name}")
    print(f"{'='*60}")

    features    = extract_features(str(csv_path))
    table_type  = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Type: {table_type}")

    # Stage 2: rule-based (LLM with rule fallback)
    rule_schema = induce_schema_rule(meta_schema, features)
    try:
        llm_schema = induce_schema_llm(str(csv_path))
        extraction_schema = {
            **rule_schema,
            "entity_types": llm_schema["entity_types"],
            "relation_types": llm_schema["relation_types"],
            "prompt_context": llm_schema.get("prompt_context", ""),
            "extraction_rules": llm_schema.get("extraction_rules", {}),
        }
        stage2_mode = "llm_enhanced"
        print(f"  LLM Stage 2 OK: {extraction_schema['entity_types']}")
    except Exception as e:
        print(f"  LLM failed ({e}), using rule-based fallback")
        extraction_schema = rule_schema
        stage2_mode = "rule_fallback"

    # Stage 3
    chunks  = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks: {len(chunks)}")

    # Persist SGE artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_schema.json").write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2))
    (output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2))
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks, 1):
        (chunks_dir / f"chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
    (output_dir / "system_prompt.txt").write_text(payload["system_prompt"])
    (output_dir / "stage2_mode.txt").write_text(stage2_mode)

    return {"chunks": chunks, "payload": payload, "schema": extraction_schema,
            "stage2_mode": stage2_mode}


# ── LightRAG runner ───────────────────────────────────────────────────────────
_original_extract_entities = _op.extract_entities


async def _sge_passthrough(chunks, knowledgebase, entity_vdb, relationships_vdb,
                            global_config, pipeline_status=None,
                            llm_response_cache=None):
    return await _original_extract_entities(
        chunks, knowledgebase, entity_vdb, relationships_vdb,
        global_config, pipeline_status=pipeline_status,
        llm_response_cache=llm_response_cache,
    )


async def run_lightrag(chunks: list[str], working_dir: Path, payload: dict) -> dict:
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG] {working_dir}")

    addon_params = payload["addon_params"]

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=5,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Inject SGE system prompt
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped
    _op.extract_entities = _sge_passthrough

    try:
        print(f"  Inserting {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            if i % 10 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}]")
            await rag.ainsert(chunk)
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt
        _op.extract_entities = _original_extract_entities

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats: dict = {"chunks": len(chunks)}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    else:
        print(f"  [warn] Graph file not found: {graph_path}")
        stats["nodes"] = 0
        stats["edges"] = 0

    await rag.finalize_storages()
    return stats


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_fc(graph_path: Path, gold_path: Path) -> dict:
    if not graph_path.exists():
        print(f"  [eval] Graph not found: {graph_path}")
        return {"ec": 0.0, "fc": 0.0, "error": "graph_missing"}
    if not gold_path.exists():
        print(f"  [eval] Gold not found: {gold_path}")
        return {"ec": 0.0, "fc": 0.0, "error": "gold_missing"}

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--graph", str(graph_path),
         "--gold", str(gold_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        err_tail = result.stderr[-300:] if result.stderr else "no stderr"
        print(f"  [eval] evaluate_coverage.py failed: {err_tail}")
        return {"ec": 0.0, "fc": 0.0, "error": err_tail}

    output = result.stdout
    try:
        marker = output.find("[JSON]")
        search_text = output[marker + 6:] if marker >= 0 else output
        j_start = search_text.find("{")
        j_end   = search_text.rfind("}") + 1
        data    = json.loads(search_text[j_start:j_end])
        ec = data["entity_coverage"]["coverage"]
        fc = data["fact_coverage"]["coverage"]
        print(f"  EC={ec:.3f}, FC={fc:.3f}")
        return {
            "ec": ec, "fc": fc,
            "ec_matched": data["entity_coverage"]["matched"],
            "ec_total":   data["entity_coverage"]["total"],
            "fc_covered": data["fact_coverage"]["covered"],
            "fc_total":   data["fact_coverage"]["total"],
        }
    except Exception as parse_err:
        print(f"  [eval] JSON parse error: {parse_err}")
        return {"ec": 0.0, "fc": 0.0, "error": f"parse_failed: {parse_err}"}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main_async() -> None:
    print(f"Model  : {MODEL}")
    print(f"Dataset: Inpatient 2023 (HK)")
    print(f"CSV    : {CSV_PATH}")
    print(f"Gold   : {GOLD_PATH}")

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not GOLD_PATH.exists():
        print(f"ERROR: Gold not found: {GOLD_PATH}")
        sys.exit(1)

    # Clean output dir for a fresh run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1→2→3
    sge_result = run_sge_pipeline(CSV_PATH, OUTPUT_DIR)
    chunks  = sge_result["chunks"]
    payload = sge_result["payload"]

    # LightRAG injection
    lightrag_dir = OUTPUT_DIR / "lightrag_storage"
    graph_stats  = await run_lightrag(chunks, lightrag_dir, payload)

    # Evaluate
    graph_path   = lightrag_dir / "graph_chunk_entity_relation.graphml"
    eval_result  = evaluate_fc(graph_path, GOLD_PATH)

    result = {
        "label": "Inpatient HK 2023",
        "model": MODEL,
        "stage2_mode": sge_result["stage2_mode"],
        "chunks": len(chunks),
        "nodes": graph_stats.get("nodes", 0),
        "edges": graph_stats.get("edges", 0),
        "ec": eval_result.get("ec", 0.0),
        "fc": eval_result.get("fc", 0.0),
        "ec_matched": eval_result.get("ec_matched"),
        "ec_total":   eval_result.get("ec_total"),
        "fc_covered": eval_result.get("fc_covered"),
        "fc_total":   eval_result.get("fc_total"),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "crossmodel_gemini_inpatient.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 50)
    print("RESULT SUMMARY")
    print("=" * 50)
    print(f"  Dataset : {result['label']}")
    print(f"  Model   : {result['model']}")
    print(f"  Stage2  : {result['stage2_mode']}")
    print(f"  Chunks  : {result['chunks']}")
    print(f"  Nodes   : {result['nodes']}")
    print(f"  Edges   : {result['edges']}")
    print(f"  EC      : {result['ec']:.3f}")
    print(f"  FC      : {result['fc']:.3f}")
    print("=" * 50)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
