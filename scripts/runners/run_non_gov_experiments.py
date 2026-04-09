#!/usr/bin/env python3
"""
Run SGE + Baseline experiments on non-government datasets.

Usage:
    python3 scripts/runners/run_non_gov_experiments.py fortune500
    python3 scripts/runners/run_non_gov_experiments.py the_ranking
    python3 scripts/runners/run_non_gov_experiments.py all
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
import hashlib
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

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

DATASET_DIR = PROJECT_ROOT.parent / "dataset" / "non_gov"
OUTPUT_DIR  = PROJECT_ROOT / "output"

DATASETS = {
    "fortune500": {
        "csv": DATASET_DIR / "fortune500_revenue.csv",
        "sge_output": OUTPUT_DIR / "fortune500_revenue",
        "serial_only_output": OUTPUT_DIR / "serial_only_fortune500_revenue",
        "naive_baseline_output": OUTPUT_DIR / "naive_baseline_fortune500_revenue",
        "language": "English",
    },
    "the_ranking": {
        "csv": DATASET_DIR / "the_university_ranking.csv",
        "sge_output": OUTPUT_DIR / "the_university_ranking",
        "serial_only_output": OUTPUT_DIR / "serial_only_the_university_ranking",
        "naive_baseline_output": OUTPUT_DIR / "naive_baseline_the_university_ranking",
        "language": "English",
    },
}


# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding (Ollama via urllib3 to bypass macOS proxy) ─────────────────────
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


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)


# ── Naive baseline: raw CSV text (same as run_decoupled_ablation.py) ─────────
def read_csv_as_text(csv_path: Path) -> str:
    """Read raw CSV into plain text string using pandas."""
    import pandas as pd
    from stage1.features import _detect_encoding

    encoding = _detect_encoding(str(csv_path))
    df = None
    for enc in [encoding, "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        raise ValueError(f"Cannot read CSV: {csv_path}")

    print(f"  CSV shape (naive): {df.shape}")
    return df.to_string()


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks at line boundaries, respecting max_chars."""
    lines = text.split("\n")
    chunks, current_lines, current_size = [], [], 0

    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines, current_size = [], 0
        current_lines.append(line)
        current_size += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


# ── Monkey-patch for SGE schema injection ────────────────────────────────────
_original_extract_entities = _op.extract_entities


async def _sge_extract_entities(chunks, knowledgebase, entity_vdb, relationships_vdb,
                                 global_config, pipeline_status=None,
                                 llm_response_cache=None):
    return await _original_extract_entities(
        chunks, knowledgebase, entity_vdb, relationships_vdb,
        global_config, pipeline_status=pipeline_status,
        llm_response_cache=llm_response_cache,
    )


# ── SGE Pipeline ─────────────────────────────────────────────────────────────
def run_sge_pipeline(csv_path: Path, output_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"SGE PIPELINE: {csv_path.name}")
    print(f"{'='*60}")

    features = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Type: {table_type}")

    # Stage 2: LLM with rule fallback
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
        print(f"  LLM failed ({e}), using rule-based")
        extraction_schema = rule_schema
        stage2_mode = "rule_fallback"

    # Stage 3
    chunks = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks: {len(chunks)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_schema.json").write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2))
    (output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2))
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, c in enumerate(chunks, 1):
        (chunks_dir / f"chunk_{i:04d}.txt").write_text(c, encoding="utf-8")
    (output_dir / "system_prompt.txt").write_text(payload["system_prompt"])
    (output_dir / "stage2_mode.txt").write_text(stage2_mode)

    return {"chunks": chunks, "payload": payload, "schema": extraction_schema,
            "stage2_mode": stage2_mode}


# ── LightRAG runner ──────────────────────────────────────────────────────────
async def run_lightrag(chunks: list[str], working_dir: Path,
                       addon_params: dict, label: str) -> dict:
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG:{label}] {working_dir}")

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

    print(f"  Inserting {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks, 1):
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}]")
        await rag.ainsert(chunk)

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats = {"label": label, "chunks": len(chunks)}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    await rag.finalize_storages()
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
async def run_dataset(name: str, config: dict, naive_only: bool = False):
    csv_path = config["csv"]
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    sge_stats, serial_only_stats = {}, {}

    if naive_only:
        # Skip SGE and Serial-only, only run naive baseline
        print(f"\n{'='*60}")
        print(f"NAIVE BASELINE ONLY: {name}")
        print(f"{'='*60}")
    else:
        # 1. SGE pipeline
        result = run_sge_pipeline(csv_path, config["sge_output"])
        chunks = result["chunks"]
        payload = result["payload"]

        # 2. SGE-enhanced LightRAG
        original_prompt = PROMPTS["entity_extraction_system_prompt"]
        raw_prompt = payload["system_prompt"]
        escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                    "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped
        _op.extract_entities = _sge_extract_entities

        try:
            sge_stats = await run_lightrag(
                chunks, config["sge_output"] / "lightrag_storage",
                payload["addon_params"], f"SGE-{name}",
            )
        finally:
            PROMPTS["entity_extraction_system_prompt"] = original_prompt
            _op.extract_entities = _original_extract_entities

    # 3. Serial-only (SGE chunks + default prompt, no schema — for factorial data)
    if not naive_only:
        serial_only_chunks = [f"[SERIAL_ONLY]\n{c}" for c in chunks]
        serial_only_stats = await run_lightrag(
            serial_only_chunks, config["serial_only_output"] / "lightrag_storage",
            {"language": config["language"]}, f"SerialOnly-{name}",
        )

    # 4. True Naive Baseline (raw CSV text via pd.to_string(), no SGE, no schema)
    csv_text = read_csv_as_text(csv_path)
    naive_chunks = chunk_text(csv_text)
    naive_chunks = [f"[NAIVE_BASELINE]\n{c}" for c in naive_chunks]
    print(f"\n  Naive baseline: {len(naive_chunks)} chunks from raw CSV text")
    naive_stats = await run_lightrag(
        naive_chunks, config["naive_baseline_output"] / "lightrag_storage",
        {"language": config["language"]}, f"NaiveBaseline-{name}",
    )

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": name, "csv": str(csv_path),
        "sge": sge_stats,
        "serial_only": serial_only_stats,
        "naive_baseline": naive_stats,
    }
    out_dir = config["sge_output"] if not naive_only else config["naive_baseline_output"]
    (out_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n{'='*60}")
    print(f"DONE: {name}")
    if not naive_only:
        print(f"  SGE:           {sge_stats.get('nodes', '?')} nodes, {sge_stats.get('edges', '?')} edges")
        print(f"  Serial-only:   {serial_only_stats.get('nodes', '?')} nodes, {serial_only_stats.get('edges', '?')} edges")
    print(f"  Naive Baseline:{naive_stats.get('nodes', '?')} nodes, {naive_stats.get('edges', '?')} edges")
    print(f"{'='*60}")


async def main_async(targets: list[str], naive_only: bool = False):
    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        await run_dataset(name, DATASETS[name], naive_only=naive_only)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["all"])
    parser.add_argument("--naive-only", action="store_true",
                        help="Only run naive baseline (skip SGE and serial-only)")
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    mode = "naive-only" if args.naive_only else "full (SGE + serial-only + naive)"
    print(f"Running: {targets} [{mode}]")
    asyncio.run(main_async(targets, naive_only=args.naive_only))


if __name__ == "__main__":
    main()
