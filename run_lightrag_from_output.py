#!/usr/bin/env python3
"""
run_lightrag_from_output.py — Feed pre-generated SGE output into LightRAG.

Takes an existing SGE pipeline output directory (with chunks/ and extraction_schema.json)
and runs LightRAG with schema-aware prompt injection + a vanilla baseline.

Usage:
    python3 run_lightrag_from_output.py output/llm_budget
    python3 run_lightrag_from_output.py output/llm_food --label food_llm
"""

from __future__ import annotations

import sys
import json
import asyncio
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.integrator import patch_lightrag
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM          = 1024


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL,
        timeout=120,  # 防止代理静默断开导致无限等待
        **kwargs,
    )


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec

_raw_openai_embed = openai_embed.func

async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    # Use hash embedding directly — Ollama returns 503 under concurrent load,
    # causing excessive retries. Our evaluation reads graphml directly (not vdb),
    # so embedding quality does not affect EC/FC/η metrics.
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)

EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)

_original_extract_entities = _op.extract_entities


async def run_lightrag(chunks, working_dir, addon_params, label, baseline=False):
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG:{label}] working_dir={working_dir}")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=10,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    insert_chunks = chunks if not baseline else [f"[BASELINE]\n{c}" for c in chunks]
    print(f"  Inserting {len(insert_chunks)} chunks...")
    for i, chunk in enumerate(insert_chunks, 1):
        print(f"  [{i}/{len(insert_chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats = {
        "label": label,
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
            print(f"  [warn] Could not parse graphml: {e}")

    await rag.finalize_storages()
    return stats


async def main_async(sge_output_dir: Path, label: str):
    # Load pre-generated chunks
    chunks_dir = sge_output_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    if not chunk_files:
        print(f"Error: No chunks found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")

    # Load extraction schema
    schema_path = sge_output_dir / "extraction_schema.json"
    extraction_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(f"Entity types: {extraction_schema['entity_types']}")
    print(f"Relation types: {extraction_schema['relation_types']}")

    # Build LightRAG payload
    payload = patch_lightrag(extraction_schema)

    sge_work = sge_output_dir / "lightrag_storage"
    baseline_work = sge_output_dir.parent / f"baseline_{label}" / "lightrag_storage"

    # ── SGE-enhanced run ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"LIGHTRAG RUN — SGE-Enhanced ({label})")
    print("=" * 60)

    original_system_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        sge_stats = await run_lightrag(
            chunks=chunks, working_dir=sge_work,
            addon_params=payload["addon_params"], label=f"SGE-{label}",
        )
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_system_prompt

    (sge_output_dir / "lightrag_stats.json").write_text(
        json.dumps(sge_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Baseline run ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"LIGHTRAG RUN — Baseline ({label})")
    print("=" * 60)

    baseline_dir = sge_output_dir.parent / f"baseline_{label}"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_stats = await run_lightrag(
        chunks=chunks, working_dir=baseline_work,
        addon_params={"language": "Chinese"}, label=f"Baseline-{label}",
        baseline=True,
    )

    (baseline_dir / "lightrag_stats.json").write_text(
        json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for s in [sge_stats, baseline_stats]:
        print(f"  {s['label']}: nodes={s.get('node_count','?')}, edges={s.get('edge_count','?')}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "sge": sge_stats,
        "baseline": baseline_stats,
        "entity_types": extraction_schema["entity_types"],
    }
    report_path = sge_output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LightRAG from pre-generated SGE output")
    parser.add_argument("sge_output_dir", help="Path to SGE pipeline output directory")
    parser.add_argument("--label", "-l", default=None, help="Label for this run (default: dir name)")
    args = parser.parse_args()

    sge_dir = Path(args.sge_output_dir).expanduser().resolve()
    label = args.label or sge_dir.name

    asyncio.run(main_async(sge_dir, label))


if __name__ == "__main__":
    main()
