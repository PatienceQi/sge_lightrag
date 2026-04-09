#!/usr/bin/env python3
"""
run_baseline_only.py — Run ONLY the baseline LightRAG for an existing SGE output dir.

Avoids the SGE→Baseline reset cycle in run_lightrag_from_output.py.
Uses existing chunks from sge_output_dir/chunks/ without touching the SGE storage.

Usage:
    python3 run_baseline_only.py output/inpatient_2023 --label inpatient23
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024


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


async def main_async(sge_output_dir: Path, label: str, fresh: bool):
    chunks_dir = sge_output_dir / "chunks"
    if not chunks_dir.exists():
        print(f"ERROR: chunks dir not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")

    baseline_work = sge_output_dir.parent / f"baseline_{label}" / "lightrag_storage"
    baseline_work.mkdir(parents=True, exist_ok=True)

    if fresh:
        # Remove existing graph so we get a clean start
        graph_file = baseline_work / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            graph_file.unlink()
            print(f"Removed existing graph for fresh run.")

    print(f"\n[Baseline:{label}] working_dir={baseline_work}")
    print(f"  (LLM cache will be reused if available)")

    rag = LightRAG(
        working_dir=str(baseline_work),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": "Chinese"},
        llm_model_max_async=5,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    insert_chunks = [f"[BASELINE]\n{c}" for c in chunks]
    print(f"  Inserting {len(insert_chunks)} chunks with [BASELINE] prefix...")

    for i, chunk in enumerate(insert_chunks, 1):
        print(f"  [{i}/{len(insert_chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    graph_path = baseline_work / "graph_chunk_entity_relation.graphml"
    import networkx as nx
    if graph_path.exists():
        G = nx.read_graphml(str(graph_path))
        print(f"\nBaseline graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("\nWARNING: graph file not found")

    stats = {
        "label": f"Baseline-{label}",
        "timestamp": datetime.now().isoformat(),
        "chunks_inserted": len(insert_chunks),
    }
    stats_path = sge_output_dir.parent / f"baseline_{label}" / "baseline_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline-only LightRAG")
    parser.add_argument("sge_output_dir", help="Path to SGE pipeline output directory")
    parser.add_argument("--label", "-l", default=None)
    parser.add_argument("--fresh", action="store_true", help="Delete existing graph before running")
    args = parser.parse_args()

    sge_dir = Path(args.sge_output_dir).expanduser().resolve()
    label = args.label or sge_dir.name

    asyncio.run(main_async(sge_dir, label, args.fresh))


if __name__ == "__main__":
    main()
