#!/usr/bin/env python3
"""
run_food_rerun.py — Re-run SGE + Baseline LightRAG for food safety dataset.

Uses sge_food/ chunks (re-generated with fixed serializer) and re-runs
both SGE-enhanced and Baseline LightRAG with llm_model_max_async=5.
"""

from __future__ import annotations

import sys
import json
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.integrator import patch_lightrag
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS

API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

SGE_OUTPUT_DIR    = PROJECT_ROOT / "output" / "sge_food_adaptive"
BASELINE_WORK_DIR = PROJECT_ROOT / "output" / "baseline_food_adaptive" / "lightrag_storage"


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


async def run_lightrag(chunks, working_dir, addon_params, label):
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG:{label}] working_dir={working_dir}")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=5,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    print(f"  Inserting {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] ({len(chunk)} chars)")
        await rag.ainsert(chunk)

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats = {
        "label": label,
        "chunks_inserted": len(chunks),
        "graph_file_exists": graph_path.exists(),
        "timestamp": datetime.now().isoformat(),
    }
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["node_count"] = G.number_of_nodes()
        stats["edge_count"] = G.number_of_edges()
        print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges")

    await rag.finalize_storages()
    return stats


async def main():
    # Load chunks
    chunks_dir = SGE_OUTPUT_DIR / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")
    for i, c in enumerate(chunks, 1):
        print(f"  Chunk {i}: {c[:80]}...")

    # Load extraction schema
    schema_path = SGE_OUTPUT_DIR / "extraction_schema.json"
    extraction_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(f"\nEntity types: {extraction_schema['entity_types']}")
    print(f"Relation types: {extraction_schema['relation_types']}")

    # Build LightRAG payload
    payload = patch_lightrag(extraction_schema)

    # ── SGE run ───────────────────────────────────────────────────────────────
    is_baseline_mode = payload.get("use_baseline_mode", False)
    mode_label = "SGE-Adaptive-Baseline" if is_baseline_mode else "SGE-Enhanced"

    print("\n" + "=" * 60)
    print(f"LIGHTRAG RUN — {mode_label} (food)")
    print("=" * 60)

    sge_work = SGE_OUTPUT_DIR / "lightrag_storage"

    original_system_prompt = PROMPTS["entity_extraction_system_prompt"]

    # Only override PROMPTS if schema injection is active (non-baseline mode)
    if not is_baseline_mode and payload["system_prompt"] is not None:
        raw_prompt = payload["system_prompt"]
        escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped
    else:
        print("  [Adaptive] Using LightRAG default prompts (no schema injection)")

    try:
        sge_stats = await run_lightrag(
            chunks=chunks,
            working_dir=sge_work,
            addon_params=payload["addon_params"],
            label=mode_label,
        )
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_system_prompt

    (SGE_OUTPUT_DIR / "lightrag_stats.json").write_text(
        json.dumps(sge_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Baseline run ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTRAG RUN — Baseline (food)")
    print("=" * 60)

    baseline_chunks = [f"[BASELINE]\n{c}" for c in chunks]
    baseline_stats = await run_lightrag(
        chunks=baseline_chunks,
        working_dir=BASELINE_WORK_DIR,
        addon_params={"language": "Chinese"},
        label="Baseline-food",
    )

    baseline_dir = PROJECT_ROOT / "output" / "baseline_food_adaptive"
    (baseline_dir / "lightrag_stats.json").write_text(
        json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for s in [sge_stats, baseline_stats]:
        print(f"  {s['label']}: nodes={s.get('node_count','?')}, edges={s.get('edge_count','?')}")


if __name__ == "__main__":
    asyncio.run(main())
