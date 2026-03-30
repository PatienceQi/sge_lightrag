#!/usr/bin/env python3
"""
insert_missing_gold_chunks.py — Insert only the missing gold-country chunks
into existing LightRAG storages for wb_population and wb_maternal.
"""
from __future__ import annotations
import sys, json, asyncio, hashlib, numpy as np
from pathlib import Path

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

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL,
        timeout=120,  # 防止代理静默断开导致无限等待
        **kwargs,
    )

def _hash_embed(text):
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec

async def safe_embed(texts):
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)

EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embed
)

MISSING = {
    "wb_population": {
        "chunks_dir":   "output/wb_population/chunks",
        "sge_dir":      "output/wb_population",
        "baseline_dir": "output/baseline_wb_population",
        "schema_path":  "output/wb_population/extraction_schema.json",
        "missing_chunks": [
            "chunk_0082.txt",   # United Kingdom
            "chunk_0110.txt",   # India
            "chunk_0119.txt",   # Japan
            "chunk_0251.txt",   # United States
        ],
    },
    "wb_maternal": {
        "chunks_dir":   "output/wb_maternal/chunks",
        "sge_dir":      "output/wb_maternal",
        "baseline_dir": "output/baseline_wb_maternal",
        "schema_path":  "output/wb_maternal/extraction_schema.json",
        "missing_chunks": [
            "chunk_0075.txt",   # United Kingdom
            "chunk_0098.txt",   # India
            "chunk_0107.txt",   # Japan
            "chunk_0231.txt",   # United States
        ],
    },
}


async def insert_into_rag(working_dir: Path, chunks: list[str], addon_params: dict, label: str):
    """Create LightRAG instance on existing storage and insert chunks."""
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

    print(f"  [{label}] Inserting {len(chunks)} chunks into {working_dir.name}...")
    for i, chunk in enumerate(chunks, 1):
        first = chunk.split('\n')[0][:60]
        print(f"    [{i}/{len(chunks)}] {first}")
        await rag.ainsert(chunk)
    print(f"  [{label}] Done!")


async def process_dataset(dataset_name: str, config: dict):
    chunks_dir = PROJECT_ROOT / config["chunks_dir"]
    schema_path = PROJECT_ROOT / config["schema_path"]
    sge_work = PROJECT_ROOT / config["sge_dir"] / "lightrag_storage"
    baseline_work = PROJECT_ROOT / config["baseline_dir"] / "lightrag_storage"

    schema = json.loads(schema_path.read_text())
    payload = patch_lightrag(schema)

    chunks_to_insert = []
    for fname in config["missing_chunks"]:
        chunk_path = chunks_dir / fname
        if chunk_path.exists():
            content = chunk_path.read_text()
            chunks_to_insert.append(content)
            print(f"  Queued: {fname} ({content.split(chr(10))[0][:60]})")
        else:
            print(f"  WARNING: {fname} not found")

    if not chunks_to_insert:
        print(f"[{dataset_name}] No chunks to insert")
        return

    # SGE run — with schema-guided system prompt
    orig_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        await insert_into_rag(sge_work, chunks_to_insert, payload["addon_params"], f"SGE-{dataset_name}")
    finally:
        PROMPTS["entity_extraction_system_prompt"] = orig_prompt

    # Baseline run — plain, no schema
    baseline_work.mkdir(parents=True, exist_ok=True)
    baseline_chunks = [f"[BASELINE]\n{c}" for c in chunks_to_insert]

    # Check if baseline storage exists (may need to init from scratch)
    baseline_graphml = baseline_work / "graph_chunk_entity_relation.graphml"
    print(f"\n  Baseline storage exists: {baseline_graphml.exists()}")

    await insert_into_rag(baseline_work, baseline_chunks, {"language": "English"}, f"Baseline-{dataset_name}")


async def main():
    for ds_name, config in MISSING.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        await process_dataset(ds_name, config)
    print("\n\nAll insertions complete!")


if __name__ == "__main__":
    asyncio.run(main())
