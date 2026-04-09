#!/usr/bin/env python3
"""
nanographrag_baseline.py — nano-GraphRAG baseline for multi-system comparison.

Builds a knowledge graph from serialized CSV chunks using nano-graphrag, then
exports the graph as GraphML so it can be evaluated with evaluate_coverage.py.

nano-graphrag outputs graph_{namespace}.graphml in its working_dir.
The namespace used by GraphRAG for the entity-relation graph is
"chunk_entity_relation", so the output file is:
  <working_dir>/graph_chunk_entity_relation.graphml

This is the same filename pattern as LightRAG, so load_graphml() in
graph_loaders.py works without modification.

Chunk sourcing strategy:
  1. Use pre-computed chunks from output/<sge_dir>/chunks/*.txt (preferred).
     These are the same chunks as the SGE/Baseline LightRAG runs, produced by
     the LLM-enhanced Stage 2 serializer.
  2. If no pre-computed chunks exist, fall back to rule-based serialization from CSV.

Usage:
    python3 evaluation/nanographrag_baseline.py --dataset who
    python3 evaluation/nanographrag_baseline.py --dataset who --output output/nanographrag_who

Configuration:
    LLM  : Claude Haiku via wolfai proxy (OpenAI-compatible)
    Embed: mxbai-embed-large via Ollama (local)

Requires:
    pip install --break-system-packages --no-deps nano-graphrag neo4j hnswlib xxhash graspologic
    (graspologic leiden is monkey-patched to a no-op since the full dep chain
     is incompatible with Python 3.14; entity extraction still works correctly)
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Patch leiden clustering BEFORE importing GraphRAG.
# graspologic has deep deps that won't install on Python 3.14;
# community reports are not needed for FC evaluation.
# ---------------------------------------------------------------------------
from nano_graphrag._storage.gdb_networkx import NetworkXStorage


async def _noop_leiden(self) -> None:
    """No-op replacement for leiden clustering (graspologic unavailable)."""
    return


NetworkXStorage._leiden_clustering = _noop_leiden  # type: ignore[method-assign]

# Now safe to import GraphRAG
from nano_graphrag import GraphRAG
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_API_KEY = os.environ.get("SGE_API_KEY", "")
LLM_BASE_URL = "https://wolfai.top/v1"
LLM_MODEL = "claude-haiku-4-5-20251001"

EMBED_MODEL = "mxbai-embed-large"
EMBED_BASE_URL = "http://127.0.0.1:11434"
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 512

LLM_MAX_ASYNC = 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset → pre-computed SGE chunks directory (primary source)
# These chunks were produced by the LLM-enhanced Stage 2 serializer and are
# identical to what the existing LightRAG Baseline runs received.
DATASET_CHUNKS_DIR = {
    "who":       "output/who_life_expectancy/chunks",
    "wb_cm":     "output/wb_child_mortality/chunks",
    "wb_pop":    "output/wb_population/chunks",
    "wb_mat":    "output/wb_maternal/chunks",
    "inpatient": "output/inpatient_2023/chunks",
}

# Dataset → CSV path (fallback serialization if pre-computed chunks unavailable)
DATASET_CSV = {
    "who":       "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
    "wb_cm":     "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
    "wb_pop":    "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
    "wb_mat":    "dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
    "inpatient": None,  # inpatient: use pre-computed chunks only
}

# ---------------------------------------------------------------------------
# LLM function (OpenAI-compatible proxy)
# ---------------------------------------------------------------------------

async def _haiku_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Call Claude Haiku via wolfai OpenAI-compatible proxy."""
    from openai import AsyncOpenAI

    if history_messages is None:
        history_messages = []

    # nano-graphrag passes hashing_kv via kwargs for LLM cache
    hashing_kv = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Check cache
    if hashing_kv is not None:
        from nano_graphrag._utils import compute_args_hash
        args_hash = compute_args_hash(LLM_MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        **kwargs,
    )
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": LLM_MODEL}}
        )
        await hashing_kv.index_done_callback()

    return result


# ---------------------------------------------------------------------------
# Embedding function (Ollama mxbai-embed-large)
# ---------------------------------------------------------------------------

@wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=EMBED_MAX_TOKENS)
async def _ollama_embed(texts: List[str]) -> np.ndarray:
    """Embed texts via Ollama mxbai-embed-large using urllib3 (bypass macOS proxy)."""
    import urllib3 as _urllib3

    _pool = _urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)
    embeddings = []
    for text in texts:
        if len(text) > 1000:
            text = text[:1000]
        body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
        resp = _pool.urlopen(
            "POST", "/api/embeddings",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        if resp.status != 200:
            raise RuntimeError(f"Ollama {resp.status}: {resp.data[:200]}")
        emb = json.loads(resp.data)["embedding"]
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Chunk loading: pre-computed chunks (primary) or CSV fallback
# ---------------------------------------------------------------------------

def _load_precomputed_chunks(chunks_dir: Path) -> list[str]:
    """Load pre-computed text chunks from a directory of *.txt files."""
    chunk_files = sorted(chunks_dir.glob("*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"  Loaded {len(chunks)} pre-computed chunks from {chunks_dir}")
    return chunks


def _serialize_csv_fallback(csv_path: str) -> list[str]:
    """
    Fallback: serialize a CSV using rule-based Stage 1+2 pipeline.
    Note: rule-based Stage 2 may produce fewer chunks than LLM Stage 2
    for some dataset types (e.g., Type-II World Bank CSVs).
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    from stage1.features import extract_features
    from stage1.classifier import classify
    from stage1.schema import build_meta_schema
    from stage2.inductor import induce_schema_from_meta
    from stage3.serializer import serialize_csv

    features = extract_features(csv_path)
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    schema = induce_schema_from_meta(features, table_type, meta_schema)
    schema["_n_rows"] = features.n_rows
    if "time_dimension" not in schema:
        schema["time_dimension"] = meta_schema.get("time_dimension", {})

    chunks = serialize_csv(csv_path, schema)
    print(f"  [Fallback] Serialized {len(chunks)} chunks from {Path(csv_path).name}")
    return chunks


def _load_chunks(dataset: str) -> list[str]:
    """
    Load chunks for a dataset.
    Strategy:
      1. Pre-computed chunks from output/<dataset>/chunks/ (preferred)
      2. Rule-based CSV serialization (fallback)
    """
    chunks_rel = DATASET_CHUNKS_DIR.get(dataset)
    if chunks_rel:
        chunks_dir = PROJECT_ROOT / chunks_rel
        if chunks_dir.exists():
            chunks = _load_precomputed_chunks(chunks_dir)
            if chunks:
                return chunks
            print(f"  WARNING: chunks dir empty: {chunks_dir}")

    csv_rel = DATASET_CSV.get(dataset)
    if not csv_rel:
        raise FileNotFoundError(
            f"No pre-computed chunks found and no CSV fallback for dataset '{dataset}'. "
            f"Run the SGE pipeline first: python3 run_pipeline.py <csv_path> --output output/{dataset}"
        )

    csv_path = PROJECT_ROOT / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"  Pre-computed chunks not found, falling back to rule-based serialization...")
    return _serialize_csv_fallback(str(csv_path))


# ---------------------------------------------------------------------------
# Main indexing function
# ---------------------------------------------------------------------------

def run_index(dataset: str, output_dir: str) -> None:
    """Build nano-graphrag knowledge graph from a dataset CSV."""
    csv_path_rel = DATASET_CSV.get(dataset)
    if csv_path_rel is None:
        raise ValueError(
            f"Dataset '{dataset}' not supported. "
            f"Supported: {list(k for k, v in DATASET_CSV.items() if v)}"
        )

    csv_path = PROJECT_ROOT / csv_path_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    working_dir = PROJECT_ROOT / output_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[nano-GraphRAG Baseline] dataset={dataset}")
    print(f"  CSV         : {csv_path}")
    print(f"  Working dir : {working_dir}")
    print(f"  LLM         : {LLM_MODEL} via {LLM_BASE_URL}")
    print(f"  Embedding   : {EMBED_MODEL} via {EMBED_BASE_URL}")

    # Load chunks (prefer pre-computed, fallback to CSV serialization)
    print("\nLoading chunks...")
    chunks = _load_chunks(dataset)
    if not chunks:
        raise ValueError(f"No chunks produced for dataset={dataset}")

    # Initialize GraphRAG
    rag = GraphRAG(
        working_dir=str(working_dir),
        best_model_func=_haiku_complete,
        cheap_model_func=_haiku_complete,
        best_model_max_async=LLM_MAX_ASYNC,
        cheap_model_max_async=LLM_MAX_ASYNC,
        embedding_func=_ollama_embed,
        embedding_func_max_async=4,
        enable_llm_cache=True,
    )

    # Insert all chunks
    print(f"\nIndexing {len(chunks)} chunks into nano-graphrag...")
    start = datetime.now()
    rag.insert(chunks)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Indexing complete in {elapsed:.1f}s")

    # Report graph stats
    graphml_path = working_dir / "graph_chunk_entity_relation.graphml"
    if graphml_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graphml_path))
        print(f"\nGraph stats:")
        print(f"  Nodes : {G.number_of_nodes()}")
        print(f"  Edges : {G.number_of_edges()}")
        print(f"  GraphML: {graphml_path}")
    else:
        print(f"\nWARNING: GraphML not found at {graphml_path}")

    # Save run metadata
    stats = {
        "timestamp": datetime.now().isoformat(),
        "system": "nano_graphrag_baseline",
        "dataset": dataset,
        "csv_path": str(csv_path),
        "working_dir": str(working_dir),
        "config": {
            "llm_model": LLM_MODEL,
            "llm_base_url": LLM_BASE_URL,
            "embedding_model": EMBED_MODEL,
            "embedding_base_url": EMBED_BASE_URL,
        },
        "chunks_inserted": len(chunks),
        "elapsed_seconds": elapsed,
        "graphml_path": str(graphml_path) if graphml_path.exists() else None,
    }
    stats_path = working_dir / "nanographrag_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stats saved to {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="nano-GraphRAG baseline: index a CSV dataset and export GraphML"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[k for k, v in DATASET_CSV.items() if v],
        help="Dataset to index",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for nano-graphrag index (default: output/nanographrag_<dataset>)",
    )
    args = parser.parse_args()

    output_dir = args.output or f"output/nanographrag_{args.dataset}"
    run_index(args.dataset, output_dir)


if __name__ == "__main__":
    main()
