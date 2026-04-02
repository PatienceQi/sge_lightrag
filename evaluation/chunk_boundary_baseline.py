#!/usr/bin/env python3
"""
chunk_boundary_baseline.py — Chunk-Boundary-Aware Baseline for SGE-LightRAG

Tests whether SGE's advantage over naive baseline comes from better chunking
(repeating headers per chunk) rather than Schema constraints.

Approach: feed header-aware chunks to LightRAG with DEFAULT prompt (no Schema),
then evaluate FC against the WHO Life Expectancy gold standard.

Two conditions:
  - n1: 1 data row per chunk, each chunk prefixed with full header row
  - n5: 5 data rows per chunk, each chunk prefixed with full header row

Usage:
    python3 evaluation/chunk_boundary_baseline.py
    python3 evaluation/chunk_boundary_baseline.py --condition n1
    python3 evaluation/chunk_boundary_baseline.py --condition n5
"""

from __future__ import annotations

import sys
import json
import asyncio
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.features import _detect_encoding
from stage1.preprocessor import preprocess_csv

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

CSV_PATH  = PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"
GOLD_PATH = PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_v2.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "chunk_boundary_results.json"

# Reference FC numbers from authoritative results
NAIVE_BASELINE_FC  = 0.167
SERIAL_ONLY_FC     = 0.013
FULL_SGE_FC        = 1.000


# ── LLM function ──────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding (Ollama via urllib3 to bypass macOS proxy) ──────────────────────
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


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_who_csv(csv_path: Path):
    """Load WHO Life Expectancy CSV using stage1 preprocessor (handles metadata)."""
    df, metadata = preprocess_csv(str(csv_path))
    print(f"  CSV shape: {df.shape} (stripped {metadata['rows_stripped']} title rows)")
    print(f"  Encoding: {metadata['original_encoding']}")
    return df


# ── Chunk construction ────────────────────────────────────────────────────────
def build_header_aware_chunks(df, batch_size: int = 1) -> list[str]:
    """
    Build chunks where each chunk starts with the full column header row
    followed by batch_size data rows.

    This tests whether SGE's advantage comes purely from header repetition
    (ensuring each chunk has column context) rather than Schema constraints.
    """
    header = "\t".join(str(c) for c in df.columns)
    chunks = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        chunk_text = header + "\n" + batch.to_string(index=False, header=False)
        chunks.append(f"[HEADER_AWARE]\n{chunk_text}")
    return chunks


# ── LightRAG runner ───────────────────────────────────────────────────────────
async def run_lightrag_with_chunks(
    chunks: list[str],
    work_dir: Path,
    label: str,
) -> dict:
    """Insert chunks into LightRAG (default prompt, no Schema) and return stats."""
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG:{label}] working_dir={work_dir}")
    print(f"  Inserting {len(chunks)} chunks (default prompt, no Schema)...")

    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": "English"},
        llm_model_max_async=5,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    for i, chunk in enumerate(chunks, 1):
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}]")
        await rag.ainsert(chunk)

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    stats = {"label": label, "chunks": len(chunks)}

    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    else:
        print("  WARNING: graph file not found after insertion")
        stats["nodes"] = 0
        stats["edges"] = 0

    await rag.finalize_storages()
    return stats


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_condition(graph_path: Path, gold_path: Path) -> dict:
    """Load graph and gold standard, compute EC and FC metrics."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph,
        check_entity_coverage, check_fact_coverage,
    )

    if not graph_path.exists():
        print(f"  WARNING: graph not found at {graph_path}")
        return {"EC": 0.0, "FC": 0.0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph(str(graph_path))

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  Gold: {len(gold_entities)} entities, {len(facts)} facts")
    print(f"  EC: {len(matched_entities)}/{len(gold_entities)} = {ec:.4f}")
    print(f"  FC: {len(covered)}/{len(facts)} = {fc:.4f}")

    return {
        "EC": round(ec, 4),
        "FC": round(fc, 4),
        "matched_entities": len(matched_entities),
        "covered_facts": len(covered),
        "total_entities": len(gold_entities),
        "total_facts": len(facts),
    }


# ── Condition runner ──────────────────────────────────────────────────────────
async def run_condition(df, batch_size: int, condition_name: str) -> dict:
    """Run one header-aware chunking condition end-to-end."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name} (batch_size={batch_size})")
    print(f"{'='*60}")

    chunks = build_header_aware_chunks(df, batch_size=batch_size)
    print(f"  Built {len(chunks)} header-aware chunks (batch_size={batch_size})")

    work_dir = OUTPUT_DIR / f"chunk_boundary_who_{condition_name}" / "lightrag_storage"
    stats = await run_lightrag_with_chunks(chunks, work_dir, label=condition_name)

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    eval_results = evaluate_condition(graph_path, GOLD_PATH)

    return {
        "chunks": stats["chunks"],
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0),
        "EC": eval_results["EC"],
        "FC": eval_results["FC"],
        "matched_entities": eval_results.get("matched_entities", 0),
        "covered_facts": eval_results.get("covered_facts", 0),
        "total_entities": eval_results.get("total_entities", 0),
        "total_facts": eval_results.get("total_facts", 0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main_async(conditions: list[str]) -> None:
    print(f"\n{'='*60}")
    print("CHUNK-BOUNDARY BASELINE EXPERIMENT")
    print(f"CSV:  {CSV_PATH}")
    print(f"Gold: {GOLD_PATH}")
    print(f"{'='*60}")

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    if not GOLD_PATH.exists():
        print(f"ERROR: Gold standard not found: {GOLD_PATH}", file=sys.stderr)
        sys.exit(1)

    df = load_who_csv(CSV_PATH)

    condition_configs = {"n1": 1, "n5": 5}
    results = {}

    for cond_name in conditions:
        if cond_name not in condition_configs:
            print(f"WARNING: Unknown condition '{cond_name}', skipping")
            continue
        batch_size = condition_configs[cond_name]
        results[cond_name] = await run_condition(df, batch_size, cond_name)

    output = {
        "dataset": "who_life_expectancy",
        "timestamp": datetime.now().isoformat(),
        "conditions": results,
        "comparison": {
            "naive_baseline_fc": NAIVE_BASELINE_FC,
            "serial_only_fc": SERIAL_ONLY_FC,
            "full_sge_fc": FULL_SGE_FC,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {RESULTS_PATH}")
    print(f"{'='*60}")

    print("\nSUMMARY:")
    print(f"  {'Condition':<12} {'Chunks':>8} {'Nodes':>8} {'Edges':>8} {'EC':>8} {'FC':>8}")
    print(f"  {'-'*52}")
    for cond, data in results.items():
        print(f"  {cond:<12} {data['chunks']:>8} {data['nodes']:>8} "
              f"{data['edges']:>8} {data['EC']:>8.4f} {data['FC']:>8.4f}")
    print(f"\n  Reference:")
    print(f"  {'naive_baseline':<20} FC={NAIVE_BASELINE_FC}")
    print(f"  {'serial_only':<20} FC={SERIAL_ONLY_FC}")
    print(f"  {'full_sge':<20} FC={FULL_SGE_FC}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Header-aware chunking baseline for WHO Life Expectancy"
    )
    parser.add_argument(
        "--condition",
        choices=["n1", "n5", "all"],
        default="all",
        help="Which condition(s) to run: n1 (1 row/chunk), n5 (5 rows/chunk), all",
    )
    args = parser.parse_args()

    conditions = ["n1", "n5"] if args.condition == "all" else [args.condition]
    asyncio.run(main_async(conditions))


if __name__ == "__main__":
    main()
