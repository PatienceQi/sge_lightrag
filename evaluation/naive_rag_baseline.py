#!/usr/bin/env python3
"""
naive_rag_baseline.py — Naive RAG (vector-only) baseline for multi-system comparison.

Strategy: Pure vector retrieval, no knowledge graph construction.
  1. Serialize CSV → text chunks (same naive serialization used by LightRAG baseline)
  2. Embed chunks using mxbai-embed-large (Ollama, same as SGE project)
  3. For each QA question: embed query → retrieve top-k chunks → LLM answer
  4. Score using the same score_answer() logic from run_qa_eval.py

Usage:
    python3 evaluation/naive_rag_baseline.py \
        --questions evaluation/gold/qa_questions.jsonl \
        --output evaluation/results/naive_rag_results.json \
        --verbose
"""

from __future__ import annotations

import re
import sys
import json
import asyncio
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import requests as _requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag.llm.openai import openai_complete_if_cache

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM = 1024
TOP_K = 5

SYSTEM_ZH = (
    "你是一个问答助手。根据下面提供的文档上下文，"
    "用简洁的中文回答问题。如果上下文包含具体数值，请直接引用该数值。"
    "只根据给定上下文回答，不要编造信息。"
)

SYSTEM_EN = (
    "You are a QA assistant. Answer the question concisely "
    "based on the provided document context. If the context contains "
    "specific numeric values, quote them directly. Only use the provided context."
)

# ── Dataset → chunk directory mapping ─────────────────────────────────────────
# Chunks are the raw SGE-serialized text files (same data fed to LightRAG baseline)
DATASET_CHUNKS = {
    "who": "output/who_life_expectancy/chunks",
    "wb_cm": "output/wb_child_mortality/chunks",
    "wb_pop": "output/wb_population/chunks",
    "wb_mat": "output/wb_maternal/chunks",
    "inpatient": "output/inpatient_2023/chunks",
    "budget": "output/sge_budget/chunks",
    "food": "output/sge_food_adaptive/chunks",
    "health": "output/sge_health/chunks",
}


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_text_sync(text: str) -> np.ndarray:
    """Embed a single text using Ollama (sync, via requests)."""
    resp = _requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


async def embed_text(text: str) -> np.ndarray:
    """Async wrapper around sync embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_text_sync, text)


async def embed_batch(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Embed a batch of texts with controlled parallelism."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tasks = [embed_text(t) for t in batch]
        batch_embs = await asyncio.gather(*tasks)
        embeddings.extend(batch_embs)
        if (i + batch_size) % 50 < batch_size:
            print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
    return np.stack(embeddings)


def cosine_similarity(query_emb: np.ndarray, chunk_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all chunks."""
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    return chunk_norms @ query_norm


# ── Vector Index ──────────────────────────────────────────────────────────────

class NaiveVectorIndex:
    """Simple in-memory vector index for chunk retrieval."""

    def __init__(self, chunks: list[str], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings

    async def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        query_emb = await embed_text(query)
        scores = cosine_similarity(query_emb, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


# ── Index cache (per dataset) ─────────────────────────────────────────────────

_index_cache: dict[str, NaiveVectorIndex] = {}


async def get_or_build_index(dataset: str) -> NaiveVectorIndex | None:
    """Get cached index or build one from chunk files."""
    if dataset in _index_cache:
        return _index_cache[dataset]

    chunks_dir_rel = DATASET_CHUNKS.get(dataset)
    if not chunks_dir_rel:
        print(f"  [WARN] No chunk directory configured for dataset '{dataset}'")
        return None

    chunks_dir = PROJECT_ROOT / chunks_dir_rel
    if not chunks_dir.exists():
        print(f"  [WARN] Chunks dir not found: {chunks_dir}")
        return None

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    if not chunk_files:
        print(f"  [WARN] No chunk files in {chunks_dir}")
        return None

    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"  Building vector index for {dataset}: {len(chunks)} chunks...")

    # Check for cached embeddings
    cache_file = PROJECT_ROOT / "output" / f"naive_rag_cache_{dataset}.npy"
    if cache_file.exists():
        embeddings = np.load(str(cache_file))
        if len(embeddings) == len(chunks):
            print(f"  Loaded cached embeddings from {cache_file.name}")
            index = NaiveVectorIndex(chunks, embeddings)
            _index_cache[dataset] = index
            return index

    embeddings = await embed_batch(chunks)
    np.save(str(cache_file), embeddings)
    print(f"  Cached embeddings to {cache_file.name}")

    index = NaiveVectorIndex(chunks, embeddings)
    _index_cache[dataset] = index
    return index


# ── Scoring (reused from run_qa_eval.py) ──────────────────────────────────────

def _normalize_number(s: str) -> str:
    s = s.replace(',', '').replace('，', '')
    s = re.sub(r'\.0+\b', '', s)
    return s


def _match(answer: str, value: str) -> bool:
    if not value:
        return False
    al, vl = answer.lower(), value.lower()
    if vl in al:
        return True
    return _normalize_number(vl) in _normalize_number(al)


def score_answer(answer: str, q: dict) -> bool:
    expected = q.get("expected_value", "")
    if _match(answer, expected):
        return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


# ── LLM call ──────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system_prompt: str) -> str:
    return await openai_complete_if_cache(
        MODEL, prompt,
        system_prompt=system_prompt,
        api_key=API_KEY, base_url=BASE_URL,
        max_tokens=256,
        timeout=120,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

async def eval_question(q: dict, index: NaiveVectorIndex, language: str, verbose: bool) -> dict:
    """Evaluate a single question using naive vector retrieval."""
    query = q["question"]
    retrieved_chunks = await index.retrieve(query, top_k=TOP_K)

    context = "\n---\n".join(retrieved_chunks)
    # Limit context to ~3000 chars (same as graph-based eval)
    context = context[:3000]

    system_prompt = SYSTEM_ZH if language == "zh" else SYSTEM_EN
    prompt = (
        f"文档上下文：\n{context}\n\n问题：{query}"
        if language == "zh"
        else f"Document Context:\n{context}\n\nQuestion: {query}"
    )

    try:
        answer = await call_llm(prompt, system_prompt)
    except Exception as e:
        answer = f"[LLM Error: {e}]"

    correct = score_answer(answer, q)

    if verbose:
        print(f"    Context ({len(context)} chars): {context[:150]}...")
        print(f"    Answer [{'✓' if correct else '✗'}]: {answer[:200]}")

    return {
        "answer": answer,
        "correct": correct,
        "context_length": len(context),
        "num_chunks_retrieved": len(retrieved_chunks),
    }


async def run_eval(questions_path: str, output_path: str | None, verbose: bool):
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions")
    print(f"Naive RAG: vector retrieval (top-{TOP_K}) + LLM answering\n")

    results = []
    datasets = list(dict.fromkeys(q["dataset"] for q in questions))

    for ds in datasets:
        ds_qs = [q for q in questions if q["dataset"] == ds]
        lang = ds_qs[0]["language"]

        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds.upper()} ({lang}) — {len(ds_qs)} questions")
        print(f"{'=' * 60}")

        index = await get_or_build_index(ds)
        if not index:
            print(f"  SKIPPED (no index available)")
            for q in ds_qs:
                results.append({
                    "id": q["id"], "dataset": ds,
                    "question": q["question"],
                    "expected_value": q["expected_value"],
                    "naive_rag_answer": "[NO INDEX]",
                    "naive_rag_correct": False,
                })
            continue

        for q in ds_qs:
            print(f"\n  [{q['id']}] {q['question']}")
            print(f"  Expected: {q['expected_value']}")

            result = await eval_question(q, index, lang, verbose)
            mark = '✓' if result['correct'] else '✗'

            if not verbose:
                print(f"  NaiveRAG [{mark}]: {result['answer'][:120]}")

            results.append({
                "id": q["id"],
                "dataset": ds,
                "question": q["question"],
                "expected_value": q["expected_value"],
                "entity": q.get("entity", ""),
                "naive_rag_answer": result["answer"],
                "naive_rag_correct": result["correct"],
                "context_length": result["context_length"],
            })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("NAIVE RAG EVALUATION RESULTS")
    print(f"{'=' * 60}")

    header = f"{'Dataset':<12} | {'NaiveRAG':^9}"
    print(header)
    print("-" * len(header))

    summary = {}
    for ds in datasets:
        ds_r = [r for r in results if r["dataset"] == ds]
        n = len(ds_r)
        correct = sum(r["naive_rag_correct"] for r in ds_r)
        print(f"{ds:<12} | {correct}/{n} ({correct / n:.0%})")
        summary[ds] = {"n": n, "correct": correct, "accuracy": correct / n}

    total = len(results)
    total_correct = sum(r["naive_rag_correct"] for r in results)
    print("-" * len(header))
    print(f"{'OVERALL':<12} | {total_correct}/{total} ({total_correct / total:.0%})")
    print(f"{'=' * 60}")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "system": "naive_rag",
        "config": {
            "embedding_model": EMBED_MODEL,
            "top_k": TOP_K,
            "llm_model": MODEL,
        },
        "questions_file": questions_path,
        "total_questions": total,
        "summary": summary,
        "overall": {
            "correct": total_correct,
            "accuracy": total_correct / total,
        },
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Naive RAG baseline evaluation")
    parser.add_argument("--questions", default="evaluation/gold/qa_questions.jsonl")
    parser.add_argument("--output", "-o", default="evaluation/results/naive_rag_results.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_eval(args.questions, args.output, args.verbose))


if __name__ == "__main__":
    main()
