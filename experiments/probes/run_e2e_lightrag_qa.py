#!/usr/bin/env python3
"""
run_e2e_lightrag_qa.py — End-to-end LightRAG Query QA Evaluation.

Unlike run_qa_eval.py (direct graph context retrieval), this script uses
LightRAG's full query pipeline: vector retrieval → graph exploration → LLM synthesis.

This tests the complete system, not just graph construction quality.

Usage:
    python3 experiments/run_e2e_lightrag_qa.py \
        --questions evaluation/gold/qa_questions.jsonl \
        --output experiments/results/e2e_qa_results.json \
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM = 1024


# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
):
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


# ── Embedding function ───────────────────────────────────────────────────────
import aiohttp


async def _ollama_embed_native(texts: list[str]) -> np.ndarray:
    """Call Ollama's native API directly, bypassing OpenAI-compat layer."""
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            async with session.post(
                "http://localhost:11434/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
                else:
                    raise RuntimeError(f"Ollama embedding failed: {resp.status}")
    return np.array(embeddings, dtype=np.float32)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        return await _ollama_embed_native(texts)
    except Exception as e:
        print(f"  [warn] Ollama native embedding failed ({e}), using hash fallback")
        return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=safe_embedding_func,
)


# ── Scoring (reused from run_qa_eval.py) ─────────────────────────────────────
def _normalize_number(s: str) -> str:
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\.0+\b", "", s)
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


# ── LightRAG query modes to test ────────────────────────────────────────────
QUERY_MODES = ["hybrid", "local", "global"]


async def init_rag(working_dir: str) -> LightRAG:
    """Initialize a LightRAG instance from existing storage."""
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": "Chinese"},
        llm_model_max_async=10,
        embedding_func_max_async=2,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    return rag


async def query_rag(rag: LightRAG, question: str, mode: str = "hybrid") -> str:
    """Query LightRAG using its full pipeline with retry."""
    for attempt in range(2):
        try:
            result = await rag.aquery(
                question,
                param=QueryParam(mode=mode, only_need_context=False),
            )
            if result:
                return str(result)
            # If None/empty, try local mode as fallback
            if mode == "hybrid" and attempt == 0:
                result = await rag.aquery(
                    question,
                    param=QueryParam(mode="local", only_need_context=False),
                )
                if result:
                    return str(result)
            return "[No context found]"
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(2)
            else:
                return f"[Query error: {e}]"
    return "[Query failed]"


async def eval_question_e2e(
    q: dict,
    sge_rag: LightRAG,
    baseline_rag: LightRAG,
    mode: str,
    verbose: bool,
) -> dict:
    """Evaluate a single question using full LightRAG query pipeline."""
    question = q["question"]

    sge_answer = await query_rag(sge_rag, question, mode)
    baseline_answer = await query_rag(baseline_rag, question, mode)

    sge_correct = score_answer(sge_answer, q)
    baseline_correct = score_answer(baseline_answer, q)

    if verbose:
        sge_mark = "✓" if sge_correct else "✗"
        base_mark = "✓" if baseline_correct else "✗"
        print(f"    SGE [{sge_mark}]: {sge_answer[:150]}")
        print(f"    Base[{base_mark}]: {baseline_answer[:150]}")

    return {
        "sge_answer": sge_answer,
        "sge_correct": sge_correct,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_correct,
    }


async def run_eval(questions_path: str, output_path: str | None, verbose: bool, mode: str):
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions")
    print(f"Query mode: {mode}")

    # Group questions by (sge_graph, baseline_graph) to reuse RAG instances
    graph_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = []
    for q in questions:
        sge_path = str(PROJECT_ROOT / q["sge_graph"])
        base_path = str(PROJECT_ROOT / q["baseline_graph"])
        graphml_sge = Path(sge_path) / "graph_chunk_entity_relation.graphml"
        graphml_base = Path(base_path) / "graph_chunk_entity_relation.graphml"
        if not graphml_sge.exists() or not graphml_base.exists():
            skipped.append(q["id"])
            continue
        graph_groups[(q["sge_graph"], q["baseline_graph"])].append(q)

    if skipped:
        print(f"Skipped {len(skipped)} questions (missing graphs): {skipped[:5]}...")

    results = []
    datasets = list(dict.fromkeys(q["dataset"] for q in questions))

    for (sge_graph_dir, base_graph_dir), group_qs in graph_groups.items():
        sge_path = str(PROJECT_ROOT / sge_graph_dir)
        base_path = str(PROJECT_ROOT / base_graph_dir)

        ds_name = group_qs[0]["dataset"]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()} — {len(group_qs)} questions")
        print(f"  SGE:  {sge_graph_dir}")
        print(f"  Base: {base_graph_dir}")
        print(f"{'='*60}")

        # Initialize RAG instances
        print("  Initializing LightRAG instances...")
        sge_rag = await init_rag(sge_path)
        baseline_rag = await init_rag(base_path)

        for q in group_qs:
            print(f"\n  [{q['id']}] {q['question']}")
            print(f"  Expected: {q['expected_value']}")

            result = await eval_question_e2e(q, sge_rag, baseline_rag, mode, verbose)
            sge_mark = "✓" if result["sge_correct"] else "✗"
            base_mark = "✓" if result["baseline_correct"] else "✗"

            if not verbose:
                print(f"  SGE [{sge_mark}]: {result['sge_answer'][:120]}")
                print(f"  Base[{base_mark}]: {result['baseline_answer'][:120]}")

            results.append({
                "id": q["id"],
                "dataset": q["dataset"],
                "type": q.get("type", "direct"),
                "question": q["question"],
                "expected_value": q["expected_value"],
                "entity": q.get("entity", ""),
                "sge_answer": result["sge_answer"],
                "sge_correct": result["sge_correct"],
                "baseline_answer": result["baseline_answer"],
                "baseline_correct": result["baseline_correct"],
                "query_mode": mode,
            })

        # Cleanup
        await sge_rag.finalize_storages()
        await baseline_rag.finalize_storages()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"E2E LIGHTRAG QA RESULTS (mode={mode})")
    print(f"{'='*60}")

    header = f"{'Dataset':<20} | {'SGE':^9} | {'Baseline':^9} | Δ"
    print(header)
    print("-" * len(header))

    summary = {}
    for ds in datasets:
        ds_r = [r for r in results if r["dataset"] == ds]
        if not ds_r:
            continue
        n = len(ds_r)
        sge = sum(r["sge_correct"] for r in ds_r)
        base = sum(r["baseline_correct"] for r in ds_r)
        delta = sge - base
        sign = "+" if delta > 0 else ""
        print(f"{ds:<20} | {sge}/{n} ({sge/n:.0%})  | {base}/{n} ({base/n:.0%}) | {sign}{delta}")
        summary[ds] = {
            "n": n,
            "sge_correct": sge,
            "baseline_correct": base,
            "sge_accuracy": round(sge / n, 4),
            "baseline_accuracy": round(base / n, 4),
        }

    total = len(results)
    sge_t = sum(r["sge_correct"] for r in results)
    base_t = sum(r["baseline_correct"] for r in results)
    delta_t = sge_t - base_t
    sign_t = "+" if delta_t > 0 else ""
    print("-" * len(header))
    print(f"{'OVERALL':<20} | {sge_t}/{total} ({sge_t/total:.0%}) | {base_t}/{total} ({base_t/total:.0%}) | {sign_t}{delta_t}")
    print(f"{'='*60}")

    if skipped:
        print(f"\nSkipped {len(skipped)} questions (missing graph storage)")

    # Compare with direct context results
    direct_results_path = PROJECT_ROOT / "evaluation" / "results" / "qa_results_v3_100q.json"
    if direct_results_path.exists():
        with open(direct_results_path) as f:
            direct = json.load(f)
        print(f"\n--- Comparison with Direct Graph Context ---")
        print(f"  Direct: SGE {direct['overall']['sge_correct']}/{direct['overall'].get('total', 100)} "
              f"({direct['overall']['sge_accuracy']:.0%}) | "
              f"Base {direct['overall']['baseline_correct']}/{direct['overall'].get('total', 100)} "
              f"({direct['overall']['baseline_accuracy']:.0%})")
        print(f"  E2E:    SGE {sge_t}/{total} ({sge_t/total:.0%}) | Base {base_t}/{total} ({base_t/total:.0%})")

    # By question type
    print(f"\n--- By Question Type ---")
    for qtype in ["direct", "comparison", "trend"]:
        type_r = [r for r in results if r.get("type") == qtype]
        if type_r:
            n = len(type_r)
            sge = sum(r["sge_correct"] for r in type_r)
            base = sum(r["baseline_correct"] for r in type_r)
            print(f"  {qtype:<12}: SGE {sge}/{n} ({sge/n:.0%}) | Base {base}/{n} ({base/n:.0%})")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "end_to_end_lightrag_query",
        "query_mode": mode,
        "questions_file": questions_path,
        "total_questions": total,
        "skipped_questions": len(skipped),
        "skipped_ids": skipped,
        "summary": summary,
        "overall": {
            "sge_correct": sge_t,
            "baseline_correct": base_t,
            "total": total,
            "sge_accuracy": round(sge_t / total, 4),
            "baseline_accuracy": round(base_t / total, 4),
        },
        "by_type": {},
        "results": results,
    }

    for qtype in ["direct", "comparison", "trend"]:
        type_r = [r for r in results if r.get("type") == qtype]
        if type_r:
            n = len(type_r)
            output_data["by_type"][qtype] = {
                "n": n,
                "sge_correct": sum(r["sge_correct"] for r in type_r),
                "baseline_correct": sum(r["baseline_correct"] for r in type_r),
            }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="End-to-end LightRAG QA evaluation")
    parser.add_argument("--questions", default="evaluation/gold/qa_questions.jsonl")
    parser.add_argument("--output", "-o", default="experiments/results/e2e_qa_results.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "local", "global"],
                        help="LightRAG query mode (default: hybrid)")
    args = parser.parse_args()

    asyncio.run(run_eval(args.questions, args.output, args.verbose, args.mode))


if __name__ == "__main__":
    main()
