#!/usr/bin/env python3
"""
run_ggcr_eval.py — Main evaluation: 6 systems × ~97 questions.

Systems:
  1. sge_ggcr      — Graph BFS entity enumeration → compact chunks → LLM
  2. pure_compact   — Vector retrieval on compact chunks → LLM (no graph)
  3. graph_native   — Pure graph traversal + deterministic rules (no LLM)
  4. naive_rag      — Vector retrieval on naive serialization → LLM
  5. concat_all     — Concatenate ALL compact chunks → LLM (no retrieval)
  6. baseline_ggcr  — Same as sge_ggcr but entity enumeration from Baseline graph

Usage:
    python3 experiments/ggcr/run_ggcr_eval.py [--systems sge_ggcr,pure_compact] [--verbose]
"""

from __future__ import annotations

import re
import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ggcr.compact_chunks import build_compact_index, _embed_text_sync, _embed_batch_sync, cosine_similarity
from experiments.ggcr.graph_guided_retriever import (
    retrieve_ggcr, retrieve_pure_compact, retrieve_concat_all,
    retrieve_naive_rag, retrieve_graph_native, retrieve_baseline_ggcr,
)

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "gold" / "ggcr_benchmark.jsonl"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ggcr_results.json"

# Naive RAG chunk directories (same serialization as LightRAG baseline)
NAIVE_CHUNKS_DIRS = {
    "who": "output/who_life_expectancy/chunks",
    "wb_cm": "output/wb_child_mortality/chunks",
    "inpatient": "output/inpatient_2023/chunks",
}


# ---------------------------------------------------------------------------
# Scoring functions (adapted from run_qa_eval.py)
# ---------------------------------------------------------------------------

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


def score_answer(answer: str, question: dict) -> bool:
    """Score an answer against a benchmark question."""
    eval_type = question["evaluation_type"]
    ref = question["reference_answer"]

    if eval_type == "exact_match":
        # Check if reference answer appears in the answer
        if _match(answer, ref):
            return True
        # Also check reference_set items
        ref_set = question.get("reference_set") or []
        for item in ref_set:
            if _match(answer, item):
                return True
        return False

    elif eval_type == "direction":
        # Check yes/no or 是/否
        answer_lower = answer.lower().strip()
        ref_lower = ref.lower().strip()
        if ref_lower in ("yes", "是"):
            return any(w in answer_lower for w in ["yes", "是", "correct", "true", "对", "consistently"])
        else:
            return any(w in answer_lower for w in ["no", "否", "not", "false", "没有", "fluctuat", "不是"])

    elif eval_type == "set_overlap":
        # Check how many reference set items appear in the answer
        ref_set = question.get("reference_set", [])
        if not ref_set:
            return False
        matched = sum(1 for item in ref_set if _match(answer, item))
        # Require at least 60% overlap
        threshold = max(1, int(len(ref_set) * 0.6))
        return matched >= threshold

    elif eval_type == "numeric_tolerance":
        ref_num = question.get("reference_numeric")
        if ref_num is None:
            return _match(answer, ref)
        # Extract numbers from answer
        numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        for num_str in numbers:
            try:
                predicted = float(num_str)
                if ref_num == 0:
                    if abs(predicted) < 0.1:
                        return True
                elif abs(predicted - ref_num) / abs(ref_num) < 0.10:  # 10% tolerance
                    return True
            except ValueError:
                continue
        # Fallback: substring match
        return _match(answer, ref)

    return False


# ---------------------------------------------------------------------------
# Naive RAG index builder
# ---------------------------------------------------------------------------

def _load_naive_chunks(dataset: str) -> tuple[list[str], np.ndarray] | None:
    """Load and embed naive serialization chunks for a dataset."""
    chunks_dir_rel = NAIVE_CHUNKS_DIRS.get(dataset)
    if not chunks_dir_rel:
        return None
    chunks_dir = PROJECT_ROOT / chunks_dir_rel
    if not chunks_dir.exists():
        print(f"  [WARN] Naive chunks dir not found: {chunks_dir}")
        return None

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    if not chunk_files:
        return None

    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]

    # Check for cached embeddings
    cache_dir = PROJECT_ROOT / "output" / "ggcr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"naive_emb_{dataset}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(chunks):
            return chunks, embeddings

    print(f"  Embedding {len(chunks)} naive chunks for {dataset}...")
    embeddings = _embed_batch_sync(chunks)
    np.save(cache_path, embeddings)
    return chunks, embeddings


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

ALL_SYSTEMS = ["sge_ggcr", "pure_compact", "graph_native", "naive_rag", "concat_all", "baseline_ggcr"]


def run_evaluation(systems: list[str] | None = None, verbose: bool = False):
    """Run full GGCR evaluation."""
    systems = systems or ALL_SYSTEMS

    # Load benchmark
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} benchmark questions")

    # Group by dataset
    datasets = sorted(set(q["dataset"] for q in questions))
    print(f"Datasets: {datasets}")

    # Pre-build indices
    compact_indices = {}
    naive_indices = {}

    for ds in datasets:
        print(f"\nBuilding compact index for {ds}...")
        compact_indices[ds] = build_compact_index(ds, embed=True)

        if "naive_rag" in systems:
            print(f"Loading naive chunks for {ds}...")
            result = _load_naive_chunks(ds)
            if result:
                naive_indices[ds] = result

    # Run evaluation with concurrency
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    detailed_results = []
    results_lock = threading.Lock()

    total = len(questions) * len(systems)
    done_count = [0]  # mutable counter for threads

    def _run_one(q, system):
        """Run one (question, system) pair. Thread-safe."""
        ds = q["dataset"]
        qid = q["id"]
        answer = "[SKIP]"
        try:
            if system == "sge_ggcr":
                answer = retrieve_ggcr(q, compact_indices[ds], ds)
            elif system == "pure_compact":
                answer = retrieve_pure_compact(q, compact_indices[ds])
            elif system == "concat_all":
                answer = retrieve_concat_all(q, compact_indices[ds])
            elif system == "graph_native":
                answer = retrieve_graph_native(q, ds)
            elif system == "naive_rag":
                if ds in naive_indices:
                    chunks, embs = naive_indices[ds]
                    answer = retrieve_naive_rag(q, chunks, embs)
                else:
                    answer = "[No naive chunks]"
            elif system == "baseline_ggcr":
                answer = retrieve_baseline_ggcr(q, compact_indices[ds], ds)
        except Exception as e:
            answer = f"[Error: {e}]"
        correct = score_answer(answer, q)
        return q, system, answer, correct

    # Submit all tasks to thread pool (max_workers controls concurrency)
    MAX_WORKERS = 15
    print(f"\nRunning {total} evaluations with {MAX_WORKERS} concurrent workers...")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for q in questions:
            for system in systems:
                futures.append(executor.submit(_run_one, q, system))

        for future in as_completed(futures):
            q, system, answer, correct = future.result()
            ds = q["dataset"]
            level = q["level"]
            qid = q["id"]

            with results_lock:
                results[system][ds][level].append(correct)
                detailed_results.append({
                    "id": qid,
                    "dataset": ds,
                    "level": level,
                    "system": system,
                    "question": q["question"],
                    "reference_answer": q["reference_answer"],
                    "answer": answer[:500],
                    "correct": correct,
                })
                done_count[0] += 1
                if done_count[0] % 40 == 0:
                    print(f"  Progress: {done_count[0]}/{total} ({done_count[0]/total:.0%})")

            if verbose:
                mark = "✓" if correct else "✗"
                print(f"  [{mark}] {qid} [{system}] {q['question'][:60]}...")

    # Compute summary
    summary = {}
    for system in systems:
        summary[system] = {"by_dataset": {}, "by_level": {}, "overall": {}}

        all_correct = []
        for ds in datasets:
            ds_correct = []
            for level in ["L1", "L2", "L3", "L4"]:
                level_results = results[system][ds][level]
                if level_results:
                    acc = sum(level_results) / len(level_results)
                    n = len(level_results)
                    summary[system]["by_dataset"].setdefault(ds, {})[level] = {
                        "correct": sum(level_results), "total": n, "accuracy": round(acc, 4),
                    }
                    ds_correct.extend(level_results)
                    all_correct.extend(level_results)

            if ds_correct:
                summary[system]["by_dataset"][ds]["total"] = {
                    "correct": sum(ds_correct), "total": len(ds_correct),
                    "accuracy": round(sum(ds_correct) / len(ds_correct), 4),
                }

        # By level (aggregated across datasets)
        for level in ["L1", "L2", "L3", "L4"]:
            level_all = []
            for ds in datasets:
                level_all.extend(results[system][ds][level])
            if level_all:
                summary[system]["by_level"][level] = {
                    "correct": sum(level_all), "total": len(level_all),
                    "accuracy": round(sum(level_all) / len(level_all), 4),
                }

        if all_correct:
            summary[system]["overall"] = {
                "correct": sum(all_correct), "total": len(all_correct),
                "accuracy": round(sum(all_correct) / len(all_correct), 4),
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("GGCR Evaluation Results")
    print("=" * 80)

    # Per-level summary
    print(f"\n{'System':<16} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'Overall':>10}")
    print("-" * 60)
    for system in systems:
        parts = []
        for level in ["L1", "L2", "L3", "L4"]:
            data = summary[system]["by_level"].get(level, {})
            if data:
                parts.append(f"{data['accuracy']:.0%}")
            else:
                parts.append("  —")
        overall = summary[system]["overall"]
        overall_str = f"{overall['accuracy']:.0%}" if overall else "—"
        print(f"{system:<16} {'  '.join(parts):>32} {overall_str:>10}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": str(BENCHMARK_PATH),
        "total_questions": len(questions),
        "systems": systems,
        "summary": summary,
        "detailed_results": detailed_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="GGCR evaluation")
    parser.add_argument("--systems", type=str, default=None,
                        help="Comma-separated systems to evaluate")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    systems = args.systems.split(",") if args.systems else None
    run_evaluation(systems=systems, verbose=args.verbose)


if __name__ == "__main__":
    main()
