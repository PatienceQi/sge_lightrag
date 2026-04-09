#!/usr/bin/env python3
"""
run_eval.py — Graph-first retrieval evaluation runner.

Runs graph-first retrieval for both SGE and Baseline graphs over the full
benchmark, then scores answers and writes results.

Scoring:
  - numeric_tolerance: exact numeric match ±1% (or ±0.5 for small numbers)
  - exact_match: case-insensitive substring containment
  - set_overlap: Jaccard overlap ≥ 0.5 of reference set against answer tokens
  - direction: "yes"/"no" (or 是/否) match

Statistical test:
  - McNemar test on paired (SGE correct, Baseline correct) matrix

Output: experiments/results/graph_first_results.json

Usage:
    python3 experiments/graph_first/run_eval.py [--dry-run] [--dataset DATASET]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.graph_first.retriever import retrieve_graph_first
from experiments.graph_first.benchmark import generate_benchmark

BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "graph_first" / "benchmark_statistical.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize answer text for comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _score_numeric_tolerance(answer: str, reference_numeric: float, tol: float = 0.01) -> bool:
    """Check if any number in the answer matches reference within tolerance."""
    numbers = re.findall(r"-?\d+(?:[,_]\d+)*(?:\.\d+)?", answer.replace(",", ""))
    if not numbers:
        return False
    for num_str in numbers:
        try:
            val = float(num_str)
        except ValueError:
            continue
        if abs(reference_numeric) < 1.0:
            if abs(val - reference_numeric) < 0.05:
                return True
        elif abs(val - reference_numeric) / (abs(reference_numeric) + 1e-9) <= tol:
            return True
    return False


def _score_exact_match(answer: str, reference_answer: str) -> bool:
    """Check if reference answer matches the answer via substring or key-word overlap."""
    a = _normalize(answer)
    r = _normalize(reference_answer)
    # Direct substring check
    if r in a or a in r:
        return True
    # Key-word overlap: if ≥60% of reference words appear in answer, accept
    r_words = [w for w in r.split() if len(w) > 2]
    if r_words:
        matched = sum(1 for w in r_words if w in a)
        if matched / len(r_words) >= 0.6:
            return True
    return False


def _score_set_overlap(answer: str, reference_set: list[str]) -> bool:
    """Check Jaccard overlap ≥ 0.5 between reference set and answer tokens."""
    if not reference_set:
        return False
    a_lower = _normalize(answer)
    matched = sum(1 for item in reference_set if _normalize(item) in a_lower)
    return matched / len(reference_set) >= 0.5


def _score_direction(answer: str, reference_answer: str) -> bool:
    """Check yes/no or 是/否 direction match."""
    a = _normalize(answer)
    r = _normalize(reference_answer)
    # Map Chinese to English
    a = a.replace("是", "yes").replace("否", "no")
    r = r.replace("是", "yes").replace("否", "no")
    if r in ("yes", "no"):
        return r in a[:30]
    return r in a or a in r


def score_answer(answer: str, question: dict) -> bool:
    """Score a single answer against reference."""
    eval_type = question.get("evaluation_type", "exact_match")
    ref_answer = str(question.get("reference_answer", ""))
    ref_numeric = question.get("reference_numeric")
    ref_set = question.get("reference_set")

    if eval_type == "numeric_tolerance" and ref_numeric is not None:
        return _score_numeric_tolerance(answer, float(ref_numeric))
    elif eval_type == "set_overlap" and ref_set:
        return _score_set_overlap(answer, ref_set)
    elif eval_type == "direction":
        return _score_direction(answer, ref_answer)
    else:
        return _score_exact_match(answer, ref_answer)


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def mcnemar_test(sge_correct: list[bool], base_correct: list[bool]) -> dict:
    """
    McNemar test for paired binary outcomes.
    Returns {statistic, p_value, n01 (SGE wrong, Base right), n10 (SGE right, Base wrong)}.
    """
    n01 = sum(1 for s, b in zip(sge_correct, base_correct) if not s and b)
    n10 = sum(1 for s, b in zip(sge_correct, base_correct) if s and not b)
    n = n01 + n10
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": 0, "n10": 0}
    # Continuity-corrected McNemar
    stat = (abs(n10 - n01) - 1) ** 2 / (n01 + n10) if n > 0 else 0.0
    # Chi-squared with 1 df → p-value approximation
    from scipy.stats import chi2
    p_value = chi2.sf(stat, df=1)
    return {"statistic": float(stat), "p_value": float(p_value), "n01": n01, "n10": n10}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    questions: list[dict],
    system: str,
    dry_run: bool = False,
    delay: float = 0.5,
) -> list[dict]:
    """
    Run graph-first retrieval for all questions and score answers.

    Args:
        questions: List of benchmark question dicts.
        system: "sge" or "baseline".
        dry_run: If True, only process the first 5 questions per dataset.
        delay: Seconds to wait between LLM calls (rate limiting).

    Returns:
        List of result dicts with answer, score, and metadata.
    """
    results = []
    total = len(questions)

    for i, q in enumerate(questions):
        dataset = q["dataset"]
        question_text = q["question"]

        if dry_run and i >= 5:
            continue

        print(f"  [{i + 1}/{total}] {system.upper()} | {dataset} | {q['query_type']}", end="", flush=True)

        try:
            retrieval = retrieve_graph_first(
                question=question_text,
                dataset=dataset,
                system=system,
                max_context_tokens=3000,
            )
            answer = retrieval["answer"]
            context_source = retrieval["context_source"]
            correct = score_answer(answer, q)
        except Exception as exc:
            answer = f"[Error: {exc}]"
            context_source = "error"
            correct = False

        print(f" → {'PASS' if correct else 'FAIL'} ({context_source})")

        results.append({
            "id": q["id"],
            "dataset": dataset,
            "query_type": q["query_type"],
            "question": question_text,
            "reference_answer": q.get("reference_answer"),
            "answer": answer,
            "correct": correct,
            "context_source": context_source,
            "system": system,
            "evaluation_type": q.get("evaluation_type"),
        })

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict]) -> dict:
    """Compute per-dataset and per-query-type accuracy."""
    by_dataset: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    context_sources: dict = defaultdict(int)

    for r in results:
        ds = r["dataset"]
        qt = r["query_type"]
        by_dataset[ds]["total"] += 1
        by_type[qt]["total"] += 1
        context_sources[r.get("context_source", "unknown")] += 1
        if r["correct"]:
            by_dataset[ds]["correct"] += 1
            by_type[qt]["correct"] += 1

    total_correct = sum(r["correct"] for r in results)
    total = len(results)

    return {
        "overall": {
            "correct": total_correct,
            "total": total,
            "accuracy": round(total_correct / total, 4) if total > 0 else 0.0,
        },
        "by_dataset": {
            ds: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
            }
            for ds, v in sorted(by_dataset.items())
        },
        "by_query_type": {
            qt: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
            }
            for qt, v in sorted(by_type.items())
        },
        "context_sources": dict(context_sources),
    }


def print_summary(sge_agg: dict, base_agg: dict, mcnemar: dict):
    """Print human-readable comparison summary."""
    print("\n" + "=" * 70)
    print("GRAPH-FIRST RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    sge_acc = sge_agg["overall"]["accuracy"]
    base_acc = base_agg["overall"]["accuracy"]
    delta = sge_acc - base_acc
    print(f"\nOverall Accuracy:")
    print(f"  SGE:      {sge_acc:.1%} ({sge_agg['overall']['correct']}/{sge_agg['overall']['total']})")
    print(f"  Baseline: {base_acc:.1%} ({base_agg['overall']['correct']}/{base_agg['overall']['total']})")
    print(f"  Delta:    {delta:+.1%}")

    print("\nPer-Dataset:")
    all_datasets = sorted(set(list(sge_agg["by_dataset"]) + list(base_agg["by_dataset"])))
    for ds in all_datasets:
        s = sge_agg["by_dataset"].get(ds, {})
        b = base_agg["by_dataset"].get(ds, {})
        s_acc = s.get("accuracy", 0.0)
        b_acc = b.get("accuracy", 0.0)
        print(f"  {ds:15s}: SGE={s_acc:.1%}  Base={b_acc:.1%}  Δ={s_acc - b_acc:+.1%}")

    print("\nPer-Query-Type:")
    all_types = sorted(set(list(sge_agg["by_query_type"]) + list(base_agg["by_query_type"])))
    for qt in all_types:
        s = sge_agg["by_query_type"].get(qt, {})
        b = base_agg["by_query_type"].get(qt, {})
        s_acc = s.get("accuracy", 0.0)
        b_acc = b.get("accuracy", 0.0)
        print(f"  {qt:15s}: SGE={s_acc:.1%}  Base={b_acc:.1%}  Δ={s_acc - b_acc:+.1%}")

    print(f"\nMcNemar Test: χ²={mcnemar['statistic']:.3f}, p={mcnemar['p_value']:.4f}")
    print(f"  SGE-only-correct: {mcnemar['n10']}  |  Base-only-correct: {mcnemar['n01']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Graph-First Retrieval Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Process only first 5 questions (test mode)")
    parser.add_argument("--dataset", type=str, default=None, help="Run on single dataset only")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between LLM calls in seconds")
    args = parser.parse_args()

    # Load or generate benchmark
    if BENCHMARK_PATH.exists():
        print(f"Loading benchmark from {BENCHMARK_PATH}")
        questions = []
        with open(BENCHMARK_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        print("Generating benchmark...")
        from experiments.graph_first.benchmark import main as gen_main
        gen_main()
        questions = []
        with open(BENCHMARK_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

    # Filter by dataset if specified
    if args.dataset:
        questions = [q for q in questions if q["dataset"] == args.dataset]
        print(f"Filtered to dataset={args.dataset}: {len(questions)} questions")

    print(f"\nTotal questions: {len(questions)}")
    if args.dry_run:
        print("DRY RUN: processing first 5 questions only")

    # Run SGE evaluation
    print("\n--- SGE Graph-First ---")
    sge_results = run_evaluation(questions, "sge", args.dry_run, args.delay)

    # Run Baseline evaluation
    print("\n--- Baseline Graph-First ---")
    base_results = run_evaluation(questions, "baseline", args.dry_run, args.delay)

    # Aggregate
    sge_agg = aggregate_results(sge_results)
    base_agg = aggregate_results(base_results)

    # Align for McNemar (same questions, same order)
    sge_correct_map = {r["id"]: r["correct"] for r in sge_results}
    base_correct_map = {r["id"]: r["correct"] for r in base_results}
    common_ids = [r["id"] for r in sge_results if r["id"] in base_correct_map]
    sge_correct_list = [sge_correct_map[qid] for qid in common_ids]
    base_correct_list = [base_correct_map[qid] for qid in common_ids]

    try:
        mcnemar = mcnemar_test(sge_correct_list, base_correct_list)
    except Exception:
        mcnemar = {"statistic": 0.0, "p_value": 1.0, "n01": 0, "n10": 0}

    # Print summary
    print_summary(sge_agg, base_agg, mcnemar)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "graph_first_results.json"
    output = {
        "meta": {
            "total_questions": len(questions),
            "dry_run": args.dry_run,
            "dataset_filter": args.dataset,
        },
        "sge": {
            "aggregated": sge_agg,
            "results": sge_results,
        },
        "baseline": {
            "aggregated": base_agg,
            "results": base_results,
        },
        "mcnemar": mcnemar,
        "comparison": {
            "sge_accuracy": sge_agg["overall"]["accuracy"],
            "baseline_accuracy": base_agg["overall"]["accuracy"],
            "delta": round(sge_agg["overall"]["accuracy"] - base_agg["overall"]["accuracy"], 4),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
