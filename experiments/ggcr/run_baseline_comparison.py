#!/usr/bin/env python3
"""
run_baseline_comparison.py — Focused SGE+GGCR vs Baseline+GGCR comparison.

Runs only two systems to isolate the effect of graph quality on downstream QA:
  - sge_ggcr      — Entity enumeration from SGE graph
  - baseline_ggcr — Entity enumeration from Baseline graph

Key insight: L1/L2 results are IDENTICAL for both systems (same vector retrieval,
no graph used). Differences appear only in L3/L4 (cross-entity questions requiring
complete entity enumeration). This isolates the graph-quality signal.

Statistical test: McNemar's test on paired binary outcomes.

Usage:
    python3 experiments/ggcr/run_baseline_comparison.py [--verbose]

Output:
    experiments/results/ggcr_baseline_comparison.json
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ggcr.compact_chunks import build_compact_index
from experiments.ggcr.graph_guided_retriever import retrieve_ggcr, retrieve_baseline_ggcr, retrieve_oracle_ggcr

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "gold" / "ggcr_benchmark.jsonl"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ggcr_baseline_comparison.json"

SYSTEMS = ["sge_ggcr", "baseline_ggcr", "oracle_ggcr"]
MAX_WORKERS = 3


# ---------------------------------------------------------------------------
# Scoring (same logic as run_ggcr_eval.py)
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
        if _match(answer, ref):
            return True
        ref_set = question.get("reference_set") or []
        return any(_match(answer, item) for item in ref_set)

    elif eval_type == "direction":
        answer_lower = answer.lower().strip()
        ref_lower = ref.lower().strip()
        if ref_lower in ("yes", "是"):
            return any(w in answer_lower for w in ["yes", "是", "correct", "true", "对", "consistently"])
        else:
            return any(w in answer_lower for w in ["no", "否", "not", "false", "没有", "fluctuat", "不是"])

    elif eval_type == "set_overlap":
        ref_set = question.get("reference_set", [])
        if not ref_set:
            return False
        matched = sum(1 for item in ref_set if _match(answer, item))
        threshold = max(1, int(len(ref_set) * 0.6))
        return matched >= threshold

    elif eval_type == "numeric_tolerance":
        ref_num = question.get("reference_numeric")
        if ref_num is None:
            return _match(answer, ref)
        numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        for num_str in numbers:
            try:
                predicted = float(num_str)
                if ref_num == 0:
                    if abs(predicted) < 0.1:
                        return True
                elif abs(predicted - ref_num) / abs(ref_num) < 0.10:
                    return True
            except ValueError:
                continue
        return _match(answer, ref)

    return False


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(sge_correct: list[bool], baseline_correct: list[bool]) -> dict:
    """
    McNemar's test for paired binary outcomes.

    b = SGE correct, Baseline wrong  (SGE-only correct)
    c = SGE wrong,   Baseline correct (Baseline-only correct)

    chi2 = (|b - c| - 1)^2 / (b + c)  [with continuity correction]
    """
    b = sum(1 for s, bl in zip(sge_correct, baseline_correct) if s and not bl)
    c = sum(1 for s, bl in zip(sge_correct, baseline_correct) if not s and bl)

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "note": "no discordant pairs"}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with 1 df
    # Use scipy if available, else a simple approximation
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=1))
    except ImportError:
        # Approximation: chi2(1) ≈ normal^2; p ≈ 2 * Phi(-sqrt(chi2))
        import math
        z = chi2 ** 0.5
        # Abramowitz & Stegun approximation for normal CDF tail
        t = 1.0 / (1.0 + 0.2316419 * z)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        p_one_tail = 0.3989422804 * math.exp(-0.5 * z * z) * poly
        p = float(2 * p_one_tail)

    return {"b": b, "c": c, "chi2": round(chi2, 4), "p": round(p, 6)}


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(verbose: bool = False) -> None:
    """Run SGE+GGCR vs Baseline+GGCR comparison and save results."""

    # Load benchmark
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} benchmark questions")

    datasets = sorted(set(q["dataset"] for q in questions))
    print(f"Datasets: {datasets}")

    # Build compact indices (same Gold-Standard-derived chunks for both systems)
    compact_indices: dict = {}
    for ds in datasets:
        print(f"\nBuilding compact index for {ds}...")
        compact_indices[ds] = build_compact_index(ds, embed=True)

    # Thread-safe accumulators
    # per_question_results: {qid: {system: bool}}
    per_question_results: dict[str, dict[str, bool]] = defaultdict(dict)
    per_question_meta: dict[str, dict] = {}
    lock = threading.Lock()

    total_tasks = len(questions) * len(SYSTEMS)
    done_count = [0]

    def _run_one(q: dict, system: str) -> tuple[dict, str, str, bool]:
        ds = q["dataset"]
        try:
            if system == "sge_ggcr":
                answer = retrieve_ggcr(q, compact_indices[ds], ds)
            elif system == "baseline_ggcr":
                answer = retrieve_baseline_ggcr(q, compact_indices[ds], ds)
            elif system == "oracle_ggcr":
                answer = retrieve_oracle_ggcr(q, compact_indices[ds], ds)
            else:
                answer = f"[Unknown system: {system}]"
        except Exception as e:
            answer = f"[Error: {e}]"
        correct = score_answer(answer, q)
        return q, system, answer, correct

    print(f"\nRunning {total_tasks} evaluations ({MAX_WORKERS} concurrent workers)...")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for q in questions:
            for system in SYSTEMS:
                futures.append(executor.submit(_run_one, q, system))

        for future in as_completed(futures):
            q, system, answer, correct = future.result()
            qid = q["id"]

            with lock:
                per_question_results[qid][system] = correct
                if qid not in per_question_meta:
                    per_question_meta[qid] = {
                        "id": qid,
                        "dataset": q["dataset"],
                        "level": q["level"],
                        "question": q["question"],
                        "reference_answer": q["reference_answer"],
                    }
                done_count[0] += 1
                if done_count[0] % 20 == 0:
                    print(f"  Progress: {done_count[0]}/{total_tasks} ({done_count[0]/total_tasks:.0%})")

            if verbose:
                mark = "v" if correct else "x"
                print(f"  [{mark}] {qid} [{system}] {q['question'][:60]}...")

    # Aggregate per-level stats
    levels = ["L1", "L2", "L3", "L4"]
    level_stats: dict[str, dict] = {sys: {lvl: {"correct": 0, "total": 0} for lvl in levels}
                                     for sys in SYSTEMS}

    for qid, meta in per_question_meta.items():
        lvl = meta["level"]
        for sys in SYSTEMS:
            correct = per_question_results[qid].get(sys, False)
            level_stats[sys][lvl]["correct"] += int(correct)
            level_stats[sys][lvl]["total"] += 1

    # Overall stats
    for sys in SYSTEMS:
        total_c = sum(v["correct"] for v in level_stats[sys].values())
        total_t = sum(v["total"] for v in level_stats[sys].values())
        level_stats[sys]["overall"] = {"correct": total_c, "total": total_t}

    # McNemar's test (all questions paired)
    qids_sorted = sorted(per_question_meta.keys())
    sge_vec = [per_question_results[qid].get("sge_ggcr", False) for qid in qids_sorted]
    baseline_vec = [per_question_results[qid].get("baseline_ggcr", False) for qid in qids_sorted]
    mcnemar = mcnemar_test(sge_vec, baseline_vec)

    # McNemar's test restricted to L3/L4 only (where the systems differ)
    l34_qids = [qid for qid in qids_sorted if per_question_meta[qid]["level"] in ("L3", "L4")]
    sge_l34 = [per_question_results[qid].get("sge_ggcr", False) for qid in l34_qids]
    baseline_l34 = [per_question_results[qid].get("baseline_ggcr", False) for qid in l34_qids]
    mcnemar_l34 = mcnemar_test(sge_l34, baseline_l34)

    # Per-question summary list
    per_question_list = [
        {
            "id": qid,
            "dataset": per_question_meta[qid]["dataset"],
            "level": per_question_meta[qid]["level"],
            "sge_correct": per_question_results[qid].get("sge_ggcr", False),
            "baseline_correct": per_question_results[qid].get("baseline_ggcr", False),
        }
        for qid in qids_sorted
    ]

    # Print summary
    print("\n" + "=" * 70)
    print("SGE+GGCR vs Baseline+GGCR Comparison")
    print("=" * 70)
    header = f"{'System':<16} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'Overall':>10}"
    print(header)
    print("-" * 60)
    for sys in SYSTEMS:
        parts = []
        for lvl in levels:
            d = level_stats[sys][lvl]
            if d["total"] > 0:
                acc = d["correct"] / d["total"]
                parts.append(f"{acc:.0%}({d['correct']}/{d['total']})")
            else:
                parts.append("—")
        ov = level_stats[sys]["overall"]
        ov_str = f"{ov['correct']/ov['total']:.0%}" if ov["total"] > 0 else "—"
        print(f"{sys:<16} {parts[0]:>14} {parts[1]:>14} {parts[2]:>14} {parts[3]:>14} {ov_str:>10}")

    print(f"\nMcNemar (all):  b={mcnemar['b']}, c={mcnemar['c']}, "
          f"chi2={mcnemar['chi2']}, p={mcnemar['p']}")
    print(f"McNemar (L3/L4): b={mcnemar_l34['b']}, c={mcnemar_l34['c']}, "
          f"chi2={mcnemar_l34['chi2']}, p={mcnemar_l34['p']}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": str(BENCHMARK_PATH),
        "total_questions": len(questions),
        "systems": SYSTEMS,
        "sge_ggcr": level_stats["sge_ggcr"],
        "baseline_ggcr": level_stats["baseline_ggcr"],
        "mcnemar": mcnemar,
        "mcnemar_l34_only": mcnemar_l34,
        "per_question": per_question_list,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SGE+GGCR vs Baseline+GGCR comparison")
    parser.add_argument("--verbose", action="store_true", help="Print per-question results")
    args = parser.parse_args()
    run_comparison(verbose=args.verbose)


if __name__ == "__main__":
    main()
