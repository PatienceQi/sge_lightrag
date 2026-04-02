#!/usr/bin/env python3
"""
run_fullscale_eval.py — 190-country scale experiment (L3/L4 only).

Compares SGE+GGCR vs Concat-All vs Pure Compact at 190 entity scale.
Key hypothesis: Concat-All degrades at scale while GGCR stays stable.

Usage:
    python3 experiments/ggcr/run_fullscale_eval.py [--verbose]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ggcr.compact_chunks import build_compact_index
from experiments.ggcr.graph_guided_retriever import (
    retrieve_ggcr, retrieve_pure_compact, retrieve_concat_all,
)
from experiments.ggcr.run_ggcr_eval import score_answer

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "gold" / "ggcr_benchmark_full.jsonl"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ggcr_results_fullscale.json"

SYSTEMS = ["sge_ggcr", "pure_compact", "concat_all"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load benchmark
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} full-scale benchmark questions")

    # Build who_full compact index (190 entities)
    print("\nBuilding compact index for who_full (190 countries)...")
    compact_index = build_compact_index("who_full", embed=True)
    print(f"  Chunks: {len(compact_index.chunks)}")
    print(f"  Entity map entries: {len(compact_index.entity_map)}")

    # Run evaluation
    total = len(questions) * len(SYSTEMS)
    print(f"\nRunning {total} evaluations with 15 concurrent workers...")

    results = defaultdict(lambda: defaultdict(list))
    detailed = []

    def _run_one(q, system):
        answer = "[SKIP]"
        try:
            if system == "sge_ggcr":
                answer = retrieve_ggcr(q, compact_index, "who_full")
            elif system == "pure_compact":
                answer = retrieve_pure_compact(q, compact_index)
            elif system == "concat_all":
                answer = retrieve_concat_all(q, compact_index)
        except Exception as e:
            answer = f"[Error: {e}]"
        correct = score_answer(answer, q)
        return q, system, answer, correct

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(_run_one, q, s) for q in questions for s in SYSTEMS]
        done = 0
        for future in as_completed(futures):
            q, system, answer, correct = future.result()
            results[system][q["level"]].append(correct)
            detailed.append({
                "id": q["id"], "level": q["level"], "system": system,
                "question": q["question"], "reference_answer": q["reference_answer"],
                "answer": answer[:500], "correct": correct,
            })
            done += 1
            if args.verbose:
                mark = "✓" if correct else "✗"
                print(f"  [{mark}] {q['id']} [{system}] {q['question'][:60]}...")
            elif done % 10 == 0:
                print(f"  Progress: {done}/{total}")

    # Summary
    print("\n" + "=" * 60)
    print("Full-Scale (190 countries) Evaluation Results")
    print("=" * 60)
    print(f"\n{'System':<16} {'L3':>8} {'L4':>8} {'Overall':>10}")
    print("-" * 45)

    summary = {}
    for system in SYSTEMS:
        s = {}
        all_correct = []
        for level in ["L3", "L4"]:
            lr = results[system][level]
            if lr:
                acc = sum(lr) / len(lr)
                s[level] = {"correct": sum(lr), "total": len(lr), "accuracy": round(acc, 4)}
                all_correct.extend(lr)
        if all_correct:
            s["overall"] = {
                "correct": sum(all_correct), "total": len(all_correct),
                "accuracy": round(sum(all_correct) / len(all_correct), 4),
            }
        summary[system] = s
        l3 = f"{s.get('L3', {}).get('accuracy', 0):.0%}" if "L3" in s else "—"
        l4 = f"{s.get('L4', {}).get('accuracy', 0):.0%}" if "L4" in s else "—"
        ov = f"{s['overall']['accuracy']:.0%}" if "overall" in s else "—"
        print(f"{system:<16} {l3:>8} {l4:>8} {ov:>10}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": str(BENCHMARK_PATH),
        "scale": "190_countries",
        "total_questions": len(questions),
        "systems": SYSTEMS,
        "summary": summary,
        "detailed_results": detailed,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
