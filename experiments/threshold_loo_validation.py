"""Leave-one-out cross-validation for edge/node ratio degradation threshold.

Demonstrates that the gap between success (e/n ≥ 0.9431) and failure (e/n ≤ 0.7934)
groups is wide enough that any threshold in [0.80, 0.94] achieves perfect separation,
making the choice of θ=0.90 robust and non-sensitive.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_ood_sge_data():
    """Load OOD SGE variant edge/node ratios and ground-truth labels."""
    with open(RESULTS_DIR / "ood_degradation_detection.json") as f:
        data = json.load(f)

    records = []
    for item in data["ood_datasets"]:
        if item["variant"] == "sge":
            records.append({
                "dataset": item["dataset"],
                "edge_node_ratio": item["edge_node_ratio"],
                "sge_better": item["sge_better"],
                "sge_fc": item["sge_fc"],
                "base_fc": item["base_fc"],
            })
    return records


def find_optimal_threshold(records):
    """Find threshold as midpoint between failure-max and success-min.

    This is more principled than gap-based search: it maximizes the margin
    to both groups, making the threshold robust to unseen boundary cases.
    """
    success_ratios = [r["edge_node_ratio"] for r in records if r["sge_better"]]
    failure_ratios = [r["edge_node_ratio"] for r in records if not r["sge_better"]]

    if not failure_ratios:
        # No failures in training set — use conservative threshold below success min
        return min(success_ratios) - 0.05, 1.0
    if not success_ratios:
        return max(failure_ratios) + 0.05, 1.0

    failure_max = max(failure_ratios)
    success_min = min(success_ratios)
    threshold = (failure_max + success_min) / 2

    correct = sum(
        1 for r in records
        if (r["edge_node_ratio"] >= threshold) == r["sge_better"]
    )
    return threshold, correct / len(records)


def run_loo():
    """Leave-one-out cross-validation."""
    records = load_ood_sge_data()
    n = len(records)

    loo_results = []
    for i in range(n):
        held_out = records[i]
        train_set = records[:i] + records[i + 1:]
        threshold, train_acc = find_optimal_threshold(train_set)
        predicted_success = held_out["edge_node_ratio"] >= threshold
        correct = predicted_success == held_out["sge_better"]
        loo_results.append({
            "held_out": held_out["dataset"],
            "e_n_ratio": held_out["edge_node_ratio"],
            "actual_success": held_out["sge_better"],
            "threshold_from_9": round(threshold, 4),
            "predicted_success": predicted_success,
            "correct": correct,
        })

    # Compute separation gap
    success_ratios = [r["edge_node_ratio"] for r in records if r["sge_better"]]
    failure_ratios = [r["edge_node_ratio"] for r in records if not r["sge_better"]]
    gap_low = max(failure_ratios)
    gap_high = min(success_ratios)

    loo_accuracy = sum(1 for r in loo_results if r["correct"]) / n

    output = {
        "method": "leave-one-out cross-validation for e/n threshold",
        "n_datasets": n,
        "loo_accuracy": loo_accuracy,
        "loo_correct": sum(1 for r in loo_results if r["correct"]),
        "separation_gap": {"failure_max": gap_low, "success_min": gap_high},
        "any_threshold_in_range_works": f"[{gap_low:.4f}, {gap_high:.4f}]",
        "chosen_threshold": 0.90,
        "chosen_within_gap": gap_low < 0.90 < gap_high,
        "per_dataset": loo_results,
    }

    out_path = RESULTS_DIR / "threshold_loo_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"LOO accuracy: {loo_accuracy:.0%} ({output['loo_correct']}/{n})")
    print(f"Separation gap: [{gap_low:.4f}, {gap_high:.4f}]")
    print(f"θ=0.90 within gap: {output['chosen_within_gap']}")
    print(f"Results saved to {out_path}")
    return output


if __name__ == "__main__":
    run_loo()
