#!/usr/bin/env python3
"""
Run FC evaluation on all OOD datasets.

Scans output/ood_*/sge_*/lightrag_storage and output/ood_*/baseline_*/lightrag_storage
for graph data, matches against gold/gold_ood_*.jsonl.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate_coverage import evaluate_coverage

EVAL_DIR = Path(__file__).parent
GOLD_DIR = EVAL_DIR / "gold"
OUTPUT_BASE = Path(__file__).parent.parent / "output" / "ood"


def find_graph_dir(dataset_dir, prefix="sge"):
    """Find the LightRAG storage directory."""
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            storage = d / "lightrag_storage"
            if storage.exists():
                return storage
    return None


def main():
    results = []

    for gold_file in sorted(GOLD_DIR.glob("gold_ood_*.jsonl")):
        dataset_name = gold_file.stem.replace("gold_ood_", "")
        dataset_dir = OUTPUT_BASE / dataset_name

        if not dataset_dir.exists():
            print(f"  SKIP {dataset_name}: no output directory")
            continue

        # Find SGE and Baseline graph directories
        sge_storage = find_graph_dir(dataset_dir, "sge")
        baseline_storage = find_graph_dir(dataset_dir, "baseline")

        row = {"dataset": dataset_name}

        for label, storage in [("sge", sge_storage), ("baseline", baseline_storage)]:
            if storage is None:
                print(f"  SKIP {dataset_name}/{label}: no graph storage")
                row[f"{label}_ec"] = None
                row[f"{label}_fc"] = None
                continue

            try:
                ec, fc, details = evaluate_coverage(
                    graph_dir=str(storage),
                    gold_path=str(gold_file),
                )
                row[f"{label}_ec"] = round(ec, 3)
                row[f"{label}_fc"] = round(fc, 3)
                row[f"{label}_nodes"] = details.get("n_nodes", 0)
                print(f"  {dataset_name}/{label}: EC={ec:.3f} FC={fc:.3f}")
            except Exception as e:
                print(f"  ERROR {dataset_name}/{label}: {e}")
                row[f"{label}_ec"] = None
                row[f"{label}_fc"] = None

        if row.get("sge_fc") is not None and row.get("baseline_fc") is not None:
            if row["baseline_fc"] > 0:
                row["ratio"] = round(row["sge_fc"] / row["baseline_fc"], 2)
            else:
                row["ratio"] = "inf" if row["sge_fc"] > 0 else "N/A"

        results.append(row)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Dataset':<30} {'SGE EC':>8} {'SGE FC':>8} {'Base FC':>8} {'Ratio':>8}")
    print("-" * 80)
    for r in results:
        sge_ec = f"{r.get('sge_ec', 'N/A')}"
        sge_fc = f"{r.get('sge_fc', 'N/A')}"
        base_fc = f"{r.get('baseline_fc', 'N/A')}"
        ratio = f"{r.get('ratio', 'N/A')}"
        print(f"{r['dataset']:<30} {sge_ec:>8} {sge_fc:>8} {base_fc:>8} {ratio:>8}")

    # Save results
    out_path = EVAL_DIR / "results" / "ood_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
