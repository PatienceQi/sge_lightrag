#!/usr/bin/env python3
"""
run_expanded_eval.py — Evaluation runner for expanded dataset coverage.

Evaluates SGE-LightRAG and baseline on three new datasets spanning different
table topologies not covered by the primary evaluation:
  A. IMF Fiscal Cross-Tabulation (Type-III)
  B. UN Census Cross-Tabulation (Type-III, no time dimension)
  C. WB Indicators Long-Format (Type-II-Long, melted format — previously broken)

Also reports the stage1 classification type for each dataset so reviewers
can see that the long-format fix correctly identifies Type-II-Long CSVs.

Usage:
    python3 evaluation/run_expanded_eval.py
    python3 evaluation/run_expanded_eval.py --bootstrap 2000
    python3 evaluation/run_expanded_eval.py --dry-run   # show config only

Output:
    evaluation/results/expanded_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "evaluate_coverage.py"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
EXPANDED_DIR = PROJECT_ROOT.parent / "dataset" / "expanded"

# Dataset configuration for expanded evaluation
EXPANDED_DATASETS = [
    {
        "label": "IMF Fiscal Cross-Tab (Type-III, hierarchical financial)",
        "short": "imf_fiscal",
        "topology": "Type-III",
        "csv": "dataset/expanded/imf_fiscal_cross_tab.csv",
        "sge": (
            "output/sge_imf_fiscal/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline": (
            "output/baseline_imf_fiscal/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "gold": "evaluation/gold/gold_imf_fiscal_cross_tab.jsonl",
        "gold_facts": 75,
    },
    {
        "label": "UN Census Cross-Tab (Type-III, no time dimension)",
        "short": "un_census",
        "topology": "Type-III",
        "csv": "dataset/expanded/un_census_cross_tab.csv",
        "sge": (
            "output/sge_un_census/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline": (
            "output/baseline_un_census/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "gold": "evaluation/gold/gold_un_census_cross_tab.jsonl",
        "gold_facts": 60,
    },
    {
        "label": "WB Indicators Long-Format (Type-II-Long, melted time-series)",
        "short": "wb_long",
        "topology": "Type-II-Long",
        "csv": "dataset/expanded/wb_indicators_long_format.csv",
        "sge": (
            "output/sge_wb_long/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline": (
            "output/baseline_wb_long/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "gold": "evaluation/gold/gold_wb_indicators_long.jsonl",
        "gold_facts": 90,
    },
]


def classify_dataset(csv_relative_path: str) -> str:
    """Run Stage 1 classification on the given CSV and return type string."""
    csv_path = PROJECT_ROOT.parent / csv_relative_path
    if not csv_path.exists():
        # Try relative to PROJECT_ROOT
        csv_path = PROJECT_ROOT / csv_relative_path
    if not csv_path.exists():
        return "CSV_NOT_FOUND"

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from stage1.features import extract_features
        from stage1.classifier import classify
        features = extract_features(str(csv_path))
        return classify(features)
    except Exception as exc:
        return f"ERROR: {exc}"


def run_eval(graph_relative_path: str, gold_relative_path: str) -> dict | None:
    """Run evaluate_coverage.py and return parsed JSON results dict.

    Returns None if the graph file does not exist.
    Returns {"error": "..."} if the script fails or output cannot be parsed.
    """
    graph_path = PROJECT_ROOT / graph_relative_path
    gold_path = PROJECT_ROOT / gold_relative_path

    if not graph_path.exists():
        return None

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--graph", str(graph_path),
         "--gold", str(gold_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {"error": result.stderr[-400:]}

    output = result.stdout
    try:
        json_marker = output.find("[JSON]")
        if json_marker >= 0:
            after_marker = output[json_marker + 6:]
            json_start = after_marker.find("{")
            json_end = after_marker.rfind("}") + 1
            return json.loads(after_marker[json_start:json_end])
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(output[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return {"error": "parse_failed", "output_sample": output[:200]}


def bootstrap_ci(
    matched: int,
    total: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a proportion matched/total."""
    if total == 0:
        return (0.0, 0.0)
    p = matched / total
    samples = sorted(
        sum(random.random() < p for _ in range(total)) / total
        for _ in range(n_bootstrap)
    )
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    return (samples[lo_idx], samples[hi_idx])


def extract_metrics(result: dict | None) -> tuple[float | None, float | None, int, int]:
    """Extract (EC, FC, matched_facts, total_facts) from evaluate_coverage output."""
    if result is None or "error" in result:
        return None, None, 0, 0
    ec = result.get("entity_coverage", {}).get("coverage")
    fc = result.get("fact_coverage", {}).get("coverage")
    fc_matched = result.get("fact_coverage", {}).get("covered", 0)
    fc_total = result.get("fact_coverage", {}).get("total", 0)
    return ec, fc, fc_matched, fc_total


def fmt(val: float | None, default: str = "N/A") -> str:
    return f"{val:.3f}" if val is not None else default


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def print_classification_report(datasets: list[dict]) -> dict[str, str]:
    """Print Stage 1 classification results for all expanded datasets."""
    print("\n--- Stage 1 Classification Report ---")
    print(f"{'Dataset':<20} | {'Expected':>15} | {'Actual':>20}")
    print("-" * 62)
    actual_types: dict[str, str] = {}
    # Mapping from compact topology labels used in config to classifier output strings
    topology_to_classifier = {
        "Type-I": "Flat-Entity",
        "Type-II": "Time-Series-Matrix",
        "Type-II-Long": "Time-Series-Long",
        "Type-III": "Hierarchical-Hybrid",
    }
    for ds in datasets:
        actual = classify_dataset(ds["csv"])
        expected = ds["topology"]
        expected_classifier = topology_to_classifier.get(expected, expected)
        match_flag = "OK" if actual == expected_classifier else "MISMATCH"
        print(f"{ds['short']:<20} | {expected:>15} | {actual:>20}  [{match_flag}]")
        actual_types[ds["short"]] = actual
    return actual_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SGE + Baseline on expanded datasets."
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Bootstrap resamples for CI (default: 1000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print dataset config and classification only; skip evaluation"
    )
    args = parser.parse_args()

    random.seed(42)

    print("=" * 70)
    print("EXPANDED EVALUATION — SGE-LightRAG vs Baseline")
    print("Datasets: IMF Fiscal | UN Census | WB Long-Format")
    print("=" * 70)

    # Step 1: Report Stage 1 classification for all expanded CSVs
    actual_types = print_classification_report(EXPANDED_DATASETS)

    if args.dry_run:
        print("\n[dry-run] Skipping graph evaluation. Config verified.")
        print("\nGold standard files:")
        for ds in EXPANDED_DATASETS:
            gold_path = PROJECT_ROOT / ds["gold"]
            exists = gold_path.exists()
            print(f"  {ds['short']:20} {gold_path.name:<45} {'EXISTS' if exists else 'MISSING'}")
        return

    print("\n--- FC / EC Evaluation ---")
    header = f"{'Dataset':<22} | {'SGE EC':>7} {'SGE FC':>7} | {'Base EC':>7} {'Base FC':>7}"
    print(header)
    print("-" * 70)

    all_results: list[dict] = []

    for ds in EXPANDED_DATASETS:
        short = ds["short"]
        sge_r = run_eval(ds["sge"], ds["gold"])
        base_r = run_eval(ds["baseline"], ds["gold"])

        sge_ec, sge_fc, sge_fc_n, sge_fc_tot = extract_metrics(sge_r)
        base_ec, base_fc, base_fc_n, base_fc_tot = extract_metrics(base_r)

        print(
            f"{short:<22} | "
            f"{fmt(sge_ec):>7} {fmt(sge_fc):>7} | "
            f"{fmt(base_ec):>7} {fmt(base_fc):>7}"
        )

        # Compute bootstrap CI if results are available
        sge_fc_ci = (None, None)
        base_fc_ci = (None, None)
        if sge_r and "error" not in sge_r and sge_fc_tot > 0:
            sge_fc_ci = bootstrap_ci(sge_fc_n, sge_fc_tot, args.bootstrap)
        if base_r and "error" not in base_r and base_fc_tot > 0:
            base_fc_ci = bootstrap_ci(base_fc_n, base_fc_tot, args.bootstrap)

        # Log errors if any
        if sge_r and "error" in sge_r:
            print(f"  [WARN] SGE error: {sge_r['error'][:80]}")
        if base_r and "error" in base_r:
            print(f"  [WARN] Baseline error: {base_r['error'][:80]}")

        result_entry = {
            "dataset": short,
            "label": ds["label"],
            "topology": ds["topology"],
            "actual_type": actual_types.get(short, "unknown"),
            "gold_facts": ds["gold_facts"],
            "sge": {
                "ec": sge_ec,
                "fc": sge_fc,
                "fc_matched": sge_fc_n,
                "fc_total": sge_fc_tot,
                "fc_ci_95": list(sge_fc_ci),
                "graph_found": sge_r is not None,
            },
            "baseline": {
                "ec": base_ec,
                "fc": base_fc,
                "fc_matched": base_fc_n,
                "fc_total": base_fc_tot,
                "fc_ci_95": list(base_fc_ci),
                "graph_found": base_r is not None,
            },
        }
        all_results.append(result_entry)

    # Write results to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "expanded_results.json"
    output_data = {
        "description": (
            "Expanded evaluation on 3 new datasets covering different topologies. "
            "Includes long-format (Type-II-Long) detection fix validation."
        ),
        "bootstrap_samples": args.bootstrap,
        "datasets": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to: {output_path}")

    # Print CI summary for datasets with results
    print("\n--- Bootstrap 95% CI for FC ---")
    for entry in all_results:
        sge_ci = entry["sge"]["fc_ci_95"]
        base_ci = entry["baseline"]["fc_ci_95"]
        sge_ci_str = fmt_ci(*sge_ci) if sge_ci[0] is not None else "N/A"
        base_ci_str = fmt_ci(*base_ci) if base_ci[0] is not None else "N/A"
        print(
            f"  {entry['dataset']:<22} SGE: {sge_ci_str}  Base: {base_ci_str}"
        )

    print("\n" + "=" * 70)
    total_gold = sum(ds["gold_facts"] for ds in EXPANDED_DATASETS)
    available = sum(1 for r in all_results if r["sge"]["graph_found"])
    print(f"Datasets with graph outputs: {available}/{len(EXPANDED_DATASETS)}")
    print(f"Total gold facts available:  {total_gold}")
    print("=" * 70)


if __name__ == "__main__":
    main()
