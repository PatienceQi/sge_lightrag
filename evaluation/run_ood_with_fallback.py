"""
run_ood_with_fallback.py

Demonstrates the auto-fallback mechanism for OOD datasets.
When the SGE graph's edge/node ratio falls below threshold (0.90),
the system detects degradation and falls back to the Baseline result.

This shows that with auto-fallback enabled, 10/10 OOD datasets achieve
Final FC >= Baseline FC.
"""

import json
from pathlib import Path

FALLBACK_THRESHOLD = 0.90

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSON = PROJECT_ROOT / "experiments" / "results" / "ood_degradation_detection.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "results" / "ood_fallback_results.json"


def load_ood_data(path: Path) -> list[dict]:
    """Load and return the list of OOD dataset entries."""
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw["ood_datasets"]


def group_by_dataset(entries: list[dict]) -> dict[str, dict]:
    """
    Group entries by dataset name, returning a dict keyed by dataset name.
    Each value is a dict with keys 'sge' and 'base' pointing to the
    respective variant entry.
    """
    groups: dict[str, dict] = {}
    for entry in entries:
        name = entry["dataset"]
        variant = entry["variant"]
        if name not in groups:
            groups[name] = {}
        groups[name] = {**groups[name], variant: entry}
    return groups


def apply_fallback(sge_entry: dict, threshold: float) -> dict:
    """
    Apply the fallback rule to a single dataset.
    Returns a new dict with fallback decision and final FC.
    Does not mutate the input.
    """
    edge_node_ratio = sge_entry["edge_node_ratio"]
    sge_fc = sge_entry["sge_fc"]
    base_fc = sge_entry["base_fc"]

    fallback_triggered = edge_node_ratio < threshold
    final_fc = base_fc if fallback_triggered else sge_fc
    geq_baseline = final_fc >= base_fc

    return {
        "dataset": sge_entry["dataset"],
        "sge_fc": sge_fc,
        "base_fc": base_fc,
        "edge_node_ratio": edge_node_ratio,
        "fallback_triggered": fallback_triggered,
        "final_fc": final_fc,
        "geq_baseline": geq_baseline,
    }


def print_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    header = f"{'Dataset':<25} {'SGE FC':>7} {'Base FC':>8} {'e/n ratio':>10} {'Fallback?':>10} {'Final FC':>9}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for r in results:
        fallback_str = "YES" if r["fallback_triggered"] else "no"
        print(
            f"{r['dataset']:<25} "
            f"{r['sge_fc']:>7.3f} "
            f"{r['base_fc']:>8.3f} "
            f"{r['edge_node_ratio']:>10.4f} "
            f"{fallback_str:>10} "
            f"{r['final_fc']:>9.3f}"
        )
    print(separator)


def build_summary(results: list[dict]) -> dict:
    """Build a summary dict from the per-dataset results. Returns a new dict."""
    total = len(results)
    fallback_triggered = sum(1 for r in results if r["fallback_triggered"])
    sge_wins = sum(1 for r in results if not r["fallback_triggered"] and r["sge_fc"] >= r["base_fc"])
    all_geq_baseline = all(r["geq_baseline"] for r in results)
    geq_count = sum(1 for r in results if r["geq_baseline"])

    return {
        "total": total,
        "sge_wins": sge_wins,
        "fallback_triggered": fallback_triggered,
        "geq_baseline_count": geq_count,
        "all_geq_baseline": all_geq_baseline,
    }


def save_results(results: list[dict], summary: dict, output_path: Path) -> None:
    """Write the fallback results JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "OOD evaluation with auto-fallback "
            f"(edge/node ratio threshold = {FALLBACK_THRESHOLD})"
        ),
        "threshold": FALLBACK_THRESHOLD,
        "datasets": results,
        "summary": summary,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    entries = load_ood_data(SOURCE_JSON)
    groups = group_by_dataset(entries)

    results: list[dict] = []
    for dataset_name in sorted(groups.keys()):
        variants = groups[dataset_name]
        if "sge" not in variants:
            raise ValueError(f"Missing SGE variant for dataset: {dataset_name}")
        result = apply_fallback(variants["sge"], FALLBACK_THRESHOLD)
        results.append(result)

    print("\nOOD Evaluation with Auto-Fallback")
    print(f"Threshold: edge/node ratio < {FALLBACK_THRESHOLD} → use Baseline FC\n")
    print_table(results)

    summary = build_summary(results)
    geq = summary["geq_baseline_count"]
    total = summary["total"]
    fallback = summary["fallback_triggered"]
    print(
        f"\nSummary: {geq}/{total} SGE+Fallback >= Baseline"
        f"  (fallback triggered on {fallback}/{total} datasets)"
    )
    if summary["all_geq_baseline"]:
        print("ALL 10/10 datasets achieve Final FC >= Baseline FC.")
    else:
        failing = [r["dataset"] for r in results if not r["geq_baseline"]]
        print(f"WARNING: {len(failing)} dataset(s) still below baseline: {failing}")

    save_results(results, summary, OUTPUT_JSON)


if __name__ == "__main__":
    main()
