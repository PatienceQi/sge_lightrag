#!/usr/bin/env python3
"""
run_evaluations_v2.py — Expanded evaluation with v2 gold standards (600 facts).

B区 (COLING 2027) quality target:
  - 25 countries × 5-6 years per international dataset
  - Values read directly from source CSVs (fixes WB Population precision issue)
  - Outputs results + bootstrap 95% CI for statistical significance

Usage:
    python3 evaluation/run_evaluations_v2.py [--bootstrap N]
"""

from __future__ import annotations

import json
import sys
import subprocess
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "evaluate_coverage.py"

# v2 datasets: 4 international datasets with expanded gold standards
V2_DATASETS = [
    {
        "label": "WHO Life Expectancy (EN, Type-II) [v2: 25c×6y=150f]",
        "short": "WHO",
        "sge":      "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "graphrag": "output/graphrag_who/output/graph.graphml",
    },
    {
        "label": "WB Child Mortality (EN, Type-II) [v2: 25c×6y=150f]",
        "short": "WB_CM",
        "sge":      "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "graphrag": None,
    },
    {
        "label": "WB Population (EN, Type-II) [v2: 25c×6y=150f, precision fixed]",
        "short": "WB_Pop",
        "sge":      "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold/gold_wb_population_v2.jsonl",
        "graphrag": None,
    },
    {
        "label": "WB Maternal Mortality (EN, Type-II) [v2: 25c×5y=150f]",
        "short": "WB_Mat",
        "sge":      "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "graphrag": None,
    },
]


def run_eval(graph_path: str, gold_path: str) -> dict | None:
    """Run evaluate_coverage.py and return parsed JSON results."""
    full_graph = PROJECT_ROOT / graph_path
    full_gold = PROJECT_ROOT / gold_path

    if not full_graph.exists():
        return None

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--graph", str(full_graph),
         "--gold", str(full_gold)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr[-300:]}

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
    return {"error": "parse_failed"}


def bootstrap_ci(matched_count: int, total: int, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """
    Bootstrap 95% CI for a proportion (matched / total).
    Simulates resampling: each trial randomly samples `total` items with
    replacement from a Bernoulli population where p = matched/total.
    Returns (lower, upper).
    """
    if total == 0:
        return (0.0, 0.0)
    p = matched_count / total
    boot_proportions = []
    for _ in range(n_bootstrap):
        sample = sum(random.random() < p for _ in range(total))
        boot_proportions.append(sample / total)
    boot_proportions.sort()
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    return (boot_proportions[lo_idx], boot_proportions[hi_idx])


def fmt(val, default="—"):
    return f"{val:.3f}" if val is not None else default


def fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap samples for CI (default 1000)")
    args = parser.parse_args()

    random.seed(42)
    n_boot = args.bootstrap

    print(f"Running v2 evaluations (gold standard: 150 facts per dataset)\n")
    print(f"{'Dataset':<30} | {'SGE EC':>8} {'SGE FC':>8} | {'Base EC':>8} {'Base FC':>8}")
    print("-" * 90)

    all_results = []
    for ds in V2_DATASETS:
        short = ds["short"]
        sge_r = run_eval(ds["sge"], ds["gold"])
        base_r = run_eval(ds["baseline"], ds["gold"])
        gr_r = run_eval(ds.get("graphrag"), ds["gold"]) if ds.get("graphrag") else None

        sge_ec = sge_r["entity_coverage"]["coverage"] if sge_r and "entity_coverage" in sge_r else None
        sge_fc = sge_r["fact_coverage"]["coverage"] if sge_r and "fact_coverage" in sge_r else None
        sge_ec_n = sge_r["entity_coverage"]["matched"] if sge_r and "entity_coverage" in sge_r else 0
        sge_fc_n = sge_r["fact_coverage"]["covered"] if sge_r and "fact_coverage" in sge_r else 0
        sge_tot_e = sge_r["entity_coverage"]["total"] if sge_r and "entity_coverage" in sge_r else 0
        sge_tot_f = sge_r["fact_coverage"]["total"] if sge_r and "fact_coverage" in sge_r else 0

        base_ec = base_r["entity_coverage"]["coverage"] if base_r and "entity_coverage" in base_r else None
        base_fc = base_r["fact_coverage"]["coverage"] if base_r and "fact_coverage" in base_r else None
        base_ec_n = base_r["entity_coverage"]["matched"] if base_r and "entity_coverage" in base_r else 0
        base_fc_n = base_r["fact_coverage"]["covered"] if base_r and "fact_coverage" in base_r else 0
        base_tot_e = base_r["entity_coverage"]["total"] if base_r and "entity_coverage" in base_r else 0
        base_tot_f = base_r["fact_coverage"]["total"] if base_r and "fact_coverage" in base_r else 0

        gr_ec = gr_r["entity_coverage"]["coverage"] if gr_r and "entity_coverage" in gr_r else None
        gr_fc = gr_r["fact_coverage"]["coverage"] if gr_r and "fact_coverage" in gr_r else None

        # Bootstrap CIs
        sge_ec_ci = bootstrap_ci(sge_ec_n, sge_tot_e, n_boot) if sge_tot_e else (0, 0)
        sge_fc_ci = bootstrap_ci(sge_fc_n, sge_tot_f, n_boot) if sge_tot_f else (0, 0)
        base_ec_ci = bootstrap_ci(base_ec_n, base_tot_e, n_boot) if base_tot_e else (0, 0)
        base_fc_ci = bootstrap_ci(base_fc_n, base_tot_f, n_boot) if base_tot_f else (0, 0)

        row = {
            "dataset": short,
            "label": ds["label"],
            "sge_ec": sge_ec, "sge_fc": sge_fc,
            "sge_ec_matched": sge_ec_n, "sge_ec_total": sge_tot_e,
            "sge_fc_covered": sge_fc_n, "sge_fc_total": sge_tot_f,
            "sge_ec_ci": list(sge_ec_ci), "sge_fc_ci": list(sge_fc_ci),
            "baseline_ec": base_ec, "baseline_fc": base_fc,
            "baseline_ec_matched": base_ec_n, "baseline_ec_total": base_tot_e,
            "baseline_fc_covered": base_fc_n, "baseline_fc_total": base_tot_f,
            "baseline_ec_ci": list(base_ec_ci), "baseline_fc_ci": list(base_fc_ci),
            "graphrag_ec": gr_ec, "graphrag_fc": gr_fc,
        }
        all_results.append(row)

        status = "✓" if sge_ec is not None else "?"
        print(f"{status} {short:<28} | {fmt(sge_ec):>8} {fmt(sge_fc):>8} | {fmt(base_ec):>8} {fmt(base_fc):>8}")

    print()
    print("=" * 90)
    print("Bootstrap 95% Confidence Intervals:")
    print()
    for r in all_results:
        ds = r["dataset"]
        if r["sge_ec"] is None:
            continue
        print(f"  {ds}:")
        print(f"    SGE   EC={r['sge_ec']:.3f} ({r['sge_ec_matched']}/{r['sge_ec_total']})  "
              f"CI={fmt_ci(*r['sge_ec_ci'])}")
        print(f"    SGE   FC={r['sge_fc']:.3f} ({r['sge_fc_covered']}/{r['sge_fc_total']})  "
              f"CI={fmt_ci(*r['sge_fc_ci'])}")
        print(f"    Base  EC={r['baseline_ec']:.3f} ({r['baseline_ec_matched']}/{r['baseline_ec_total']})  "
              f"CI={fmt_ci(*r['baseline_ec_ci'])}")
        print(f"    Base  FC={r['baseline_fc']:.3f} ({r['baseline_fc_covered']}/{r['baseline_fc_total']})  "
              f"CI={fmt_ci(*r['baseline_fc_ci'])}")
        print()

    # Save
    out_path = PROJECT_ROOT / "evaluation" / "results" / "all_results_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
