#!/usr/bin/env python3
"""
wilcoxon_effect_sizes.py — Compute Wilcoxon effect sizes + Bonferroni correction.

Reads entity_level_wilcoxon_results.json, computes:
  1. Effect size r = Z / sqrt(N) for each dataset
  2. Bonferroni-corrected p-values (k=4 comparisons)
  3. Matched-pairs rank-biserial correlation
  4. Summary table for paper

Usage:
    python3 experiments/wilcoxon_effect_sizes.py
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, norm

RESULTS_PATH = Path(__file__).parent / "entity_level_wilcoxon_results.json"
OUTPUT_PATH = Path(__file__).parent / "wilcoxon_effect_sizes_results.json"

K_COMPARISONS = 4  # 4 international datasets


def compute_effect_size(sge_rates: dict, base_rates: dict) -> dict:
    """Compute Wilcoxon test with effect sizes from raw per-entity rates."""
    entities = sorted(sge_rates.keys())
    sge_list = [sge_rates[e] for e in entities]
    base_list = [base_rates.get(e, 0.0) for e in entities]
    n = len(entities)

    differences = [s - b for s, b in zip(sge_list, base_list)]
    nonzero_diffs = [d for d in differences if d != 0]
    n_nonzero = len(nonzero_diffs)

    if n_nonzero == 0:
        return {
            "n": n,
            "n_nonzero": 0,
            "W": None,
            "p_value": 1.0,
            "p_bonferroni": 1.0,
            "Z": None,
            "r_effect_size": None,
            "rank_biserial": None,
            "interpretation": "no_difference",
        }

    # Run Wilcoxon with method='approx' to get z-statistic
    result = wilcoxon(sge_list, base_list, alternative="greater", method="approx")
    W = float(result.statistic)
    p_value = float(result.pvalue)
    z_stat = float(result.zstatistic) if hasattr(result, "zstatistic") else None

    # If zstatistic not available, compute from p-value
    if z_stat is None:
        # For one-sided test: Z = norm.ppf(1 - p_value)
        z_stat = float(norm.ppf(1.0 - p_value)) if p_value < 1.0 else 0.0

    # Effect size r = Z / sqrt(N) where N = number of non-zero pairs
    r_effect = z_stat / math.sqrt(n_nonzero) if n_nonzero > 0 else 0.0

    # Matched-pairs rank-biserial correlation
    # r_rb = 1 - (2 * W_minus) / (n_nonzero * (n_nonzero + 1) / 2)
    # where W_minus = sum of negative ranks
    # Since W = W_plus for alternative="greater", W_minus = T - W_plus
    # T = n_nonzero * (n_nonzero + 1) / 2
    T = n_nonzero * (n_nonzero + 1) / 2
    # scipy's W is the sum of positive ranks for alternative="greater"
    rank_biserial = (2 * W / T) - 1 if T > 0 else 0.0

    # Bonferroni correction
    p_bonferroni = min(p_value * K_COMPARISONS, 1.0)

    # Interpretation (Cohen's benchmarks for r)
    if abs(r_effect) >= 0.5:
        interpretation = "large"
    elif abs(r_effect) >= 0.3:
        interpretation = "medium"
    elif abs(r_effect) >= 0.1:
        interpretation = "small"
    else:
        interpretation = "negligible"

    return {
        "n": n,
        "n_nonzero": n_nonzero,
        "W": round(W, 2),
        "p_value": p_value,
        "p_bonferroni": p_bonferroni,
        "Z": round(z_stat, 4),
        "r_effect_size": round(r_effect, 4),
        "rank_biserial": round(rank_biserial, 4),
        "interpretation": interpretation,
    }


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run entity_level_wilcoxon.py first.")
        return

    with open(RESULTS_PATH) as f:
        wilcoxon_data = json.load(f)

    print("=" * 90)
    print("WILCOXON EFFECT SIZES + BONFERRONI CORRECTION")
    print(f"k = {K_COMPARISONS} comparisons, α_adjusted = {0.05/K_COMPARISONS:.4f}")
    print("=" * 90)

    all_results = []

    for ds in wilcoxon_data:
        name = ds["dataset"]
        sge_rates = ds["entity_sge_rates"]
        base_rates = ds["entity_base_rates"]

        effect = compute_effect_size(sge_rates, base_rates)
        effect["dataset"] = name
        effect["mean_sge_rate"] = ds["mean_sge_rate"]
        effect["mean_base_rate"] = ds["mean_base_rate"]
        all_results.append(effect)

    # Print summary table
    print(f"\n{'Dataset':<25} {'n':>4} {'W':>8} {'Z':>8} {'p':>12} {'p_Bonf':>12} "
          f"{'r':>8} {'r_rb':>8} {'Size':>8} {'Sig':>5}")
    print("-" * 110)

    for r in all_results:
        sig_raw = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        sig_bonf = "***" if r["p_bonferroni"] < 0.001 else "**" if r["p_bonferroni"] < 0.01 else "*" if r["p_bonferroni"] < 0.05 else "ns"
        W_str = f"{r['W']:.1f}" if r["W"] is not None else "N/A"
        Z_str = f"{r['Z']:.4f}" if r["Z"] is not None else "N/A"
        r_str = f"{r['r_effect_size']:.4f}" if r["r_effect_size"] is not None else "N/A"
        rb_str = f"{r['rank_biserial']:.4f}" if r["rank_biserial"] is not None else "N/A"

        print(f"{r['dataset']:<25} {r['n']:>4} {W_str:>8} {Z_str:>8} "
              f"{r['p_value']:>12.2e} {r['p_bonferroni']:>12.2e} "
              f"{r_str:>8} {rb_str:>8} {r['interpretation']:>8} {sig_bonf:>5}")

    # Print paper-ready summary
    print(f"\n{'='*90}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*90}")
    print(f"Bonferroni correction: k={K_COMPARISONS}, α_adjusted=0.0125")
    print()

    for r in all_results:
        bonf_sig = r["p_bonferroni"] < 0.05
        print(f"  {r['dataset']}:")
        print(f"    W={r['W']}, Z={r['Z']}, p={r['p_value']:.2e}, "
              f"p_Bonferroni={r['p_bonferroni']:.2e} {'(sig)' if bonf_sig else '(ns)'}")
        print(f"    Effect size r={r['r_effect_size']} ({r['interpretation']}), "
              f"rank-biserial r_rb={r['rank_biserial']}")
        print(f"    Mean FC: SGE={r['mean_sge_rate']:.4f}, Baseline={r['mean_base_rate']:.4f}")
        print()

    # Ceiling effect analysis
    print(f"\n{'='*90}")
    print("CEILING EFFECT ANALYSIS")
    print(f"{'='*90}")
    for ds in wilcoxon_data:
        sge_rates = ds["entity_sge_rates"]
        at_ceiling = sum(1 for v in sge_rates.values() if v >= 1.0)
        total = len(sge_rates)
        print(f"  {ds['dataset']}: {at_ceiling}/{total} entities at FC=1.0 ceiling "
              f"({at_ceiling/total:.0%})")

    # Save results
    output = {
        "k_comparisons": K_COMPARISONS,
        "alpha_adjusted": 0.05 / K_COMPARISONS,
        "results": all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
