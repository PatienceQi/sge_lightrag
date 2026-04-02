"""
McNemar's test and Cohen's g effect sizes for GGCR evaluation.

Compares SGE+GGCR vs other systems on L3/L4 questions for:
  - 25-entity experiment (97 questions total, L3+L4 subset)
  - 190-country full-scale experiment (20 questions, all L3+L4)

McNemar's test is the appropriate paired test for binary (correct/incorrect)
outcomes where the same questions are evaluated by multiple systems.

Contingency table:
              System B correct   System B wrong
System A correct       a                 b
System A wrong         c                 d

McNemar statistic = (b - c)^2 / (b + c)  [chi-squared, df=1]
  - With continuity correction: (|b - c| - 1)^2 / (b + c)
  - We use the exact binomial (sign test) when b+c < 25, else chi-squared

Cohen's g = (p - 0.5) where p = b / (b + c)
  - Measures asymmetry in discordant pairs
  - g = 0: no effect; g ≈ 0.05: small; g ≈ 0.15: medium; g ≈ 0.25: large
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from scipy.stats import chi2, binomtest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAIN_FILE = RESULTS_DIR / "ggcr_results.json"
FULLSCALE_FILE = RESULTS_DIR / "ggcr_results_fullscale.json"
OUTPUT_FILE = RESULTS_DIR / "ggcr_statistical_tests.json"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_paired_results(detailed_results: list, levels: list[str]) -> dict:
    """
    Build a mapping: question_id -> {system: correct_bool}
    filtered to the requested levels.
    """
    paired: dict = defaultdict(dict)
    for entry in detailed_results:
        if entry["level"] in levels:
            paired[entry["id"]][entry["system"]] = entry["correct"]
    return dict(paired)


def build_contingency(paired: dict, system_a: str, system_b: str) -> dict:
    """
    Build McNemar contingency table for system_a vs system_b.
    Only includes questions answered by both systems.

    Returns dict with keys: a, b, c, d, n
      a = both correct
      b = A correct, B wrong   (discordant, favours A)
      c = A wrong, B correct   (discordant, favours B)
      d = both wrong
    """
    a = b = c = d = 0
    for qid, sys_results in paired.items():
        if system_a not in sys_results or system_b not in sys_results:
            continue
        ca = sys_results[system_a]
        cb = sys_results[system_b]
        if ca and cb:
            a += 1
        elif ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
        else:
            d += 1
    return {"a": a, "b": b, "c": c, "d": d, "n": a + b + c + d}


def mcnemar_test(contingency: dict, continuity_correction: bool = True) -> dict:
    """
    Compute McNemar's test statistic and p-value.

    Uses exact binomial when b+c < 25, chi-squared (with optional continuity
    correction) otherwise.

    Returns dict with: chi2_stat, p_value, method, b, c, discordant_total
    """
    b = contingency["b"]
    c = contingency["c"]
    discordant = b + c

    if discordant == 0:
        return {
            "chi2_stat": None,
            "p_value": 1.0,
            "method": "exact_binomial",
            "b": b,
            "c": c,
            "discordant_total": 0,
            "note": "No discordant pairs — systems agree on all questions",
        }

    if discordant < 25:
        # Exact sign test: probability of observing b or more successes
        # under H0: p=0.5 (two-tailed)
        p_value = float(binomtest(b, discordant, 0.5, alternative="two-sided").pvalue)
        return {
            "chi2_stat": None,
            "p_value": p_value,
            "method": "exact_binomial",
            "b": b,
            "c": c,
            "discordant_total": discordant,
        }

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / discordant
        method = "chi2_with_continuity_correction"
    else:
        stat = (b - c) ** 2 / discordant
        method = "chi2_no_correction"

    p_value = float(1 - chi2.cdf(stat, df=1))
    return {
        "chi2_stat": round(stat, 4),
        "p_value": p_value,
        "method": method,
        "b": b,
        "c": c,
        "discordant_total": discordant,
    }


def cohen_g(contingency: dict) -> dict:
    """
    Compute Cohen's g effect size.

    g = p - 0.5  where p = b / (b + c)

    Interpretation thresholds (Cohen 1988):
      small:  g ≈ 0.05
      medium: g ≈ 0.15
      large:  g ≈ 0.25
    """
    b = contingency["b"]
    c = contingency["c"]
    discordant = b + c

    if discordant == 0:
        return {"g": None, "p_discordant": None, "interpretation": "no_discordant_pairs"}

    p = b / discordant
    g = abs(p - 0.5)

    if g < 0.05:
        interp = "negligible"
    elif g < 0.15:
        interp = "small"
    elif g < 0.25:
        interp = "medium"
    else:
        interp = "large"

    return {
        "g": round(g, 4),
        "p_discordant": round(p, 4),
        "interpretation": interp,
    }


def run_comparison(
    paired: dict,
    system_a: str,
    system_b: str,
    label: str,
) -> dict:
    """Run McNemar + Cohen's g for one system pair and return result dict."""
    cont = build_contingency(paired, system_a, system_b)
    test = mcnemar_test(cont)
    effect = cohen_g(cont)

    significant = test["p_value"] < 0.05

    return {
        "comparison": label,
        "system_a": system_a,
        "system_b": system_b,
        "n_questions": cont["n"],
        "contingency": cont,
        "mcnemar": test,
        "cohen_g": effect,
        "significant_p05": significant,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load data
    with open(MAIN_FILE) as f:
        main_data = json.load(f)
    with open(FULLSCALE_FILE) as f:
        fullscale_data = json.load(f)

    results: dict = {
        "experiment_25entity": {},
        "experiment_190country": {},
        "summary_table": [],
    }

    # ------------------------------------------------------------------
    # 25-entity experiment: L3+L4 questions
    # ------------------------------------------------------------------
    paired_25 = load_paired_results(
        main_data["detailed_results"], levels=["L3", "L4"]
    )
    n_l3l4 = len(paired_25)

    comparisons_25 = [
        ("sge_ggcr", "pure_compact", "GGCR vs Pure_Compact"),
        ("sge_ggcr", "naive_rag", "GGCR vs Naive_RAG"),
        ("sge_ggcr", "concat_all", "GGCR vs Concat_All"),
    ]

    exp25_results = []
    for sys_a, sys_b, label in comparisons_25:
        r = run_comparison(paired_25, sys_a, sys_b, label)
        exp25_results.append(r)

    results["experiment_25entity"] = {
        "description": "25-entity experiment, L3+L4 questions only",
        "n_l3l4_questions": n_l3l4,
        "comparisons": exp25_results,
    }

    # ------------------------------------------------------------------
    # 190-country experiment: all questions (all L3+L4)
    # ------------------------------------------------------------------
    paired_190 = load_paired_results(
        fullscale_data["detailed_results"], levels=["L3", "L4"]
    )
    n_190 = len(paired_190)

    comparisons_190 = [
        ("sge_ggcr", "pure_compact", "GGCR vs Pure_Compact"),
        ("sge_ggcr", "concat_all", "GGCR vs Concat_All"),
    ]

    exp190_results = []
    for sys_a, sys_b, label in comparisons_190:
        r = run_comparison(paired_190, sys_a, sys_b, label)
        exp190_results.append(r)

    results["experiment_190country"] = {
        "description": "190-country full-scale experiment, all questions (L3+L4)",
        "n_questions": n_190,
        "comparisons": exp190_results,
    }

    # ------------------------------------------------------------------
    # Build summary table rows
    # ------------------------------------------------------------------
    summary_rows = []
    for r in exp25_results:
        mc = r["mcnemar"]
        cg = r["cohen_g"]
        summary_rows.append({
            "experiment": "25-entity",
            "comparison": r["comparison"],
            "n": r["n_questions"],
            "b_favours_A": r["contingency"]["b"],
            "c_favours_B": r["contingency"]["c"],
            "discordant": r["contingency"]["b"] + r["contingency"]["c"],
            "chi2_or_method": mc.get("chi2_stat", mc["method"]),
            "p_value": round(mc["p_value"], 4),
            "significant": r["significant_p05"],
            "cohen_g": cg["g"],
            "effect_size": cg["interpretation"],
        })
    for r in exp190_results:
        mc = r["mcnemar"]
        cg = r["cohen_g"]
        summary_rows.append({
            "experiment": "190-country",
            "comparison": r["comparison"],
            "n": r["n_questions"],
            "b_favours_A": r["contingency"]["b"],
            "c_favours_B": r["contingency"]["c"],
            "discordant": r["contingency"]["b"] + r["contingency"]["c"],
            "chi2_or_method": mc.get("chi2_stat", mc["method"]),
            "p_value": round(mc["p_value"], 4),
            "significant": r["significant_p05"],
            "cohen_g": cg["g"],
            "effect_size": cg["interpretation"],
        })

    results["summary_table"] = summary_rows

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {OUTPUT_FILE}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("McNemar's Test Results: SGE+GGCR vs Baseline Systems")
    print("=" * 80)
    print(
        f"{'Experiment':<15} {'Comparison':<30} {'N':>4} {'b':>4} {'c':>4} "
        f"{'Disc':>4} {'Stat':>10} {'p-val':>8} {'Sig?':>5} {'g':>6} {'Effect':<12}"
    )
    print("-" * 110)
    for row in summary_rows:
        stat_str = (
            f"chi2={row['chi2_or_method']:.2f}"
            if isinstance(row["chi2_or_method"], float)
            else str(row["chi2_or_method"])
        )
        sig_str = "YES *" if row["significant"] else "no"
        g_str = f"{row['cohen_g']:.4f}" if row["cohen_g"] is not None else "N/A"
        print(
            f"{row['experiment']:<15} {row['comparison']:<30} {row['n']:>4} "
            f"{row['b_favours_A']:>4} {row['c_favours_B']:>4} {row['discordant']:>4} "
            f"{stat_str:>10} {row['p_value']:>8.4f} {sig_str:>5} {g_str:>6} "
            f"{row['effect_size']:<12}"
        )
    print("=" * 110)
    print("\nNotes:")
    print("  b = discordant pairs where GGCR correct, baseline wrong  (favours GGCR)")
    print("  c = discordant pairs where GGCR wrong, baseline correct  (favours baseline)")
    print("  Stat: exact_binomial used when discordant < 25; chi2 with continuity correction otherwise")
    print("  Cohen's g: negligible<0.05, small<0.15, medium<0.25, large>=0.25")
    print("  * p < 0.05 (two-tailed)")
    print()


if __name__ == "__main__":
    main()
