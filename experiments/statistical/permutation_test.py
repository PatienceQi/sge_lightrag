"""
Permutation Test for Fact Coverage: SGE vs Baseline.

The entity-level Wilcoxon test assumes independence across the 25 countries,
but that assumption is violated because the same LLM call extracts all
countries simultaneously. This permutation test avoids the independence
assumption by using the observed data distribution directly.

For each dataset:
  - Compute per-entity (country) coverage rates for SGE and Baseline
  - Observed statistic: mean(SGE_rates) - mean(Baseline_rates)
  - Permutation test (10,000 iterations, seed=42):
      For each entity, randomly swap its (SGE_rate, Baseline_rate) pair
      with probability 0.5, then recompute the mean difference
  - p-value = proportion of permuted statistics >= observed (one-sided)
  - Effect size r from Wilcoxon (for comparison with existing results)

H0: mean difference (SGE_rate - Base_rate) = 0
H1: mean difference > 0 (one-sided: SGE is better)
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from evaluation.evaluate_coverage import load_gold, check_fact_coverage
from evaluation.graph_loaders import load_graph_auto

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
OUT_DIR = PROJECT_ROOT / "output"

DATASETS = [
    (
        "WHO Life Expectancy",
        EVAL_DIR / "gold" / "gold_who_life_expectancy_v2.jsonl",
        OUT_DIR / "who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        OUT_DIR / "baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
    ),
    (
        "WB Child Mortality",
        EVAL_DIR / "gold" / "gold_wb_child_mortality_v2.jsonl",
        OUT_DIR / "wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        OUT_DIR / "baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    ),
    (
        "WB Population",
        EVAL_DIR / "gold" / "gold_wb_population_v2.jsonl",
        OUT_DIR / "wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        OUT_DIR / "baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
    ),
    (
        "WB Maternal Mortality",
        EVAL_DIR / "gold" / "gold_wb_maternal_v2.jsonl",
        OUT_DIR / "wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        OUT_DIR / "baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
    ),
]

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_entity_coverage_rates(facts: list, covered_facts: list) -> dict:
    """
    Compute per-entity (country) coverage rates.

    Returns a dict mapping entity name -> coverage_rate (float in [0, 1]).
    """
    total_per_entity: dict[str, int] = defaultdict(int)
    covered_per_entity: dict[str, int] = defaultdict(int)

    for fact in facts:
        total_per_entity[fact["subject"]] += 1

    for fact in covered_facts:
        covered_per_entity[fact["subject"]] += 1

    return {
        entity: covered_per_entity.get(entity, 0) / total
        for entity, total in total_per_entity.items()
        if total > 0
    }


def run_permutation_test(
    sge_rates: np.ndarray,
    base_rates: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """
    Vectorized permutation test for paired observations.

    For each permutation, each entity's (SGE, Baseline) pair is swapped
    with probability 0.5 (equivalent to randomly flipping the sign of
    each difference). Computes one-sided p-value (H1: SGE > Baseline).

    Returns:
        observed_diff   — mean(SGE) - mean(Baseline) on original data
        p_value         — proportion of permuted stats >= observed_diff
        perm_stats      — array of all permuted test statistics
    """
    n = len(sge_rates)
    observed_diff = float(np.mean(sge_rates) - np.mean(base_rates))

    rng = np.random.default_rng(seed)
    # swap_mask[i, j] == True means entity j is swapped in permutation i
    swap_mask = rng.random((n_permutations, n)) < 0.5

    # Build permuted SGE and Baseline arrays using broadcasting
    # Shape: (n_permutations, n)
    perm_sge = np.where(swap_mask, base_rates, sge_rates)
    perm_base = np.where(swap_mask, sge_rates, base_rates)

    perm_stats = np.mean(perm_sge, axis=1) - np.mean(perm_base, axis=1)
    p_value = float(np.mean(perm_stats >= observed_diff))

    return observed_diff, p_value, perm_stats


def compute_wilcoxon_p_and_effect_size(
    sge_rates: np.ndarray,
    base_rates: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Compute Wilcoxon signed-rank p-value (one-sided) and effect size r.

    Effect size r = Z / sqrt(n), where Z is derived from the W statistic
    using the normal approximation. Returns (None, None) when all differences
    are zero (test not applicable).
    """
    differences = sge_rates - base_rates
    if np.all(differences == 0):
        return None, None

    result = wilcoxon(sge_rates, base_rates, alternative="greater")
    w_stat = result.statistic
    p_value = float(result.pvalue)

    # Effect size r = Z / sqrt(n) via normal approximation of W
    n = len(differences[differences != 0])
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z_score = (w_stat - mean_w) / std_w if std_w > 0 else 0.0
    effect_r = float(abs(z_score) / np.sqrt(len(sge_rates)))

    return p_value, round(effect_r, 4)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_permutation_for_dataset(
    name: str,
    gold_path: Path,
    sge_graph_path: Path,
    base_graph_path: Path,
) -> dict | None:
    """Run permutation test for a single dataset. Returns result dict or None."""
    print(f"\n--- {name} ---")

    for label, path in [
        ("Gold", gold_path),
        ("SGE graph", sge_graph_path),
        ("Baseline graph", base_graph_path),
    ]:
        if not path.exists():
            print(f"  SKIP: {label} not found: {path}")
            return None

    entities, facts = load_gold(str(gold_path))
    print(f"  Gold: {len(entities)} entities, {len(facts)} facts")

    _, sge_nodes, sge_text = load_graph_auto(str(sge_graph_path))
    sge_covered, _ = check_fact_coverage(facts, sge_nodes, sge_text)

    _, base_nodes, base_text = load_graph_auto(str(base_graph_path))
    base_covered, _ = check_fact_coverage(facts, base_nodes, base_text)

    sge_rate_map = compute_entity_coverage_rates(facts, sge_covered)
    base_rate_map = compute_entity_coverage_rates(facts, base_covered)

    all_entities = sorted(sge_rate_map.keys())
    n_entities = len(all_entities)

    sge_arr = np.array([sge_rate_map[e] for e in all_entities])
    base_arr = np.array([base_rate_map.get(e, 0.0) for e in all_entities])

    print(f"  Entities: {n_entities}")
    print(f"  Mean SGE: {np.mean(sge_arr):.4f}  Mean Base: {np.mean(base_arr):.4f}")

    observed_diff, perm_p, _ = run_permutation_test(
        sge_arr, base_arr, N_PERMUTATIONS, RANDOM_SEED
    )

    sig_perm = (
        "***" if perm_p < 0.001
        else "**" if perm_p < 0.01
        else "*" if perm_p < 0.05
        else "ns"
    )
    print(f"  Observed diff: {observed_diff:.4f}")
    print(f"  Permutation p={perm_p:.6f}  {sig_perm}  (n={N_PERMUTATIONS})")

    wilcoxon_p, effect_r = compute_wilcoxon_p_and_effect_size(sge_arr, base_arr)
    if wilcoxon_p is not None:
        sig_w = (
            "***" if wilcoxon_p < 0.001
            else "**" if wilcoxon_p < 0.01
            else "*" if wilcoxon_p < 0.05
            else "ns"
        )
        print(f"  Wilcoxon p={wilcoxon_p:.6f}  {sig_w}  effect_r={effect_r:.4f}")
    else:
        print("  Wilcoxon: all differences zero — not applicable")

    return {
        "dataset": name,
        "n_entities": n_entities,
        "observed_diff": round(observed_diff, 6),
        "permutation_p": round(perm_p, 6),
        "n_permutations": N_PERMUTATIONS,
        "wilcoxon_p": round(wilcoxon_p, 6) if wilcoxon_p is not None else None,
        "effect_size_r": effect_r,
        "mean_sge_rate": round(float(np.mean(sge_arr)), 4),
        "mean_base_rate": round(float(np.mean(base_arr)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("PERMUTATION TEST: SGE vs Baseline Fact Coverage (One-Sided)")
    print(f"n_permutations={N_PERMUTATIONS}, seed={RANDOM_SEED}")
    print("=" * 80)

    all_results = []

    for name, gold_path, sge_path, base_path in DATASETS:
        result = run_permutation_for_dataset(name, gold_path, sge_path, base_path)
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = (
        f"{'Dataset':<25} {'n':>4} {'Obs.Diff':>10} "
        f"{'Perm-p':>10} {'Wilcox-p':>10} {'r':>6} {'sig':>5}"
    )
    print(header)
    print("-" * 80)

    for r in all_results:
        perm_sig = (
            "***" if r["permutation_p"] < 0.001
            else "**" if r["permutation_p"] < 0.01
            else "*" if r["permutation_p"] < 0.05
            else "ns"
        )
        w_p_str = f"{r['wilcoxon_p']:.6f}" if r["wilcoxon_p"] is not None else "N/A"
        r_str = f"{r['effect_size_r']:.4f}" if r["effect_size_r"] is not None else "N/A"
        print(
            f"{r['dataset']:<25} {r['n_entities']:>4} {r['observed_diff']:>10.4f} "
            f"{r['permutation_p']:>10.6f} {w_p_str:>10} {r_str:>6} {perm_sig:>5}"
        )

    out_path = PROJECT_ROOT / "experiments" / "results" / "permutation_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
