#!/usr/bin/env python3
"""
interaction_ci_analysis.py — Interaction term CI + permutation test for SGE 2x2 ablation.

Computes:
  1. Interaction term confidence intervals for the 2x2 factorial ablation.
     Interaction = Full - Serial - Schema + Baseline  (for each dataset)
     Bootstrap 95% CI (n=10000) using entity-level fact vectors.

  2. 50-country permutation test (Wilcoxon signed-rank on v3 gold, if graphs exist;
     otherwise falls back to 25-country entity-level Wilcoxon on v2 gold).

  3. Bootstrap 95% CI for the effect size r from each Wilcoxon test.

Saves results to experiments/results/statistical_improvements.json.

Usage:
    python3 experiments/statistical/interaction_ci_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import wilcoxon, norm

from evaluation.evaluate_coverage import load_gold, load_graph, check_fact_coverage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_OUT = PROJECT_ROOT / "experiments" / "results" / "statistical_improvements.json"

# Bootstrap parameters
N_BOOTSTRAP = 10_000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

# Maps dataset key -> (label, gold_v2_file, n_gold_facts, entity key in entity_level)
# Condition graph dir names: full_sge, serial_only (c4), schema_only, baseline
DATASETS = [
    {
        "key": "WHO",
        "label": "WHO Life Expectancy",
        "gold_v2": EVAL_GOLD_DIR / "gold_who_life_expectancy_v2.jsonl",
        "gold_v3": EVAL_GOLD_DIR / "gold_who_life_expectancy_v3.jsonl",
        "graph_full_sge": OUTPUT_DIR / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_serial_only": OUTPUT_DIR / "ablation_c4_serial_only_who" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_schema_only": OUTPUT_DIR / "ablation_schema_only_who" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_baseline": OUTPUT_DIR / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        # Paper Table 4 aggregate FC values (for sanity check)
        "paper_fc": {"full": 1.000, "serial": 0.013, "schema": 0.380, "baseline": 0.167},
    },
    {
        "key": "WB_CM",
        "label": "WB Child Mortality",
        "gold_v2": EVAL_GOLD_DIR / "gold_wb_child_mortality_v2.jsonl",
        "gold_v3": EVAL_GOLD_DIR / "gold_wb_child_mortality_v3.jsonl",
        "graph_full_sge": OUTPUT_DIR / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_serial_only": OUTPUT_DIR / "ablation_c4_serial_only_wb_cm" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_schema_only": OUTPUT_DIR / "ablation_schema_only_wb_cm" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_baseline": OUTPUT_DIR / "baseline_wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "paper_fc": {"full": 1.000, "serial": 0.760, "schema": 0.000, "baseline": 0.473},
    },
    {
        "key": "Inpatient",
        "label": "HK Inpatient 2023",
        "gold_v2": EVAL_GOLD_DIR / "gold_inpatient_2023.jsonl",
        "gold_v3": None,  # No v3 for inpatient
        "graph_full_sge": OUTPUT_DIR / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_serial_only": OUTPUT_DIR / "ablation_c4_serial_only_inpatient" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_schema_only": OUTPUT_DIR / "ablation_schema_only_inpatient" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_baseline": OUTPUT_DIR / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "paper_fc": {"full": 0.938, "serial": 0.438, "schema": 0.750, "baseline": 0.438},
    },
    {
        "key": "WB_Pop",
        "label": "WB Population",
        "gold_v2": EVAL_GOLD_DIR / "gold_wb_population_v2.jsonl",
        "gold_v3": EVAL_GOLD_DIR / "gold_wb_population_v3.jsonl",
        "graph_full_sge": OUTPUT_DIR / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_serial_only": OUTPUT_DIR / "ablation_c4_serial_only_wb_pop" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_schema_only": OUTPUT_DIR / "ablation_schema_only_wb_pop" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_baseline": OUTPUT_DIR / "baseline_wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "paper_fc": {"full": 1.000, "serial": 0.000, "schema": 0.007, "baseline": 0.187},
    },
    {
        "key": "WB_Mat",
        "label": "WB Maternal Mortality",
        "gold_v2": EVAL_GOLD_DIR / "gold_wb_maternal_v2.jsonl",
        "gold_v3": EVAL_GOLD_DIR / "gold_wb_maternal_v3.jsonl",
        "graph_full_sge": OUTPUT_DIR / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_serial_only": OUTPUT_DIR / "ablation_c4_serial_only_wb_mat" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_schema_only": OUTPUT_DIR / "ablation_schema_only_wb_mat" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "graph_baseline": OUTPUT_DIR / "baseline_wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "paper_fc": {"full": 0.967, "serial": 0.827, "schema": 0.000, "baseline": 0.787},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_graph_safe(path: Path):
    """Load graph or return None if path missing."""
    if not path.exists():
        return None, None, None
    _, nodes, entity_text = load_graph(str(path))
    return nodes, entity_text, True


def compute_fact_vector(facts: list, nodes, entity_text) -> np.ndarray:
    """
    Return binary vector (length = len(facts)) indicating which facts are covered.

    Reuses check_fact_coverage from evaluate_coverage.py.
    Returns array of 0/1 floats aligned with the facts list.
    """
    if nodes is None:
        return np.zeros(len(facts), dtype=float)

    covered, _ = check_fact_coverage(facts, nodes, entity_text)
    covered_set = set(
        (f["subject"], f["value"], f.get("year", "")) for f in covered
    )
    vec = np.array([
        1.0 if (f["subject"], f["value"], f.get("year", "")) in covered_set
        else 0.0
        for f in facts
    ])
    return vec


def interaction_point_estimate(vec_full, vec_serial, vec_schema, vec_base) -> float:
    """Compute interaction = mean(Full) - mean(Serial) - mean(Schema) + mean(Baseline)."""
    return float(vec_full.mean() - vec_serial.mean() - vec_schema.mean() + vec_base.mean())


def bootstrap_interaction_ci(
    vec_full: np.ndarray,
    vec_serial: np.ndarray,
    vec_schema: np.ndarray,
    vec_base: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for interaction term by resampling facts with replacement.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(vec_full)

    boot_interactions = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_interactions[i] = interaction_point_estimate(
            vec_full[idx], vec_serial[idx], vec_schema[idx], vec_base[idx]
        )

    point = interaction_point_estimate(vec_full, vec_serial, vec_schema, vec_base)
    ci_lo = float(np.percentile(boot_interactions, 2.5))
    ci_hi = float(np.percentile(boot_interactions, 97.5))
    return point, ci_lo, ci_hi


def compute_entity_rates(facts: list, covered: list) -> dict[str, float]:
    """Compute per-entity (subject) coverage rate."""
    total: dict[str, int] = defaultdict(int)
    covered_cnt: dict[str, int] = defaultdict(int)

    for f in facts:
        total[f["subject"]] += 1
    for f in covered:
        covered_cnt[f["subject"]] += 1

    return {
        entity: covered_cnt.get(entity, 0) / total_cnt
        for entity, total_cnt in total.items()
    }


def run_wilcoxon_with_effect(
    sge_rates: dict[str, float],
    base_rates: dict[str, float],
) -> dict:
    """Run Wilcoxon signed-rank (greater) and compute effect size r = Z/sqrt(N)."""
    entities = sorted(sge_rates.keys())
    sge_list = [sge_rates[e] for e in entities]
    base_list = [base_rates.get(e, 0.0) for e in entities]
    n = len(entities)

    diffs = [s - b for s, b in zip(sge_list, base_list)]
    n_nonzero = sum(1 for d in diffs if d != 0)

    if n_nonzero == 0:
        return {
            "n": n,
            "n_nonzero": 0,
            "W": None,
            "p_value": 1.0,
            "Z": None,
            "r_effect_size": None,
            "interpretation": "no_difference",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = wilcoxon(sge_list, base_list, alternative="greater", method="approx")

    W = float(result.statistic)
    p_value = float(result.pvalue)

    if hasattr(result, "zstatistic") and result.zstatistic is not None:
        z_stat = float(result.zstatistic)
    else:
        z_stat = float(norm.ppf(1.0 - p_value)) if p_value < 1.0 else 0.0

    r_effect = z_stat / math.sqrt(n_nonzero) if n_nonzero > 0 else 0.0

    if abs(r_effect) >= 0.5:
        interp = "large"
    elif abs(r_effect) >= 0.3:
        interp = "medium"
    elif abs(r_effect) >= 0.1:
        interp = "small"
    else:
        interp = "negligible"

    return {
        "n": n,
        "n_nonzero": n_nonzero,
        "W": round(W, 2),
        "p_value": round(p_value, 8),
        "Z": round(z_stat, 4),
        "r_effect_size": round(r_effect, 4),
        "interpretation": interp,
    }


def bootstrap_effect_size_ci(
    sge_rates: dict[str, float],
    base_rates: dict[str, float],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for effect size r by resampling paired (sge, base) observations.

    Returns (ci_lower, ci_upper).
    """
    entities = sorted(sge_rates.keys())
    sge_arr = np.array([sge_rates[e] for e in entities])
    base_arr = np.array([base_rates.get(e, 0.0) for e in entities])
    n = len(entities)

    rng = np.random.default_rng(seed)
    boot_r = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        s = sge_arr[idx].tolist()
        b = base_arr[idx].tolist()

        diffs = [si - bi for si, bi in zip(s, b)]
        n_nz = sum(1 for d in diffs if d != 0)
        if n_nz == 0:
            boot_r.append(0.0)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = wilcoxon(s, b, alternative="greater", method="approx")
            if hasattr(res, "zstatistic") and res.zstatistic is not None:
                z = float(res.zstatistic)
            else:
                z = float(norm.ppf(1.0 - float(res.pvalue))) if float(res.pvalue) < 1.0 else 0.0
            boot_r.append(z / math.sqrt(n_nz))
        except Exception:
            boot_r.append(0.0)

    boot_r_arr = np.array(boot_r)
    return float(np.percentile(boot_r_arr, 2.5)), float(np.percentile(boot_r_arr, 97.5))


# ---------------------------------------------------------------------------
# Part 1: Interaction term bootstrap CI
# ---------------------------------------------------------------------------

def run_interaction_analysis(ds: dict) -> dict:
    """Load four condition graphs, compute interaction CI for one dataset."""
    key = ds["key"]
    label = ds["label"]
    gold_path = ds["gold_v2"]
    print(f"\n[Interaction CI] {label}")

    if not gold_path.exists():
        return {"dataset": key, "label": label, "error": f"Gold not found: {gold_path}"}

    _, facts = load_gold(str(gold_path))
    if not facts:
        return {"dataset": key, "label": label, "error": "No facts in gold file"}

    print(f"  Gold facts: {len(facts)}")

    # Load four condition graphs
    conditions = {
        "full_sge": ds["graph_full_sge"],
        "serial_only": ds["graph_serial_only"],
        "schema_only": ds["graph_schema_only"],
        "baseline": ds["graph_baseline"],
    }

    cond_vectors: dict[str, np.ndarray] = {}
    cond_fc: dict[str, float] = {}
    missing_conditions = []

    for cond, graph_path in conditions.items():
        if not graph_path.exists():
            print(f"  WARN: graph missing for {cond}: {graph_path}")
            missing_conditions.append(cond)
            continue
        nodes, entity_text, _ = _load_graph_safe(graph_path)
        vec = compute_fact_vector(facts, nodes, entity_text)
        cond_vectors[cond] = vec
        cond_fc[cond] = float(vec.mean())
        print(f"  {cond}: FC={cond_fc[cond]:.4f}  (graph: {graph_path.parent.parent.name})")

    if missing_conditions:
        return {
            "dataset": key,
            "label": label,
            "error": f"Missing graphs: {missing_conditions}",
            "computed_fc": cond_fc,
        }

    # Paper sanity check
    paper = ds["paper_fc"]
    for cond, expected in [("full_sge", paper["full"]), ("serial_only", paper["serial"]),
                            ("schema_only", paper["schema"]), ("baseline", paper["baseline"])]:
        diff = abs(cond_fc[cond] - expected)
        if diff > 0.05:
            print(f"  WARN: {cond} FC={cond_fc[cond]:.3f} vs paper {expected:.3f} (diff={diff:.3f})")

    # Point estimate
    point, ci_lo, ci_hi = bootstrap_interaction_ci(
        cond_vectors["full_sge"],
        cond_vectors["serial_only"],
        cond_vectors["schema_only"],
        cond_vectors["baseline"],
        n_resamples=N_BOOTSTRAP,
    )

    # Analytical interaction from aggregate FC values
    analytic_interaction = (
        cond_fc["full_sge"] - cond_fc["serial_only"]
        - cond_fc["schema_only"] + cond_fc["baseline"]
    )

    # Proportion of bootstrap resamples > 0 (one-sided p-value proxy)
    rng = np.random.default_rng(RNG_SEED)
    n = len(facts)
    boot_vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_vals.append(interaction_point_estimate(
            cond_vectors["full_sge"][idx],
            cond_vectors["serial_only"][idx],
            cond_vectors["schema_only"][idx],
            cond_vectors["baseline"][idx],
        ))
    p_bootstrap = float(np.mean(np.array(boot_vals) <= 0))

    print(f"  Interaction: point={point:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Bootstrap p (H0: interaction <= 0): {p_bootstrap:.4f}")

    return {
        "dataset": key,
        "label": label,
        "n_facts": len(facts),
        "computed_fc": {c: round(v, 4) for c, v in cond_fc.items()},
        "paper_fc": paper,
        "interaction_point_estimate": round(point, 4),
        "interaction_analytic": round(analytic_interaction, 4),
        "interaction_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "bootstrap_p_interaction_positive": round(1.0 - p_bootstrap, 4),
        "bootstrap_p_value_one_sided": round(p_bootstrap, 4),
        "n_bootstrap": N_BOOTSTRAP,
    }


# ---------------------------------------------------------------------------
# Part 2: Permutation test (Wilcoxon) + effect size CI
# ---------------------------------------------------------------------------

def run_permutation_analysis(ds: dict) -> dict:
    """
    Run entity-level Wilcoxon signed-rank test for one dataset.

    Tries v3 gold (50 countries) first. If v3 graphs don't exist, falls back
    to v2 gold (25 countries) with existing graphs.
    """
    key = ds["key"]
    label = ds["label"]

    # Determine which gold + graphs to use
    v3_gold = ds.get("gold_v3")
    v3_sge_path = None
    v3_base_path = None

    # v3 graphs would follow naming convention: {dataset}_v3/lightrag_storage/...
    # They don't exist in this project, so we check and fall back.
    if v3_gold and Path(v3_gold).exists():
        # Check whether v3 SGE and baseline graphs exist
        v3_sge_candidates = [
            OUTPUT_DIR / f"{key.lower()}_v3" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
            OUTPUT_DIR / f"{key.lower().replace('_', '')}_v3" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        ]
        v3_base_candidates = [
            OUTPUT_DIR / f"baseline_{key.lower()}_v3" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        ]
        for p in v3_sge_candidates:
            if p.exists():
                v3_sge_path = p
                break
        for p in v3_base_candidates:
            if p.exists():
                v3_base_path = p
                break

    use_v3 = v3_sge_path is not None and v3_base_path is not None
    gold_version = "v3" if use_v3 else "v2"

    if use_v3:
        gold_path = v3_gold
        sge_path = v3_sge_path
        base_path = v3_base_path
        note = "50-country v3 gold + v3 graphs"
    else:
        gold_path = ds["gold_v2"]
        sge_path = ds["graph_full_sge"]
        base_path = ds["graph_baseline"]
        note = "25-country v2 gold (v3 graphs not found; falling back to entity-level permutation)"

    print(f"\n[Permutation Test] {label}  ({note})")

    if not gold_path.exists():
        return {"dataset": key, "label": label, "error": f"Gold not found: {gold_path}"}
    if not sge_path.exists():
        return {"dataset": key, "label": label, "error": f"SGE graph not found: {sge_path}"}
    if not base_path.exists():
        return {"dataset": key, "label": label, "error": f"Baseline graph not found: {base_path}"}

    _, facts = load_gold(str(gold_path))
    if not facts:
        return {"dataset": key, "label": label, "error": "No facts in gold file"}

    nodes_sge, text_sge, _ = _load_graph_safe(sge_path)
    nodes_base, text_base, _ = _load_graph_safe(base_path)

    covered_sge, _ = check_fact_coverage(facts, nodes_sge, text_sge)
    covered_base, _ = check_fact_coverage(facts, nodes_base, text_base)

    sge_rates = compute_entity_rates(facts, covered_sge)
    base_rates = compute_entity_rates(facts, covered_base)

    n_entities = len(sge_rates)
    mean_sge = float(np.mean(list(sge_rates.values())))
    mean_base = float(np.mean([base_rates.get(e, 0.0) for e in sge_rates]))

    print(f"  Entities: {n_entities}, Facts: {len(facts)}")
    print(f"  Mean SGE rate: {mean_sge:.4f},  Mean Base rate: {mean_base:.4f}")

    wilcoxon_result = run_wilcoxon_with_effect(sge_rates, base_rates)

    sig = ("***" if wilcoxon_result["p_value"] < 0.001
           else "**" if wilcoxon_result["p_value"] < 0.01
           else "*" if wilcoxon_result["p_value"] < 0.05
           else "ns")
    print(f"  Wilcoxon W={wilcoxon_result['W']}, Z={wilcoxon_result['Z']}, "
          f"p={wilcoxon_result['p_value']:.2e} {sig},  r={wilcoxon_result['r_effect_size']}")

    # Bootstrap CI for effect size r
    r_ci_lo, r_ci_hi = bootstrap_effect_size_ci(sge_rates, base_rates, n_resamples=N_BOOTSTRAP)
    print(f"  Effect size r={wilcoxon_result['r_effect_size']:.4f},  95% CI=[{r_ci_lo:.4f}, {r_ci_hi:.4f}]")

    return {
        "dataset": key,
        "label": label,
        "gold_version": gold_version,
        "note": note,
        "n_facts": len(facts),
        "n_entities": n_entities,
        "mean_sge_rate": round(mean_sge, 4),
        "mean_base_rate": round(mean_base, 4),
        "entity_sge_rates": {e: round(v, 4) for e, v in sge_rates.items()},
        "entity_base_rates": {e: round(base_rates.get(e, 0.0), 4) for e in sge_rates},
        "wilcoxon": wilcoxon_result,
        "effect_size_ci_95": [round(r_ci_lo, 4), round(r_ci_hi, 4)],
        "n_bootstrap": N_BOOTSTRAP,
    }


# ---------------------------------------------------------------------------
# Combined interaction significance test
# ---------------------------------------------------------------------------

def combined_interaction_test(interaction_results: list[dict]) -> dict:
    """
    Combine per-dataset bootstrap p-values using Fisher's method.

    H0 for each dataset: interaction <= 0.
    Fisher's: -2 * sum(log(p_i)) ~ chi-squared(2k).
    """
    from scipy.stats import chi2

    p_values = []
    for r in interaction_results:
        if "error" not in r:
            p = r.get("bootstrap_p_value_one_sided", 1.0)
            # Clip to avoid log(0)
            p_values.append(max(p, 1e-10))

    if not p_values:
        return {"error": "No valid datasets for combined test"}

    chi_stat = -2.0 * sum(math.log(p) for p in p_values)
    df = 2 * len(p_values)
    p_combined = float(1.0 - chi2.cdf(chi_stat, df=df))

    return {
        "n_datasets": len(p_values),
        "fisher_chi_squared": round(chi_stat, 4),
        "df": df,
        "p_combined": round(p_combined, 8),
        "individual_p_values": [round(p, 8) for p in p_values],
        "significant_at_0.05": p_combined < 0.05,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("INTERACTION CI ANALYSIS + PERMUTATION TEST")
    print(f"Bootstrap n={N_BOOTSTRAP}, seed={RNG_SEED}")
    print("=" * 70)

    # Part 1: Interaction term CI
    print("\n" + "=" * 70)
    print("PART 1: 2×2 FACTORIAL INTERACTION TERM BOOTSTRAP CI")
    print("=" * 70)
    print("Interaction = Full_SGE - Serial_Only - Schema_Only + Baseline")

    interaction_results = []
    for ds in DATASETS:
        result = run_interaction_analysis(ds)
        interaction_results.append(result)

    # Combined significance test
    print("\n[Combined Fisher Test]")
    combined = combined_interaction_test(interaction_results)
    print(f"  Datasets: {combined.get('n_datasets')}")
    print(f"  Fisher chi²={combined.get('fisher_chi_squared')}, df={combined.get('df')}, "
          f"p={combined.get('p_combined'):.4e}")
    print(f"  Significant at α=0.05: {combined.get('significant_at_0.05')}")

    # Part 2: Permutation test + effect size CI
    print("\n" + "=" * 70)
    print("PART 2: PERMUTATION TEST (WILCOXON SIGNED-RANK) + EFFECT SIZE CI")
    print("=" * 70)
    print("Note: V3 graphs (50-country) not found; using v2 gold (25-country) fallback.")

    permutation_results = []
    for ds in DATASETS:
        result = run_permutation_analysis(ds)
        permutation_results.append(result)

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'Dataset':<18} {'n':>4} {'Gold':>5} {'MeanSGE':>8} {'MeanBase':>9} "
          f"{'W':>7} {'p':>10} {'r':>7} {'r CI':>15}")
    print("-" * 80)
    for r in permutation_results:
        if "error" in r:
            print(f"  {r['dataset']}: ERROR — {r['error']}")
            continue
        w = r["wilcoxon"]
        r_ci = r["effect_size_ci_95"]
        W_str = f"{w['W']:.1f}" if w["W"] is not None else "N/A"
        r_str = f"{w['r_effect_size']:.4f}" if w["r_effect_size"] is not None else "N/A"
        ci_str = f"[{r_ci[0]:.3f},{r_ci[1]:.3f}]"
        print(f"{r['dataset']:<18} {r['n_entities']:>4} {r['gold_version']:>5} "
              f"{r['mean_sge_rate']:>8.4f} {r['mean_base_rate']:>9.4f} "
              f"{W_str:>7} {w['p_value']:>10.2e} {r_str:>7} {ci_str:>15}")

    # Save results
    output = {
        "metadata": {
            "description": "Interaction CI + permutation test for SGE 2x2 ablation",
            "n_bootstrap": N_BOOTSTRAP,
            "rng_seed": RNG_SEED,
        },
        "interaction_analysis": {
            "description": (
                "Bootstrap 95% CI for 2x2 factorial interaction term. "
                "Interaction = Full_SGE - Serial_Only - Schema_Only + Baseline. "
                "CI computed by resampling facts with replacement (n=10000)."
            ),
            "per_dataset": interaction_results,
            "combined_fisher_test": combined,
        },
        "permutation_analysis": {
            "description": (
                "Entity-level Wilcoxon signed-rank test (alternative='greater') "
                "on per-entity (country/subject) fact coverage rates. "
                "V3 graphs (50-country) not found; used v2 gold (25-country) with "
                "existing SGE and Baseline graphs. "
                "Effect size r = Z/sqrt(N_nonzero). "
                "CI for r computed via bootstrap resampling of entity pairs."
            ),
            "per_dataset": permutation_results,
        },
    }

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
