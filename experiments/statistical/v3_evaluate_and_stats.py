#!/usr/bin/env python3
"""
v3 Gold Standard evaluation + Wilcoxon + Bootstrap CI.

Evaluates existing SGE and Baseline graphs against v3 (50-country) Gold Standards.
No graph rebuilding needed.
"""

import json, sys, math
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_coverage import load_gold, load_graph, check_entity_coverage, check_fact_coverage

# Dataset configurations: (name, gold_v3, sge_graph, base_graph, subject_field)
DATASETS = [
    {
        "name": "WHO Life Expectancy",
        "gold": "evaluation/gold/gold_who_life_expectancy_v3.jsonl",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "base_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    {
        "name": "WB Child Mortality",
        "gold": "evaluation/gold/gold_wb_child_mortality_v3.jsonl",
        "sge_graph": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "base_graph": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    {
        "name": "WB Population",
        "gold": "evaluation/gold/gold_wb_population_v3.jsonl",
        "sge_graph": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "base_graph": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    {
        "name": "WB Maternal",
        "gold": "evaluation/gold/gold_wb_maternal_v3.jsonl",
        "sge_graph": "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "base_graph": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
]


def per_entity_fc(gold_path, graph_path):
    """Compute per-entity FC rates using the official evaluate_coverage functions."""
    full_gold = str(PROJECT_ROOT / gold_path)
    full_graph = str(PROJECT_ROOT / graph_path)

    entities, facts = load_gold(full_gold)
    G, nodes, entity_text_2hop = load_graph(full_graph)

    # Overall EC/FC
    matched_ents = check_entity_coverage(entities, nodes)
    covered, not_covered = check_fact_coverage(facts, nodes, entity_text_2hop)
    ec = len(matched_ents) / len(entities) if entities else 0.0
    fc = len(covered) / len(facts) if facts else 0.0

    # Per-entity FC
    entity_facts = {}
    for f in facts:
        s = f["subject"]
        entity_facts.setdefault(s, []).append(f)

    covered_set = set()
    for c in covered:
        key = (c["subject"], c["value"], c.get("year", ""))
        covered_set.add(key)

    entity_fc = {}
    for subj, subj_facts in entity_facts.items():
        matched = 0
        for f in subj_facts:
            key = (f["subject"], f["value"], f.get("year", ""))
            if key in covered_set:
                matched += 1
        entity_fc[subj] = matched / len(subj_facts) if subj_facts else 0.0

    result = {"ec": round(ec, 3), "fc": round(fc, 3),
              "matched_entities": len(matched_ents), "total_entities": len(entities),
              "matched_facts": len(covered), "total_facts": len(facts)}
    return entity_fc, result


def wilcoxon_test(sge_fc_dict, base_fc_dict):
    """Wilcoxon signed-rank test on paired entity-level FC."""
    common = sorted(set(sge_fc_dict.keys()) & set(base_fc_dict.keys()))
    sge_vals = [sge_fc_dict[k] for k in common]
    base_vals = [base_fc_dict[k] for k in common]
    diffs = [s - b for s, b in zip(sge_vals, base_vals)]

    # Filter zero diffs (Wilcoxon ignores ties)
    nonzero = [(s, b) for s, b, d in zip(sge_vals, base_vals, diffs) if d != 0]
    n_nonzero = len(nonzero)

    if n_nonzero < 2:
        return {"W": None, "Z": None, "p": None, "p_bonf": None, "r": None,
                "n": len(common), "n_nonzero": n_nonzero}

    try:
        result = stats.wilcoxon([s - b for s, b in nonzero], alternative='greater')
        W = result.statistic
        # Compute Z approximation
        n = n_nonzero
        mean_W = n * (n + 1) / 4
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        Z = (W - mean_W) / std_W if std_W > 0 else 0
        r = abs(Z) / math.sqrt(n)
        p = result.pvalue
        return {
            "W": float(W), "Z": round(Z, 3), "p": float(p),
            "p_bonf": min(float(p) * 4, 1.0), "r": round(r, 3),
            "n": len(common), "n_nonzero": n_nonzero,
            "effect": "large" if r >= 0.5 else ("medium" if r >= 0.3 else "small"),
        }
    except Exception as e:
        return {"error": str(e), "n": len(common), "n_nonzero": n_nonzero}


def bootstrap_ci(gold_path, graph_path, n_boot=1000, seed=42):
    """Bootstrap 95% CI for FC using official evaluation."""
    full_gold = str(PROJECT_ROOT / gold_path)
    full_graph = str(PROJECT_ROOT / graph_path)

    entities, facts = load_gold(full_gold)
    G, nodes, entity_text_2hop = load_graph(full_graph)
    covered, not_covered = check_fact_coverage(facts, nodes, entity_text_2hop)

    covered_set = set()
    for c in covered:
        covered_set.add((c["subject"], c["value"], c.get("year", "")))

    scores = []
    for f in facts:
        key = (f["subject"], f["value"], f.get("year", ""))
        scores.append(1 if key in covered_set else 0)

    rng = np.random.RandomState(seed)
    scores_arr = np.array(scores)
    boot_fcs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(scores_arr), len(scores_arr))
        boot_fcs.append(scores_arr[idx].mean())

    return {
        "fc": float(scores_arr.mean()),
        "ci_lower": float(np.percentile(boot_fcs, 2.5)),
        "ci_upper": float(np.percentile(boot_fcs, 97.5)),
        "matched": int(scores_arr.sum()),
        "total": len(scores_arr),
    }


def main():
    print("=" * 70)
    print("v3 Gold Standard Evaluation (50 countries, 1200 facts)")
    print("=" * 70)

    all_results = {}

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n--- {name} ---")

        # SGE evaluation
        sge_entity_fc, sge_result = per_entity_fc(ds["gold"], ds["sge_graph"])
        sge_boot = bootstrap_ci(ds["gold"], ds["sge_graph"])
        print(f"  SGE:  EC={sge_result['ec']:.3f}  FC={sge_boot['fc']:.3f}  "
              f"CI=[{sge_boot['ci_lower']:.3f}, {sge_boot['ci_upper']:.3f}]  "
              f"({sge_boot['matched']}/{sge_boot['total']})")

        # Baseline evaluation
        base_entity_fc, base_result = per_entity_fc(ds["gold"], ds["base_graph"])
        base_boot = bootstrap_ci(ds["gold"], ds["base_graph"])
        print(f"  Base: EC={base_result['ec']:.3f}  FC={base_boot['fc']:.3f}  "
              f"CI=[{base_boot['ci_lower']:.3f}, {base_boot['ci_upper']:.3f}]  "
              f"({base_boot['matched']}/{base_boot['total']})")

        # Wilcoxon
        wilc = wilcoxon_test(sge_entity_fc, base_entity_fc)
        if wilc.get("p") is not None:
            print(f"  Wilcoxon: W={wilc['W']:.1f}  Z={wilc['Z']}  "
                  f"p={wilc['p']:.2e}  p_Bonf={wilc['p_bonf']:.2e}  "
                  f"r={wilc['r']}  ({wilc['effect']})")
        else:
            print(f"  Wilcoxon: insufficient non-zero diffs (n_nonzero={wilc['n_nonzero']})")

        # Ratio
        ratio = sge_boot['fc'] / base_boot['fc'] if base_boot['fc'] > 0 else float('inf')
        print(f"  Ratio: {ratio:.2f}×")

        all_results[name] = {
            "sge": {"ec": sge_result["ec"], "fc": sge_boot["fc"],
                    "ci": [sge_boot["ci_lower"], sge_boot["ci_upper"]],
                    "matched": sge_boot["matched"], "total": sge_boot["total"]},
            "baseline": {"ec": base_result["ec"], "fc": base_boot["fc"],
                         "ci": [base_boot["ci_lower"], base_boot["ci_upper"]],
                         "matched": base_boot["matched"], "total": base_boot["total"]},
            "wilcoxon": wilc,
            "ratio": round(ratio, 2),
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: v2 (25c/150f) vs v3 (50c/300f)")
    print(f"{'='*70}")
    print(f"{'Dataset':<22} {'v3 SGE FC':>10} {'v3 Base FC':>11} {'Ratio':>7} {'p_Bonf':>10} {'r':>6}")
    print("-" * 70)
    for name, r in all_results.items():
        p_str = f"{r['wilcoxon']['p_bonf']:.2e}" if r['wilcoxon'].get('p_bonf') else "N/A"
        r_str = f"{r['wilcoxon']['r']}" if r['wilcoxon'].get('r') else "N/A"
        print(f"{name:<22} {r['sge']['fc']:>10.3f} {r['baseline']['fc']:>11.3f} "
              f"{r['ratio']:>6.2f}× {p_str:>10} {r_str:>6}")

    # Save
    out_path = PROJECT_ROOT / "experiments" / "v3_gold_standard_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
