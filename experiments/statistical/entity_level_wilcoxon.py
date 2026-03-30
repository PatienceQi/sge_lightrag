"""
Entity-Level Wilcoxon Signed-Rank Test for Fact Coverage.

For each international dataset (WHO, WB Child Mortality, WB Population, WB Maternal),
computes per-entity (country) fact coverage rates for SGE and Baseline, then applies
Wilcoxon signed-rank test to determine whether SGE coverage is significantly higher.

H0: median difference (SGE_rate - Base_rate) = 0
H1: median difference > 0 (SGE is better, one-sided)
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from collections import defaultdict

from scipy.stats import wilcoxon
from evaluation.evaluate_coverage import load_gold, load_graph, check_fact_coverage

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

DATASETS = [
    ("WHO Life Expectancy",
     EVAL_DIR / "gold_who_life_expectancy_v2.jsonl",
     OUT_DIR / "who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml"),
    ("WB Child Mortality",
     EVAL_DIR / "gold_wb_child_mortality_v2.jsonl",
     OUT_DIR / "wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml"),
    ("WB Population",
     EVAL_DIR / "gold_wb_population_v2.jsonl",
     OUT_DIR / "wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml"),
    ("WB Maternal Mortality",
     EVAL_DIR / "gold_wb_maternal_v2.jsonl",
     OUT_DIR / "wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml"),
]


def compute_entity_coverage_rates(facts, covered_facts):
    """
    Compute per-entity coverage rates.

    Returns a dict: entity -> coverage_rate (covered / total for that entity).
    """
    total_per_entity = defaultdict(int)
    covered_per_entity = defaultdict(int)

    for fact in facts:
        total_per_entity[fact["subject"]] += 1

    for fact in covered_facts:
        covered_per_entity[fact["subject"]] += 1

    rates = {}
    for entity, total in total_per_entity.items():
        covered = covered_per_entity.get(entity, 0)
        rates[entity] = covered / total if total > 0 else 0.0

    return rates


def run_wilcoxon_for_dataset(name, gold_path, sge_graph_path, base_graph_path):
    """Run entity-level Wilcoxon test for a single dataset."""
    print(f"\n--- {name} ---")

    for label, path in [("Gold", gold_path), ("SGE graph", sge_graph_path), ("Baseline graph", base_graph_path)]:
        if not path.exists():
            print(f"  SKIP: {label} not found: {path}")
            return None

    entities, facts = load_gold(str(gold_path))
    print(f"  Gold: {len(entities)} entities, {len(facts)} facts")

    _, sge_nodes, sge_text = load_graph(str(sge_graph_path))
    sge_covered, _ = check_fact_coverage(facts, sge_nodes, sge_text)

    _, base_nodes, base_text = load_graph(str(base_graph_path))
    base_covered, _ = check_fact_coverage(facts, base_nodes, base_text)

    sge_rates = compute_entity_coverage_rates(facts, sge_covered)
    base_rates = compute_entity_coverage_rates(facts, base_covered)

    # Align entities: use only those present in gold facts
    all_entities = sorted(sge_rates.keys())
    sge_list = [sge_rates[e] for e in all_entities]
    base_list = [base_rates.get(e, 0.0) for e in all_entities]

    n_entities = len(all_entities)
    mean_sge = sum(sge_list) / n_entities if n_entities > 0 else 0.0
    mean_base = sum(base_list) / n_entities if n_entities > 0 else 0.0

    print(f"  Entities: {n_entities}")
    print(f"  Mean SGE coverage rate: {mean_sge:.4f}")
    print(f"  Mean Base coverage rate: {mean_base:.4f}")

    # Wilcoxon signed-rank test (one-sided: SGE > Base)
    differences = [s - b for s, b in zip(sge_list, base_list)]
    nonzero_diffs = [d for d in differences if d != 0]

    if len(nonzero_diffs) == 0:
        print("  All differences are zero — test not applicable.")
        w_stat, p_value = None, 1.0
    else:
        result = wilcoxon(sge_list, base_list, alternative="greater")
        w_stat = result.statistic
        p_value = result.pvalue
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  Wilcoxon W={w_stat:.1f}, p={p_value:.6f}  {sig}")

    return {
        "dataset": name,
        "n_entities": n_entities,
        "mean_sge_rate": round(mean_sge, 4),
        "mean_base_rate": round(mean_base, 4),
        "entity_sge_rates": {e: round(sge_rates[e], 4) for e in all_entities},
        "entity_base_rates": {e: round(base_rates.get(e, 0.0), 4) for e in all_entities},
        "wilcoxon_W": round(w_stat, 2) if w_stat is not None else None,
        "wilcoxon_p": round(p_value, 6),
    }


def main():
    print("=" * 80)
    print("ENTITY-LEVEL WILCOXON SIGNED-RANK TEST: SGE vs Baseline Coverage")
    print("=" * 80)

    all_results = []

    for name, gold_path, sge_path, base_path in DATASETS:
        result = run_wilcoxon_for_dataset(name, gold_path, sge_path, base_path)
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Dataset':<25} {'n':>4} {'Mean SGE':>10} {'Mean Base':>10} {'W':>8} {'p-value':>10} {'sig':>5}")
    print("-" * 80)
    for r in all_results:
        sig = "***" if r["wilcoxon_p"] < 0.001 else \
              "**" if r["wilcoxon_p"] < 0.01 else \
              "*" if r["wilcoxon_p"] < 0.05 else "ns"
        w_str = f"{r['wilcoxon_W']:.1f}" if r["wilcoxon_W"] is not None else "N/A"
        print(f"{r['dataset']:<25} {r['n_entities']:>4} {r['mean_sge_rate']:>10.4f} "
              f"{r['mean_base_rate']:>10.4f} {w_str:>8} {r['wilcoxon_p']:>10.6f} {sig:>5}")

    out_path = Path(__file__).parent / "entity_level_wilcoxon_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
