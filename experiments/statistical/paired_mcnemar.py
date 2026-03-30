"""
Paired McNemar Test for Fact Coverage (FC).

For each gold standard fact, checks whether SGE covers it AND whether Baseline
covers it, constructing a 2×2 paired contingency table:

              Baseline+  Baseline-
    SGE+         a          b
    SGE-         c          d

Then runs exact McNemar test: H0: b == c (marginal homogeneity).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from evaluation.evaluate_coverage import load_gold, load_graph, check_fact_coverage

# Dataset configurations: (name, gold_file, sge_graph, baseline_graph)
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
    ("Inpatient 2023",
     EVAL_DIR / "gold_inpatient_2023.jsonl",
     OUT_DIR / "inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml"),
    ("Health Stats",
     EVAL_DIR / "gold_health.jsonl",
     OUT_DIR / "sge_health/lightrag_storage/graph_chunk_entity_relation.graphml",
     OUT_DIR / "baseline_health/lightrag_storage/graph_chunk_entity_relation.graphml"),
]


def fact_key(fact):
    """Create a unique key for a fact."""
    return (fact["subject"], fact["value"], fact.get("year", ""))


def mcnemar_exact(b, c):
    """Exact McNemar test using binomial distribution."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided p-value
    k = min(b, c)
    p = 2 * binom.cdf(k, n, 0.5)
    return min(p, 1.0)


def mcnemar_chi2(b, c):
    """McNemar chi-square (without continuity correction)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return chi2, p


def main():
    print("=" * 90)
    print("PAIRED McNEMAR TEST: Fact-Level Coverage (SGE vs Baseline)")
    print("=" * 90)

    all_results = []

    for name, gold_path, sge_graph_path, base_graph_path in DATASETS:
        print(f"\n--- {name} ---")

        if not gold_path.exists():
            print(f"  SKIP: Gold file not found: {gold_path}")
            continue
        if not sge_graph_path.exists():
            print(f"  SKIP: SGE graph not found: {sge_graph_path}")
            continue
        if not base_graph_path.exists():
            print(f"  SKIP: Baseline graph not found: {base_graph_path}")
            continue

        # Load gold standard
        entities, facts = load_gold(str(gold_path))
        print(f"  Gold: {len(entities)} entities, {len(facts)} facts")

        # Evaluate SGE
        _, sge_nodes, sge_text = load_graph(str(sge_graph_path))
        sge_covered, sge_not_covered = check_fact_coverage(facts, sge_nodes, sge_text)
        sge_covered_keys = {fact_key(f) for f in sge_covered}

        # Evaluate Baseline
        _, base_nodes, base_text = load_graph(str(base_graph_path))
        base_covered, base_not_covered = check_fact_coverage(facts, base_nodes, base_text)
        base_covered_keys = {fact_key(f) for f in base_covered}

        # Build paired 2×2 table
        a = b = c = d = 0
        for fact in facts:
            fk = fact_key(fact)
            sge_ok = fk in sge_covered_keys
            base_ok = fk in base_covered_keys
            if sge_ok and base_ok:
                a += 1
            elif sge_ok and not base_ok:
                b += 1
            elif not sge_ok and base_ok:
                c += 1
            else:
                d += 1

        total = a + b + c + d
        sge_fc = (a + b) / total if total > 0 else 0
        base_fc = (a + c) / total if total > 0 else 0

        print(f"  SGE FC: {a+b}/{total} = {sge_fc:.3f}")
        print(f"  Base FC: {a+c}/{total} = {base_fc:.3f}")
        print(f"  Paired table: a={a} (both+), b={b} (SGE+ Base-), c={c} (SGE- Base+), d={d} (both-)")

        # McNemar tests
        if b + c > 0:
            chi2, chi2_p = mcnemar_chi2(b, c)
            exact_p = mcnemar_exact(b, c)
            print(f"  McNemar chi2={chi2:.2f} (uncorrected), p={chi2_p:.6f}")
            print(f"  McNemar exact p={exact_p:.6f}")
            print(f"  Discordant ratio b/c = {b}/{c} = {'inf' if c == 0 else f'{b/c:.1f}'}")
        else:
            chi2, chi2_p, exact_p = 0, 1.0, 1.0
            print(f"  No discordant pairs (b=c=0), test not applicable")

        result = {
            "dataset": name,
            "n_facts": total,
            "sge_fc": round(sge_fc, 4),
            "base_fc": round(base_fc, 4),
            "a_both_pos": a, "b_sge_only": b,
            "c_base_only": c, "d_both_neg": d,
            "mcnemar_chi2": round(chi2, 2),
            "mcnemar_chi2_p": round(chi2_p, 6),
            "mcnemar_exact_p": round(exact_p, 6),
        }
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (for paper)")
    print("=" * 90)
    print(f"{'Dataset':<25} {'n':>4} {'SGE FC':>8} {'Base FC':>8} "
          f"{'b(SGE+)':>8} {'c(Base+)':>8} {'exact p':>10} {'sig?':>5}")
    print("-" * 90)
    for r in all_results:
        sig = "***" if r["mcnemar_exact_p"] < 0.001 else \
              "**" if r["mcnemar_exact_p"] < 0.01 else \
              "*" if r["mcnemar_exact_p"] < 0.05 else "ns"
        print(f"{r['dataset']:<25} {r['n_facts']:>4} {r['sge_fc']:>8.3f} {r['base_fc']:>8.3f} "
              f"{r['b_sge_only']:>8} {r['c_base_only']:>8} {r['mcnemar_exact_p']:>10.6f} {sig:>5}")

    # Save results
    out_path = Path(__file__).parent / "paired_mcnemar_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
