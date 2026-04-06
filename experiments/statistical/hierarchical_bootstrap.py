#!/usr/bin/env python3
"""
hierarchical_bootstrap.py — Hierarchical Bootstrap for FC comparison.

Standard entity-level bootstrap assumes independence between entities,
but entities from the same LLM extraction call share inference context.
This script implements a two-level hierarchical bootstrap:
  Level 1: Resample chunks (extraction units)
  Level 2: Within each chunk, include all entities

This addresses the reviewer concern about violated independence assumptions.

Usage:
    python3 experiments/statistical/hierarchical_bootstrap.py
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_coverage import load_gold, load_graph, check_fact_coverage

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DATASETS = {
    "who": {
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    "wb_cm": {
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "sge_graph": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    "wb_pop": {
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "sge_graph": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    "wb_mat": {
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "sge_graph": "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
    "inpatient": {
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "sge_graph": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
    },
}


def load_facts_by_entity(gold_path: str) -> dict[str, list]:
    """Load gold facts grouped by subject entity (= chunk cluster)."""
    clusters: dict[str, list] = {}
    with open(gold_path) as f:
        for line in f:
            record = json.loads(line.strip())
            triple = record.get("triple", {})
            subject = triple.get("subject", "").strip()
            if not subject:
                continue
            clusters.setdefault(subject, []).append(record)
    return clusters


def compute_fc_for_subset(
    facts_subset: list[dict],
    graph_nodes: dict,
    entity_text: dict,
) -> float:
    """Compute FC for a subset of gold facts."""
    if not facts_subset:
        return 0.0
    # Convert to the format check_fact_coverage expects
    formatted = []
    for record in facts_subset:
        triple = record.get("triple", {})
        formatted.append({
            "subject": triple.get("subject", ""),
            "value": triple.get("object", ""),
            "year": triple.get("attributes", {}).get("year", ""),
        })
    covered, _ = check_fact_coverage(formatted, graph_nodes, entity_text)
    return len(covered) / len(formatted)


def hierarchical_bootstrap(
    gold_path: str,
    graph_path: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Two-level hierarchical bootstrap:
      Level 1: Resample entity clusters (with replacement)
      Level 2: Include all facts from each sampled entity

    Returns: point estimate, 95% CI, bootstrap distribution stats
    """
    clusters = load_facts_by_entity(gold_path)
    entity_names = list(clusters.keys())
    n_entities = len(entity_names)

    _, graph_nodes, entity_text = load_graph(graph_path)

    # All facts for point estimate
    all_facts = [f for facts in clusters.values() for f in facts]
    point_fc = compute_fc_for_subset(all_facts, graph_nodes, entity_text)

    # Bootstrap
    rng = random.Random(seed)
    boot_fcs = []

    for _ in range(n_bootstrap):
        # Level 1: resample entities
        sampled_entities = rng.choices(entity_names, k=n_entities)
        # Level 2: collect all facts from sampled entities
        boot_facts = []
        for entity in sampled_entities:
            boot_facts.extend(clusters[entity])
        boot_fc = compute_fc_for_subset(boot_facts, graph_nodes, entity_text)
        boot_fcs.append(boot_fc)

    boot_fcs.sort()
    ci_lo = boot_fcs[int(0.025 * n_bootstrap)]
    ci_hi = boot_fcs[int(0.975 * n_bootstrap)]

    return {
        "point_estimate": round(point_fc, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "bootstrap_mean": round(sum(boot_fcs) / len(boot_fcs), 4),
        "bootstrap_std": round(
            (sum((x - sum(boot_fcs)/len(boot_fcs))**2 for x in boot_fcs) / len(boot_fcs)) ** 0.5,
            4,
        ),
        "n_entities": n_entities,
        "n_facts": len(all_facts),
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    results = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Hierarchical Bootstrap: {ds_name}")
        print(f"{'='*60}")

        ds_result = {}
        for system, graph_key in [("sge", "sge_graph"), ("baseline", "baseline_graph")]:
            graph_path = str(PROJECT_ROOT / cfg[graph_key])
            if not Path(graph_path).exists():
                print(f"  [{system}] Graph not found, skipping")
                continue

            print(f"  [{system}] Running 10,000 bootstrap iterations...")
            result = hierarchical_bootstrap(
                str(PROJECT_ROOT / cfg["gold"]),
                graph_path,
                n_bootstrap=10000,
            )
            print(f"    FC={result['point_estimate']:.4f} "
                  f"[{result['ci_95_lo']:.4f}, {result['ci_95_hi']:.4f}]")
            ds_result[system] = result

        # Compute difference CI
        if "sge" in ds_result and "baseline" in ds_result:
            diff = ds_result["sge"]["point_estimate"] - ds_result["baseline"]["point_estimate"]
            print(f"  Δ(SGE-Baseline) = {diff:+.4f}")

        results[ds_name] = ds_result

    # Summary
    print(f"\n\n{'='*70}")
    print(f"{'Dataset':<15s} {'SGE FC':>8s} {'SGE CI':>18s} {'Base FC':>8s} {'Base CI':>18s}")
    print(f"{'-'*70}")
    for ds, r in results.items():
        sge = r.get("sge", {})
        base = r.get("baseline", {})
        print(f"{ds:<15s} {sge.get('point_estimate',0):>8.4f} "
              f"[{sge.get('ci_95_lo',0):.4f}, {sge.get('ci_95_hi',0):.4f}] "
              f"{base.get('point_estimate',0):>8.4f} "
              f"[{base.get('ci_95_lo',0):.4f}, {base.get('ci_95_hi',0):.4f}]")

    # Save
    output_path = PROJECT_ROOT / "experiments" / "results" / "hierarchical_bootstrap_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "description": "Hierarchical bootstrap CI (entity-cluster resampling, n=10000)",
            "methodology": "Level 1: resample entity clusters with replacement. Level 2: include all facts from sampled entities. Addresses within-entity dependence from shared LLM extraction context.",
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
