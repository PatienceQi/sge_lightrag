#!/usr/bin/env python3
"""
run_error_taxonomy.py — Full Error Taxonomy across all datasets and systems.

Runs error analysis on 7 primary datasets × 3 systems (SGE, Baseline, Det Parser),
producing a unified error taxonomy table.

Error categories:
  - entity_missing: subject not found in graph
  - entity_isolated: entity exists but has 0 edges
  - value_missing: entity found, edges exist, but value string absent
  - year_missing: value found but year string absent
  - value_wrong_binding: value exists in graph but not reachable from entity

Usage:
    python3 evaluation/run_error_taxonomy.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import networkx as nx

from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)

# ---------------------------------------------------------------------------
# Dataset × System registry
# ---------------------------------------------------------------------------

DATASETS = {
    "who": {
        "label": "WHO Life Expectancy",
        "type": "Type-II",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "systems": {
            "sge": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
            "baseline": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_who/graph.graphml",
        },
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "type": "Type-II",
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "systems": {
            "sge": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
            "baseline": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_wb_cm/graph.graphml",
        },
    },
    "wb_pop": {
        "label": "WB Population",
        "type": "Type-II",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "systems": {
            "sge": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
            "baseline": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_wb_pop/graph.graphml",
        },
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "type": "Type-II",
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "systems": {
            "sge": "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
            "baseline": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_wb_mat/graph.graphml",
        },
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "type": "Type-III",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "systems": {
            "sge": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
            "baseline": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_inpatient/graph.graphml",
        },
    },
    "fortune500": {
        "label": "Fortune 500 Revenue",
        "type": "Type-II",
        "gold": "evaluation/gold/gold_fortune500_revenue.jsonl",
        "systems": {
            "sge": "output/fortune500_revenue/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_fortune500/graph.graphml",
        },
    },
    "the": {
        "label": "THE University Ranking",
        "type": "Type-III",
        "gold": "evaluation/gold/gold_the_university_ranking.jsonl",
        "systems": {
            "sge": "output/the_university_ranking/lightrag_storage/graph_chunk_entity_relation.graphml",
            "det_parser": "output/det_parser_the/graph.graphml",
        },
    },
}


# ---------------------------------------------------------------------------
# Error diagnosis (adapted from run_error_analysis.py)
# ---------------------------------------------------------------------------

def _find_entity(subject: str, graph_nodes: dict) -> str | None:
    subj_lower = subject.lower()
    node_names_lower = {n.lower(): n for n in graph_nodes}
    if subj_lower in node_names_lower:
        return node_names_lower[subj_lower]
    for nn in node_names_lower:
        if subj_lower in nn or nn in subj_lower:
            return node_names_lower[nn]
    for nn, orig in node_names_lower.items():
        desc = graph_nodes.get(orig, {}).get("description", "").lower()
        if subj_lower in desc:
            return orig
    return None


def diagnose_missed_fact(fact: dict, G: nx.Graph, graph_nodes: dict, entity_text: dict) -> str:
    subj = fact["subject"]
    value = fact["value"]
    year = fact.get("year", "")

    matched = _find_entity(subj, graph_nodes)
    if not matched:
        return "entity_missing"

    # Check degree
    node_id = None
    for nid, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or nid
        if str(name).strip() == matched:
            node_id = nid
            break
    if node_id is not None and G.degree(node_id) == 0:
        return "entity_isolated"

    # Check value reachability
    texts = entity_text.get(matched, [])
    node_desc = graph_nodes.get(matched, {}).get("description", "")
    all_text = " ".join(texts) + " " + node_desc

    if value in all_text:
        if year and year not in all_text:
            return "year_missing"
        return "unknown"  # shouldn't happen for missed facts

    # Value not in entity's neighborhood — check global
    global_text = " ".join(t for tl in entity_text.values() for t in tl)
    if value in global_text:
        return "value_wrong_binding"

    return "value_missing"


def analyze_system(gold_path: str, graph_path: str) -> dict:
    """Analyze error distribution for one dataset × one system."""
    if not Path(BASE_DIR / graph_path).exists():
        return {"error": f"Graph not found: {graph_path}"}

    gold_entities, facts = load_gold(str(BASE_DIR / gold_path))
    G, graph_nodes, entity_text = load_graph(str(BASE_DIR / graph_path))

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0

    covered, missed = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0

    # Diagnose missed facts
    categories = {
        "entity_missing": 0,
        "entity_isolated": 0,
        "value_missing": 0,
        "year_missing": 0,
        "value_wrong_binding": 0,
        "unknown": 0,
    }

    for fact in missed:
        reason = diagnose_missed_fact(fact, G, graph_nodes, entity_text)
        categories[reason] = categories.get(reason, 0) + 1

    return {
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "total_facts": len(facts),
        "covered": len(covered),
        "missed": len(missed),
        "error_categories": categories,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


def main() -> None:
    results = {}

    for ds_key, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"{cfg['label']} ({cfg['type']})")
        print(f"{'='*60}")

        ds_results = {"label": cfg["label"], "type": cfg["type"], "systems": {}}

        for sys_name, graph_path in cfg["systems"].items():
            print(f"\n  [{sys_name.upper()}]")
            result = analyze_system(cfg["gold"], graph_path)
            ds_results["systems"][sys_name] = result

            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    EC={result['ec']:.4f}  FC={result['fc']:.4f}  "
                      f"missed={result['missed']}/{result['total_facts']}")
                cats = result["error_categories"]
                for cat, count in cats.items():
                    if count > 0:
                        print(f"      {cat}: {count}")

        results[ds_key] = ds_results

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"ERROR TAXONOMY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20s} {'System':<10s} {'FC':>6s} {'Miss':>5s} "
          f"{'EntMiss':>8s} {'ValMiss':>8s} {'WrgBind':>8s} {'YrMiss':>7s}")
    print(f"{'-'*80}")

    for ds_key, ds in results.items():
        for sys_name, r in ds["systems"].items():
            if "error" in r:
                continue
            cats = r["error_categories"]
            print(f"{ds['label']:<20s} {sys_name:<10s} {r['fc']:>6.3f} {r['missed']:>5d} "
                  f"{cats.get('entity_missing',0):>8d} {cats.get('value_missing',0):>8d} "
                  f"{cats.get('value_wrong_binding',0):>8d} {cats.get('year_missing',0):>7d}")

    # Save
    output_path = BASE_DIR / "evaluation" / "results" / "error_taxonomy_full.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
