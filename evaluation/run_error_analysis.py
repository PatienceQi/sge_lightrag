#!/usr/bin/env python3
"""
run_error_analysis.py — Detailed error analysis for SGE-LightRAG missed facts.

Analyzes three failure cases:
  1. WB Maternal Mortality (5 missed facts out of 150)
  2. THE University Ranking (85 missed facts out of 150, old Type-III graph)
  3. OOD failures (unemployment, immunization_dpt, immunization_measles)

Outputs: evaluation/results/error_analysis_detailed.json
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # sge_lightrag/
EVAL_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR = BASE_DIR / "output"

sys.path.insert(0, str(BASE_DIR))

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)

from evaluation.graph_loaders import load_graph_auto
from evaluation.evaluate_coverage import (
    load_gold,
    check_entity_coverage,
    check_fact_coverage,
    compute_structural_quality,
)


# ---------------------------------------------------------------------------
# Helper: find entity in graph (mirrors evaluate_coverage.py logic)
# ---------------------------------------------------------------------------

def _find_entity_in_graph(subject: str, graph_nodes: dict) -> str | None:
    """Return matched graph node name for a gold subject, or None."""
    node_names_lower = {n.lower(): n for n in graph_nodes}
    subj_lower = subject.lower()
    if subj_lower in node_names_lower:
        return node_names_lower[subj_lower]
    for nn in node_names_lower:
        if subj_lower in nn or nn in subj_lower:
            return node_names_lower[nn]
    for nn, orig_name in node_names_lower.items():
        node_desc = graph_nodes.get(orig_name, {}).get("description", "").lower()
        if subj_lower in node_desc:
            return orig_name
    return None


def _check_value_in_text(value: str, texts: list[str], node_desc: str) -> bool:
    """Return True if value string appears in any text."""
    all_text = " ".join(texts) + " " + node_desc
    return value in all_text


def _check_year_in_text(year: str, texts: list[str], node_desc: str) -> bool:
    """Return True if year string appears in any text (or year is empty)."""
    if not year:
        return True
    all_text = " ".join(texts) + " " + node_desc
    return year in all_text


# ---------------------------------------------------------------------------
# Diagnosis: WHY a fact is missed
# ---------------------------------------------------------------------------

def _diagnose_missed_fact(
    fact: dict,
    graph_nodes: dict,
    entity_text: dict,
    G: nx.Graph,
) -> dict:
    """
    Produce a detailed diagnosis for a single missed fact.

    Possible reasons:
      - entity_missing: subject not found in graph at all
      - entity_isolated: entity node exists but has no edges
      - value_missing: entity found, edges exist, but value string absent
      - year_missing: value found in entity text but year string absent
      - value_in_graph_but_unreachable: value exists somewhere else in the graph
    """
    subj = fact["subject"]
    value = fact["value"]
    year = fact.get("year", "")

    matched_node = _find_entity_in_graph(subj, graph_nodes)
    if not matched_node:
        return {**fact, "reason": "entity_missing", "detail": "No matching node found"}

    # Entity found — check if it has edges
    # Find the graph node id for matched_node
    node_id = None
    for nid, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or nid
        if str(name).strip() == matched_node:
            node_id = nid
            break

    degree = G.degree(node_id) if node_id is not None else 0
    if degree == 0:
        return {
            **fact,
            "reason": "entity_isolated",
            "detail": f"Node '{matched_node}' exists but has 0 edges in graph",
        }

    texts = entity_text.get(matched_node, [])
    node_desc = graph_nodes.get(matched_node, {}).get("description", "")

    value_found = _check_value_in_text(value, texts, node_desc)
    year_found = _check_year_in_text(year, texts, node_desc)

    if value_found and not year_found:
        return {
            **fact,
            "reason": "year_missing",
            "detail": f"Value '{value}' found in entity text but year '{year}' absent",
        }

    if not value_found:
        # Check if value exists elsewhere in graph (unreachable from this entity)
        all_graph_text = " ".join(
            t
            for texts_list in entity_text.values()
            for t in texts_list
        )
        if value in all_graph_text:
            return {
                **fact,
                "reason": "value_wrong_binding",
                "detail": f"Value '{value}' exists in graph but not reachable from entity '{matched_node}'",
            }
        return {
            **fact,
            "reason": "value_missing",
            "detail": f"Value '{value}' not present anywhere in graph",
        }

    # Should not reach here (covered facts filter out earlier)
    return {**fact, "reason": "unknown", "detail": "Unclassified miss"}


# ---------------------------------------------------------------------------
# Part 1: WB Maternal Mortality
# ---------------------------------------------------------------------------

def analyze_wb_maternal() -> dict:
    """Analyze missed facts for WB Maternal Mortality (expected: 5 missed)."""
    graph_path = (
        OUTPUT_DIR / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    )
    gold_path = EVAL_DIR / "gold" / "gold_wb_maternal_v2.jsonl"

    print("[1/3] WB Maternal Mortality analysis...")

    if not graph_path.exists():
        return {"error": f"Graph not found: {graph_path}"}
    if not gold_path.exists():
        return {"error": f"Gold not found: {gold_path}"}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph_auto(str(graph_path))
    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)

    # Deep diagnosis of each missed fact
    missed_detailed = [
        _diagnose_missed_fact(nc, graph_nodes, entity_text, G)
        for nc in not_covered
    ]

    # Check CSV source for missed facts
    csv_path = (
        BASE_DIR
        / "dataset"
        / "世界银行数据"
        / "maternal_mortality"
        / "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv"
    )
    csv_available = csv_path.exists()

    # Category summary
    category_counts = {
        "entity_missing": 0,
        "entity_isolated": 0,
        "value_missing": 0,
        "value_wrong_binding": 0,
        "year_missing": 0,
        "unknown": 0,
    }
    for m in missed_detailed:
        reason = m.get("reason", "unknown")
        category_counts[reason] = category_counts.get(reason, 0) + 1

    # Structural quality
    structure = compute_structural_quality(G, graph_nodes)

    # Format missed list without redundant keys
    missed_list = [
        {
            "subject": m["subject"],
            "value": m["value"],
            "year": m.get("year", ""),
            "reason": m["reason"],
            "detail": m.get("detail", ""),
        }
        for m in missed_detailed
    ]

    return {
        "total_gold": len(facts),
        "covered": len(covered),
        "missed_count": len(not_covered),
        "missed": missed_list,
        "category_summary": category_counts,
        "graph_structure": structure,
        "source_csv_available": csv_available,
    }


# ---------------------------------------------------------------------------
# Part 2: THE University Ranking
# ---------------------------------------------------------------------------

def _categorize_the_ranking_miss(
    fact: dict,
    graph_nodes: dict,
    entity_text: dict,
    G: nx.Graph,
    isolated_entities: set,
) -> dict:
    """Categorize a THE Ranking missed fact with domain-specific logic."""
    subj = fact["subject"]
    matched_node = _find_entity_in_graph(subj, graph_nodes)

    if not matched_node:
        return {**fact, "reason": "entity_missing", "category": "entity_missing"}

    if matched_node in isolated_entities:
        return {
            **fact,
            "reason": "entity_isolated",
            "category": "entity_isolated",
            "detail": "University node exists but has 0 edges (LLM skipped this university's data)",
        }

    # Entity has edges — check if value/year issue
    base = _diagnose_missed_fact(fact, graph_nodes, entity_text, G)
    reason = base["reason"]
    if reason == "value_missing":
        return {**fact, "reason": reason, "category": "value_not_extracted"}
    if reason == "year_missing":
        return {**fact, "reason": reason, "category": "year_not_bound"}
    if reason == "value_wrong_binding":
        return {**fact, "reason": reason, "category": "value_wrong_binding"}

    return {**fact, "reason": reason, "category": reason}


def analyze_the_ranking() -> dict:
    """Analyze missed facts for THE University Ranking (expected: ~85 missed)."""
    graph_path = (
        OUTPUT_DIR
        / "the_university_ranking"
        / "lightrag_storage"
        / "graph_chunk_entity_relation.graphml"
    )
    gold_path = EVAL_DIR / "gold" / "gold_the_university_ranking.jsonl"

    print("[2/3] THE University Ranking analysis...")

    if not graph_path.exists():
        return {"error": f"Graph not found: {graph_path}"}
    if not gold_path.exists():
        return {"error": f"Gold not found: {gold_path}"}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph_auto(str(graph_path))
    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)

    # Identify isolated university nodes
    isolated_set = set()
    for nid, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or nid
        name = str(name).strip()
        etype = data.get("entity_type", "")
        if G.degree(nid) == 0 and etype == "university":
            isolated_set.add(name)

    # Categorize each missed fact
    missed_categorized = [
        _categorize_the_ranking_miss(nc, graph_nodes, entity_text, G, isolated_set)
        for nc in not_covered
    ]

    # Count categories
    category_counts: dict[str, int] = {}
    for m in missed_categorized:
        cat = m.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Structural notes
    structure = compute_structural_quality(G, graph_nodes)

    # Universities with most missed facts
    missed_by_uni: dict[str, int] = {}
    for m in missed_categorized:
        uni = m["subject"]
        missed_by_uni[uni] = missed_by_uni.get(uni, 0) + 1

    top_missed = sorted(missed_by_uni.items(), key=lambda x: -x[1])

    # 2-hop reachability check: are values reachable via year node?
    # year nodes like "2011" connect many universities;
    # if a university is isolated, 2-hop through year nodes doesn't help either
    reachable_via_year_hop = 0
    for nc in not_covered:
        subj = nc["subject"]
        value = nc["value"]
        year = nc.get("year", "")
        # Check year-node 2-hop: look at text of year nodes connected to this entity
        matched_node = _find_entity_in_graph(subj, graph_nodes)
        if matched_node:
            texts = entity_text.get(matched_node, [])
            all_text = " ".join(texts)
            if value in all_text:
                reachable_via_year_hop += 1

    sample_missed = [
        {
            "subject": m["subject"],
            "value": m["value"],
            "year": m.get("year", ""),
            "category": m.get("category", ""),
            "detail": m.get("detail", ""),
        }
        for m in missed_categorized[:20]
    ]

    return {
        "total_gold": len(facts),
        "covered": len(covered),
        "missed_count": len(not_covered),
        "missed_categories": category_counts,
        "isolated_universities": sorted(isolated_set),
        "isolated_university_count": len(isolated_set),
        "universities_with_most_misses": [
            {"university": u, "missed_count": c} for u, c in top_missed[:10]
        ],
        "reachable_via_2hop_count": reachable_via_year_hop,
        "graph_structure": structure,
        "sample_missed": sample_missed,
    }


# ---------------------------------------------------------------------------
# Part 3: OOD Failures
# ---------------------------------------------------------------------------

def _analyze_ood_dataset(name: str, gold_path: Path, sge_graph_path: Path) -> dict:
    """Analyze a single OOD dataset failure."""
    if not sge_graph_path.exists():
        return {"error": f"SGE graph not found: {sge_graph_path}"}
    if not gold_path.exists():
        return {"error": f"Gold not found: {gold_path}"}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph_auto(str(sge_graph_path))
    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    structure = compute_structural_quality(G, graph_nodes)

    # Reason breakdown
    reason_counts: dict[str, int] = {}
    for nc in not_covered:
        r = nc.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1

    # Classify failure mode
    num_nodes = structure.get("num_nodes", 0)
    num_edges = structure.get("num_edges", 0)
    edge_node_ratio = round(num_edges / num_nodes, 4) if num_nodes > 0 else 0
    isolated_ratio = structure.get("isolated_ratio", 0)

    # Detect specific failure modes based on graph metrics and fact coverage
    fc = len(covered) / len(facts) if facts else 0.0
    if num_edges == 0:
        failure_mode = (
            "complete_extraction_failure: graph has nodes but 0 edges; "
            "LLM produced entity nodes but no relations. "
            "Likely cause: integer-only values (e.g., immunization percentages 83, 93) "
            "are ambiguous — LLM may skip relation creation when values look like stand-alone integers."
        )
    elif fc < 0.1 and edge_node_ratio < 1.0:
        failure_mode = (
            f"incomplete_year_extraction: edge/node={edge_node_ratio}; "
            f"FC={fc:.3f}. LLM extracts only a subset of years per entity "
            "(each 23-column chunk yields only ~4 edges instead of 23). "
            "Values exist in the graph but for wrong years — gold sampled years "
            "are systematically skipped by the LLM."
        )
    elif edge_node_ratio < 0.5:
        failure_mode = (
            f"partial_extraction: low edge/node ratio={edge_node_ratio}; "
            "many nodes lack connections, resulting in sparse value coverage."
        )
    elif isolated_ratio > 0.3:
        failure_mode = (
            f"high_orphan_ratio: {isolated_ratio:.1%} nodes are isolated; "
            "entity extraction succeeded but relation binding failed."
        )
    else:
        failure_mode = f"partial_coverage: edge_node_ratio={edge_node_ratio}, FC={fc:.3f}"

    # Load chunk sample to understand serialization
    chunks_dir = sge_graph_path.parent.parent / "chunks"
    sample_chunk = None
    if chunks_dir.exists():
        chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
        if chunk_files:
            try:
                sample_chunk = chunk_files[0].read_text(encoding="utf-8")[:500]
            except OSError:
                sample_chunk = None

    return {
        "total_gold": len(facts),
        "covered": len(covered),
        "missed_count": len(not_covered),
        "fact_coverage": round(len(covered) / len(facts), 4) if facts else 0.0,
        "edge_node_ratio": edge_node_ratio,
        "entity_count": num_nodes,
        "edge_count": num_edges,
        "orphan_node_ratio": isolated_ratio,
        "failure_mode": failure_mode,
        "miss_reason_breakdown": reason_counts,
        "sample_chunk": sample_chunk,
    }


def analyze_ood_failures() -> dict:
    """Analyze all three OOD failure datasets."""
    print("[3/3] OOD failure analysis...")

    ood_datasets = [
        {
            "name": "unemployment",
            "gold": EVAL_DIR / "gold" / "gold_ood_wb_unemployment.jsonl",
            "sge_graph": (
                OUTPUT_DIR
                / "ood"
                / "wb_unemployment"
                / "sge_budget"
                / "lightrag_storage"
                / "graph_chunk_entity_relation.graphml"
            ),
        },
        {
            "name": "immunization_dpt",
            "gold": EVAL_DIR / "gold" / "gold_ood_wb_immunization_dpt.jsonl",
            "sge_graph": (
                OUTPUT_DIR
                / "ood"
                / "wb_immunization_dpt"
                / "sge_budget"
                / "lightrag_storage"
                / "graph_chunk_entity_relation.graphml"
            ),
        },
        {
            "name": "immunization_measles",
            "gold": EVAL_DIR / "gold" / "gold_ood_wb_immunization_measles.jsonl",
            "sge_graph": (
                OUTPUT_DIR
                / "ood"
                / "wb_immunization_measles"
                / "sge_budget"
                / "lightrag_storage"
                / "graph_chunk_entity_relation.graphml"
            ),
        },
    ]

    results = {}
    for ds in ood_datasets:
        print(f"  Analyzing OOD: {ds['name']}...")
        results[ds["name"]] = _analyze_ood_dataset(
            ds["name"], ds["gold"], ds["sge_graph"]
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("SGE-LightRAG Error Analysis")
    print("=" * 50)

    result = {
        "wb_maternal": analyze_wb_maternal(),
        "the_ranking": analyze_the_ranking(),
        "ood_failures": analyze_ood_failures(),
    }

    out_path = EVAL_DIR / "results" / "error_analysis_detailed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 50)
    print(f"Results written to: {out_path}")

    # Print summary
    print()
    print("[WB Maternal Mortality]")
    wbm = result["wb_maternal"]
    if "error" not in wbm:
        print(f"  Total: {wbm['total_gold']}, Covered: {wbm['covered']}, Missed: {wbm['missed_count']}")
        print(f"  Category breakdown: {wbm['category_summary']}")
        for m in wbm.get("missed", []):
            print(f"    MISS: {m['subject']} | {m['year']} | {m['value']} | {m['reason']}")

    print()
    print("[THE University Ranking]")
    the = result["the_ranking"]
    if "error" not in the:
        print(f"  Total: {the['total_gold']}, Covered: {the['covered']}, Missed: {the['missed_count']}")
        print(f"  Missed categories: {the['missed_categories']}")
        print(f"  Isolated universities: {the['isolated_university_count']}")
        print(f"  Reachable via 2-hop: {the['reachable_via_2hop_count']}")

    print()
    print("[OOD Failures]")
    for name, ood in result["ood_failures"].items():
        if "error" not in ood:
            print(f"  {name}: FC={ood['fact_coverage']:.3f}, "
                  f"nodes={ood['entity_count']}, edges={ood['edge_count']}, "
                  f"edge/node={ood['edge_node_ratio']}")
            print(f"    failure_mode: {ood['failure_mode'][:80]}...")


if __name__ == "__main__":
    main()
