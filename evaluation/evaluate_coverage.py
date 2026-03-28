#!/usr/bin/env python3
"""
evaluate_coverage.py — Information Coverage Evaluation for SGE-LightRAG

Unlike triple-matching evaluation, this script checks whether the *information*
in gold standard triples is captured somewhere in the graph — either as node
attributes, edge keywords, or edge descriptions.

This addresses the structural mismatch between LightRAG's graph format
(Entity → Year edges with budget values in keywords) and gold standard format
(Entity → Value edges with year as attribute).

Metrics:
  - Entity Coverage: what fraction of gold entities appear as graph nodes
  - Fact Coverage: what fraction of gold (subject, value, year) facts are
    recoverable from the graph's edges (keywords + descriptions)
  - Structural Precision: what fraction of graph nodes are meaningful
    (not noise like isolated year nodes or numeric literals)

Usage:
    python3 evaluate_coverage.py --graph <graphml> --gold <jsonl>
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)


def load_gold(jsonl_path: str):
    """Load gold triples and extract facts as (subject, value, year) tuples."""
    entities = set()
    facts = []  # list of dicts with subject, value, year, relation

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triple = record.get("triple", {})
            subj = triple.get("subject", "").strip()
            obj = triple.get("object", "").strip()
            rel = triple.get("relation", "").strip()
            attrs = triple.get("attributes", {})
            year = attrs.get("year", "")

            if subj:
                entities.add(subj)

            # Only count value-bearing relations as facts
            obj_type = triple.get("object_type", "")
            if obj_type in ("BudgetAmount", "StatValue", "Literal"):
                facts.append({
                    "subject": subj,
                    "value": obj,
                    "year": year,
                    "relation": rel,
                })

    return entities, facts


def load_graph(graphml_path: str):
    """Load graph and build a searchable text index per entity."""
    G = nx.read_graphml(graphml_path)

    nodes = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        name = str(name).strip()
        desc = str(data.get("description", ""))
        nodes[name] = {
            "type": data.get("entity_type", ""),
            "description": desc,
        }

    # Build per-entity text corpus from all connected edges
    entity_text = {}  # entity_name -> concatenated text from edges
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u
        v_name = G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v
        u_name = str(u_name).strip()
        v_name = str(v_name).strip()

        kw = str(data.get("keywords", ""))
        desc = str(data.get("description", ""))
        edge_text = f"{kw} {desc} {v_name}"

        entity_text.setdefault(u_name, []).append(edge_text)
        # Also index reverse direction
        rev_text = f"{kw} {desc} {u_name}"
        entity_text.setdefault(v_name, []).append(rev_text)

    # Build node_id -> name mapping for 2-hop expansion
    node_id_to_name = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        node_id_to_name[node_id] = str(name).strip()

    # 2-hop expansion: for each entity, also include texts from immediate neighbors.
    # This is needed for hierarchical graphs where values are stored on sub-item nodes
    # (e.g., Disease → ICD_Code_SubItem → numeric_value via HAS_VALUE/HAS_SUB_ITEM).
    entity_text_2hop = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        # Add neighbor texts and descriptions (1-hop)
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, nb_id)
            texts.extend(entity_text.get(nb_name, []))
            # Also include neighbor node description for value propagation
            nb_desc = nodes.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
        entity_text_2hop[name] = texts

    return G, nodes, entity_text_2hop


def check_entity_coverage(gold_entities: set, graph_nodes: dict):
    """Check how many gold entities appear in graph nodes (fuzzy)."""
    matched = set()
    node_names_lower = {n.lower(): n for n in graph_nodes}

    for ge in gold_entities:
        ge_lower = ge.lower()
        # Exact
        if ge_lower in node_names_lower:
            matched.add(ge)
            continue
        # Substring on node name
        found = False
        for nn in node_names_lower:
            if ge_lower in nn or nn in ge_lower:
                matched.add(ge)
                found = True
                break
        if found:
            continue
        # Fallback: check if gold entity code appears in any node description
        # (handles country codes like CHN appearing in "China ... code CHN")
        for nn, orig_name in node_names_lower.items():
            node_desc = graph_nodes.get(orig_name, {}).get("description", "").lower()
            if ge_lower in node_desc:
                matched.add(ge)
                break

    return matched


def check_fact_coverage(facts: list, graph_nodes: dict, entity_text: dict):
    """
    Check how many gold facts are recoverable from the graph.

    A fact (subject, value, year) is "covered" if:
    1. The subject entity exists in the graph (fuzzy match), AND
    2. The value appears somewhere in the entity's connected edge text, AND
    3. (If year is specified) the year also appears in the same edge text
    """
    covered = []
    not_covered = []

    node_names_lower = {n.lower(): n for n in graph_nodes}

    for fact in facts:
        subj = fact["subject"]
        value = fact["value"]
        year = fact.get("year", "")

        # Find matching node
        matched_node = None
        subj_lower = subj.lower()
        if subj_lower in node_names_lower:
            matched_node = node_names_lower[subj_lower]
        else:
            for nn in node_names_lower:
                if subj_lower in nn or nn in subj_lower:
                    matched_node = node_names_lower[nn]
                    break
        # Fallback: check if subject code appears in any node description
        if not matched_node:
            for nn, orig_name in node_names_lower.items():
                node_desc = graph_nodes.get(orig_name, {}).get("description", "").lower()
                if subj_lower in node_desc:
                    matched_node = orig_name
                    break

        if not matched_node:
            not_covered.append({**fact, "reason": "entity_not_found"})
            continue

        # Search for value in entity's edge text
        texts = entity_text.get(matched_node, [])
        # Also check node description
        node_desc = graph_nodes.get(matched_node, {}).get("description", "")
        all_text = " ".join(texts) + " " + node_desc

        value_found = value in all_text
        year_found = (not year) or (year in all_text)

        if value_found and year_found:
            covered.append(fact)
        elif value_found and not year_found:
            not_covered.append({**fact, "reason": "year_not_found"})
        else:
            not_covered.append({**fact, "reason": "value_not_found"})

    return covered, not_covered


def compute_structural_quality(G: nx.Graph, graph_nodes: dict):
    """Compute structural quality metrics."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"num_nodes": 0, "num_edges": 0, "typed_ratio": 0, "avg_degree": 0}

    # Count nodes with meaningful types (not UNKNOWN)
    typed = sum(1 for n in graph_nodes.values()
                if n["type"] and n["type"].upper() != "UNKNOWN")
    typed_ratio = typed / num_nodes

    isolated = len(list(nx.isolates(G)))
    avg_degree = sum(d for _, d in G.degree()) / num_nodes

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "typed_nodes": typed,
        "typed_ratio": round(typed_ratio, 4),
        "isolated_nodes": isolated,
        "isolated_ratio": round(isolated / num_nodes, 4) if num_nodes else 0,
        "avg_degree": round(avg_degree, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Information coverage evaluation for SGE-LightRAG graphs."
    )
    parser.add_argument("--graph", required=True, help="Path to graphml file")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print uncovered facts")
    args = parser.parse_args()

    if not Path(args.graph).exists():
        print(f"ERROR: {args.graph} not found", file=sys.stderr)
        sys.exit(1)
    if not Path(args.gold).exists():
        print(f"ERROR: {args.gold} not found", file=sys.stderr)
        sys.exit(1)

    # Load
    gold_entities, facts = load_gold(args.gold)
    G, graph_nodes, entity_text = load_graph(args.graph)

    print(f"Gold: {len(gold_entities)} entities, {len(facts)} value-facts")
    print(f"Graph: {len(graph_nodes)} nodes, {G.number_of_edges()} edges")

    # Entity coverage
    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ent_coverage = len(matched_entities) / len(gold_entities) if gold_entities else 0

    # Fact coverage
    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fact_coverage = len(covered) / len(facts) if facts else 0

    # Structural quality
    structure = compute_structural_quality(G, graph_nodes)

    # Print
    print()
    print("=" * 55)
    print("INFORMATION COVERAGE EVALUATION")
    print("=" * 55)
    print(f"\n[Entity Coverage]")
    print(f"  Matched  : {len(matched_entities)} / {len(gold_entities)}")
    print(f"  Coverage : {ent_coverage:.4f}")

    print(f"\n[Fact Coverage]  (subject + value + year)")
    print(f"  Covered  : {len(covered)} / {len(facts)}")
    print(f"  Coverage : {fact_coverage:.4f}")

    if not_covered:
        reasons = {}
        for nc in not_covered:
            r = nc.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"  Uncovered breakdown:")
        for r, c in sorted(reasons.items()):
            print(f"    {r}: {c}")

    print(f"\n[Structural Quality]")
    for k, v in structure.items():
        print(f"  {k:20s}: {v}")
    print("=" * 55)

    if args.verbose and not_covered:
        print("\n[Uncovered Facts]")
        for nc in not_covered:
            print(f"  {nc['subject']} | {nc.get('year','')} | "
                  f"{nc['value']} | reason={nc['reason']}")

    results = {
        "entity_coverage": {
            "matched": len(matched_entities),
            "total": len(gold_entities),
            "coverage": round(ent_coverage, 4),
        },
        "fact_coverage": {
            "covered": len(covered),
            "total": len(facts),
            "coverage": round(fact_coverage, 4),
        },
        "structural_quality": structure,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to: {args.output}")
    else:
        print("\n[JSON]")
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
