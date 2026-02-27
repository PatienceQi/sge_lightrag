#!/usr/bin/env python3
"""
evaluate.py — SGE-LightRAG Gold Standard Evaluation Script

Computes Precision, Recall, F1 at entity and relation level,
plus graph topology metrics (isolated node ratio, average degree).

Usage:
    python3 evaluate.py --graph <graphml_path> --gold <jsonl_path>
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed. Run: pip install networkx", file=sys.stderr)
    sys.exit(1)


def load_gold(jsonl_path: str):
    """Load gold triples from a JSONL file."""
    gold_entities = set()
    gold_relations = set()

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

            if subj:
                gold_entities.add(subj)
            # Only add object as entity if it has an entity type (not a raw literal value)
            obj_type = triple.get("object_type", "")
            if obj and obj_type not in ("Literal", "BudgetAmount", "StatValue"):
                gold_entities.add(obj)

            if subj and rel and obj:
                gold_relations.add((subj, rel, obj))

    return gold_entities, gold_relations


def load_graph(graphml_path: str):
    """Load a LightRAG graphml file and extract entities and relations."""
    G = nx.read_graphml(graphml_path)

    pred_entities = set()
    pred_relations = set()

    for node_id, data in G.nodes(data=True):
        # LightRAG stores entity name in node id or 'entity_name' attribute
        name = data.get("entity_name") or data.get("name") or node_id
        if name:
            pred_entities.add(str(name).strip())

    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u
        v_name = G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v
        rel = data.get("relation_type") or data.get("relation") or data.get("keyword") or "RELATED"
        pred_relations.add((str(u_name).strip(), str(rel).strip(), str(v_name).strip()))

    return G, pred_entities, pred_relations


def fuzzy_match_entities(gold_entities: set, pred_entities: set):
    """
    Match gold entities against predicted entities.
    Uses exact match first, then substring containment as fallback.
    Returns set of matched gold entities.
    """
    matched = set()
    pred_lower = {e.lower(): e for e in pred_entities}

    for gold_e in gold_entities:
        gold_lower = gold_e.lower()
        # Exact match
        if gold_lower in pred_lower:
            matched.add(gold_e)
            continue
        # Substring match: gold entity appears inside any predicted entity or vice versa
        for pred_e_lower in pred_lower:
            if gold_lower in pred_e_lower or pred_e_lower in gold_lower:
                matched.add(gold_e)
                break

    return matched


def fuzzy_match_relations(gold_relations: set, pred_relations: set):
    """
    Match gold relations against predicted relations.
    Uses exact triple match first, then subject+object substring match.
    """
    matched = set()
    pred_list = list(pred_relations)

    for (gs, gr, go) in gold_relations:
        gs_l, gr_l, go_l = gs.lower(), gr.lower(), go.lower()
        found = False

        # Exact match
        if (gs_l, gr_l, go_l) in {(s.lower(), r.lower(), o.lower()) for s, r, o in pred_list}:
            matched.add((gs, gr, go))
            found = True

        if not found:
            # Fuzzy: subject and object both appear (substring) in some predicted triple
            for (ps, pr, po) in pred_list:
                ps_l, po_l = ps.lower(), po.lower()
                subj_match = gs_l in ps_l or ps_l in gs_l
                obj_match = go_l in po_l or po_l in go_l
                if subj_match and obj_match:
                    matched.add((gs, gr, go))
                    break

    return matched


def compute_prf(gold: set, matched: set, pred_count: int):
    """Compute Precision, Recall, F1."""
    tp = len(matched)
    fp = max(0, pred_count - tp)
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def compute_topology(G: nx.Graph):
    """Compute graph topology metrics."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "isolated_node_count": 0,
            "isolated_node_ratio": 0.0,
            "average_degree": 0.0,
        }

    isolated = list(nx.isolates(G))
    isolated_ratio = len(isolated) / num_nodes
    avg_degree = sum(d for _, d in G.degree()) / num_nodes

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "isolated_node_count": len(isolated),
        "isolated_node_ratio": round(isolated_ratio, 4),
        "average_degree": round(avg_degree, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LightRAG graph output against gold standard triples."
    )
    parser.add_argument("--graph", required=True, help="Path to LightRAG graphml file")
    parser.add_argument("--gold", required=True, help="Path to gold standard JSONL file")
    parser.add_argument("--output", default=None, help="Optional path to write JSON results")
    parser.add_argument("--fuzzy", action="store_true", default=True,
                        help="Use fuzzy (substring) matching (default: True)")
    args = parser.parse_args()

    # Validate paths
    if not Path(args.graph).exists():
        print(f"ERROR: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.gold).exists():
        print(f"ERROR: Gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gold triples from: {args.gold}")
    gold_entities, gold_relations = load_gold(args.gold)
    print(f"  Gold entities: {len(gold_entities)}, Gold relations: {len(gold_relations)}")

    print(f"Loading graph from: {args.graph}")
    G, pred_entities, pred_relations = load_graph(args.graph)
    print(f"  Predicted entities: {len(pred_entities)}, Predicted relations: {len(pred_relations)}")

    # Match
    if args.fuzzy:
        matched_entities = fuzzy_match_entities(gold_entities, pred_entities)
        matched_relations = fuzzy_match_relations(gold_relations, pred_relations)
    else:
        matched_entities = gold_entities & pred_entities
        matched_relations = gold_relations & pred_relations

    # Compute metrics
    entity_metrics = compute_prf(gold_entities, matched_entities, len(pred_entities))
    relation_metrics = compute_prf(gold_relations, matched_relations, len(pred_relations))
    topology = compute_topology(G)

    results = {
        "match_mode": "fuzzy" if args.fuzzy else "exact",
        "entity_metrics": entity_metrics,
        "relation_metrics": relation_metrics,
        "graph_topology": topology,
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\n[Entity-Level]  (match_mode={results['match_mode']})")
    print(f"  Precision : {entity_metrics['precision']:.4f}")
    print(f"  Recall    : {entity_metrics['recall']:.4f}")
    print(f"  F1        : {entity_metrics['f1']:.4f}")
    print(f"  TP={entity_metrics['true_positives']}  FP={entity_metrics['false_positives']}  FN={entity_metrics['false_negatives']}")

    print(f"\n[Relation-Level]")
    print(f"  Precision : {relation_metrics['precision']:.4f}")
    print(f"  Recall    : {relation_metrics['recall']:.4f}")
    print(f"  F1        : {relation_metrics['f1']:.4f}")
    print(f"  TP={relation_metrics['true_positives']}  FP={relation_metrics['false_positives']}  FN={relation_metrics['false_negatives']}")

    print(f"\n[Graph Topology]")
    print(f"  Nodes              : {topology['num_nodes']}")
    print(f"  Edges              : {topology['num_edges']}")
    print(f"  Isolated nodes     : {topology['isolated_node_count']}")
    print(f"  Isolated node ratio: {topology['isolated_node_ratio']:.4f}")
    print(f"  Average degree     : {topology['average_degree']:.4f}")
    print("=" * 50)

    # Write JSON output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to: {args.output}")
    else:
        print("\n[JSON Output]")
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
