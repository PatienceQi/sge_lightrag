#!/usr/bin/env python3
"""
evaluate.py — SGE-LightRAG Gold Standard Evaluation Script

Computes Precision, Recall, F1 at entity and relation level,
plus graph topology metrics (isolated node ratio, average degree).

Relation matching supports three levels:
  - strict  : subject + relation_type + object all match
  - relaxed : subject + object match (relation type ignored)

Usage:
    python3 evaluate.py --graph <graphml_path> --gold <jsonl_path> [--relation-map <json_path>]
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


# ---------------------------------------------------------------------------
# Default keyword → gold relation-type mapping
# Keys are lowercase substrings to search for in LightRAG keyword strings.
# Values are the canonical gold relation type names.
# ---------------------------------------------------------------------------
DEFAULT_RELATION_TYPE_MAP: dict[str, str] = {
    # HAS_BUDGET
    "budget": "HAS_BUDGET",
    "预算": "HAS_BUDGET",
    # HAS_VALUE
    "value": "HAS_VALUE",
    "数值": "HAS_VALUE",
    "amount": "HAS_VALUE",
    "金额": "HAS_VALUE",
    # HAS_SUB_ITEM
    "sub": "HAS_SUB_ITEM",
    "子": "HAS_SUB_ITEM",
    "包含": "HAS_SUB_ITEM",
    "item": "HAS_SUB_ITEM",
    "分项": "HAS_SUB_ITEM",
}


def load_relation_type_map(json_path: str | None) -> dict[str, str]:
    """Load relation type map from JSON file, merging with defaults."""
    mapping = dict(DEFAULT_RELATION_TYPE_MAP)
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            custom = json.load(f)
        mapping.update({k.lower(): v for k, v in custom.items()})
    return mapping


def normalize_relation_type(keyword_str: str, rel_map: dict[str, str]) -> str | None:
    """
    Map a LightRAG keyword string to a gold relation type.
    Returns the first matching gold type, or None if no match.
    """
    kw_lower = keyword_str.lower()
    for keyword, rel_type in rel_map.items():
        if keyword in kw_lower:
            return rel_type
    return None


def load_gold(jsonl_path: str):
    """Load gold triples from a JSONL file."""
    gold_entities: set[str] = set()
    gold_relations: set[tuple[str, str, str]] = set()

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

    pred_entities: set[str] = set()
    pred_relations: set[tuple[str, str, str]] = set()

    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        if name:
            pred_entities.add(str(name).strip())

    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u
        v_name = G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v
        rel = data.get("relation_type") or data.get("relation") or data.get("keyword") or "RELATED"
        pred_relations.add((str(u_name).strip(), str(rel).strip(), str(v_name).strip()))

    return G, pred_entities, pred_relations


def fuzzy_match_entities(gold_entities: set, pred_entities: set) -> set:
    """
    Match gold entities against predicted entities.
    Uses exact match first, then substring containment as fallback.
    """
    matched: set[str] = set()
    pred_lower = {e.lower(): e for e in pred_entities}

    for gold_e in gold_entities:
        gold_lower = gold_e.lower()
        if gold_lower in pred_lower:
            matched.add(gold_e)
            continue
        for pred_e_lower in pred_lower:
            if gold_lower in pred_e_lower or pred_e_lower in gold_lower:
                matched.add(gold_e)
                break

    return matched


def fuzzy_match_relations(
    gold_relations: set,
    pred_relations: set,
    rel_map: dict[str, str],
) -> tuple[set, set]:
    """
    Match gold relations against predicted relations.

    Returns:
        (strict_matched, relaxed_matched)
        - strict_matched  : gold triples where subject, normalized-relation-type, AND object all match
        - relaxed_matched : gold triples where subject AND object match (relation type ignored)

    Matching logic per gold triple (gs, gr, go):
      1. Exact triple match (all three lowercased).
      2. Fuzzy subject+object substring match; if also relation-type matches → strict, else → relaxed only.
    """
    strict_matched: set[tuple] = set()
    relaxed_matched: set[tuple] = set()
    pred_list = list(pred_relations)

    # Pre-build lowercased pred set for fast exact lookup
    pred_lower_set = {(s.lower(), r.lower(), o.lower()) for s, r, o in pred_list}

    for (gs, gr, go) in gold_relations:
        gs_l, gr_l, go_l = gs.lower(), gr.lower(), go.lower()

        # 1. Exact triple match
        if (gs_l, gr_l, go_l) in pred_lower_set:
            strict_matched.add((gs, gr, go))
            relaxed_matched.add((gs, gr, go))
            continue

        # 2. Fuzzy subject+object match
        for (ps, pr, po) in pred_list:
            ps_l, po_l = ps.lower(), po.lower()
            subj_match = gs_l in ps_l or ps_l in gs_l
            obj_match = go_l in po_l or po_l in go_l
            if not (subj_match and obj_match):
                continue

            # Subject+object matched → relaxed hit
            relaxed_matched.add((gs, gr, go))

            # Check if relation type also matches
            normalized = normalize_relation_type(pr, rel_map)
            if normalized and normalized.lower() == gr_l:
                strict_matched.add((gs, gr, go))
            break

    return strict_matched, relaxed_matched


def compute_prf(gold: set, matched: set, pred_count: int) -> dict:
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


def compute_topology(G: nx.Graph) -> dict:
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
    parser.add_argument(
        "--relation-map",
        default=None,
        metavar="JSON_PATH",
        help=(
            "Path to a JSON file mapping lowercase keyword substrings to gold relation types. "
            "Merged with built-in defaults. "
            "Example: {\"cost\": \"HAS_BUDGET\", \"费用\": \"HAS_BUDGET\"}"
        ),
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.graph).exists():
        print(f"ERROR: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.gold).exists():
        print(f"ERROR: Gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)
    if args.relation_map and not Path(args.relation_map).exists():
        print(f"ERROR: Relation map file not found: {args.relation_map}", file=sys.stderr)
        sys.exit(1)

    rel_map = load_relation_type_map(args.relation_map)

    print(f"Loading gold triples from: {args.gold}")
    gold_entities, gold_relations = load_gold(args.gold)
    print(f"  Gold entities: {len(gold_entities)}, Gold relations: {len(gold_relations)}")

    print(f"Loading graph from: {args.graph}")
    G, pred_entities, pred_relations = load_graph(args.graph)
    print(f"  Predicted entities: {len(pred_entities)}, Predicted relations: {len(pred_relations)}")

    # Match entities
    if args.fuzzy:
        matched_entities = fuzzy_match_entities(gold_entities, pred_entities)
    else:
        matched_entities = gold_entities & pred_entities

    # Match relations (always uses rel_map for type-aware matching)
    if args.fuzzy:
        strict_matched, relaxed_matched = fuzzy_match_relations(gold_relations, pred_relations, rel_map)
    else:
        # Exact mode: strict = exact triple, relaxed = subject+object exact
        strict_matched = gold_relations & pred_relations
        relaxed_matched = {
            (gs, gr, go)
            for (gs, gr, go) in gold_relations
            if any(ps == gs and po == go for ps, _, po in pred_relations)
        }

    # Compute metrics
    entity_metrics = compute_prf(gold_entities, matched_entities, len(pred_entities))
    relation_strict_metrics = compute_prf(gold_relations, strict_matched, len(pred_relations))
    relation_relaxed_metrics = compute_prf(gold_relations, relaxed_matched, len(pred_relations))
    topology = compute_topology(G)

    results = {
        "match_mode": "fuzzy" if args.fuzzy else "exact",
        "relation_map_file": args.relation_map,
        "entity_metrics": entity_metrics,
        "relation_metrics_strict": relation_strict_metrics,
        "relation_metrics_relaxed": relation_relaxed_metrics,
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

    print(f"\n[Relation-Level — Strict  (subj + rel_type + obj match)]")
    print(f"  Precision : {relation_strict_metrics['precision']:.4f}")
    print(f"  Recall    : {relation_strict_metrics['recall']:.4f}")
    print(f"  F1        : {relation_strict_metrics['f1']:.4f}")
    print(f"  TP={relation_strict_metrics['true_positives']}  FP={relation_strict_metrics['false_positives']}  FN={relation_strict_metrics['false_negatives']}")

    print(f"\n[Relation-Level — Relaxed (subj + obj match only)]")
    print(f"  Precision : {relation_relaxed_metrics['precision']:.4f}")
    print(f"  Recall    : {relation_relaxed_metrics['recall']:.4f}")
    print(f"  F1        : {relation_relaxed_metrics['f1']:.4f}")
    print(f"  TP={relation_relaxed_metrics['true_positives']}  FP={relation_relaxed_metrics['false_positives']}  FN={relation_relaxed_metrics['false_negatives']}")

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
