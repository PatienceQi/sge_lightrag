#!/usr/bin/env python3
"""
evaluate_coverage_debiased.py — Value-First De-biased Coverage Evaluation

Removes entity-naming bias from evaluate_coverage.py by reversing the search
direction: instead of finding the subject node first (inflated by SGE's Schema
name standardization), this protocol searches the entire graph for the value
string, then verifies subject + year in the 1-hop local context.

Protocol:
  1. Search ALL graph text (nodes + edges) for the value string (case-sensitive)
  2. For each hit, collect 1-hop context text
  3. Check that subject (case-insensitive) + year appear in that context
  4. Fact is covered if ANY location passes all checks

Usage:
    python3 evaluate_coverage_debiased.py --graph <graphml> --gold <jsonl> \\
        [--output json_path] [--verbose]

    python3 evaluate_coverage_debiased.py --batch [--output json_path]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Gold loading (same logic as evaluate_coverage.py)
# ---------------------------------------------------------------------------

def load_gold(jsonl_path: str):
    """Return (entities set, facts list) from a gold JSONL file."""
    entities, facts = set(), []
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
            year = triple.get("attributes", {}).get("year", "")
            if subj:
                entities.add(subj)
            if triple.get("object_type", "") in ("BudgetAmount", "StatValue", "Literal"):
                facts.append({"subject": subj, "value": obj, "year": year, "relation": rel})
    return entities, facts


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(graphml_path: str):
    """Return (G, nodes_dict).  nodes_dict: name -> {type, description}."""
    G = nx.read_graphml(graphml_path)
    nodes = {}
    for nid, data in G.nodes(data=True):
        name = str(data.get("entity_name") or data.get("name") or nid).strip()
        nodes[name] = {"type": data.get("entity_type", ""),
                       "description": str(data.get("description", ""))}
    return G, nodes


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _node_name(G, nid: str) -> str:
    d = G.nodes[nid]
    return str(d.get("entity_name") or d.get("name") or nid).strip()


def _node_text(G, nid: str) -> str:
    d = G.nodes[nid]
    return f"{_node_name(G, nid)} {d.get('description', '')}"


def _edge_text(G, u: str, v: str) -> str:
    d = G.edges[u, v]
    return f"{d.get('keywords', '')} {d.get('description', '')}"


def _safe_edge_text(G, u: str, v: str) -> str:
    """Return edge text regardless of direction."""
    if G.has_edge(u, v):
        return _edge_text(G, u, v)
    if G.has_edge(v, u):
        return _edge_text(G, v, u)
    return ""


# ---------------------------------------------------------------------------
# Full-text index and 1-hop context
# ---------------------------------------------------------------------------

def build_full_text_index(G):
    """Build a searchable list of all text locations in the graph.

    Each entry: {location_type, location_id, text}
    - location_type: 'node' or 'edge'
    - location_id: node_id string, or (u, v) tuple
    - text: name+desc for nodes; keywords+desc for edges
    """
    index = []
    for nid, data in G.nodes(data=True):
        name = str(data.get("entity_name") or data.get("name") or nid).strip()
        index.append({
            "location_type": "node",
            "location_id": nid,
            "text": f"{name} {data.get('description', '')}",
        })
    for u, v, data in G.edges(data=True):
        index.append({
            "location_type": "edge",
            "location_id": (u, v),
            "text": f"{data.get('keywords', '')} {data.get('description', '')}",
        })
    return index


def get_node_context(G, node_id: str) -> str:
    """1-hop context for a node: own text + connected edges + neighbor texts."""
    parts = [_node_text(G, node_id)]
    seen_neighbors = set()

    for nb_id in G.neighbors(node_id):
        parts.append(_safe_edge_text(G, node_id, nb_id))
        parts.append(_node_text(G, nb_id))
        seen_neighbors.add(nb_id)

    if G.is_directed():
        for pred_id in G.predecessors(node_id):
            if pred_id not in seen_neighbors:
                parts.append(_safe_edge_text(G, pred_id, node_id))
                parts.append(_node_text(G, pred_id))

    return " ".join(parts)


def get_edge_context(G, u: str, v: str) -> str:
    """Context for an edge: own text + endpoint texts + their other edges."""
    parts = [_edge_text(G, u, v), _node_text(G, u), _node_text(G, v)]
    for nb_id in G.neighbors(u):
        if nb_id != v:
            parts.append(_safe_edge_text(G, u, nb_id))
    for nb_id in G.neighbors(v):
        if nb_id != u:
            parts.append(_safe_edge_text(G, v, nb_id))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# De-biased (value-first) fact coverage
# ---------------------------------------------------------------------------

def check_fact_coverage_debiased(facts: list, G, text_index: list):
    """Value-first coverage: search graph for value, then verify subject+year.

    Failure reasons:
    - value_not_in_graph: value not found anywhere in graph text
    - subject_not_in_context: value found but no context contains subject
    - year_not_in_context: value+subject found but year missing from context
    """
    covered, not_covered = [], []

    for fact in facts:
        value = fact["value"]
        year = fact.get("year", "")
        subj_lower = fact["subject"].lower()

        hits = [e for e in text_index if value in e["text"]]
        if not hits:
            not_covered.append({**fact, "reason": "value_not_in_graph"})
            continue

        subject_seen = False
        fact_covered = False
        for entry in hits:
            if entry["location_type"] == "node":
                ctx = get_node_context(G, entry["location_id"])
            else:
                u, v = entry["location_id"]
                ctx = get_edge_context(G, u, v)

            if subj_lower not in ctx.lower():
                continue
            subject_seen = True
            if (not year) or (year in ctx):
                fact_covered = True
                break

        if fact_covered:
            covered.append(fact)
        elif not subject_seen:
            not_covered.append({**fact, "reason": "subject_not_in_context"})
        else:
            not_covered.append({**fact, "reason": "year_not_in_context"})

    return covered, not_covered


# ---------------------------------------------------------------------------
# Entity-first protocol (original, kept for direct comparison)
# ---------------------------------------------------------------------------

def check_entity_coverage(gold_entities: set, graph_nodes: dict):
    """Fuzzy entity coverage: fraction of gold entities found in graph nodes."""
    matched = set()
    nn_lower = {n.lower(): n for n in graph_nodes}
    for ge in gold_entities:
        gel = ge.lower()
        if gel in nn_lower:
            matched.add(ge); continue
        for nn in nn_lower:
            if gel in nn or nn in gel:
                matched.add(ge); break
        else:
            for nn, orig in nn_lower.items():
                if gel in graph_nodes.get(orig, {}).get("description", "").lower():
                    matched.add(ge); break
    return matched


def _build_entity_text_2hop(G, graph_nodes: dict) -> dict:
    """Per-entity 2-hop text corpus (mirrors evaluate_coverage.py logic)."""
    nid2name = {
        nid: str(G.nodes[nid].get("entity_name") or G.nodes[nid].get("name") or nid).strip()
        for nid in G.nodes()
    }
    entity_text = {}
    for u, v, data in G.edges(data=True):
        kw, desc = str(data.get("keywords", "")), str(data.get("description", ""))
        entity_text.setdefault(nid2name[u], []).append(f"{kw} {desc} {nid2name[v]}")
        entity_text.setdefault(nid2name[v], []).append(f"{kw} {desc} {nid2name[u]}")

    result = {}
    for nid in G.nodes():
        name = nid2name[nid]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(nid):
            nb_name = nid2name[nb_id]
            texts.extend(entity_text.get(nb_name, []))
            nb_desc = graph_nodes.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
        result[name] = texts
    return result


def check_fact_coverage_entity_first(facts: list, graph_nodes: dict,
                                     entity_text_2hop: dict):
    """Original entity-first protocol (same logic as evaluate_coverage.py)."""
    covered, not_covered = [], []
    nn_lower = {n.lower(): n for n in graph_nodes}

    for fact in facts:
        subj_lower = fact["subject"].lower()
        value, year = fact["value"], fact.get("year", "")

        node = nn_lower.get(subj_lower)
        if not node:
            for nn in nn_lower:
                if subj_lower in nn or nn in subj_lower:
                    node = nn_lower[nn]; break
        if not node:
            for nn, orig in nn_lower.items():
                if subj_lower in graph_nodes.get(orig, {}).get("description", "").lower():
                    node = orig; break

        if not node:
            not_covered.append({**fact, "reason": "entity_not_found"}); continue

        all_text = (" ".join(entity_text_2hop.get(node, []))
                    + " " + graph_nodes.get(node, {}).get("description", ""))
        value_found = value in all_text
        year_found = (not year) or (year in all_text)

        if value_found and year_found:
            covered.append(fact)
        elif value_found:
            not_covered.append({**fact, "reason": "year_not_found"})
        else:
            not_covered.append({**fact, "reason": "value_not_found"})

    return covered, not_covered


# ---------------------------------------------------------------------------
# Single-dataset evaluation
# ---------------------------------------------------------------------------

def _breakdown(not_covered_list: list) -> dict:
    counts = {}
    for nc in not_covered_list:
        r = nc.get("reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


def evaluate_single(graph_path: str, gold_path: str, verbose: bool = False) -> dict:
    """Run both protocols on one (graph, gold) pair; return result dict."""
    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes = load_graph(graph_path)

    print(f"Gold: {len(gold_entities)} entities, {len(facts)} value-facts")
    print(f"Graph: {len(graph_nodes)} nodes, {G.number_of_edges()} edges")

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ent_cov = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    entity_text_2hop = _build_entity_text_2hop(G, graph_nodes)
    ef_covered, ef_not = check_fact_coverage_entity_first(facts, graph_nodes, entity_text_2hop)
    ef_fc = len(ef_covered) / len(facts) if facts else 0.0

    text_index = build_full_text_index(G)
    vf_covered, vf_not = check_fact_coverage_debiased(facts, G, text_index)
    vf_fc = len(vf_covered) / len(facts) if facts else 0.0

    ef_ids = {(f["subject"], f["value"], f["year"]) for f in ef_covered}
    vf_ids = {(f["subject"], f["value"], f["year"]) for f in vf_covered}
    rescued, lost = vf_ids - ef_ids, ef_ids - vf_ids

    print()
    print("=" * 60)
    print("VALUE-FIRST DE-BIASED COVERAGE EVALUATION")
    print("=" * 60)
    print(f"\n[Entity Coverage]  {len(matched_entities)}/{len(gold_entities)}  = {ent_cov:.4f}")
    print(f"\n[Constrained FC — entity-first]")
    print(f"  Covered: {len(ef_covered)}/{len(facts)}  FC={ef_fc:.4f}")
    for r, c in sorted(_breakdown(ef_not).items()):
        print(f"    {r}: {c}")
    print(f"\n[De-biased FC — value-first]")
    print(f"  Covered: {len(vf_covered)}/{len(facts)}  FC={vf_fc:.4f}")
    for r, c in sorted(_breakdown(vf_not).items()):
        print(f"    {r}: {c}")
    print(f"\n[Delta]  rescued={len(rescued)}  lost={len(lost)}")
    print("=" * 60)

    if verbose:
        if rescued:
            print("\n[Rescued by de-biased]")
            for s, v, y in sorted(rescued):
                print(f"  {s} | {y} | {v}")
        if lost:
            print("\n[Lost in de-biased]")
            for s, v, y in sorted(lost):
                print(f"  {s} | {y} | {v}")
        if vf_not:
            print("\n[De-biased: Uncovered Facts]")
            for nc in vf_not:
                print(f"  {nc['subject']} | {nc.get('year','')} | "
                      f"{nc['value']} | {nc['reason']}")

    return {
        "entity_coverage": {"matched": len(matched_entities),
                            "total": len(gold_entities),
                            "coverage": round(ent_cov, 4)},
        "entity_first_fc": {"covered": len(ef_covered),
                            "total": len(facts),
                            "coverage": round(ef_fc, 4),
                            "failure_breakdown": _breakdown(ef_not)},
        "debiased_fc": {"covered": len(vf_covered),
                        "total": len(facts),
                        "coverage": round(vf_fc, 4),
                        "failure_breakdown": _breakdown(vf_not)},
        "delta": {"rescued_by_debiased": len(rescued),
                  "lost_by_debiased": len(lost)},
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(config_list: list) -> dict:
    """Run evaluation on multiple dataset configurations.

    config_list: list of dicts with keys:
    - name:  dataset label
    - graph: path to graphml
    - gold:  path to gold jsonl
    """
    all_results = {}
    for cfg in config_list:
        name = cfg["name"]
        print(f"\n{'='*60}\nDATASET: {name}")
        missing = [k for k in ("graph", "gold") if not Path(cfg[k]).exists()]
        if missing:
            print(f"  ERROR: missing files: {missing}", file=sys.stderr)
            all_results[name] = {"error": f"missing: {missing}"}
            continue
        all_results[name] = evaluate_single(cfg["graph"], cfg["gold"])
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BATCH_CONFIG = [
    # WHO Life Expectancy
    {"name": "WHO_SGE", "graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl"},
    {"name": "WHO_BASE", "graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl"},
    # WB Population
    {"name": "WB_Pop_SGE", "graph": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_population_v2.jsonl"},
    {"name": "WB_Pop_BASE", "graph": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_population_v2.jsonl"},
    # WB Child Mortality
    {"name": "WB_CM_SGE", "graph": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl"},
    {"name": "WB_CM_BASE", "graph": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl"},
    # WB Maternal
    {"name": "WB_Mat_SGE", "graph": "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl"},
    {"name": "WB_Mat_BASE", "graph": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl"},
    # Inpatient 2023
    {"name": "Inpatient_SGE", "graph": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_inpatient_2023.jsonl"},
    {"name": "Inpatient_BASE", "graph": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml", "gold": "evaluation/gold/gold_inpatient_2023.jsonl"},
]


def main():
    parser = argparse.ArgumentParser(
        description="Value-first de-biased coverage evaluation for SGE-LightRAG."
    )
    parser.add_argument("--graph", default=None, help="Path to graphml file")
    parser.add_argument("--gold", default=None, help="Path to gold JSONL file")
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print rescued/lost/uncovered facts")
    parser.add_argument("--batch", action="store_true",
                        help="Run predefined BATCH_CONFIG (edit constant above)")
    args = parser.parse_args()

    if args.batch:
        results = run_batch(BATCH_CONFIG)
    else:
        if not args.graph or not args.gold:
            parser.error("--graph and --gold are required (or use --batch)")
        for p, label in ((args.graph, "--graph"), (args.gold, "--gold")):
            if not Path(p).exists():
                print(f"ERROR: {label} not found: {p}", file=sys.stderr)
                sys.exit(1)
        results = evaluate_single(args.graph, args.gold, verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to: {args.output}")
    else:
        print("\n[JSON]")
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
