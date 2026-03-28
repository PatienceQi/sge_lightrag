#!/usr/bin/env python3
"""
run_precision_analysis.py — Precision Analysis for SGE-LightRAG

Computes precision-oriented metrics for SGE and Baseline graphs across 5 datasets:
  - Node/Edge counts
  - Isolated node ratio
  - Entity-level precision (gold entities found in graph nodes)
  - Signal-to-noise ratio (gold facts covered / total edges)
  - Typed entity ratio (nodes with non-empty entity_type)

Also samples 25 edges each from WHO and inpatient_2023 SGE graphs
for human annotation of edge-level precision.

Usage:
    python3 evaluation/run_precision_analysis.py

Run from sge_lightrag/ directory.
"""

import json
import random
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # sge_lightrag/
EVAL_DIR = Path(__file__).parent         # sge_lightrag/evaluation/
OUTPUT_DIR = BASE_DIR / "output"

DATASETS = [
    {
        "name": "WHO Life Expectancy",
        "key": "who_life_expectancy",
        "sge_graph": OUTPUT_DIR / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "base_graph": OUTPUT_DIR / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "gold": EVAL_DIR / "gold_who_life_expectancy_v2.jsonl",
    },
    {
        "name": "WB Child Mortality",
        "key": "wb_child_mortality",
        "sge_graph": OUTPUT_DIR / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "base_graph": OUTPUT_DIR / "baseline_wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "gold": EVAL_DIR / "gold_wb_child_mortality_v2.jsonl",
    },
    {
        "name": "WB Population",
        "key": "wb_population",
        "sge_graph": OUTPUT_DIR / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "base_graph": OUTPUT_DIR / "baseline_wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "gold": EVAL_DIR / "gold_wb_population_v2.jsonl",
    },
    {
        "name": "WB Maternal",
        "key": "wb_maternal",
        "sge_graph": OUTPUT_DIR / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "base_graph": OUTPUT_DIR / "baseline_wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "gold": EVAL_DIR / "gold_wb_maternal_v2.jsonl",
    },
    {
        "name": "Inpatient 2023",
        "key": "inpatient_2023",
        "sge_graph": OUTPUT_DIR / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "base_graph": OUTPUT_DIR / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "gold": EVAL_DIR / "gold_inpatient_2023.jsonl",
    },
]

SAMPLE_DATASETS = {"who_life_expectancy", "inpatient_2023"}
SAMPLE_SIZE = 25
RANDOM_SEED = 42

VALUE_OBJECT_TYPES = {"BudgetAmount", "StatValue", "Literal"}


# ---------------------------------------------------------------------------
# Data loading helpers (immutable — always return new objects)
# ---------------------------------------------------------------------------

def load_gold(jsonl_path: Path) -> tuple[set, list]:
    """Return (gold_entities: set[str], facts: list[dict]).

    Only facts whose object_type is in VALUE_OBJECT_TYPES are included.
    """
    entities: set[str] = set()
    facts: list[dict] = []

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
            obj_type = triple.get("object_type", "")

            if subj:
                entities.add(subj)

            if obj_type in VALUE_OBJECT_TYPES:
                facts.append({
                    "subject": subj,
                    "value": obj,
                    "year": year,
                    "relation": rel,
                })

    return entities, facts


def load_graph(graphml_path: Path) -> tuple:
    """Load graph and build per-entity text index.

    Returns:
        (G, node_attrs, entity_text_2hop)
        G: networkx Graph
        node_attrs: dict[name -> {type, description}]
        entity_text_2hop: dict[name -> list[str]]
    """
    G = nx.read_graphml(str(graphml_path))

    node_attrs: dict[str, dict] = {}
    for node_id, data in G.nodes(data=True):
        name = str(data.get("entity_name") or data.get("name") or node_id).strip()
        node_attrs[name] = {
            "type": data.get("entity_type", ""),
            "description": str(data.get("description", "")),
        }

    # Build 1-hop edge text index
    entity_text: dict[str, list[str]] = {}
    for u, v, data in G.edges(data=True):
        u_name = str(G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u).strip()
        v_name = str(G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v).strip()
        kw = str(data.get("keywords", ""))
        desc = str(data.get("description", ""))
        edge_text = f"{kw} {desc} {v_name}"
        entity_text.setdefault(u_name, []).append(edge_text)
        rev_text = f"{kw} {desc} {u_name}"
        entity_text.setdefault(v_name, []).append(rev_text)

    # Build node_id -> name map for 2-hop expansion
    node_id_to_name: dict[str, str] = {
        nid: str(d.get("entity_name") or d.get("name") or nid).strip()
        for nid, d in G.nodes(data=True)
    }

    # 2-hop expansion
    entity_text_2hop: dict[str, list[str]] = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, nb_id)
            texts.extend(entity_text.get(nb_name, []))
            nb_desc = node_attrs.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
        entity_text_2hop[name] = texts

    return G, node_attrs, entity_text_2hop


# ---------------------------------------------------------------------------
# Metric computation helpers (pure functions — no side effects)
# ---------------------------------------------------------------------------

def compute_entity_precision(gold_entities: set, node_attrs: dict) -> dict:
    """Entity-level precision: gold entities found as substring in graph node names.

    precision = |{graph nodes that contain at least one gold entity as substring}|
                / |total graph nodes|

    This measures what fraction of graph nodes are "relevant" to the gold standard.
    """
    node_names = list(node_attrs.keys())
    gold_lower = [ge.lower() for ge in gold_entities]

    relevant_nodes = sum(
        1 for nn in node_names
        if any(ge in nn.lower() or nn.lower() in ge for ge in gold_lower)
    )

    total = len(node_names)
    precision = relevant_nodes / total if total else 0.0
    return {
        "relevant_nodes": relevant_nodes,
        "total_nodes": total,
        "entity_precision": round(precision, 4),
    }


def _find_matching_node(subject: str, node_names_lower: dict) -> str | None:
    """Fuzzy match subject to a node name. Returns original-case name or None."""
    subj_lower = subject.lower()
    if subj_lower in node_names_lower:
        return node_names_lower[subj_lower]
    for nn, orig in node_names_lower.items():
        if subj_lower in nn or nn in subj_lower:
            return orig
    return None


def compute_fact_coverage(facts: list, node_attrs: dict, entity_text_2hop: dict) -> int:
    """Count how many gold facts are recoverable via 2-hop neighbor text search.

    Mirrors the logic in evaluate_coverage.py: check_fact_coverage.
    """
    node_names_lower = {n.lower(): n for n in node_attrs}
    covered_count = 0

    for fact in facts:
        matched_node = _find_matching_node(fact["subject"], node_names_lower)
        if not matched_node:
            continue

        texts = entity_text_2hop.get(matched_node, [])
        node_desc = node_attrs.get(matched_node, {}).get("description", "")
        all_text = " ".join(texts) + " " + node_desc

        value_found = fact["value"] in all_text
        year = fact.get("year", "")
        year_found = (not year) or (year in all_text)

        if value_found and year_found:
            covered_count += 1

    return covered_count


def compute_graph_metrics(G: nx.Graph, node_attrs: dict, gold_entities: set, facts: list, entity_text_2hop: dict) -> dict:
    """Compute all precision-oriented metrics for a single graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Isolated node ratio
    isolated = sum(1 for _, deg in G.degree() if deg == 0)
    isolated_ratio = isolated / num_nodes if num_nodes else 0.0

    # Entity-level precision
    ep = compute_entity_precision(gold_entities, node_attrs)

    # Signal-to-noise ratio: gold_facts_covered / total_edges
    covered_facts = compute_fact_coverage(facts, node_attrs, entity_text_2hop)
    snr = covered_facts / num_edges if num_edges else 0.0

    # Typed entity ratio
    typed_count = sum(
        1 for info in node_attrs.values()
        if info["type"] and info["type"].strip()
    )
    typed_ratio = typed_count / num_nodes if num_nodes else 0.0

    return {
        "node_count": num_nodes,
        "edge_count": num_edges,
        "isolated_nodes": isolated,
        "isolated_ratio": round(isolated_ratio, 4),
        "entity_precision_relevant_nodes": ep["relevant_nodes"],
        "entity_precision": ep["entity_precision"],
        "gold_facts_covered": covered_facts,
        "total_gold_facts": len(facts),
        "signal_to_noise_ratio": round(snr, 4),
        "typed_nodes": typed_count,
        "typed_entity_ratio": round(typed_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Edge sampling for human annotation
# ---------------------------------------------------------------------------

def sample_edges(G: nx.Graph, n: int, seed: int) -> list[dict]:
    """Randomly sample n edges from graph G. Returns list of annotation records."""
    rng = random.Random(seed)
    all_edges = list(G.edges(data=True))
    sampled = rng.sample(all_edges, min(n, len(all_edges)))

    records = []
    for u, v, data in sampled:
        u_name = str(G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u).strip()
        v_name = str(G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v).strip()
        records.append({
            "source_node": u_name,
            "target_node": v_name,
            "keywords": data.get("keywords", ""),
            "description": data.get("description", ""),
            "correct": "",
        })
    return records


def write_sample(records: list[dict], output_path: Path) -> None:
    """Write sampled edges as JSONL to output_path."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """Print a formatted summary table to stdout."""
    col_w = 24
    sep = "-" * 100

    print("\n" + "=" * 100)
    print("SGE-LightRAG PRECISION ANALYSIS SUMMARY")
    print("=" * 100)

    header = (
        f"{'Dataset':<{col_w}} {'System':<10} {'Nodes':>7} {'Edges':>7} "
        f"{'Iso%':>7} {'EntPrec':>9} {'SNR':>9} {'TypedR':>8}"
    )
    print(header)
    print(sep)

    for ds_key, ds_result in results["datasets"].items():
        ds_name = ds_result["dataset_name"]
        for system in ("sge", "baseline"):
            m = ds_result[system]
            if m is None:
                print(f"{'  ' + ds_name:<{col_w}} {system.upper():<10} {'N/A':>7}")
                continue
            print(
                f"{'  ' + ds_name:<{col_w}} {system.upper():<10} "
                f"{m['node_count']:>7} {m['edge_count']:>7} "
                f"{m['isolated_ratio']:>7.3f} "
                f"{m['entity_precision']:>9.4f} "
                f"{m['signal_to_noise_ratio']:>9.4f} "
                f"{m['typed_entity_ratio']:>8.4f}"
            )
        print(sep)

    print("\nMetric definitions:")
    print("  Iso%     : isolated nodes / total nodes")
    print("  EntPrec  : gold-entity-relevant nodes / total nodes")
    print("  SNR      : gold facts covered / total edges (signal-to-noise ratio)")
    print("  TypedR   : nodes with non-empty entity_type / total nodes")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results: dict = {"datasets": {}}

    for ds in DATASETS:
        key = ds["key"]
        name = ds["name"]
        print(f"\nProcessing: {name} ...", flush=True)

        # Load gold standard
        if not ds["gold"].exists():
            print(f"  WARNING: gold file not found: {ds['gold']}", file=sys.stderr)
            results["datasets"][key] = {"dataset_name": name, "error": "gold_missing", "sge": None, "baseline": None}
            continue

        gold_entities, facts = load_gold(ds["gold"])
        print(f"  Gold: {len(gold_entities)} entities, {len(facts)} value-facts")

        ds_result: dict = {
            "dataset_name": name,
            "gold_entities": len(gold_entities),
            "gold_facts": len(facts),
            "sge": None,
            "baseline": None,
        }

        # Process SGE graph
        if ds["sge_graph"].exists():
            G_sge, node_attrs_sge, et_sge = load_graph(ds["sge_graph"])
            ds_result["sge"] = compute_graph_metrics(G_sge, node_attrs_sge, gold_entities, facts, et_sge)
            print(f"  SGE  : {ds_result['sge']['node_count']} nodes, {ds_result['sge']['edge_count']} edges")

            # Edge sampling for designated datasets
            if key in SAMPLE_DATASETS:
                sample_out = EVAL_DIR / f"precision_sample_{key.replace('_2023', '')}.jsonl"
                sampled = sample_edges(G_sge, SAMPLE_SIZE, RANDOM_SEED)
                write_sample(sampled, sample_out)
                print(f"  Sampled {len(sampled)} edges → {sample_out}")
        else:
            print(f"  WARNING: SGE graph not found: {ds['sge_graph']}", file=sys.stderr)

        # Process Baseline graph
        if ds["base_graph"].exists():
            G_base, node_attrs_base, et_base = load_graph(ds["base_graph"])
            ds_result["baseline"] = compute_graph_metrics(G_base, node_attrs_base, gold_entities, facts, et_base)
            print(f"  Base : {ds_result['baseline']['node_count']} nodes, {ds_result['baseline']['edge_count']} edges")
        else:
            print(f"  WARNING: Baseline graph not found: {ds['base_graph']}", file=sys.stderr)

        results["datasets"][key] = ds_result

    # Write JSON results
    output_path = EVAL_DIR / "precision_analysis_v1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to: {output_path}")

    # Print summary table
    print_summary(results)


if __name__ == "__main__":
    main()
