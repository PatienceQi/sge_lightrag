"""
OOD Degradation Detection — Pure Graph Topology Analysis
=========================================================
Analyzes graph topology metrics for OOD and main evaluation datasets to identify
an edge/node ratio threshold that separates SGE success from failure cases.

No LLM calls required.
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OOD_OUTPUT_DIR = PROJECT_ROOT / "output" / "ood"
MAIN_OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# FC results from paper experiments (OOD datasets)
OOD_FC_RESULTS = {
    "cereal_production": {"sge_fc": 0.950, "base_fc": 0.050},
    "co2_emissions":     {"sge_fc": 0.700, "base_fc": 0.025},
    "population_growth": {"sge_fc": 0.625, "base_fc": 0.075},
    "education_spending": {"sge_fc": 0.609, "base_fc": 0.000},
    "literacy_rate":     {"sge_fc": 0.600, "base_fc": 0.100},
    "health_expenditure": {"sge_fc": 0.550, "base_fc": 0.050},
    "gdp_growth":        {"sge_fc": 0.475, "base_fc": 0.025},
    "unemployment":      {"sge_fc": 0.025, "base_fc": 0.125},
    "immunization_dpt":  {"sge_fc": 0.000, "base_fc": 0.300},
    "immunization_measles": {"sge_fc": 0.000, "base_fc": 0.325},
}

# OOD directory names map to FC result keys (strip "wb_" prefix)
OOD_DIR_TO_KEY = {
    "wb_cereal_production": "cereal_production",
    "wb_co2_emissions":     "co2_emissions",
    "wb_population_growth": "population_growth",
    "wb_education_spending": "education_spending",
    "wb_literacy_rate":     "literacy_rate",
    "wb_health_expenditure": "health_expenditure",
    "wb_gdp_growth":        "gdp_growth",
    "wb_unemployment":      "unemployment",
    "wb_immunization_dpt":  "immunization_dpt",
    "wb_immunization_measles": "immunization_measles",
}

# Main evaluation datasets (SGE FC known from paper)
MAIN_DATASETS = {
    "who_life_expectancy":  {"sge_fc": 1.000, "base_fc": 0.167,
                             "path": MAIN_OUTPUT_DIR / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"},
    "wb_child_mortality":   {"sge_fc": 1.000, "base_fc": 0.473,
                             "path": MAIN_OUTPUT_DIR / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"},
    "wb_population":        {"sge_fc": 1.000, "base_fc": 0.187,
                             "path": MAIN_OUTPUT_DIR / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"},
    "wb_maternal":          {"sge_fc": 0.967, "base_fc": 0.787,
                             "path": MAIN_OUTPUT_DIR / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"},
    "inpatient_2023":       {"sge_fc": 0.938, "base_fc": 0.438,
                             "path": MAIN_OUTPUT_DIR / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"},
}


def load_graph(graphml_path: Path) -> Optional[nx.Graph]:
    """Load a graphml file and return an undirected graph, or None if missing."""
    if not graphml_path.exists():
        warnings.warn(f"GraphML not found, skipping: {graphml_path}")
        return None
    try:
        return nx.read_graphml(str(graphml_path))
    except Exception as exc:
        warnings.warn(f"Failed to parse {graphml_path}: {exc}")
        return None


def compute_topology_metrics(graph: nx.Graph) -> dict:
    """Compute topology metrics for a graph."""
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    edge_node_ratio = n_edges / n_nodes if n_nodes > 0 else 0.0

    degrees = [d for _, d in graph.degree()]
    isolated_nodes = sum(1 for d in degrees if d == 0)
    isolated_pct = isolated_nodes / n_nodes if n_nodes > 0 else 0.0
    avg_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0.0

    n_components = nx.number_connected_components(graph) if not graph.is_directed() else \
        nx.number_weakly_connected_components(graph)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "edge_node_ratio": round(edge_node_ratio, 4),
        "isolated_nodes": isolated_nodes,
        "isolated_pct": round(isolated_pct, 4),
        "n_components": n_components,
        "avg_degree": round(avg_degree, 4),
    }


def analyze_ood_datasets() -> list[dict]:
    """Analyze all OOD datasets — SGE and Baseline graphs."""
    records = []
    for dir_name, fc_key in sorted(OOD_DIR_TO_KEY.items()):
        dataset_dir = OOD_OUTPUT_DIR / dir_name
        fc_info = OOD_FC_RESULTS[fc_key]

        for variant, subdir in [("sge", "sge_budget"), ("base", "baseline_budget")]:
            graphml_path = dataset_dir / subdir / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
            graph = load_graph(graphml_path)
            if graph is None:
                continue
            metrics = compute_topology_metrics(graph)
            records.append({
                "dataset": fc_key,
                "variant": variant,
                "sge_fc": fc_info["sge_fc"],
                "base_fc": fc_info["base_fc"],
                "sge_better": fc_info["sge_fc"] > fc_info["base_fc"],
                **metrics,
            })
    return records


def analyze_main_datasets() -> list[dict]:
    """Analyze the 5 main evaluation datasets (SGE only)."""
    records = []
    for name, info in sorted(MAIN_DATASETS.items()):
        graph = load_graph(info["path"])
        if graph is None:
            continue
        metrics = compute_topology_metrics(graph)
        records.append({
            "dataset": name,
            "variant": "sge",
            "sge_fc": info["sge_fc"],
            "base_fc": info["base_fc"],
            "sge_better": info["sge_fc"] > info["base_fc"],
            **metrics,
        })
    return records


def find_optimal_threshold(records: list[dict], variant: str = "sge") -> dict:
    """
    Scan candidate thresholds and find the one that best separates
    SGE-better from SGE-worse datasets by edge/node ratio.
    """
    sge_records = [r for r in records if r["variant"] == variant]
    if not sge_records:
        return {}

    ratios = sorted({r["edge_node_ratio"] for r in sge_records})
    candidates = []
    for threshold in ratios:
        tp = sum(1 for r in sge_records if r["sge_better"] and r["edge_node_ratio"] >= threshold)
        fp = sum(1 for r in sge_records if not r["sge_better"] and r["edge_node_ratio"] >= threshold)
        tn = sum(1 for r in sge_records if not r["sge_better"] and r["edge_node_ratio"] < threshold)
        fn = sum(1 for r in sge_records if r["sge_better"] and r["edge_node_ratio"] < threshold)
        accuracy = (tp + tn) / len(sge_records)
        candidates.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(accuracy, 4),
        })

    best = max(candidates, key=lambda x: x["accuracy"])
    return {"best": best, "all_candidates": candidates}


def print_summary(ood_records: list[dict], main_records: list[dict], threshold_analysis: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 75)
    print("OOD DEGRADATION DETECTION — TOPOLOGY ANALYSIS SUMMARY")
    print("=" * 75)

    print("\n--- OOD Datasets (SGE graphs) ---")
    header = f"{'Dataset':<24} {'Nodes':>6} {'Edges':>6} {'E/N':>6} {'IsoN%':>7} {'Comps':>5} {'AvgDeg':>7} {'SGE>Base':>9}"
    print(header)
    print("-" * len(header))
    sge_ood = [r for r in ood_records if r["variant"] == "sge"]
    for r in sorted(sge_ood, key=lambda x: x["edge_node_ratio"]):
        mark = "YES" if r["sge_better"] else "NO"
        print(
            f"{r['dataset']:<24} {r['n_nodes']:>6} {r['n_edges']:>6} "
            f"{r['edge_node_ratio']:>6.3f} {r['isolated_pct']:>7.1%} "
            f"{r['n_components']:>5} {r['avg_degree']:>7.3f} {mark:>9}"
        )

    print("\n--- OOD Datasets (Baseline graphs) ---")
    base_ood = [r for r in ood_records if r["variant"] == "base"]
    print(header)
    print("-" * len(header))
    for r in sorted(base_ood, key=lambda x: x["edge_node_ratio"]):
        mark = "YES" if r["sge_better"] else "NO"
        print(
            f"{r['dataset']:<24} {r['n_nodes']:>6} {r['n_edges']:>6} "
            f"{r['edge_node_ratio']:>6.3f} {r['isolated_pct']:>7.1%} "
            f"{r['n_components']:>5} {r['avg_degree']:>7.3f} {mark:>9}"
        )

    print("\n--- Main Evaluation Datasets (SGE graphs, validation) ---")
    print(header)
    print("-" * len(header))
    for r in sorted(main_records, key=lambda x: x["edge_node_ratio"]):
        mark = "YES" if r["sge_better"] else "NO"
        print(
            f"{r['dataset']:<24} {r['n_nodes']:>6} {r['n_edges']:>6} "
            f"{r['edge_node_ratio']:>6.3f} {r['isolated_pct']:>7.1%} "
            f"{r['n_components']:>5} {r['avg_degree']:>7.3f} {mark:>9}"
        )

    if threshold_analysis:
        best = threshold_analysis["best"]
        print("\n--- Threshold Optimization (OOD SGE graphs) ---")
        print(f"  Optimal threshold : edge/node ratio >= {best['threshold']:.3f}")
        print(f"  Accuracy          : {best['accuracy']:.1%}")
        print(f"  TP={best['tp']}  FP={best['fp']}  TN={best['tn']}  FN={best['fn']}")

    print("\n" + "=" * 75)


def main() -> None:
    ood_records = analyze_ood_datasets()
    main_records = analyze_main_datasets()
    threshold_analysis = find_optimal_threshold(ood_records, variant="sge")

    results = {
        "ood_datasets": ood_records,
        "main_datasets": main_records,
        "threshold_analysis": threshold_analysis,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ood_degradation_detection.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print_summary(ood_records, main_records, threshold_analysis)


if __name__ == "__main__":
    main()
