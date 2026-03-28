#!/usr/bin/env python3
"""
run_all_evaluations.py — Run coverage evaluation for all datasets.

Generates a combined results table comparing:
  - SGE (schema-guided extraction via LightRAG)
  - Baseline (plain LightRAG, no schema)
  - GraphRAG (MS GraphRAG community-based extraction)
"""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "evaluate_coverage.py"

# Dataset registry: (label, sge_graph, baseline_graph, gold, graphrag_root)
DATASETS = [
    {
        "label": "Annual Budget (ZH, Type-II)",
        "sge":      "output/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_budget/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_budget.jsonl",
        "graphrag": "output/graphrag_budget/output/graph.graphml",
    },
    {
        "label": "WHO Life Expectancy (EN, Type-II)",
        "sge":      "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_who_life_expectancy.jsonl",
        "graphrag": "output/graphrag_who/output/graph.graphml",
    },
    {
        "label": "Inpatient Stats (ZH, Type-III)",
        "sge":      "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_inpatient_2023.jsonl",
        "graphrag": "output/graphrag_inpatient/output/graph.graphml",
    },
    {
        "label": "WB Child Mortality (EN, Type-II)",
        "sge":      "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_wb_child_mortality.jsonl",
        "graphrag": None,
    },
    {
        "label": "Health Stats (ZH, Type-II-Trans)",
        "sge":      "output/sge_health_v5/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_health_v5/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_health.jsonl",
        "graphrag": None,
    },
    {
        "label": "WB Population (EN, Type-II)",
        "sge":      "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_wb_population.jsonl",
        "graphrag": None,
    },
    {
        "label": "WB Maternal Mortality (EN, Type-II)",
        "sge":      "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "gold":     "evaluation/gold_wb_maternal.jsonl",
        "graphrag": None,
    },
]


def run_eval(graph_path: str, gold_path: str) -> dict | None:
    """Run evaluate_coverage.py and return parsed JSON results."""
    full_graph = PROJECT_ROOT / graph_path
    full_gold = PROJECT_ROOT / gold_path

    if not full_graph.exists():
        return None

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--graph", str(full_graph), "--gold", str(full_gold)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        return {"error": result.stderr[-200:]}

    # Parse JSON from output (look for [JSON] marker or first { ... last } block)
    output = result.stdout
    try:
        # Prefer [JSON] section
        json_marker = output.find("[JSON]")
        if json_marker >= 0:
            after_marker = output[json_marker + 6:]
            json_start = after_marker.find("{")
            json_end = after_marker.rfind("}") + 1
        else:
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            after_marker = output

        if json_start >= 0 and json_end > json_start:
            json_str = after_marker[json_start:json_end] if json_marker >= 0 else output[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {"error": "parse_failed"}


def fmt(val, default="—"):
    """Format a float value as percentage string."""
    if val is None:
        return default
    return f"{val:.3f}"


def main():
    print(f"Running evaluations from {PROJECT_ROOT}\n")
    print(f"{'Dataset':<45} | {'SGE EC':>7} {'SGE FC':>7} | {'Base EC':>7} {'Base FC':>7} | {'GR EC':>6} {'GR FC':>6}")
    print("-" * 110)

    summary = []
    for ds in DATASETS:
        label = ds["label"]
        sge_r = run_eval(ds["sge"], ds["gold"])
        base_r = run_eval(ds["baseline"], ds["gold"])
        gr_r = run_eval(ds["graphrag"], ds["gold"]) if ds.get("graphrag") else None

        sge_ec = sge_r["entity_coverage"]["coverage"] if sge_r and "entity_coverage" in sge_r else None
        sge_fc = sge_r["fact_coverage"]["coverage"] if sge_r and "fact_coverage" in sge_r else None
        base_ec = base_r["entity_coverage"]["coverage"] if base_r and "entity_coverage" in base_r else None
        base_fc = base_r["fact_coverage"]["coverage"] if base_r and "fact_coverage" in base_r else None
        gr_ec = gr_r["entity_coverage"]["coverage"] if gr_r and "entity_coverage" in gr_r else None
        gr_fc = gr_r["fact_coverage"]["coverage"] if gr_r and "fact_coverage" in gr_r else None

        row = {
            "dataset": label,
            "sge_ec": sge_ec, "sge_fc": sge_fc,
            "baseline_ec": base_ec, "baseline_fc": base_fc,
            "graphrag_ec": gr_ec, "graphrag_fc": gr_fc,
        }
        summary.append(row)

        status = "✓" if sge_r and "entity_coverage" in sge_r else "?"
        print(f"{status} {label:<43} | {fmt(sge_ec):>7} {fmt(sge_fc):>7} | {fmt(base_ec):>7} {fmt(base_fc):>7} | {fmt(gr_ec):>6} {fmt(gr_fc):>6}")

    # Save results
    out_path = PROJECT_ROOT / "evaluation" / "all_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
