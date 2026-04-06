#!/usr/bin/env python3
"""
csv_verified_precision.py — CSV-Verified Precision Audit.

Deterministically verifies graph edges against source CSV data.
No LLM needed — pure rule-based CSV lookup.

For each sampled edge:
  1. Parse (subject, year, value) from graph edge metadata
  2. Look up the original CSV row for that subject
  3. Check if the value appears in the correct year column
  4. Report verified precision = correct / total_sampled

Usage:
    python3 evaluation/csv_verified_precision.py
    python3 evaluation/csv_verified_precision.py --sample-size 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.preprocessor import preprocess_csv

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DATASETS = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "subject_col": "Country Code",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv_path": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "sge_graph": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "subject_col": "Country Code",
    },
    "wb_pop": {
        "label": "WB Population",
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "sge_graph": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "subject_col": "Country Code",
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "sge_graph": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
        "subject_col": None,  # Type-III, needs different matching
    },
}


# ---------------------------------------------------------------------------
# Edge parsing
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# Matches "yearXXXX=VALUE" or "_XXXX=VALUE" in node names (SGE compact format)
_YEAR_EQ_VALUE_RE = re.compile(r"(?:year|_)((?:19|20)\d{2})\s*=\s*(\d[\d.]*)")


def parse_edge_triple(G: nx.Graph, source: str, target: str, edge_data: dict) -> dict:
    """Extract (subject, year, value) from a graph edge."""
    keywords = edge_data.get("keywords", "")
    description = edge_data.get("description", "")
    source_label = G.nodes[source].get("label", source) if source in G.nodes else source
    target_label = G.nodes[target].get("label", target) if target in G.nodes else target

    # Check for yearXXXX=VALUE pattern in node names (SGE compact format)
    # e.g., "population_year2012=7391448_persons"
    for node_label, other_label in [(target_label, source_label), (source_label, target_label)]:
        m = _YEAR_EQ_VALUE_RE.search(node_label)
        if m:
            return {
                "subject": other_label,
                "year": m.group(1),
                "value": m.group(2),
                "source_node": source,
                "target_node": target,
                "keywords": keywords,
            }

    # Find year in keywords or description
    year = ""
    year_match = _YEAR_RE.search(keywords)
    if year_match:
        year = year_match.group(1)
    elif _YEAR_RE.search(description):
        year = _YEAR_RE.search(description).group(1)

    # Find numeric value in target node label or description
    value = ""
    for text in [target_label, description, keywords]:
        nums = _NUMBER_RE.findall(text)
        for n in nums:
            clean = n.replace(",", "")
            try:
                float(clean)
                if clean != year:  # don't confuse year with value
                    value = clean
                    break
            except ValueError:
                continue
        if value:
            break

    return {
        "subject": source_label,
        "year": year,
        "value": value,
        "source_node": source,
        "target_node": target,
        "keywords": keywords,
    }


# ---------------------------------------------------------------------------
# CSV verification
# ---------------------------------------------------------------------------

def verify_against_csv(
    triple: dict,
    df: pd.DataFrame,
    subject_col: str | None,
) -> dict:
    """Check if a (subject, year, value) triple exists in the source CSV."""
    subject = triple["subject"].strip()
    year = triple["year"]
    value = triple["value"]

    result = {
        "subject": subject,
        "year": year,
        "value": value,
        "verified": False,
        "reason": "",
    }

    if not value:
        result["reason"] = "no_value_extracted"
        return result

    # Find subject row in CSV — try all text columns
    matched_rows = pd.DataFrame()
    search_cols = []
    if subject_col and subject_col in df.columns:
        search_cols.append(subject_col)
    # Also search all text columns as fallback
    for col in df.columns:
        if col not in search_cols and df[col].dtype == object:
            search_cols.append(col)

    for col in search_cols:
        # Exact match
        mask = df[col].astype(str).str.strip().str.upper() == subject.upper()
        matched_rows = df[mask]
        if not matched_rows.empty:
            break
        # Substring match
        if len(subject) >= 3:
            mask = df[col].astype(str).str.strip().str.upper().str.contains(
                re.escape(subject.upper()), na=False
            )
            matched_rows = df[mask]
            if not matched_rows.empty:
                break

    if matched_rows.empty:
        result["reason"] = "subject_not_found_in_csv"
        return result

    # Normalize value for comparison
    try:
        target_val = float(value.replace(",", ""))
    except ValueError:
        result["reason"] = "value_not_numeric"
        return result

    # Check if value exists in the matched rows
    for _, row in matched_rows.iterrows():
        # If year specified, check that specific year column
        if year and year in df.columns:
            cell = str(row[year]).strip()
            try:
                csv_val = float(cell.replace(",", ""))
                if abs(csv_val - target_val) < 0.01 * max(abs(csv_val), 1):
                    result["verified"] = True
                    result["reason"] = f"exact_match_year_{year}"
                    return result
            except (ValueError, TypeError):
                pass

        # Check all numeric columns for the value
        for col in df.columns:
            cell = str(row[col]).strip()
            try:
                csv_val = float(cell.replace(",", ""))
                if abs(csv_val - target_val) < 0.01 * max(abs(csv_val), 1):
                    result["verified"] = True
                    result["reason"] = f"value_found_in_col_{col}"
                    return result
            except (ValueError, TypeError):
                continue

    result["reason"] = "value_not_found_in_matched_rows"
    return result


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit_graph(
    graph_path: str,
    csv_path: str,
    subject_col: str | None,
    sample_size: int,
    seed: int = 42,
) -> dict:
    """Audit a graph against its source CSV."""
    if not Path(graph_path).exists():
        return {"error": f"Graph not found: {graph_path}"}

    G = nx.read_graphml(graph_path)
    df, _ = preprocess_csv(str(PROJECT_ROOT / csv_path))

    # Get all edges
    edges = list(G.edges(data=True))
    if not edges:
        return {"error": "No edges in graph"}

    # Sample edges
    rng = random.Random(seed)
    sampled = rng.sample(edges, min(sample_size, len(edges)))

    results = []
    for source, target, data in sampled:
        triple = parse_edge_triple(G, source, target, data)
        verification = verify_against_csv(triple, df, subject_col)
        results.append({**triple, **verification})

    verified_count = sum(1 for r in results if r["verified"])
    total = len(results)
    precision = verified_count / total if total > 0 else 0.0

    # Error breakdown
    reasons = {}
    for r in results:
        reason = r["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "total_sampled": total,
        "verified_correct": verified_count,
        "precision": round(precision, 4),
        "reason_breakdown": reasons,
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV-verified precision audit")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Edges to sample per dataset per system (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_results = {}

    for ds_key, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"PRECISION AUDIT: {cfg['label']}")
        print(f"{'='*60}")

        for system, graph_key in [("sge", "sge_graph"), ("baseline", "baseline_graph")]:
            graph_path = str(PROJECT_ROOT / cfg[graph_key])
            print(f"\n  [{system.upper()}] {graph_path}")

            result = audit_graph(
                graph_path=graph_path,
                csv_path=cfg["csv_path"],
                subject_col=cfg["subject_col"],
                sample_size=args.sample_size,
                seed=args.seed,
            )

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Precision: {result['verified_correct']}/{result['total_sampled']} "
                      f"= {result['precision']:.4f}")
                print(f"  Reasons: {result['reason_breakdown']}")

            all_results[f"{ds_key}_{system}"] = {
                "dataset": cfg["label"],
                "system": system,
                "precision": result.get("precision", 0),
                "verified": result.get("verified_correct", 0),
                "total": result.get("total_sampled", 0),
                "reasons": result.get("reason_breakdown", {}),
            }

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Dataset':<25s} {'System':<10s} {'Precision':>10s} {'Verified':>10s}")
    print(f"{'-'*60}")
    for key, r in all_results.items():
        print(f"{r['dataset']:<25s} {r['system']:<10s} "
              f"{r['precision']:.4f}     {r['verified']}/{r['total']}")
    print(f"{'='*60}")

    # Aggregate
    sge_total = sum(r["verified"] for k, r in all_results.items() if "_sge" in k)
    sge_sampled = sum(r["total"] for k, r in all_results.items() if "_sge" in k)
    base_total = sum(r["verified"] for k, r in all_results.items() if "_baseline" in k)
    base_sampled = sum(r["total"] for k, r in all_results.items() if "_baseline" in k)

    print(f"\nAggregate SGE: {sge_total}/{sge_sampled} = {sge_total/sge_sampled:.4f}" if sge_sampled else "")
    print(f"Aggregate Baseline: {base_total}/{base_sampled} = {base_total/base_sampled:.4f}" if base_sampled else "")

    # Save
    output_path = PROJECT_ROOT / "evaluation" / "results" / "csv_verified_precision.json"
    save_data = {
        "description": "CSV-verified precision audit: sampled edges checked against source CSV",
        "sample_size_per_dataset": args.sample_size,
        "seed": args.seed,
        "results": all_results,
        "aggregate": {
            "sge": {"verified": sge_total, "sampled": sge_sampled,
                    "precision": round(sge_total / sge_sampled, 4) if sge_sampled else 0},
            "baseline": {"verified": base_total, "sampled": base_sampled,
                        "precision": round(base_total / base_sampled, 4) if base_sampled else 0},
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
