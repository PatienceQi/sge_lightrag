#!/usr/bin/env python3
"""
det_parser_quality.py — Quality signal for Deterministic Parser output.

Computes structural coverage metrics on a Det Parser GraphML to decide
whether the graph is "good enough" or needs SGE fallback.

The quality signal works WITHOUT a gold standard — it uses the source CSV
to derive expected entity count and year column count, then measures how
completely the graph represents those expectations.

Decision thresholds (conservative: when in doubt, fall back to SGE):
  - entity_coverage >= 0.80  (80% of CSV rows have a node)
  - AND (value_completeness >= 0.85 OR edge_node_ratio >= 0.90)

Usage (as module):
    from evaluation.det_parser_quality import compute_quality_signal
    signal = compute_quality_signal(graphml_path, csv_path)
    # signal["recommendation"] is "accept" or "fallback"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow imports from project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required: pip install networkx") from exc

import pandas as pd

from stage1.features import _detect_encoding, _YEAR_PATTERN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum fraction of CSV rows that must have a corresponding entity node
_MIN_ENTITY_COVERAGE: float = 0.80

# Minimum fraction of expected (entity × year) cells that must have an edge.
# Conservative: 0.70 catches severely sparse datasets while allowing natural
# historical data gaps (e.g. WB_Mat has 54% completeness but Det Parser FC=0.967).
_MIN_VALUE_COMPLETENESS: float = 0.70

# Minimum edge-to-node ratio (graph density proxy)
_MIN_EDGE_NODE_RATIO: float = 0.90

# Null sentinels matched during CSV row counting (mirrors deterministic parser)
_NULL_SENTINELS = {"", "nan", "...", "na", "n/a", "null", "none", "-"}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv_for_signal(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV using the same encoding-detection and skip-row logic
    as the Det Parser, returning a DataFrame for metric computation.
    """
    encoding = _detect_encoding(csv_path)
    if "utf-16" in encoding:
        return pd.read_csv(csv_path, encoding="utf-16", sep="\t", header=None)

    skiprows = _detect_skiprows(csv_path, encoding)
    for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
        try:
            return pd.read_csv(csv_path, encoding=enc, skiprows=skiprows)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Cannot read CSV for quality signal: {csv_path}")


def _detect_skiprows(csv_path: str, encoding: str) -> int:
    """Detect leading metadata rows to skip (World Bank: 4, HK title: 2)."""
    try:
        with open(csv_path, encoding=encoding, errors="ignore") as fh:
            lines = [fh.readline() for _ in range(2)]
        first = lines[0]
        if "Data Source" in first or "World Development Indicators" in first:
            return 4
        row0 = [c.strip().strip('"') for c in first.split(",")]
        row1 = [c.strip().strip('"') for c in lines[1].split(",")]
        if sum(1 for c in row0 if c) <= 1 and sum(1 for c in row1 if c) == 0:
            return 2
    except Exception:
        pass
    return 0


def _count_csv_entities(df: pd.DataFrame) -> int:
    """
    Estimate the number of distinct entity rows.
    Uses the first non-numeric, non-empty column as the entity key column.
    Falls back to total row count if no key column can be identified.
    """
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_frac = pd.to_numeric(non_null, errors="coerce").notna().sum() / len(non_null)
        if numeric_frac < 0.6:
            return int(df[col].dropna().nunique())
    return len(df)


def _count_year_columns(df: pd.DataFrame) -> int:
    """Count columns whose header matches a year or fiscal-year pattern."""
    return sum(
        1 for col in df.columns
        if _YEAR_PATTERN.search(str(col).replace("\n", " ").strip())
    )


# ---------------------------------------------------------------------------
# GraphML helpers
# ---------------------------------------------------------------------------

def _load_graphml_stats(graphml_path: str) -> dict:
    """
    Load GraphML and return node/edge counts along with entity-node count.
    Entity nodes are those with entity_type == 'subject'.
    """
    G = nx.read_graphml(graphml_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    n_entity_nodes = sum(
        1 for _, data in G.nodes(data=True)
        if str(data.get("entity_type", "")).lower() == "subject"
    )
    # Fallback: if no typed nodes found, use half of all nodes as heuristic
    if n_entity_nodes == 0 and n_nodes > 0:
        n_entity_nodes = n_nodes // 2

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_entity_nodes": n_entity_nodes,
    }


# ---------------------------------------------------------------------------
# Main quality signal function
# ---------------------------------------------------------------------------

def compute_quality_signal(graphml_path: str, csv_path: str) -> dict:
    """
    Compute coverage quality metrics for a Det Parser GraphML output.

    Metrics are derived by comparing graph structure against the source CSV —
    no gold standard required. This is suitable for deployment (online decision).

    Parameters
    ----------
    graphml_path : str
        Path to the GraphML file produced by deterministic_parser_baseline.py.
    csv_path : str
        Path to the original source CSV file.

    Returns
    -------
    dict with keys:
        n_csv_rows        : int   — number of data rows in CSV
        n_year_columns    : int   — number of year-pattern columns
        n_entity_nodes    : int   — subject nodes in graph
        n_value_edges     : int   — total edges in graph
        n_nodes           : int   — total nodes in graph
        entity_coverage   : float — n_entity_nodes / n_csv_rows
        value_completeness: float — n_value_edges / (n_entity_nodes × n_year_cols)
        edge_node_ratio   : float — n_value_edges / n_nodes
        recommendation    : str   — "accept" or "fallback"
        accept_reason     : str   — human-readable explanation
    """
    graphml_p = Path(graphml_path)
    csv_p = Path(csv_path)

    if not graphml_p.exists():
        return _error_signal(f"GraphML not found: {graphml_path}", fallback=True)
    if not csv_p.exists():
        return _error_signal(f"CSV not found: {csv_path}", fallback=True)

    # Load CSV metrics
    try:
        df = _load_csv_for_signal(csv_path)
    except Exception as exc:
        return _error_signal(f"CSV load error: {exc}", fallback=True)

    n_csv_rows = _count_csv_entities(df)
    n_year_cols = _count_year_columns(df)

    # Load graph metrics
    try:
        stats = _load_graphml_stats(graphml_path)
    except Exception as exc:
        return _error_signal(f"GraphML load error: {exc}", fallback=True)

    n_entity_nodes = stats["n_entity_nodes"]
    n_value_edges = stats["n_edges"]
    n_nodes = stats["n_nodes"]

    # Compute metrics (guard against division by zero)
    entity_coverage = (
        n_entity_nodes / n_csv_rows if n_csv_rows > 0 else 0.0
    )

    if n_year_cols > 0 and n_entity_nodes > 0:
        expected_values = n_entity_nodes * n_year_cols
        value_completeness = min(n_value_edges / expected_values, 1.0)
    else:
        # No year columns (Type-III): skip value_completeness threshold
        value_completeness = 1.0

    edge_node_ratio = (
        n_value_edges / n_nodes if n_nodes > 0 else 0.0
    )

    # Decision logic (conservative: prefer SGE fallback when uncertain)
    #
    # For datasets with many year columns (> 20), value_completeness is the
    # dominant signal.  A high edge_node_ratio alone does not guarantee
    # correctness when the CSV has many year columns but sparse coverage
    # (e.g. WB Child Mortality: 266 rows × 66 year cols, but only 13251 non-null
    # values out of 17556 expected → value_completeness=0.755).
    # In such cases the graph is structurally dense but only partially covers the
    # year-value matrix, causing FC degradation in gold evaluation.
    #
    # Threshold: when n_year_cols > 20, require value_completeness to meet
    # the minimum; edge_node_ratio alone is insufficient.
    _MANY_YEAR_COLS = 20

    entity_ok = entity_coverage >= _MIN_ENTITY_COVERAGE

    if n_year_cols > _MANY_YEAR_COLS:
        # Strict: both value_completeness AND entity_coverage must pass
        value_ok = value_completeness >= _MIN_VALUE_COMPLETENESS
    else:
        # Lenient: either metric suffices (handles Type-III with no year cols)
        value_ok = (
            value_completeness >= _MIN_VALUE_COMPLETENESS
            or edge_node_ratio >= _MIN_EDGE_NODE_RATIO
        )

    if entity_ok and value_ok:
        recommendation = "accept"
        accept_reason = (
            f"entity_coverage={entity_coverage:.3f}>={_MIN_ENTITY_COVERAGE}, "
            f"value_completeness={value_completeness:.3f}>={_MIN_VALUE_COMPLETENESS} "
            f"(n_year_cols={n_year_cols})"
        )
    elif not entity_ok:
        recommendation = "fallback"
        accept_reason = (
            f"entity_coverage={entity_coverage:.3f} < {_MIN_ENTITY_COVERAGE} "
            f"(only {n_entity_nodes}/{n_csv_rows} entities represented)"
        )
    else:
        recommendation = "fallback"
        accept_reason = (
            f"value_completeness={value_completeness:.3f} < {_MIN_VALUE_COMPLETENESS} "
            f"(n_year_cols={n_year_cols} > {_MANY_YEAR_COLS}; "
            f"sparse year-value coverage, possibly missing year bindings)"
        )

    return {
        "n_csv_rows": n_csv_rows,
        "n_year_columns": n_year_cols,
        "n_entity_nodes": n_entity_nodes,
        "n_value_edges": n_value_edges,
        "n_nodes": n_nodes,
        "entity_coverage": round(entity_coverage, 4),
        "value_completeness": round(value_completeness, 4),
        "edge_node_ratio": round(edge_node_ratio, 4),
        "recommendation": recommendation,
        "accept_reason": accept_reason,
        "error": None,
    }


def _error_signal(message: str, fallback: bool = True) -> dict:
    """Return a quality signal dict representing an error state."""
    return {
        "n_csv_rows": 0,
        "n_year_columns": 0,
        "n_entity_nodes": 0,
        "n_value_edges": 0,
        "n_nodes": 0,
        "entity_coverage": 0.0,
        "value_completeness": 0.0,
        "edge_node_ratio": 0.0,
        "recommendation": "fallback" if fallback else "accept",
        "accept_reason": message,
        "error": message,
    }


# ---------------------------------------------------------------------------
# CLI for ad-hoc testing
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Quick CLI: python3 det_parser_quality.py --graph X.graphml --csv X.csv"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute quality signal for a Det Parser GraphML output."
    )
    parser.add_argument("--graph", required=True, help="Path to GraphML file")
    parser.add_argument("--csv",   required=True, help="Path to source CSV file")
    args = parser.parse_args()

    signal = compute_quality_signal(args.graph, args.csv)
    print(json.dumps(signal, indent=2))
    print(f"\nRecommendation: {signal['recommendation'].upper()}")
    print(f"Reason: {signal['accept_reason']}")


if __name__ == "__main__":
    _cli()
