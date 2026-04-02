#!/usr/bin/env python3
"""
deterministic_parser_baseline.py — Deterministic Parser Upper Bound Baseline

Answers the reviewer question: "If facts are deterministic, why not use rules
directly to build the graph?"

This script converts a CSV file into a GraphML knowledge graph using ONLY
deterministic rules — no LLM, no SGE pipeline. It handles:
  - Type-II (Time-Series-Matrix): rows are entities, columns are year headers
  - Type-III (Hierarchical-Hybrid): composite key columns + sparse forward-fill

The generated GraphML is compatible with evaluate_coverage.py and
evaluate_triple_f1.py so it can be directly compared against SGE and Baseline.

Usage:
    python3 deterministic_parser_baseline.py \\
        --csv <path> \\
        --type <type-ii|type-iii|auto> \\
        --output <graphml_path> \\
        [--sample-rows 25] \\
        [--sample-seed 42]
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Allow importing Stage 1 modules when run from evaluation/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stage1.features import extract_features, _detect_encoding, _YEAR_PATTERN, _YEAR_STRICT
from stage1.classifier import classify, TYPE_II, TYPE_III

# Sentinel values to skip during parsing
_NULL_SENTINELS = {"", "nan", "...", "na", "n/a", "null", "none", "-"}

# GraphML XML namespace
_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

# Node key definitions for the output GraphML
_NODE_KEYS = [
    ("d0", "entity_id",   "string"),
    ("d1", "entity_type", "string"),
    ("d2", "description", "string"),
]
# Edge key definitions
_EDGE_KEYS = [
    ("d7", "weight",      "double"),
    ("d8", "description", "string"),
    ("d9", "keywords",    "string"),
]


# ---------------------------------------------------------------------------
# CSV reading (reuses Stage 1 encoding detection)
# ---------------------------------------------------------------------------

def _read_csv_full(csv_path: str) -> pd.DataFrame:
    """
    Read the full CSV (no row limit) with encoding detection and skip-row logic.

    Handles UTF-16 tab-separated HK gov files and World Bank metadata headers.
    Returns a DataFrame with string-typed columns (no header=None for UTF-16).
    """
    encoding = _detect_encoding(csv_path)

    if "utf-16" in encoding:
        df = pd.read_csv(csv_path, encoding="utf-16", sep="\t", header=None)
        return df

    # Detect how many leading metadata rows to skip
    skiprows = _detect_skiprows_full(csv_path, encoding)

    for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
        try:
            return pd.read_csv(csv_path, encoding=enc, skiprows=skiprows)
        except (UnicodeDecodeError, Exception):
            continue

    raise ValueError(f"Cannot read CSV: {csv_path}")


def _detect_skiprows_full(csv_path: str, encoding: str) -> int:
    """Detect leading metadata rows to skip (World Bank 4-row, HK title+empty=2)."""
    try:
        with open(csv_path, encoding=encoding, errors="ignore") as fh:
            lines = [fh.readline() for _ in range(2)]
        first_line = lines[0]
        if "Data Source" in first_line or "World Development Indicators" in first_line:
            return 4
        row0_cells = [c.strip().strip('"') for c in first_line.split(",")]
        if len(lines) > 1:
            row1_cells = [c.strip().strip('"') for c in lines[1].split(",")]
            if sum(1 for c in row0_cells if c) <= 1 and sum(1 for c in row1_cells if c) == 0:
                return 2
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Value validation helpers
# ---------------------------------------------------------------------------

def _is_valid_value(val) -> bool:
    """Return True if val is non-null, non-empty, and not a known sentinel."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s not in _NULL_SENTINELS


def _is_numeric_value(val) -> bool:
    """Return True if val can be parsed as a float."""
    if not _is_valid_value(val):
        return False
    try:
        float(str(val).strip().replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_value(val) -> str:
    """Return the original string value, preserving full numeric precision."""
    s = str(val).strip()
    # Remove thousands-separators only if clearly numeric
    try:
        cleaned = s.replace(",", "")
        float(cleaned)
        return cleaned
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Year column detection
# ---------------------------------------------------------------------------

def _identify_year_columns(df: pd.DataFrame) -> list[str]:
    """
    Return column names whose header matches a year or fiscal-year pattern.
    E.g. "2020", "2000", "2022-23", "2022-2023".
    """
    year_cols = []
    for col in df.columns:
        col_str = str(col).replace("\n", " ").strip()
        if _YEAR_PATTERN.search(col_str):
            year_cols.append(col)
    return year_cols


def _identify_key_columns(df: pd.DataFrame, year_cols: list) -> list[str]:
    """
    Return non-year columns that are mostly non-numeric (i.e. entity key columns).
    Stops collecting once a clearly numeric column is encountered.
    """
    year_col_set = set(str(c) for c in year_cols)
    key_cols = []
    for col in df.columns:
        if str(col) in year_col_set:
            break
        # Skip if mostly numeric
        non_null = df[col].dropna()
        if len(non_null) == 0:
            key_cols.append(col)
            continue
        numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
        frac = numeric_count / len(non_null)
        if frac >= 0.6:
            break
        key_cols.append(col)
    return key_cols


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _sample_entities(df: pd.DataFrame, key_cols: list[str],
                     n: int, seed: int) -> pd.DataFrame:
    """
    Randomly sample N unique entities (by the first key column) from the DataFrame.
    Returns a filtered DataFrame preserving all rows for sampled entities.
    """
    if not key_cols or n <= 0:
        return df

    primary_key = key_cols[0]
    unique_vals = df[primary_key].dropna().unique()
    if len(unique_vals) <= n:
        return df

    rng = pd.Series(unique_vals).sample(n=n, random_state=seed)
    sampled = set(rng.values)
    return df[df[primary_key].isin(sampled)].copy()


# ---------------------------------------------------------------------------
# GraphML builder
# ---------------------------------------------------------------------------

class GraphMLBuilder:
    """Builds a GraphML document compatible with evaluate_coverage.py."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}   # node_id -> {entity_type, description}
        self._edges: list[tuple] = []       # (src_id, tgt_id, weight, keywords, description)
        self._edge_set: set[tuple] = set()  # dedup by (src, tgt, keywords)

    def add_node(self, node_id: str, entity_type: str, description: str) -> None:
        """Add or update a node. Later additions win for description."""
        if node_id not in self._nodes:
            self._nodes[node_id] = {
                "entity_type": entity_type,
                "description": description,
            }

    def add_edge(self, src: str, tgt: str, keywords: str,
                 description: str, weight: float = 1.0) -> None:
        """Add a directed edge; skip exact duplicates."""
        dedup_key = (src, tgt, keywords)
        if dedup_key in self._edge_set:
            return
        self._edge_set.add(dedup_key)
        self._edges.append((src, tgt, weight, keywords, description))

    def to_graphml_string(self) -> str:
        """Serialize the graph to a GraphML XML string."""
        root = ET.Element("graphml")
        root.set("xmlns", _GRAPHML_NS)

        # Node key declarations
        for kid, attr_name, attr_type in _NODE_KEYS:
            key_el = ET.SubElement(root, "key")
            key_el.set("id", kid)
            key_el.set("for", "node")
            key_el.set("attr.name", attr_name)
            key_el.set("attr.type", attr_type)

        # Edge key declarations
        for kid, attr_name, attr_type in _EDGE_KEYS:
            key_el = ET.SubElement(root, "key")
            key_el.set("id", kid)
            key_el.set("for", "edge")
            key_el.set("attr.name", attr_name)
            key_el.set("attr.type", attr_type)

        graph_el = ET.SubElement(root, "graph", edgedefault="undirected")

        for node_id, attrs in self._nodes.items():
            node_el = ET.SubElement(graph_el, "node", id=_xml_safe(node_id))
            _data(node_el, "d0", node_id)
            _data(node_el, "d1", attrs["entity_type"])
            _data(node_el, "d2", attrs["description"])

        for i, (src, tgt, weight, keywords, desc) in enumerate(self._edges):
            edge_el = ET.SubElement(graph_el, "edge",
                                    id=f"e{i}",
                                    source=_xml_safe(src),
                                    target=_xml_safe(tgt))
            _data(edge_el, "d7", str(weight))
            _data(edge_el, "d8", desc)
            _data(edge_el, "d9", keywords)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _data(parent: ET.Element, key: str, value: str) -> ET.Element:
    """Append a <data key=...>value</data> child element."""
    el = ET.SubElement(parent, "data", key=key)
    el.text = value
    return el


def _xml_safe(s: str) -> str:
    """Replace characters invalid in XML attribute values / node IDs."""
    # XML node ids cannot start with digits; prefix with 'n_' if needed
    s = s.strip()
    if s and s[0].isdigit():
        s = "n_" + s
    # Replace XML-unsafe characters
    return re.sub(r"[^\w\-.]", "_", s)


# ---------------------------------------------------------------------------
# Type-II parser
# ---------------------------------------------------------------------------

def parse_type_ii(df: pd.DataFrame, csv_path: str) -> GraphMLBuilder:
    """
    Parse a Type-II (Time-Series-Matrix) CSV into a GraphMLBuilder.

    For each (entity, year) pair with a numeric value, creates:
      - Subject node  (entity_type = "subject")
      - StatValue node (entity_type = "StatValue", description includes value+year)
      - Edge: subject --[HAS_VALUE_IN_YEAR]--> stat_value node

    The stat_value node id encodes "{subject}_{year}_{value}" to avoid
    collisions when the same numeric value appears for different entities.
    """
    year_cols = _identify_year_columns(df)
    key_cols = _identify_key_columns(df, year_cols)

    if not year_cols:
        print("WARNING: No year columns detected for Type-II parsing.", file=sys.stderr)
        return GraphMLBuilder()

    builder = GraphMLBuilder()
    csv_name = Path(csv_path).name

    for _, row in df.iterrows():
        subject_name = _build_subject_name(row, key_cols)
        if not subject_name:
            continue

        builder.add_node(
            node_id=subject_name,
            entity_type="subject",
            description=f"Entity from {csv_name}: {subject_name}",
        )

        for year_col in year_cols:
            val = row.get(year_col)
            if not _is_numeric_value(val):
                continue

            year_label = _extract_year_label(str(year_col))
            norm_val = _normalize_value(val)

            # Unique node id per (subject, year) to avoid value collisions
            stat_node_id = f"{subject_name}_{year_label}_{norm_val}"
            stat_desc = (
                f"{subject_name} in {year_label}: {norm_val} "
                f"(source: {csv_name})"
            )
            builder.add_node(
                node_id=stat_node_id,
                entity_type="StatValue",
                description=stat_desc,
            )

            keywords = f"HAS_VALUE_IN_YEAR, {year_label}, {norm_val}"
            edge_desc = (
                f"{subject_name} has value {norm_val} in year {year_label}"
            )
            builder.add_edge(
                src=subject_name,
                tgt=stat_node_id,
                keywords=keywords,
                description=edge_desc,
            )

    return builder


# ---------------------------------------------------------------------------
# Type-III parser
# ---------------------------------------------------------------------------

def parse_type_iii(df: pd.DataFrame, csv_path: str) -> GraphMLBuilder:
    """
    Parse a Type-III (Hierarchical-Hybrid) CSV into a GraphMLBuilder.

    Applies forward-fill on composite key columns to recover sparse hierarchies,
    then for each (composite_key, value_column) pair with a numeric value, creates:
      - Subject node  (entity_type = "subject", name = joined key path)
      - StatValue node (entity_type = "StatValue")
      - Edge: subject --[HAS_VALUE]--> stat_value node

    Two column modes:
      1. Year columns present (e.g. THE Ranking with "2016", "2017"...):
         uses year label as relationship context.
      2. No year columns (e.g. HK Inpatient where columns are category totals):
         treats all numeric columns as value columns using column name as label.
    """
    year_cols = _identify_year_columns(df)
    key_cols = _identify_key_columns(df, year_cols)

    if not key_cols:
        key_cols = [df.columns[0]] if len(df.columns) > 0 else []

    # When no year columns, use all non-key numeric columns as value columns
    if not year_cols:
        key_col_set = set(str(c) for c in key_cols)
        value_cols = [
            col for col in df.columns
            if str(col) not in key_col_set
            and _col_is_mostly_numeric_series(df[col])
        ]
    else:
        value_cols = list(year_cols)

    # Forward-fill sparse composite key columns
    df_filled = _forward_fill_keys(df, key_cols)

    builder = GraphMLBuilder()
    csv_name = Path(csv_path).name

    for _, row in df_filled.iterrows():
        subject_name = _build_subject_name(row, key_cols)
        if not subject_name:
            continue

        builder.add_node(
            node_id=subject_name,
            entity_type="subject",
            description=f"Hierarchical entity from {csv_name}: {subject_name}",
        )

        for val_col in value_cols:
            val = row.get(val_col)
            if not _is_numeric_value(val):
                continue

            norm_val = _normalize_value(val)
            col_label = _extract_year_label(str(val_col))

            stat_node_id = f"{subject_name}_{col_label}_{norm_val}"
            stat_desc = (
                f"{subject_name} {col_label}: {norm_val} "
                f"(source: {csv_name})"
            )
            builder.add_node(
                node_id=stat_node_id,
                entity_type="StatValue",
                description=stat_desc,
            )

            keywords = f"HAS_VALUE, {col_label}, {norm_val}"
            edge_desc = f"{subject_name} has {col_label} value {norm_val}"
            builder.add_edge(
                src=subject_name,
                tgt=stat_node_id,
                keywords=keywords,
                description=edge_desc,
            )

    return builder


def _col_is_mostly_numeric_series(series: pd.Series) -> bool:
    """Return True if >= 60% of non-null values in the series are numeric."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    return (numeric_count / len(non_null)) >= 0.6


def _forward_fill_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    Forward-fill null/empty cells in composite key columns.

    Sparse hierarchical CSVs typically leave higher-level category cells empty
    after the first occurrence (e.g. "Disease A" appears once, rows below are
    blank). Forward-fill restores the full composite key for every data row.
    Returns a new DataFrame (immutable pattern).
    """
    df_copy = df.copy()
    for col in key_cols:
        if col not in df_copy.columns:
            continue
        # Replace sentinel strings with NaN so ffill works
        df_copy[col] = df_copy[col].replace(
            list(_NULL_SENTINELS), pd.NA
        )
        df_copy[col] = df_copy[col].ffill()
    return df_copy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_subject_name(row: pd.Series, key_cols: list) -> str:
    """
    Build a subject name string from one or more key column values.
    Returns empty string if no valid key values found.
    """
    parts = []
    for col in key_cols:
        val = row.get(col)
        if _is_valid_value(val):
            parts.append(str(val).strip())
    return " / ".join(parts) if parts else ""


def _extract_year_label(col_str: str) -> str:
    """
    Extract the year (or fiscal-year) token from a column name string.
    Returns the full match, e.g. "2022-23" or "2020".
    Falls back to the raw column string if no match found.
    """
    col_clean = col_str.replace("\n", " ").strip()
    match = _YEAR_PATTERN.search(col_clean)
    return match.group(0) if match else col_clean


# ---------------------------------------------------------------------------
# Auto-classification
# ---------------------------------------------------------------------------

def auto_classify(csv_path: str) -> str:
    """
    Use Stage 1 feature extraction + classifier to determine the CSV type.
    Returns "type-ii" or "type-iii" (or "type-i" as fallback).
    """
    features = extract_features(csv_path)
    result = classify(features)
    type_map = {
        "Time-Series-Matrix": "type-ii",
        "Hierarchical-Hybrid": "type-iii",
        "Flat-Entity": "type-i",
    }
    return type_map.get(result, "type-ii")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deterministic Parser Baseline: CSV → GraphML without LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--type", default="auto",
                        choices=["type-ii", "type-iii", "auto"],
                        help="Table type (default: auto-detect via Stage 1)")
    parser.add_argument("--output", required=True,
                        help="Output GraphML file path")
    parser.add_argument("--sample-rows", type=int, default=0,
                        help="Randomly sample N entities (0 = use all rows)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for entity sampling (default: 42)")
    return parser.parse_args()


def run(csv_path: str, table_type: str, output_path: str,
        sample_rows: int, sample_seed: int) -> None:
    """
    Core pipeline: read CSV → classify (if auto) → parse → write GraphML.

    Parameters
    ----------
    csv_path    : path to input CSV
    table_type  : "type-ii", "type-iii", or "auto"
    output_path : where to write the GraphML file
    sample_rows : if > 0, sample this many entities before parsing
    sample_seed : random seed for sampling
    """
    print(f"Reading CSV: {csv_path}")
    df = _read_csv_full(csv_path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    # Resolve auto type
    if table_type == "auto":
        table_type = auto_classify(csv_path)
        print(f"  Auto-classified as: {table_type}")
    else:
        print(f"  Using specified type: {table_type}")

    if table_type not in ("type-ii", "type-iii"):
        print(f"WARNING: Type '{table_type}' is not Type-II or Type-III. "
              f"Falling back to Type-II.", file=sys.stderr)
        table_type = "type-ii"

    # Identify key columns for sampling
    year_cols = _identify_year_columns(df)
    key_cols = _identify_key_columns(df, year_cols)

    # Optional entity sampling
    if sample_rows > 0:
        df = _sample_entities(df, key_cols, sample_rows, sample_seed)
        print(f"  Sampled {len(df)} rows ({sample_rows} entities, seed={sample_seed})")

    # Dispatch to the appropriate parser
    if table_type == "type-ii":
        builder = parse_type_ii(df, csv_path)
    else:
        builder = parse_type_iii(df, csv_path)

    node_count = len(builder._nodes)
    edge_count = len(builder._edges)
    print(f"  Extracted {node_count} nodes, {edge_count} edges")

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graphml_str = builder.to_graphml_string()
    out_path.write_text(graphml_str, encoding="utf-8")
    print(f"  GraphML written to: {out_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run(
        csv_path=args.csv,
        table_type=args.type,
        output_path=args.output,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
