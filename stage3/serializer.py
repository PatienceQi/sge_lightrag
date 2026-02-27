"""
serializer.py — CSV row serialization for Stage 3.

Converts CSV rows into natural-language text chunks suitable for LightRAG
ingestion, using the column_roles from the Stage 2 extraction schema.

Supported table types:
  - Type I  (Flat-Entity):        "Entity: X, Attr1: V1, Attr2: V2"
  - Type II (Time-Series-Matrix): one line per entity, time values inline
  - Type III (Hierarchical-Hybrid): hierarchical key path + year values
"""

from __future__ import annotations

import re
import pandas as pd
from pathlib import Path
from typing import Optional

TYPE_I   = "Flat-Entity"
TYPE_II  = "Time-Series-Matrix"
TYPE_III = "Hierarchical-Hybrid"

# Batch size for Type I (multiple rows per chunk)
TYPE_I_BATCH_SIZE = 5


def serialize_csv(csv_path: str, schema: dict) -> list[str]:
    """
    Serialize a CSV file into a list of text chunks using the extraction schema.

    Parameters
    ----------
    csv_path : path to the CSV file
    schema   : Stage 2 extraction schema dict (output of stage2.inducer.induce_schema)

    Returns
    -------
    list[str] — text chunks ready for LightRAG ingestion
    """
    table_type   = schema.get("table_type", TYPE_II)
    column_roles = schema.get("column_roles", {})

    df = _read_csv(csv_path)

    if table_type == TYPE_I:
        return _serialize_type_i(df, column_roles)
    elif table_type == TYPE_II:
        return _serialize_type_ii(df, column_roles, schema)
    elif table_type == TYPE_III:
        return _serialize_type_iii(df, column_roles)
    else:
        # Fallback: treat as Type II
        return _serialize_type_ii(df, column_roles, schema)


# ---------------------------------------------------------------------------
# Type I — Flat-Entity-Attribute
# ---------------------------------------------------------------------------

def _serialize_type_i(df: pd.DataFrame, column_roles: dict) -> list[str]:
    """
    Serialize Type I rows.
    Format: "Entity: X, Attr1: V1, Attr2: V2"
    Batched: TYPE_I_BATCH_SIZE rows per chunk.
    """
    subject_cols = [c for c, r in column_roles.items() if r == "subject"]
    value_cols   = [c for c, r in column_roles.items() if r == "value"]

    chunks = []
    batch  = []

    for _, row in df.iterrows():
        # Build entity identifier
        if subject_cols:
            entity_val = " / ".join(
                str(row[c]) for c in subject_cols
                if c in row and _is_valid(row[c])
            )
        else:
            entity_val = str(row.iloc[0]) if len(row) > 0 else "Unknown"

        if not entity_val or entity_val == "nan":
            continue

        parts = [f"Entity: {entity_val}"]
        for col in value_cols:
            if col in row and _is_valid(row[col]):
                clean_col = _clean_col_name(col)
                parts.append(f"{clean_col}: {row[col]}")

        line = ", ".join(parts)
        batch.append(line)

        if len(batch) >= TYPE_I_BATCH_SIZE:
            chunks.append("\n".join(batch))
            batch = []

    if batch:
        chunks.append("\n".join(batch))

    return chunks


# ---------------------------------------------------------------------------
# Type II — Time-Series-Matrix
# ---------------------------------------------------------------------------

def _serialize_type_ii(df: pd.DataFrame, column_roles: dict, schema: dict) -> list[str]:
    """
    Serialize Type II rows.
    Format: "Entity: X | Year: 2022-23 | Status: 实际 | Value: 91.5"
    One chunk per row (each row = one entity with multiple time values).
    """
    subject_cols   = [c for c, r in column_roles.items() if r == "subject"]
    time_value_cols = [c for c, r in column_roles.items() if r == "time_value"]
    metadata_cols  = [c for c, r in column_roles.items() if r == "metadata"]

    # Parse time headers for richer output
    parsed_headers = {ph["raw"]: ph for ph in schema.get("parsed_time_headers", [])}

    chunks = []

    for _, row in df.iterrows():
        # Entity identifier
        if subject_cols:
            entity_val = " / ".join(
                str(row[c]) for c in subject_cols
                if c in row and _is_valid(row[c])
            )
        else:
            entity_val = str(row.iloc[0]) if len(row) > 0 else "Unknown"

        if not entity_val or entity_val == "nan":
            continue

        lines = [f"Entity: {entity_val}"]

        for col in time_value_cols:
            if col not in row or not _is_valid(row[col]):
                continue

            val = row[col]
            ph  = parsed_headers.get(col)

            if ph:
                year   = ph.get("year", "")
                status = ph.get("status", "")
                unit   = ph.get("unit", "")
                parts  = [f"Year: {year}"]
                if status:
                    parts.append(f"Status: {status}")
                parts.append(f"Value: {val}")
                if unit:
                    parts.append(f"Unit: {unit}")
                lines.append(" | ".join(parts))
            else:
                # Fallback: use raw column name
                clean = _clean_col_name(col)
                lines.append(f"{clean}: {val}")

        # Metadata
        for col in metadata_cols:
            if col in row and _is_valid(row[col]):
                lines.append(f"Remark: {row[col]}")

        chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# Type III — Hierarchical-Hybrid
# ---------------------------------------------------------------------------

def _serialize_type_iii(df: pd.DataFrame, column_roles: dict) -> list[str]:
    """
    Serialize Type III rows with sparse-fill for composite key columns.
    Format: "Category: X > SubCategory: Y > Item: Z | 2021: V1 | 2022: V2 | Remark: ..."
    One chunk per row.
    """
    # Collect key levels in order
    key_levels: list[tuple[int, str]] = []
    for col, role in column_roles.items():
        m = re.match(r"key_level_(\d+)", role)
        if m:
            key_levels.append((int(m.group(1)), col))
    key_levels.sort(key=lambda x: x[0])
    key_cols = [col for _, col in key_levels]

    value_cols    = [c for c, r in column_roles.items() if r == "value"]
    metadata_cols = [c for c, r in column_roles.items() if r == "metadata"]

    # Sparse-fill: track last non-empty value per key column
    last_key_vals: dict[str, str] = {col: "" for col in key_cols}

    chunks = []

    for _, row in df.iterrows():
        # Update sparse-fill state
        for col in key_cols:
            if col in row and _is_valid(row[col]):
                last_key_vals[col] = str(row[col])

        # Build hierarchy path
        key_parts = []
        level_labels = ["Category", "SubCategory", "Item", "SubItem"]
        for i, col in enumerate(key_cols):
            val = last_key_vals.get(col, "")
            if val:
                label = level_labels[i] if i < len(level_labels) else f"Level{i}"
                key_parts.append(f"{label}: {val}")

        if not key_parts:
            continue

        hierarchy_str = " > ".join(key_parts)

        # Build value parts
        value_parts = []
        for col in value_cols:
            if col in row and _is_valid(row[col]):
                clean = _clean_col_name(col)
                value_parts.append(f"{clean}: {row[col]}")

        # Build metadata parts
        meta_parts = []
        for col in metadata_cols:
            if col in row and _is_valid(row[col]):
                meta_parts.append(f"Remark: {row[col]}")

        # Assemble chunk
        all_parts = [hierarchy_str] + value_parts + meta_parts
        chunk = " | ".join(all_parts)
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(csv_path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback."""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk", "big5"):
        try:
            return pd.read_csv(csv_path, encoding=enc, header=0)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Cannot read CSV: {csv_path}")


def _is_valid(val) -> bool:
    """Return True if val is non-null and non-empty."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() != "nan"


def _clean_col_name(col: str) -> str:
    """Strip newlines and extra whitespace from a column name."""
    return re.sub(r"\s+", " ", str(col).replace("\n", " ").replace("\r", "")).strip()
