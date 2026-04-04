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

from .compact_representation import should_use_compact, compact_serialize_type_ii

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

    # Compact mode: large Type-II tables collapse all year-value pairs into
    # one chunk per entity to prevent StatValue node explosion in LightRAG.
    if should_use_compact(schema, len(df)):
        return compact_serialize_type_ii(df, schema)

    # Align column_roles keys with DataFrame column types.
    # Stage 1 stores column names as strings (e.g. '0'), but headerless
    # CSVs produce integer column names. Convert roles keys to match.
    if column_roles and len(df.columns) > 0:
        sample_col = df.columns[0]
        if isinstance(sample_col, (int, float)) or hasattr(sample_col, "__int__"):
            aligned = {}
            for k, v in column_roles.items():
                try:
                    aligned[int(k)] = v
                except (ValueError, TypeError):
                    aligned[k] = v
            column_roles = aligned

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

    # Detect transposed tables (time in rows, metrics in columns)
    transposed = (schema.get("time_dimension", schema.get("_meta_schema", {}).get("time_dimension", {})).get("location") == "rows")
    value_cols = [c for c, r in column_roles.items() if r == "value"]

    if transposed and not time_value_cols and value_cols:
        # Transposed Type II: rows are metrics, value columns hold yearly data.
        # Find the year-label row: scan first few rows for one with mostly year-like values.
        year_labels = {}  # col_index -> year string
        year_row_idx = None
        import re
        _yr = re.compile(r"^(19|20)\d{2}(\.\d)?$")
        for i in range(min(8, len(df))):
            hits = 0
            for vc in value_cols:
                v = str(df.iloc[i].get(vc, "")).strip()
                if _yr.match(v):
                    hits += 1
            if hits >= 3:
                year_row_idx = i
                for vc in value_cols:
                    v = str(df.iloc[i].get(vc, "")).strip()
                    if _yr.match(v):
                        year_labels[vc] = v.replace(".0", "")
                break

        # Entity name column: first subject column
        name_col = subject_cols[0] if subject_cols else df.columns[0]
        # Secondary name columns for composite keys (e.g. "病床" + "A系列")
        extra_name_cols = subject_cols[1:] if len(subject_cols) > 1 else []

        # Data rows start after the year-label row
        data_start = (year_row_idx + 1) if year_row_idx is not None else 0

        for i in range(data_start, len(df)):
            row = df.iloc[i]
            # Build entity name from subject columns
            name_parts = []
            for nc in [name_col] + extra_name_cols:
                val = row.get(nc, "")
                if _is_valid(val):
                    name_parts.append(str(val).strip())
            entity_name = " / ".join(name_parts) if name_parts else f"Row_{i}"

            # Skip note/annotation rows
            if entity_name.startswith(("注释", "资料来源", "1.", "2.", "3.", "4.")):
                continue
            if not entity_name or entity_name == "nan":
                continue

            lines = [f"Entity: {entity_name}"]
            has_data = False
            for vc in value_cols:
                data_val = row.get(vc, "")
                if not _is_valid(data_val):
                    continue
                try:
                    float(str(data_val))
                except ValueError:
                    continue
                year = year_labels.get(vc, str(vc))
                lines.append(f"Year: {year} | Value: {data_val}")
                has_data = True

            if has_data:
                chunks.append("\n".join(lines))
        return chunks

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

        # Metadata — use column name as label for named columns,
        # fall back to "Remark:" for anonymous/remark columns.
        extra_meta = set(schema.get("extra_metadata_columns", []))
        for col in metadata_cols:
            if col in row and _is_valid(row[col]):
                if col in extra_meta:
                    clean = _clean_col_name(col)
                    lines.append(f"{clean}: {row[col]}")
                else:
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

    Metadata columns whose values look like indicator/metric names (non-numeric,
    no leading annotation markers, reasonable length) are promoted into the
    hierarchy path as additional key levels, so the LLM treats them as entity
    identifiers rather than remarks.
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

    # Separate metadata cols into "promotable" (indicator names) vs true remarks.
    # A metadata column is promotable if:
    #   1. Most of its non-empty values look like indicator names
    #   2. It has a high fill rate (>= 70%) — remarks columns are usually sparse
    promotable_cols: list[str] = []
    remark_cols: list[str] = []
    for col in metadata_cols:
        if col not in df.columns:
            remark_cols.append(col)
            continue
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        total_rows = len(df)
        fill_rate = len(vals) / total_rows if total_rows > 0 else 0
        if len(vals) == 0 or fill_rate < 0.7:
            remark_cols.append(col)
            continue
        indicator_like = vals.apply(_looks_like_indicator)
        if indicator_like.mean() >= 0.5:
            promotable_cols.append(col)
        else:
            remark_cols.append(col)

    # Build effective key columns: original keys + promoted metadata cols
    next_level = len(key_cols)
    promoted_levels: list[tuple[int, str]] = []
    for col in promotable_cols:
        promoted_levels.append((next_level, col))
        next_level += 1

    all_key_cols = key_cols + [col for _, col in promoted_levels]

    # Sparse-fill: track last non-empty value per key column
    last_key_vals: dict[str, str] = {col: "" for col in all_key_cols}

    chunks = []

    for _, row in df.iterrows():
        # Update sparse-fill state
        for col in all_key_cols:
            if col in row and _is_valid(row[col]):
                last_key_vals[col] = str(row[col])

        # Build hierarchy path
        key_parts = []
        level_labels = ["Category", "SubCategory", "Item", "SubItem", "Detail"]
        for i, col in enumerate(all_key_cols):
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

        # Build metadata parts (only true remark columns)
        meta_parts = []
        for col in remark_cols:
            if col in row and _is_valid(row[col]):
                meta_parts.append(f"Remark: {row[col]}")

        # Assemble chunk
        all_parts = [hierarchy_str] + value_parts + meta_parts
        chunk = " | ".join(all_parts)
        chunks.append(chunk)

    return chunks


def _looks_like_indicator(val: str) -> bool:
    """
    Heuristic: does this string look like a metric/indicator name?
    True if: non-numeric, no leading annotation marker (* # †), length 2-80.
    """
    val = val.strip()
    if not val or len(val) < 2 or len(val) > 80:
        return False
    # Pure number or number with unit
    if re.match(r"^[\d,.\-+%]+\s*\S{0,4}$", val):
        return False
    # Annotation markers
    if val[0] in ("*", "#", "†", "‡", "※"):
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_worldbank_format(csv_path: str, enc: str) -> bool:
    """Return True if the file has the World Bank 4-row metadata header."""
    try:
        with open(csv_path, encoding=enc, errors="ignore") as fh:
            first_line = fh.readline()
        return "Data Source" in first_line or "World Development Indicators" in first_line
    except Exception:
        return False


def _detect_skiprows(csv_path: str, enc: str) -> int:
    """
    Detect how many leading rows to skip before the real header.

    Heuristics:
    1. World Bank 4-row metadata (first line contains "Data Source"): skip 4
    2. Title-then-empty pattern (first row has ≤1 non-empty field, second row
       is blank): skip 2 — handles HK government inpatient/hierarchical CSVs
    """
    try:
        with open(csv_path, encoding=enc, errors="ignore") as fh:
            lines = [fh.readline() for _ in range(2)]
        first_line = lines[0]

        # World Bank
        if "Data Source" in first_line or "World Development Indicators" in first_line:
            return 4

        # Title + empty row pattern
        row0_cells = [c.strip().strip('"') for c in first_line.split(",")]
        row0_nonempty = sum(1 for c in row0_cells if c)
        if len(lines) > 1:
            row1_cells = [c.strip().strip('"') for c in lines[1].split(",")]
            row1_nonempty = sum(1 for c in row1_cells if c)
            if row0_nonempty <= 1 and row1_nonempty == 0:
                return 2
    except Exception:
        pass
    return 0


def _read_csv(csv_path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback.

    UTF-16 files (common in HK gov data) are read with header=None and
    sep=tab to match Stage 1 feature extraction, ensuring column names
    are consistent (integer indices) across the pipeline.

    Automatically skips leading title/metadata rows:
    - World Bank API_*.csv: skiprows=4
    - HK government tables with title row + empty row: skiprows=2
    """
    import chardet
    with open(csv_path, "rb") as f:
        raw = f.read(8192)
    det = chardet.detect(raw)
    detected_enc = (det.get("encoding") or "utf-8").lower().replace("-", "")

    if "utf16" in detected_enc:
        return pd.read_csv(csv_path, encoding="utf-16", sep="	", header=None)

    for enc in ("utf-8", "utf-8-sig", "gbk", "big5hkscs", "big5"):
        try:
            skiprows = _detect_skiprows(csv_path, enc)
            return pd.read_csv(csv_path, encoding=enc, header=0, skiprows=skiprows)
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
