"""
preprocessor.py — CSV Preprocessor for Stage 1 pipeline.

Handles:
1. Encoding detection (UTF-16 LE/BE, UTF-8, GBK, Big5HKSCS)
2. UTF-16 tab-separated → clean UTF-8 CSV conversion
3. Title/subtitle row stripping (rows before the actual data header)
4. Merged cell / empty row handling (forward-fill merged cells)
5. Outputs a clean pandas DataFrame ready for Stage 1 feature extraction

Usage:
    from preprocessor import preprocess_csv
    df, metadata = preprocess_csv("path/to/file.csv")
"""

import re
import io
import pandas as pd
from pathlib import Path

# Reuse encoding detection and skiprows detection from stage1.features
from stage1.features import _detect_encoding, _detect_skiprows


# ---------------------------------------------------------------------------
# Heuristics for detecting the "real" header row
# ---------------------------------------------------------------------------

# A row is likely a title/subtitle if it has very few non-null cells
_MIN_HEADER_FILL_RATIO = 0.3  # at least 30% of cells must be non-null for a real header

# Patterns that indicate a title row (not a data header)
_TITLE_PATTERNS = re.compile(
    r"^表\s*\d|^Table\s+\d|^图\s*\d|^Figure\s+\d|^\d{4}年|^\d{4}/\d{2}",
    re.IGNORECASE,
)


def _is_title_row(row: pd.Series, n_cols: int) -> bool:
    """
    Return True if this row looks like a title/subtitle rather than a data header.

    Criteria:
    - Very sparse (< 30% non-null cells), OR
    - First non-null cell matches a title pattern
    """
    non_null = row.dropna()
    if len(non_null) == 0:
        return True  # blank row

    fill_ratio = len(non_null) / max(n_cols, 1)
    if fill_ratio < _MIN_HEADER_FILL_RATIO:
        # Sparse row — check if first value looks like a title
        first_val = str(non_null.iloc[0]).strip()
        if _TITLE_PATTERNS.match(first_val):
            return True
        # Also treat as title if it's a single cell spanning the whole row
        if len(non_null) == 1:
            return True

    return False


def _find_header_row(df: pd.DataFrame) -> int:
    """
    Scan the first 10 rows to find the index of the actual column header row.

    Returns the 0-based row index of the header, or 0 if not found.
    """
    n_cols = df.shape[1]
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        if not _is_title_row(row, n_cols):
            return i
    return 0


def _forward_fill_merged_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill NaN values in text columns to handle merged cells.

    Only fills columns that are predominantly text (not numeric), since
    numeric NaN usually means missing data, not a merged cell.
    """
    df = df.copy()
    for col in df.columns:
        # Check if column is mostly text
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
        is_mostly_text = (numeric_count / len(non_null)) < 0.4
        if is_mostly_text:
            df[col] = df[col].ffill()
    return df


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where all values are NaN or empty strings."""
    mask = df.apply(
        lambda row: all(
            pd.isna(v) or str(v).strip() == "" for v in row
        ),
        axis=1,
    )
    return df[~mask].reset_index(drop=True)


def preprocess_csv(
    path: str,
    strip_titles: bool = True,
    fill_merged: bool = True,
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and clean a CSV file, returning a DataFrame ready for Stage 1.

    Parameters
    ----------
    path         : Path to the CSV file
    strip_titles : Strip title/subtitle rows before the real header
    fill_merged  : Forward-fill merged cells in text columns
    drop_empty   : Drop fully-empty rows

    Returns
    -------
    (df, metadata) where metadata contains:
        - original_encoding: detected encoding
        - was_utf16: whether the file was UTF-16 tab-separated
        - header_row_index: which row was used as the header (0-based in raw file)
        - rows_stripped: number of title rows removed
        - original_shape: (rows, cols) before cleaning
        - clean_shape: (rows, cols) after cleaning
    """
    path = str(path)
    encoding = _detect_encoding(path)
    was_utf16 = "utf-16" in encoding

    metadata = {
        "original_encoding": encoding,
        "was_utf16": was_utf16,
        "header_row_index": 0,
        "rows_stripped": 0,
        "original_shape": None,
        "clean_shape": None,
    }

    # ── Step 1: Read raw file ─────────────────────────────────────────────────
    if was_utf16:
        # UTF-16 tab-separated: read without header first to inspect structure
        raw_df = pd.read_csv(path, encoding="utf-16", sep="\t", header=None)
    else:
        # Detect and skip leading metadata rows (e.g., World Bank 4-row header)
        skiprows = _detect_skiprows(path, encoding)
        # Try encodings in order
        raw_df = None
        for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
            try:
                raw_df = pd.read_csv(path, encoding=enc, header=None,
                                     skiprows=skiprows)
                break
            except (UnicodeDecodeError, Exception):
                continue
        if raw_df is None:
            raise ValueError(f"Cannot read CSV: {path}")

    metadata["original_shape"] = raw_df.shape

    # ── Step 2: Find and strip title rows ─────────────────────────────────────
    header_row = 0
    if strip_titles:
        header_row = _find_header_row(raw_df)
        metadata["header_row_index"] = header_row
        metadata["rows_stripped"] = header_row

    # Promote the detected header row to column names
    if header_row > 0:
        new_columns = raw_df.iloc[header_row].tolist()
        # Clean column names: strip whitespace, replace NaN with positional index
        cleaned_cols = []
        for i, c in enumerate(new_columns):
            if pd.isna(c) or str(c).strip() == "":
                cleaned_cols.append(f"_col_{i}")
            else:
                cleaned_cols.append(str(c).strip())
        df = raw_df.iloc[header_row + 1:].copy()
        df.columns = cleaned_cols
        df = df.reset_index(drop=True)
    else:
        # Use first row as header (standard CSV)
        df = raw_df.copy()
        # If headerless (UTF-16), keep integer column indices
        if not was_utf16:
            # Re-read with proper header (applying same skiprows as initial read)
            for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
                try:
                    df = pd.read_csv(path, encoding=enc, skiprows=skiprows)
                    break
                except (UnicodeDecodeError, Exception):
                    continue

    # ── Step 3: Drop fully-empty rows ─────────────────────────────────────────
    if drop_empty:
        df = _drop_empty_rows(df)

    # ── Step 4: Forward-fill merged cells ─────────────────────────────────────
    if fill_merged:
        df = _forward_fill_merged_cells(df)

    metadata["clean_shape"] = df.shape
    return df, metadata


def preprocess_to_tempfile(path: str, **kwargs) -> tuple[str, dict]:
    """
    Preprocess a CSV and write the result to a temporary UTF-8 CSV file.

    Returns (temp_path, metadata). The caller is responsible for deleting
    the temp file when done.
    """
    import tempfile
    import os

    df, metadata = preprocess_csv(path, **kwargs)

    suffix = Path(path).suffix or ".csv"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="sge_preprocessed_")
    os.close(fd)

    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    metadata["temp_path"] = temp_path
    return temp_path, metadata


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    df, meta = preprocess_csv(csv_path)

    print(f"Encoding  : {meta['original_encoding']}")
    print(f"UTF-16    : {meta['was_utf16']}")
    print(f"Header row: {meta['header_row_index']} ({meta['rows_stripped']} title rows stripped)")
    print(f"Shape     : {meta['original_shape']} → {meta['clean_shape']}")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(max_cols=6, max_colwidth=30))
