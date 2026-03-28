"""
features.py — Feature extraction from a CSV file.

Reads the CSV (with automatic encoding detection) and computes a
FeatureSet that the classifier and schema builder consume.

Dependencies: standard library + pandas only (no chardet).
"""

import re
import pandas as pd
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Regex patterns for time-dimension detection
# ---------------------------------------------------------------------------
# Matches bare 4-digit years (2020) or fiscal-year ranges (2022-23, 2022-2023)
_YEAR_PATTERN = re.compile(r"\b\d{4}(?:-\d{2,4})?\b")

# Strict match: the entire string is a year or fiscal-year token
_YEAR_STRICT = re.compile(r"^\d{4}(?:-\d{2,4})?$")

# Remarks / notes column name keywords (Chinese + English)
_REMARKS_KEYWORDS = re.compile(
    r"备注|注释|说明|remark|note|comment", re.IGNORECASE
)

# Maximum number of rows to sample for feature extraction
_SAMPLE_ROWS = 20


@dataclass
class FeatureSet:
    """All heuristic signals extracted from a single CSV file."""

    # Raw data (first N rows)
    df: pd.DataFrame

    # Column names as strings (newlines stripped for matching)
    header_strings: list[str]

    # --- Time-dimension signals ---
    time_cols_in_headers: list[str]   # header names that match year pattern
    time_cols_in_first_col: bool      # first column values look like years (transposed)
    time_in_data_body: bool           # a data row is entirely year-like values (headerless)

    # --- Column-type counts ---
    n_text_cols: int                  # columns whose values are mostly text
    n_numeric_cols: int               # columns whose values are mostly numeric

    # --- Composite-key signals ---
    leading_text_col_count: int       # consecutive text cols before first numeric col

    # --- Metadata signals ---
    remarks_cols: list[str]           # columns identified as remarks/notes

    # --- Raw column names (original, with newlines) ---
    raw_columns: list[str]

    # --- Whether the file had no real header (auto-indexed columns) ---
    headerless: bool

    # --- Actual data row count (exact for small CSVs, min(_SAMPLE_ROWS+5) for large) ---
    n_rows: int = 0


def _detect_encoding(path: str) -> str:
    """
    Detect file encoding using BOM sniffing (stdlib only, no chardet).

    BOM signatures checked:
      UTF-16 LE  : FF FE
      UTF-16 BE  : FE FF
      UTF-8 BOM  : EF BB BF

    Falls back to 'utf-8-sig' for plain UTF-8 / GBK files.
    """
    with open(path, "rb") as fh:
        bom = fh.read(4)

    if bom[:2] == b"\xff\xfe":
        return "utf-16-le"
    if bom[:2] == b"\xfe\xff":
        return "utf-16-be"
    if bom[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"

    # Try decoding a sample as UTF-8; fall back to GBK or Big5 for legacy Chinese files
    with open(path, "rb") as fh:
        sample = fh.read(4096)
    try:
        sample.decode("utf-8")
        return "utf-8-sig"   # utf-8-sig also handles plain utf-8
    except UnicodeDecodeError:
        pass

    # Try GBK (Simplified Chinese)
    try:
        sample.decode("gbk")
        return "gbk"
    except UnicodeDecodeError:
        pass

    # Fall back to Big5HKSCS (Traditional Chinese, used in HK government files)
    return "big5hkscs"


def _detect_skiprows(path: str, encoding: str) -> int:
    """
    Detect how many leading rows to skip before the real header.

    Heuristics:
    1. World Bank 4-row metadata (first line contains "Data Source"): skip 4
    2. Title-then-empty pattern (first row has ≤1 non-empty field, second row
       is blank): skip 2 — handles HK government inpatient/hierarchical CSVs
    """
    try:
        with open(path, encoding=encoding, errors="ignore") as fh:
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


def _read_csv(path: str) -> tuple[pd.DataFrame, bool]:
    """
    Try to read the CSV with automatic encoding + separator detection.
    Returns (DataFrame, headerless).

    headerless=True means the file had no real header row (e.g. UTF-16 tab files
    from HK government that start with a title row, not column names).

    Leading metadata/title rows are automatically detected and skipped:
    - World Bank API_*.csv: skip 4 rows
    - HK government tables with title row + empty row: skip 2 rows
    """
    encoding = _detect_encoding(path)

    # UTF-16 files from Hong Kong gov are tab-separated and have no real header
    if "utf-16" in encoding:
        df = pd.read_csv(path, encoding="utf-16", sep="\t",
                         header=None, nrows=_SAMPLE_ROWS + 5)
        return df, True

    skiprows = _detect_skiprows(path, encoding)

    # Standard comma-separated — try multiple encodings
    for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
        try:
            df = pd.read_csv(path, encoding=enc, skiprows=skiprows,
                             nrows=_SAMPLE_ROWS + 5)
            return df, False
        except (UnicodeDecodeError, Exception):
            continue

    raise ValueError(f"Cannot read CSV: {path}")


def _col_is_mostly_numeric(series: pd.Series) -> bool:
    """Return True if ≥60 % of non-null values in the series are numeric."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    return (numeric_count / len(non_null)) >= 0.6


def _col_has_long_text(series: pd.Series) -> bool:
    """Return True if the column contains at least one value longer than 20 chars."""
    non_null = series.dropna().astype(str)
    return any(len(v) > 20 for v in non_null)


def _header_matches_year(col_name: str) -> bool:
    """Return True if the column name contains a year or fiscal-year pattern."""
    return bool(_YEAR_PATTERN.search(str(col_name)))


def _first_col_has_years(df: pd.DataFrame) -> bool:
    """
    Return True if the first column's values look like years.
    Used to detect transposed time-series tables where years appear as row labels.
    Requires at least 2 strict year matches.
    """
    if df.empty or df.shape[1] == 0:
        return False
    first_col = df.iloc[:, 0].dropna().astype(str)
    year_hits = sum(1 for v in first_col if _YEAR_STRICT.match(v.strip()))
    return year_hits >= 2


def _data_body_has_year_row(df: pd.DataFrame) -> bool:
    """
    Detect transposed tables where a data row (not the header) contains
    mostly year values. This handles headerless files (e.g. UTF-16 HK gov
    tables) where years appear in one of the first few rows.

    A row qualifies if it has ≥3 cells that look like years (int or float
    representation of a 4-digit year in range 1900-2100).
    """
    year_range = re.compile(r"^(19|20)\d{2}(\.0)?$")
    for i in range(min(6, len(df))):
        row_vals = [str(v).strip() for v in df.iloc[i] if str(v).strip() not in ("nan", "")]
        year_hits = sum(1 for v in row_vals if year_range.match(v))
        if year_hits >= 3:
            return True
    return False


def extract_features(path: str) -> FeatureSet:
    """
    Main entry point: read the CSV at *path* and return a FeatureSet.
    """
    df, headerless = _read_csv(path)

    raw_columns = [str(c) for c in df.columns]
    # Flatten multi-line headers for pattern matching
    header_strings = [c.replace("\n", " ").strip() for c in raw_columns]

    # --- Time columns in headers ---
    # Only count headers that look like years, not auto-generated integer indices
    time_cols_in_headers: list[str] = []
    if not headerless:
        time_cols_in_headers = [
            raw_columns[i]
            for i, h in enumerate(header_strings)
            if _header_matches_year(h)
        ]

    # --- Time in first column (transposed detection) ---
    time_in_first_col = _first_col_has_years(df) if not headerless else False

    # --- Time in data body (headerless transposed tables) ---
    time_in_data_body = _data_body_has_year_row(df) if headerless else False

    # --- Column type counts ---
    n_text = 0
    n_numeric = 0
    for col in df.columns:
        if _col_is_mostly_numeric(df[col]):
            n_numeric += 1
        else:
            n_text += 1

    # --- Leading text columns (composite-key signal) ---
    leading_text = 0
    for col in df.columns:
        if _col_is_mostly_numeric(df[col]):
            break
        leading_text += 1

    # --- Remarks columns ---
    remarks_cols: list[str] = []
    for i, col in enumerate(df.columns):
        name = header_strings[i]
        if _REMARKS_KEYWORDS.search(name):
            remarks_cols.append(raw_columns[i])
        elif not _col_is_mostly_numeric(df[col]) and _col_has_long_text(df[col]):
            remarks_cols.append(raw_columns[i])

    return FeatureSet(
        df=df,
        header_strings=header_strings,
        time_cols_in_headers=time_cols_in_headers,
        time_cols_in_first_col=time_in_first_col,
        time_in_data_body=time_in_data_body,
        n_text_cols=n_text,
        n_numeric_cols=n_numeric,
        leading_text_col_count=leading_text,
        remarks_cols=remarks_cols,
        raw_columns=raw_columns,
        headerless=headerless,
        n_rows=len(df),
    )
