"""
classifier.py — Rule-based topological type classifier.

Decision logic (no LLM):

  Priority 1 — Hierarchical-Hybrid (Type III)
      Strong composite-key signal overrides a weak time-header signal.
      Condition: leading_text_col_count >= 2 AND the year columns in headers
      are few (≤ 5) relative to the total columns — meaning years are just
      value columns, not the primary structural axis.

  Priority 2 — Time-Series-Matrix (Type II)
      Time dimension found in headers, first column, or data body.

  Priority 3 — Flat-Entity-Attribute (Type I)
      Default fallback.
"""

from .features import FeatureSet

# Human-readable type labels
TYPE_I   = "Flat-Entity"
TYPE_II  = "Time-Series-Matrix"
TYPE_III = "Hierarchical-Hybrid"

# If the number of year-header columns is ≤ this threshold AND a composite
# key is present, we treat the table as Hierarchical-Hybrid (Type III).
# Rationale: a table with 2021/2022/2023/2024 as value columns + 3 text key
# columns is structurally hierarchical, not a time-series matrix.
_MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE = 6


def classify(features: FeatureSet) -> str:
    """
    Apply heuristic rules to a FeatureSet and return one of the three type strings.

    Rules (evaluated in priority order):
    ─────────────────────────────────────────────────────────────────────────────
    Rule 1 — Hierarchical-Hybrid (Type III)  [checked first to avoid false Type II]
        Triggered when:
        • leading_text_col_count >= 2  (composite key present)
        • AND the year columns in headers are few (≤ _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE)
          — meaning years are just data columns, not the structural time axis

    Rule 2 — Time-Series-Matrix (Type II)
        Triggered when:
        • ≥1 header column name matches a year/fiscal-year pattern, OR
        • The first column's values are mostly year numbers (transposed layout), OR
        • A data body row contains ≥3 year-like values (headerless transposed file)

    Rule 3 — Flat-Entity-Attribute (Type I)
        Default fallback: simple entity table, one row = one entity.
    ─────────────────────────────────────────────────────────────────────────────
    """
    n_time_headers = len(features.time_cols_in_headers)
    has_composite_key = features.leading_text_col_count >= 2

    # Rule 1: composite key takes priority when year columns are just value columns.
    # Requires numeric columns to be present — a table with only text columns is
    # a flat entity registry (Type I), not a hierarchical hybrid (Type III).
    has_numeric_values = features.n_numeric_cols > 0
    if has_composite_key and has_numeric_values and n_time_headers <= _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE:
        # But only if there's no strong time-series signal beyond the header years
        # (i.e. no transposed year rows, no fiscal-year compound headers)
        has_fiscal_year_headers = any(
            "-" in col for col in features.time_cols_in_headers
        )
        if not features.time_cols_in_first_col and not features.time_in_data_body and not has_fiscal_year_headers:
            return TYPE_III

    # Rule 2: time dimension present
    if (features.time_cols_in_headers
            or features.time_cols_in_first_col
            or features.time_in_data_body):
        return TYPE_II

    # Rule 3: default
    return TYPE_I
