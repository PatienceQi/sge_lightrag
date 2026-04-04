"""
classifier.py — Rule-based topological type classifier.

Decision logic (no LLM):

  Priority 1 — Hierarchical-Hybrid (Type III)
      Composite key (≥2 leading text columns) is the primary discriminator.
      Guard conditions exclude transposed, fiscal-year, and body-year
      layouts that are genuinely time-series despite having composite keys.

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


def classify(features: FeatureSet) -> str:
    """
    Apply heuristic rules to a FeatureSet and return one of the three type strings.

    Rules (evaluated in priority order):
    ─────────────────────────────────────────────────────────────────────────────
    Rule 1 — Hierarchical-Hybrid (Type III)  [checked first to avoid false Type II]
        Triggered when:
        • leading_text_col_count >= 2  (composite key present)
        • AND numeric value columns exist
        • AND no strong time-series counter-signal (transposed / fiscal / body-year)

    Rule 2 — Time-Series-Matrix (Type II)
        Triggered when:
        • ≥1 header column name matches a year/fiscal-year pattern, OR
        • The first column's values are mostly year numbers (transposed layout), OR
        • A data body row contains ≥3 year-like values (headerless transposed file)

    Rule 3 — Flat-Entity-Attribute (Type I)
        Default fallback: simple entity table, one row = one entity.
    ─────────────────────────────────────────────────────────────────────────────
    """
    has_composite_key = features.leading_text_col_count >= 2

    # Rule 1: composite key takes priority UNLESS the key is shallow (exactly 2)
    # AND year-pattern headers indicate a structural time axis.  With ≥3 key
    # columns the hierarchy signal is strong enough to override year headers
    # (e.g. Food Safety: 数据内容 > 内容分类 > 项目 + year value columns).
    # With exactly 2 key columns + year headers, the second column is likely
    # metadata (e.g. Country) and the table is a time-series.
    has_numeric_values = features.n_numeric_cols > 0
    has_year_headers = bool(features.time_cols_in_headers)
    if has_composite_key and has_numeric_values:
        has_fiscal_year_headers = any(
            "-" in col for col in features.time_cols_in_headers
        )
        deep_hierarchy = features.leading_text_col_count >= 3
        # Many year-pattern headers (>6) indicate a structural time axis even
        # with deep composite keys (e.g. World Bank CSVs have 4 redundant
        # identifier columns + 60+ year columns — clearly time-series).
        few_time_cols = len(features.time_cols_in_headers) <= 6
        if (not features.time_cols_in_first_col
                and not features.time_in_data_body
                and not has_fiscal_year_headers
                and (deep_hierarchy or not has_year_headers)
                and few_time_cols):
            return TYPE_III

    # Rule 2: time dimension present
    if (features.time_cols_in_headers
            or features.time_cols_in_first_col
            or features.time_in_data_body):
        return TYPE_II

    # Rule 3: default
    return TYPE_I
