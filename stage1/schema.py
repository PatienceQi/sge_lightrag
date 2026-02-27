"""
schema.py — Meta-Schema JSON generation.

Given a FeatureSet and the classified table type, builds the structured
Meta-Schema that downstream stages (Stage 2 / Stage 3) will consume.
"""

from .features import FeatureSet
from .classifier import TYPE_I, TYPE_II, TYPE_III


def build_meta_schema(features: FeatureSet, table_type: str) -> dict:
    """
    Construct the Meta-Schema dictionary for a classified table.

    Parameters
    ----------
    features   : FeatureSet produced by features.extract_features()
    table_type : one of TYPE_I / TYPE_II / TYPE_III from classifier.classify()

    Returns
    -------
    dict matching the Meta-Schema JSON format defined in the project spec.
    """
    df = features.df
    raw_cols = features.raw_columns

    # ── Value columns: mostly-numeric, not a time-header column ──────────────
    from .features import _col_is_mostly_numeric  # reuse helper
    value_cols = [
        raw_cols[i]
        for i, col in enumerate(df.columns)
        if _col_is_mostly_numeric(df[col])
    ]

    # ── Primary subject columns ───────────────────────────────────────────────
    # Text columns that are NOT remarks and NOT time-header columns
    time_col_set = set(features.time_cols_in_headers)
    remarks_set  = set(features.remarks_cols)

    primary_subject_cols = [
        raw_cols[i]
        for i, col in enumerate(df.columns)
        if (not _col_is_mostly_numeric(df[col])
            and raw_cols[i] not in time_col_set
            and raw_cols[i] not in remarks_set)
    ]

    # ── Time dimension ────────────────────────────────────────────────────────
    if table_type == TYPE_III:
        # For Hierarchical-Hybrid tables, year-named columns are value columns,
        # not a structural time axis — suppress the time dimension.
        time_info = {
            "is_present": False,
            "location": None,
            "columns": [],
        }
    elif features.time_cols_in_headers:
        time_info = {
            "is_present": True,
            "location": "headers",
            "columns": features.time_cols_in_headers,
        }
    elif features.time_cols_in_first_col:
        # Transposed: years are in the first column's values
        first_col_name = raw_cols[0] if raw_cols else None
        time_info = {
            "is_present": True,
            "location": "rows",
            "columns": [first_col_name] if first_col_name else [],
        }
    elif features.time_in_data_body:
        # Headerless transposed file: years appear in a data row, not a named column
        time_info = {
            "is_present": True,
            "location": "rows",
            "columns": [],  # column positions are unnamed in headerless files
        }
    else:
        time_info = {
            "is_present": False,
            "location": None,
            "columns": [],
        }

    # ── Composite key flag ────────────────────────────────────────────────────
    composite_key = features.leading_text_col_count >= 2

    # ── Extraction rule summary ───────────────────────────────────────────────
    summary = _build_summary(table_type, primary_subject_cols, time_info, composite_key)

    return {
        "table_type": table_type,
        "primary_subject_columns": primary_subject_cols,
        "time_dimension": time_info,
        "value_columns": value_cols,
        "metadata_columns": list(remarks_set),
        "composite_key": composite_key,
        "extraction_rule_summary": summary,
    }


def _build_summary(
    table_type: str,
    subject_cols: list[str],
    time_info: dict,
    composite_key: bool,
) -> str:
    """Generate a one-sentence human-readable extraction rule."""
    subj = ", ".join(f'"{c}"' for c in subject_cols[:3]) if subject_cols else "unknown"

    if table_type == TYPE_II:
        loc = time_info.get("location", "headers")
        time_cols = time_info.get("columns", [])
        n_time = len(time_cols)
        if loc == "headers":
            return (
                f"Each row is an entity identified by {subj}; "
                f"{n_time} time-period column(s) in the header hold numeric values."
            )
        else:
            return (
                f"Table is transposed: rows represent time periods (years in first column); "
                f"columns represent metrics identified by {subj}."
            )

    if table_type == TYPE_III:
        return (
            f"Each row is identified by a composite key across {subj}; "
            "numeric values follow the key columns; remarks column may contain annotations."
        )

    # TYPE_I
    return (
        f"Each row is a distinct entity described by {subj} and associated numeric attributes."
    )
