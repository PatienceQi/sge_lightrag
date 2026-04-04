"""
inducer.py — Main Stage 2 rule-based schema induction logic.

Takes Stage 1 outputs (meta_schema dict + features FeatureSet) and produces
a full extraction schema without requiring an LLM.

Public API:
    induce_schema(meta_schema, features) -> dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .header_parser import parse_all_headers, ParsedHeader
from .templates import (
    derive_entity_name,
    detect_relation_name,
    type_i_templates,
    type_ii_templates,
    type_iii_templates,
)

if TYPE_CHECKING:
    from stage1.features import FeatureSet

# Table type constants (mirrors stage1.classifier)
TYPE_I   = "Flat-Entity"
TYPE_II  = "Time-Series-Matrix"
TYPE_III = "Hierarchical-Hybrid"


def induce_schema(meta_schema: dict, features: "FeatureSet") -> dict:
    """
    Produce a full extraction schema from Stage 1 outputs.

    Parameters
    ----------
    meta_schema : dict
        Output of stage1.schema.build_meta_schema()
    features : FeatureSet
        Output of stage1.features.extract_features()

    Returns
    -------
    dict with keys:
        table_type, entity_types, relation_types, column_roles,
        extraction_constraints, entity_extraction_template,
        relation_extraction_template, parsed_time_headers (bonus)
    """
    table_type = meta_schema["table_type"]

    if table_type == TYPE_I:
        return _induce_type_i(meta_schema, features)
    elif table_type == TYPE_II:
        return _induce_type_ii(meta_schema, features)
    elif table_type == TYPE_III:
        return _induce_type_iii(meta_schema, features)
    else:
        raise ValueError(f"Unknown table_type: {table_type!r}")


# ---------------------------------------------------------------------------
# Type I — Flat-Entity-Attribute
# ---------------------------------------------------------------------------

def _induce_type_i(meta_schema: dict, features: "FeatureSet") -> dict:
    subject_cols = meta_schema["primary_subject_columns"]
    value_cols   = meta_schema["value_columns"]
    remarks_cols = meta_schema["metadata_columns"]

    # Primary subject: first non-numeric, non-remarks column
    subject_col = subject_cols[0] if subject_cols else (features.raw_columns[0] if features.raw_columns else "entity")
    entity_name = derive_entity_name(subject_col)

    entity_types   = [entity_name]
    relation_types = ["HAS_ATTRIBUTE"]

    # Column roles
    column_roles = _assign_column_roles_flat(
        raw_columns=features.raw_columns,
        subject_cols=subject_cols,
        value_cols=value_cols,
        remarks_cols=remarks_cols,
        time_cols=[],
    )

    templates = type_i_templates(subject_col, value_cols, entity_name)

    return {
        "table_type": TYPE_I,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "column_roles": column_roles,
        **templates,
        "parsed_time_headers": [],
    }


# ---------------------------------------------------------------------------
# Type II — Time-Series-Matrix
# ---------------------------------------------------------------------------

def _induce_type_ii(meta_schema: dict, features: "FeatureSet") -> dict:
    subject_cols = meta_schema["primary_subject_columns"]
    value_cols   = meta_schema["value_columns"]
    remarks_cols = list(meta_schema["metadata_columns"])  # copy to avoid mutation
    time_info    = meta_schema["time_dimension"]

    transposed = (time_info.get("location") == "rows")

    # When a Type-II table has multiple subject columns (e.g. University + Country),
    # only the first is the true entity identifier; the rest are metadata attributes.
    extra_subject_as_metadata: list[str] = []
    if len(subject_cols) > 1 and not transposed:
        extra_subject_as_metadata = subject_cols[1:]
        subject_cols = subject_cols[:1]

    # Parse all column headers for compound time info
    parsed_headers = parse_all_headers(features.raw_columns)
    time_parsed = [ph for ph in parsed_headers if ph.is_time_column]

    # Determine time column names
    if transposed:
        # Transposed: years are row values (first col); metric names are column headers (non-first cols).
        # Pass all non-first column names as "time_col_names" so templates can show metric examples.
        time_col_names: list[str] = features.raw_columns[1:] if len(features.raw_columns) > 1 else []
        subject_col = features.raw_columns[1] if len(features.raw_columns) > 1 else "metric"
    else:
        time_col_names = [ph.raw for ph in time_parsed]
        subject_col = subject_cols[0] if subject_cols else (features.raw_columns[0] if features.raw_columns else "subject")

    entity_name   = derive_entity_name(subject_col)
    relation_name = detect_relation_name(time_col_names, value_cols)

    entity_types   = [entity_name]
    relation_types = [relation_name]

    # Column roles — demoted subject columns become metadata
    all_remarks = remarks_cols + extra_subject_as_metadata
    column_roles = _assign_column_roles_flat(
        raw_columns=features.raw_columns,
        subject_cols=subject_cols,
        value_cols=value_cols,
        remarks_cols=all_remarks,
        time_cols=time_col_names,
    )

    templates = type_ii_templates(
        subject_col=subject_col,
        time_cols=time_col_names,
        parsed_headers=time_parsed,
        entity_name=entity_name,
        relation_name=relation_name,
        transposed=transposed,
    )

    return {
        "table_type": TYPE_II,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "column_roles": column_roles,
        "extra_metadata_columns": extra_subject_as_metadata,
        **templates,
        "parsed_time_headers": [ph.to_dict() for ph in time_parsed],
    }


# ---------------------------------------------------------------------------
# Type III — Hierarchical-Hybrid
# ---------------------------------------------------------------------------

def _induce_type_iii(meta_schema: dict, features: "FeatureSet") -> dict:
    subject_cols = meta_schema["primary_subject_columns"]
    value_cols   = meta_schema["value_columns"]
    remarks_cols = meta_schema["metadata_columns"]

    # Key columns = all leading text columns (composite key)
    key_cols = subject_cols if subject_cols else features.raw_columns[:2]

    # Derive entity names for each hierarchy level
    entity_names = [derive_entity_name(c) for c in key_cols]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_entity_names: list[str] = []
    for name in entity_names:
        if name not in seen:
            seen.add(name)
            unique_entity_names.append(name)

    relation_types = ["HAS_SUB_ITEM", "HAS_VALUE"]
    if remarks_cols:
        relation_types.append("HAS_METADATA")

    # Column roles
    column_roles: dict[str, str] = {}
    for col in features.raw_columns:
        if col in key_cols:
            level = key_cols.index(col)
            column_roles[col] = "key_level_0" if level == 0 else f"key_level_{level}"
        elif col in value_cols:
            column_roles[col] = "value"
        elif col in remarks_cols:
            column_roles[col] = "metadata"
        else:
            column_roles[col] = "unknown"

    templates = type_iii_templates(
        key_cols=key_cols,
        value_cols=value_cols,
        remarks_cols=remarks_cols,
        entity_names=unique_entity_names,
    )

    return {
        "table_type": TYPE_III,
        "entity_types": unique_entity_names,
        "relation_types": relation_types,
        "column_roles": column_roles,
        **templates,
        "parsed_time_headers": [],
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assign_column_roles_flat(
    raw_columns: list[str],
    subject_cols: list[str],
    value_cols: list[str],
    remarks_cols: list[str],
    time_cols: list[str],
) -> dict[str, str]:
    """
    Assign a role string to every column.

    Roles: "subject" | "time_value" | "value" | "metadata" | "unknown"
    """
    roles: dict[str, str] = {}
    for col in raw_columns:
        if col in subject_cols:
            roles[col] = "subject"
        elif col in time_cols:
            roles[col] = "time_value"
        elif col in remarks_cols:
            roles[col] = "metadata"
        elif col in value_cols:
            roles[col] = "value"
        else:
            roles[col] = "unknown"
    return roles
