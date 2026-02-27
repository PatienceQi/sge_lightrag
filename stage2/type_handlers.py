"""
type_handlers.py — Per-type schema induction handlers.

Each handler receives the Stage 1 FeatureSet + Meta-Schema and returns
the type-specific portions of the Stage 2 extraction schema.
"""

import re
from typing import Optional

from stage1.classifier import TYPE_I, TYPE_II, TYPE_III
from stage1.features import FeatureSet, _col_is_mostly_numeric
from .header_parser import parse_all_headers
from .templates import derive_entity_name, detect_relation_name


# ---------------------------------------------------------------------------
# Type I — Flat-Entity-Attribute
# ---------------------------------------------------------------------------

def handle_type_i(features: FeatureSet, meta_schema: dict) -> dict:
    """
    Induce schema for Type I (Flat-Entity) tables.

    - entity_types: derived from the primary subject column name
    - relation_types: ["HAS_ATTRIBUTE"]
    - attribute_mapping: {col_name → derived_attr_name}
    """
    subject_cols = meta_schema.get("primary_subject_columns", [])
    value_cols = meta_schema.get("value_columns", [])

    # Derive entity types from subject column names
    entity_types = [derive_entity_name(c) for c in subject_cols] if subject_cols else ["Entity"]
    entity_types = _dedupe(entity_types)

    # Attribute mapping: value column → snake_case attribute name
    attribute_mapping = {col: _to_attr_name(col) for col in value_cols}

    return {
        "entity_types": entity_types,
        "relation_types": ["HAS_ATTRIBUTE"],
        "attribute_mapping": attribute_mapping,
        "extraction_rules": {
            "subject_extraction": (
                f"Each row represents one entity. "
                f"Use column(s) {subject_cols} as the entity identifier."
            ),
            "value_extraction": (
                "Each numeric column becomes an attribute of the entity "
                "linked via HAS_ATTRIBUTE relation."
            ),
            "time_handling": "No time dimension — this is a static attribute table.",
            "hierarchy": "No hierarchy — flat one-level entity table.",
            "remarks": "No remarks columns detected.",
        },
    }


# ---------------------------------------------------------------------------
# Type II — Time-Series-Matrix
# ---------------------------------------------------------------------------

def handle_type_ii(features: FeatureSet, meta_schema: dict) -> dict:
    """
    Induce schema for Type II (Time-Series-Matrix) tables.

    Handles both normal (time in headers) and transposed (time in rows) layouts.
    """
    subject_cols = meta_schema.get("primary_subject_columns", [])
    value_cols = meta_schema.get("value_columns", [])
    time_info = meta_schema.get("time_dimension", {})
    time_cols = time_info.get("columns", [])
    time_location = time_info.get("location", "headers")

    transposed = (time_location == "rows")

    # Parse compound headers for time columns
    parsed_headers = parse_all_headers(features.raw_columns)
    time_parsed = [ph for ph in parsed_headers if ph.is_time_column]

    # Build time_parsing_rules from parsed headers
    time_parsing_rules = _build_time_parsing_rules(time_parsed)

    # Derive entity types
    if transposed:
        # Columns are entities; derive from column names (skip first col = time axis)
        entity_cols = [c for c in features.raw_columns[1:] if c not in features.remarks_cols]
        entity_types = _dedupe([derive_entity_name(c) for c in entity_cols[:5]]) or ["Metric"]
    else:
        entity_types = _dedupe([derive_entity_name(c) for c in subject_cols]) if subject_cols else ["Entity"]

    # Detect relation name from column semantics
    relation_name = detect_relation_name(time_cols, value_cols)

    # Build extraction rules
    if transposed:
        subject_rule = (
            "Table is transposed: rows represent time periods, columns represent metrics. "
            "The first column contains year/period labels."
        )
        value_rule = (
            "Each cell value is a numeric measurement for a (metric, time_period) pair. "
            "Extract as: (Metric entity) -[" + relation_name + " {year: <row_label>}]-> (value)."
        )
        pivot_instruction = "rows=time_periods, cols=metric_entities"
    else:
        subj_str = ", ".join(f'"{c}"' for c in subject_cols[:3])
        subject_rule = (
            f"Each row is one entity identified by {subj_str}. "
            "Time-period columns in the header hold numeric values."
        )
        value_rule = (
            f"For each (entity, time_column) pair, create a {relation_name} relation "
            "with attributes: year, status (实际/预算/修订), unit (百万元/etc.)."
        )
        pivot_instruction = None

    time_rule = (
        "Parse compound headers (newline-separated) into components: "
        "{year, status, unit}. "
        "Example: '2022-23\\n(实际)\\n(百万元)' → {year: '2022-23', status: '实际', unit: '百万元'}."
    ) if not transposed else (
        "Time periods appear as row values in the first column. "
        "Parse year values (e.g. '2020', '2020.0') as the time dimension."
    )

    result = {
        "entity_types": entity_types,
        "relation_types": [relation_name, "HAS_METRIC"],
        "time_parsing_rules": time_parsing_rules,
        "extraction_rules": {
            "subject_extraction": subject_rule,
            "value_extraction": value_rule,
            "time_handling": time_rule,
            "hierarchy": "No hierarchy — flat time-series matrix.",
            "remarks": "No remarks columns detected.",
        },
    }

    if pivot_instruction:
        result["pivot_instruction"] = pivot_instruction

    return result


# ---------------------------------------------------------------------------
# Type III — Hierarchical-Hybrid
# ---------------------------------------------------------------------------

def handle_type_iii(features: FeatureSet, meta_schema: dict) -> dict:
    """
    Induce schema for Type III (Hierarchical-Hybrid) tables.

    - entity_types: one per composite key level
    - hierarchy_relations: parent-child between key levels
    - value_mapping: year columns → numeric attributes
    - remarks_handling: attach as metadata property
    """
    subject_cols = meta_schema.get("primary_subject_columns", [])
    value_cols = meta_schema.get("value_columns", [])
    remarks_cols = meta_schema.get("metadata_columns", [])

    # All leading text columns form the composite key hierarchy
    key_cols = _get_key_cols(features)

    # Derive entity types for each key level
    entity_types = _dedupe([derive_entity_name(c) for c in key_cols]) if key_cols else ["Category"]

    # Build hierarchy relations between consecutive key levels
    hierarchy_relations = []
    for i in range(len(key_cols) - 1):
        parent = entity_types[i] if i < len(entity_types) else f"Level{i}"
        child = entity_types[i + 1] if i + 1 < len(entity_types) else f"Level{i+1}"
        hierarchy_relations.append({
            "parent_col": key_cols[i],
            "child_col": key_cols[i + 1],
            "parent_entity": parent,
            "child_entity": child,
            "relation": "HAS_SUB_ITEM",
        })

    # Value mapping: year columns → numeric attributes
    value_mapping = {}
    for col in value_cols:
        # Check if it looks like a year column
        year_match = re.match(r"^(\d{4})(?:-\d{2,4})?$", str(col).strip())
        if year_match:
            value_mapping[col] = {"type": "year_value", "year": year_match.group(1)}
        else:
            value_mapping[col] = {"type": "numeric_attribute", "attr": _to_attr_name(col)}

    # Remarks handling
    remarks_rule = (
        f"Attach columns {remarks_cols} as metadata properties on the leaf entity. "
        "Do not create separate entity nodes for remarks."
    ) if remarks_cols else "No remarks columns detected."

    key_desc = " → ".join(f'"{c}"' for c in key_cols)
    subject_rule = (
        f"Composite key columns: {key_desc}. "
        "Empty cells in key columns inherit the last non-empty value from above (sparse fill). "
        f"Top-level key '{key_cols[0]}' defines the root entity." if key_cols else
        "No composite key columns detected."
    )

    return {
        "entity_types": entity_types,
        "relation_types": ["HAS_SUB_ITEM", "HAS_VALUE"],
        "hierarchy_relations": hierarchy_relations,
        "value_mapping": value_mapping,
        "extraction_rules": {
            "subject_extraction": subject_rule,
            "value_extraction": (
                "Year-named columns (e.g. '2021', '2022') hold numeric values. "
                "Create HAS_VALUE relations from the leaf entity to each year's value, "
                "with the year as a relation attribute."
            ),
            "time_handling": (
                "Year columns are value columns, not a structural time axis. "
                "Treat each year column as a separate numeric attribute of the leaf entity."
            ),
            "hierarchy": (
                f"Hierarchy levels: {key_desc}. "
                "Use sparse-fill to propagate parent values down to child rows."
            ),
            "remarks": remarks_rule,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_key_cols(features: FeatureSet) -> list[str]:
    """Return the leading text (composite key) columns."""
    key_cols = []
    df = features.df
    raw_cols = features.raw_columns
    for i, col in enumerate(df.columns):
        if not _col_is_mostly_numeric(df[col]):
            key_cols.append(raw_cols[i])
        else:
            break
    return key_cols


def _build_time_parsing_rules(parsed_headers: list) -> dict:
    """Build a time_parsing_rules dict from parsed compound headers."""
    if not parsed_headers:
        return {}

    # Collect unique statuses and units seen
    statuses = list({ph.status for ph in parsed_headers if ph.status})
    units = list({ph.unit for ph in parsed_headers if ph.unit})

    # Build an example
    examples = []
    for ph in parsed_headers[:3]:
        examples.append({
            "raw": ph.raw,
            "parsed": {"year": ph.year, "status": ph.status, "unit": ph.unit},
        })

    return {
        "format": "newline-separated compound header",
        "components": ["year", "status", "unit"],
        "year_pattern": r"\d{4}(?:-\d{2,4})?",
        "known_statuses": statuses,
        "known_units": units,
        "examples": examples,
    }


def _to_attr_name(col: str) -> str:
    """Convert a column name to a snake_case attribute name."""
    # Strip parentheses and newlines
    clean = re.sub(r"[\n\r]", "_", str(col).strip())
    clean = re.sub(r"[()（）]", "", clean)
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^\w\u4e00-\u9fff]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "value"


def _dedupe(lst: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
