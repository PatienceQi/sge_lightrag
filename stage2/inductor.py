"""
inductor.py — Main Stage 2 schema induction entry point.

Takes a CSV path, runs Stage 1 to get the Meta-Schema, then applies
rule-based induction to produce a full extraction schema for Stage 3.
"""

from stage1.features import extract_features
from stage1.classifier import classify, TYPE_I, TYPE_II, TYPE_III
from stage1.schema import build_meta_schema

from .type_handlers import handle_type_i, handle_type_ii, handle_type_iii
from .prompt_builder import build_prompt_context


def induce_schema(csv_path: str) -> dict:
    """
    Run the full Stage 1 → Stage 2 pipeline for a CSV file.

    Parameters
    ----------
    csv_path : path to the CSV file

    Returns
    -------
    dict — the full extraction schema (Stage 2 output)
    """
    # Stage 1
    features = extract_features(csv_path)
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    return induce_schema_from_meta(features, table_type, meta_schema)


def induce_schema_from_meta(features, table_type: str, meta_schema: dict) -> dict:
    """
    Run Stage 2 given already-computed Stage 1 outputs.

    Useful for testing without re-reading the CSV.
    """
    # Dispatch to the appropriate type handler
    if table_type == TYPE_I:
        type_specific = handle_type_i(features, meta_schema)
    elif table_type == TYPE_II:
        type_specific = handle_type_ii(features, meta_schema)
    elif table_type == TYPE_III:
        type_specific = handle_type_iii(features, meta_schema)
    else:
        # Unknown type — minimal fallback
        type_specific = {
            "entity_types": ["Entity"],
            "relation_types": ["HAS_VALUE"],
            "extraction_rules": {
                "subject_extraction": "Extract entities from text columns.",
                "value_extraction": "Extract values from numeric columns.",
                "time_handling": "No time dimension detected.",
                "hierarchy": "No hierarchy detected.",
                "remarks": "No remarks columns detected.",
            },
        }

    entity_types = type_specific.get("entity_types", ["Entity"])
    relation_types = type_specific.get("relation_types", ["HAS_VALUE"])
    extraction_rules = type_specific.get("extraction_rules", {})

    # Build the natural-language prompt context
    prompt_context = build_prompt_context(
        table_type=table_type,
        entity_types=entity_types,
        relation_types=relation_types,
        extraction_rules=extraction_rules,
        type_specific=type_specific,
    )

    # Assemble the final schema (matches the spec format)
    schema = {
        "table_type": table_type,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "extraction_rules": extraction_rules,
        "prompt_context": prompt_context,
    }

    # Attach type-specific extras (time_parsing_rules, hierarchy_relations, etc.)
    for key in ("time_parsing_rules", "pivot_instruction", "attribute_mapping",
                "hierarchy_relations", "value_mapping"):
        if key in type_specific:
            schema[key] = type_specific[key]

    return schema
