"""
inductor.py — Main Stage 2 schema induction entry point.

Takes a CSV path, runs Stage 1 to get the Meta-Schema, then applies
rule-based induction to produce a full extraction schema for Stage 3.

Adaptive Mode:
    When n_rows < SMALL_TABLE_THRESHOLD, schema constraints are relaxed
    (use_baseline_mode=True) to avoid entity-name rewriting on simple
    small tables (e.g. Food Safety with 13 rows, Type-III).
"""

from stage1.features import extract_features
from stage1.classifier import classify, TYPE_I, TYPE_II, TYPE_III
from stage1.schema import build_meta_schema

from .type_handlers import handle_type_i, handle_type_ii, handle_type_iii
from .prompt_builder import build_prompt_context

# Tables with fewer than this many rows use relaxed (baseline) extraction mode
SMALL_TABLE_THRESHOLD = 20


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

    When features.n_rows < SMALL_TABLE_THRESHOLD, returns a schema with
    use_baseline_mode=True — Stage 3 will skip schema injection, letting
    LightRAG use its default prompts (which preserves original entity names).
    """
    # Adaptive mode: small tables with complex hierarchy → schema constraints
    # cause entity-name rewriting which hurts EC. Skip schema injection but
    # keep column_roles so Stage 3 serializer can still produce structured chunks.
    if features.n_rows < SMALL_TABLE_THRESHOLD and table_type == TYPE_III:
        # Compute full schema for column_roles (needed by serializer), then
        # overlay baseline flags to suppress prompt injection.
        from .inducer import induce_schema as _full_induce
        full_schema = _full_induce(meta_schema, features)
        return _baseline_schema(
            table_type,
            reason="small_table_adaptive",
            column_roles=full_schema.get("column_roles", {}),
        )

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


def _baseline_schema(table_type: str, reason: str = "",
                     column_roles: dict | None = None) -> dict:
    """
    Return a minimal schema that signals Stage 3 to skip schema injection.

    Stage 3 checks schema["use_baseline_mode"] and, when True, skips the
    system-prompt override so LightRAG uses its default (unconstrained) prompts.
    This preserves original entity names in small/simple tables.

    column_roles is preserved so the serializer can still produce structured
    text chunks (structure-aware serialization without prompt constraint).
    """
    schema = {
        "table_type": table_type,
        "entity_types": ["Entity"],
        "relation_types": ["HAS_VALUE"],
        "extraction_rules": {},
        "prompt_context": "",
        "use_baseline_mode": True,
        "adaptive_reason": reason,
    }
    if column_roles:
        schema["column_roles"] = column_roles
    return schema
