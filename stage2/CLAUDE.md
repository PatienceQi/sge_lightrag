# Stage 2: Rule-Based Schema Induction

## Purpose

Transform the Meta-Schema S (from Stage 1) into a full Extraction Schema Σ that defines entity types, relation types, column roles, and extraction constraints — all without LLM calls.

## Data Flow

```
Meta-Schema S + FeatureSet → inductor.py → Σ = (E, R, φ, ψ, Γ, δ)
```

## Files

| File | Responsibility |
|------|---------------|
| `inductor.py` | Main entry: Stage 1 → Stage 2 pipeline, adaptive mode logic |
| `inducer.py` | Schema induction orchestrator (dispatches to type handlers) |
| `type_handlers.py` | Type-specific schema generation (I/II/III) |
| `header_parser.py` | Column header parsing (temporal/hierarchical structure) |
| `prompt_builder.py` | Converts schema into LLM-friendly extraction prompts |
| `templates.py` | Schema templates per type (entity/relation name derivation) |

## Extraction Schema Σ

```python
Σ = {
    "entity_types": ["Policy_Program", "StatValue"],     # E
    "relation_types": ["HAS_BUDGET", "IN_YEAR"],          # R
    "column_roles": {"纲领": "subject", "2022-23": "time", ...},  # φ
    "subject_entity_map": {"纲领": "Policy_Program"},     # ψ
    "constraints": "...",                                  # Γ (natural language)
    "strategy_summary": "..."                              # δ
}
```

Column role enum: `{subject, time, value, metadata, ignore}`

## Adaptive Mode (inductor.py)

```python
SMALL_TABLE_THRESHOLD = 20

# When n_rows < 20 AND Type-III → use_baseline_mode=True
# Skips schema override in Stage 3, LightRAG uses default prompts
# Reason: small Type-III (e.g. food safety, 13 rows) suffers from
# entity_type constraint causing LLM to rewrite indicator names
```

## Type-Specific Strategies

| Type | Entity Derivation | Relation Pattern | Key Logic |
|------|------------------|-----------------|-----------|
| Type-I | Column names → entity types | Row-level attributes | Batch=5 rows |
| Type-II | Subject col → entity, year cols → time | `HAS_VALUE_IN_YEAR` | Fiscal year detection |
| Type-III | Composite key → hierarchical entities | `HAS_SUB_ITEM`, `HAS_VALUE` | Sparse fill handling |

## Post-Processing Validation

Deterministic rules override LLM output:
- Time columns → forced `time` role
- Composite key columns → forced `subject` role
- Remark columns → forced `metadata` role

"Deterministic rules backstop probabilistic model."
