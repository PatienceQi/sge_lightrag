# Stage 2 LLM: LLM-Enhanced Schema Induction

## Purpose

Alternative to rule-based Stage 2. Uses a single Claude Haiku call to generate domain-specific extraction schemas with richer semantic context (e.g., recognizing "纲领" as `Policy_Program`).

## Data Flow

```
CSV path → _build_csv_snippet() → Claude Haiku call → JSON parse → Σ
         ↘ (on failure) → automatic fallback to rule-based stage2/
```

## Files

| File | Responsibility |
|------|---------------|
| `inductor.py` | Orchestrator: Stage 1 → CSV snippet → LLM → parse → Σ |
| `llm_client.py` | OpenAI-compatible API wrapper |
| `prompts.py` | System/user prompt templates |

## LLM Configuration (llm_client.py)

```python
_DEFAULT_BASE_URL = "https://wolfai.top/v1"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
_MAX_RETRIES = 3
```

⚠️ **WARNING**: `llm_client.py` contains a hardcoded API key. Do NOT copy this pattern. Use environment variables for new code.

## Key Constraint: entity_types ≤ 2

LightRAG's parser truncates tuples with >4 fields. When LLM generates 3+ entity types, extraction output exceeds parser capacity → FC drops to 30% (C3 experiment). The `entity_types ≤ 2` constraint is enforced in the prompt.

## Encoding Handling (inductor.py)

CSV snippet extraction uses cascading encoding detection:
```
UTF-16 → UTF-8 → GBK → Big5HKSCS
```
Reads first `n_rows=5` for the LLM context window.

## Fallback Behavior

- LLM call timeout/error → automatic downgrade to `stage2/inductor.py` (rule-based)
- JSON parse failure → retry up to `_MAX_RETRIES`
- Invalid schema structure → fallback to rule-based

## Prompt Design (prompts.py)

- System prompt defines output JSON schema and entity_types constraint
- User prompt includes: table_type, meta_schema summary, CSV snippet (5 rows)
- Output: entity_types, relation_types, column_roles, prompt_context, extraction_templates
