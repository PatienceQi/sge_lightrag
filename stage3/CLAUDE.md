# Stage 3: Constrained Extraction & LightRAG Integration

## Purpose

Inject the Extraction Schema Σ into LightRAG via two mechanisms: (1) static prompt override, (2) dynamic context extension. Also handles row serialization (CSV → natural language chunks) with type-specific strategies.

## Data Flow

```
Σ + CSV → serializer.py → text chunks
       → prompt_injector.py → PROMPTS dict override
       → integrator.py → LightRAG ainsert() with addon_params
```

## Files

| File | Responsibility |
|------|---------------|
| `serializer.py` | CSV rows → natural language text chunks (type-specific) |
| `compact_representation.py` | Large Type-II compression (year=value; year=value; ...) |
| `integrator.py` | LightRAG injection orchestrator (system prompt + context) |
| `prompt_injector.py` | `PROMPTS["entity_extraction_system_prompt"]` override |

## Serialization Strategies (serializer.py)

| Type | Batch Size | Format Example |
|------|-----------|---------------|
| Type-I | 5 rows | Entity attributes grouped |
| Type-II | 1 row | `"纲领：防止贪污。各财年预算数据：2022-23(实际): 91.5百万元；..."` |
| Type-III | 1 row | Hierarchical with composite key context |

- `ignore` columns are skipped
- `time` columns: header (year) + value combined
- Output uses LightRAG native delimiter `<\|#\|>` for parser compatibility

## Compact Representation (compact_representation.py)

**Problem**: Large Type-II (e.g., WB Child Mortality: 244 countries × 23 years) creates ~5218 nodes (13.6× Baseline).

**Solution**: When `n_rows > COMPACT_THRESHOLD` AND Type-II:

```python
COMPACT_THRESHOLD = 100
MAX_YEARS_PER_CHUNK = 64
```

- One chunk per entity (country), all years as `year=value; year=value; ...`
- Companion system prompt: instructs LightRAG to create ONE entity node per country
- Expected compression: 5218 → ~245 nodes (21×)
- EC/FC equivalence: substring match `2022=7.1` hits value `7.1`

**Status**: Implemented + tested (27 tests pass). Full LightRAG rebuild verification pending.

## LightRAG Injection (integrator.py + prompt_injector.py)

**Static injection**: Replace `PROMPTS["entity_extraction_system_prompt"]` with SGE template containing `{schema_json}` placeholder, filled at runtime.

**Dynamic extension**: Fork `operate.py`, add `schema_json` to `context_base` dict, read from `addon_params` on each `ainsert()`.

**Non-invasive principle**: Only override PROMPTS dict + extend context_base. Never modify LightRAG core code.

## Error Handling

- Format error → retry
- Constraint violation → type filtering
- Orphan nodes → proportion monitoring + auto-cleanup
