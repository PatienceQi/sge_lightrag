# SGE-LightRAG Codebase

## Architecture

Three-stage perception pipeline injected upstream of LightRAG:

```
CSV → Stage 1 (classifier.py)  → τ ∈ {Type-I, II, III} + Meta-Schema S
    → Stage 2 (inductor.py)    → Extraction Schema Σ = (E, R, φ, ψ, Γ, δ)
    → Stage 3 (integrator.py)  → PROMPTS dict override → LightRAG ainsert()
```

### Stage 1: Topological Pattern Recognition (`stage1/`)
- `features.py` — Extracts 5 feature signals: C_T, C_key, fiscal, transposed, yearInBody
- `classifier.py` — Algorithm 1: 3 priority rules → τ assignment (complete: every CSV maps to exactly one type)
- `schema.py` — Builds Meta-Schema S from features + classification

### Stage 2: Schema Induction (`stage2/` + `stage2_llm/`)
- `stage2/inductor.py` — Rule-based induction entry point; includes adaptive mode (`SMALL_TABLE_THRESHOLD = 20`)
- `stage2/type_handlers.py` — Type-specific handlers (Type-I/II/III)
- `stage2/prompt_builder.py` — Extraction constraint prompt construction
- `stage2_llm/inductor.py` — LLM-enhanced induction (entity_types ≤ 2 constraint)
- `stage2_llm/llm_client.py` — OpenAI-compatible API wrapper

### Stage 3: Constrained Extraction (`stage3/`)
- `integrator.py` — LightRAG injection orchestrator
- `prompt_injector.py` — `PROMPTS["entity_extraction_system_prompt"]` override
- `serializer.py` — Row serialization (Type-I: batch=5, Type-II/III: batch=1)
- `compact_representation.py` — Compact timeseries for large Type-II (COMPACT_THRESHOLD=100)

### Evaluation (`evaluation/`)
- `evaluate_coverage.py` — EC/FC metrics (substring match + 2-hop neighbor search)
- `generate_gold_standards.py` — v2 Gold Standard auto-generation from CSV
- `run_evaluations_v2.py` — Full v2 evaluation with Bootstrap 95% CI
- `run_qa_eval.py` — Downstream QA (100 questions, v3: 93/100 SGE, 59/100 Baseline)
- `run_precision_analysis.py` — Graph topology + sampled precision (50/50 = 100%)
- `ablation_misclassify.py` — Stage 1 misclassification ablation (3/3 → 0 chunks)

## Key Design Decisions

1. **Non-invasive integration**: Only override `PROMPTS` dict + extend `context_base`; never modify LightRAG core
2. **Deterministic fallback**: LLM Stage 2 failure → automatic downgrade to rule-based mode
3. **Adaptive degradation**: `n_rows < 20 AND Type-III` → `use_baseline_mode=True` (skip schema override)
4. **entity_types ≤ 2**: LightRAG parser truncates tuples with >4 fields; constraining entity_types avoids this

## Running Tests

```bash
python3 -m pytest tests/ -v
```

Tests cover:
- `test_stage1.py` — Classification on all 27 files (100% accuracy)
- `test_stage2.py` — Schema induction + adaptive mode
- `test_stage3.py` — Serialization + prompt injection
- `test_compact.py` — Compact representation (27 tests)

## Important Paths

| Path | Content |
|------|---------|
| `output/` | All LightRAG graph outputs (gitignored, regenerable) |
| `evaluation/gold_*_v2.jsonl` | v2 Gold Standards (DO NOT modify without instruction) |
| `evaluation/all_results_v2.json` | Authoritative v2 evaluation numbers |
| `evaluation/qa_results_v2.json` | Authoritative QA results |
| `dataset/` | Symlink or relative path to `../dataset/` |

## Conventions

- Immutable data patterns: never mutate Meta-Schema or Extraction Schema in-place; always return new dicts
- Functions < 50 lines; files < 800 lines
- All CSV reading goes through `preprocessor.py` (handles BOM, encoding detection, WB metadata skip)
- Column role enum: `{subject, time, value, metadata, ignore}`
- Error handling: log + fallback, never silently swallow
