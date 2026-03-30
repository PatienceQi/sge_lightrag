# Stage 1: Topological Pattern Recognition

## Purpose

Given a CSV file, automatically classify its topology type (Type-I/II/III) and extract a structured Meta-Schema. This is the deterministic foundation — zero LLM calls, pure rule-based.

## Data Flow

```
CSV file → extract_features() → FeatureSet → classify() → τ → build_meta_schema() → Meta-Schema S
```

## Files

| File | Responsibility |
|------|---------------|
| `features.py` | Extract 5 feature signals from CSV header + first k rows |
| `classifier.py` | Algorithm 1: 3 priority rules → τ assignment |
| `schema.py` | Build Meta-Schema dict from FeatureSet + τ |

## Feature Signals (features.py)

| Signal | Definition | Key Regex/Threshold |
|--------|-----------|-------------------|
| `C_T` (time_cols) | Columns whose header matches year/fiscal format | `r"\b\d{4}(?:-\d{2,4})?\b"` |
| `C_key` (leading_text_col_count) | Consecutive non-numeric columns from col 0 | Stops at first numeric column |
| `fiscal` | Any C_T header contains "-" (e.g. 2022-23) | Dash in year pattern |
| `transposed` | First column has ≥2 values matching strict year | `r"^\d{4}$"` in col[0] |
| `yearInBody` | First 6 rows have ≥3 cells matching year | `r"^(19\|20)\d{2}$"` |

### Other Important Constants

- `_SAMPLE_ROWS = 20` — Max rows sampled for feature extraction
- `_MIN_HEADER_FILL_RATIO = 0.3` — Minimum non-empty header ratio
- `_REMARKS_KEYWORDS` — Chinese/English remark column detection keywords

## Classification Rules (classifier.py)

```
Rule 1 (Type-III): |C_key| ≥ 2 AND n_numeric > 0 AND |C_T| ≤ 6
                    AND ¬transposed AND ¬yearInBody AND ¬fiscal

Rule 2 (Type-II):  |C_T| > 0 OR transposed OR yearInBody

Rule 3 (Type-I):   Default fallback
```

**Critical threshold**: `_MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE = 6` — if ≤6 year columns, years are values not structure axis

**Completeness**: Every CSV maps to exactly one type (Rule 3 is catch-all).

**Accuracy**: 100% on all 33 test files (in-sample, human-verified). Generalization to new domains (e.g., industrial sensor data, financial time series) requires blind test set validation.

## Meta-Schema Output (schema.py)

```python
{
    "table_type": "Type-II",
    "time_dimension": {"columns": ["2020", "2021", "2022"]},
    "composite_key": ["纲领"],
    "n_rows": 4,
    "encoding": "utf-8",
    ...
}
```

## Encoding Detection

BOM cascade: UTF-16LE/BE → UTF-8-BOM → GBK → Big5HKSCS. World Bank metadata auto-skip (`Data Source` marker → `skiprows=4`).
