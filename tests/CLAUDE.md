# Test Suite

## Purpose

Unit tests for all three pipeline stages and the compact representation module. Tests use real CSV files from `../dataset/` where possible.

## Test Files

| File | Coverage | Key Validations |
|------|----------|----------------|
| `test_stage1.py` | Feature extraction + classification | Year pattern detection, composite key identification, remark column detection, all 33 files classified correctly (in-sample) |
| `test_stage2.py` | Schema induction (rule-based) | Entity name derivation, relation extraction, column role assignment, adaptive mode (n_rows < 20 → baseline) |
| `test_stage3.py` | Serialization + prompt injection | Chunk structure, entity-time-value patterns, delimiter compatibility, Type-I/II/III serialization |
| `test_compact.py` | Compact representation | Node reduction validation, threshold logic, year-value format, system prompt generation (27 tests) |

## Running Tests

```bash
cd ~/Desktop/SGE/sge_lightrag
python3 -m pytest tests/ -v

# Single file
python3 -m pytest tests/test_stage1.py -v

# With coverage
python3 -m pytest tests/ -v --cov=stage1 --cov=stage2 --cov=stage3
```

## Test Data

Tests reference real CSV files via relative paths to `../dataset/`. Key test datasets:
- Annual budget (Type-II, 4 entities)
- Food safety (Type-III, 17 entities, 13 rows — adaptive mode trigger)
- Health stats (Type-II transposed)
- WB Child Mortality (Type-II, 244 countries — compact mode trigger)

## Key Edge Cases Tested

- UTF-16LE/Big5HKSCS encoding detection
- World Bank metadata row skip (`skiprows=4`)
- Fiscal year format (2022-23) vs calendar year (2022)
- Transposed tables (years in first column)
- Sparse fill in Type-III hierarchical data
- Compact threshold boundary (n_rows = 99 vs 101)
- Adaptive threshold boundary (n_rows = 19 vs 21)
