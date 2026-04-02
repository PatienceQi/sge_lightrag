# Ablation Experiments

| Script | Paper Section | Purpose |
|--------|--------------|---------|
| `run_decoupled_ablation.py` | §4.3 Table 4 | Schema-only condition (raw CSV + schema prompt) |
| `run_c4_serialization_only.py` | §4.3 Table 4 | C4 condition (SGE chunks + default prompt) |
| `threshold_sensitivity.py` | §4.3 | C_T threshold sweep [3,9] + adaptive threshold [15,25] |
| `run_type1_sanity_check.py` | §4.1 | Type-I compatibility sanity check |
| `error_analysis_schema_only.py` | §4.3 | Schema-only failure mode analysis (entity type distribution, hallucination rate, LLM refusal rate) |
| `run_decoupled_ablation_extended.py` | §4.3 Table 4 | Extended factorial to 5 datasets (WB Pop + WB Mat) |

## Decoupled Ablation Matrix (Table 4)

```
               Schema=Yes    Schema=No
SGE chunks     Full SGE      C4 (serial-only)
Raw CSV        Schema-only   Baseline
```
