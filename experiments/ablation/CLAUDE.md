# Ablation Experiments

| Script | Paper Section | Purpose |
|--------|--------------|---------|
| `run_decoupled_ablation.py` | §4.3 Table 4 | Schema-only condition (raw CSV + schema prompt) |
| `run_c4_serialization_only.py` | §4.3 Table 4 | C4 condition (SGE chunks + default prompt) |
| `threshold_sensitivity.py` | §4.3 | C_T threshold sweep [3,9] + adaptive threshold [15,25] |
| `run_type1_sanity_check.py` | §4.1 | Type-I compatibility sanity check |

## Decoupled Ablation Matrix (Table 4)

```
               Schema=Yes    Schema=No
SGE chunks     Full SGE      C4 (serial-only)
Raw CSV        Schema-only   Baseline
```
