# Archived Experiment Outputs

Archived on 2026-03-29 during Round 6 paper review cleanup.

These 46 directories contain intermediate/old experiment graph outputs that are
NOT referenced in the final paper (main_zh.md). They include:

- **Version iterations**: `sge_health_v3`–`v5`, `baseline_health_v3`–`v5`, etc.
- **LLM variants**: `llm_budget`, `llm_food`, `baseline_llm_*`
- **Full-dataset runs**: `annualbudget__*`, `inpatient__*` (individual year files)
- **Old food safety**: `sge_food`, `sge_food_v2`, `baseline_food`, `foodsafety__*`

All outputs are regenerable by re-running the SGE pipeline. Safe to delete if
disk space is needed.

## Active Outputs (in parent directory)

The 26 directories in `output/` (excluding this archive) are actively referenced
by the paper's evaluation scripts and contain the authoritative graph data.
