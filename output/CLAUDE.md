# Output — LightRAG Graph Outputs

All directories are gitignored and regenerable. Each contains a `lightrag_storage/` subdirectory with `graph_chunk_entity_relation.graphml` and vector DB files.

## Main SGE Graphs (Paper Table 1)

| Directory | Dataset | Type | Nodes | Paper FC |
|-----------|---------|------|------:|---------|
| `who_life_expectancy/` | WHO Life Expectancy | II | 4508 | 1.000 |
| `wb_child_mortality/` | WB Child Mortality | II | 5218 | 1.000 |
| `wb_population/` | WB Population | II | 6551 | 1.000 |
| `wb_maternal/` | WB Maternal Mortality | II | 3764 | 0.967 |
| `inpatient_2023/` | HK Inpatient | III | 1190 | 0.938 |
| `sge_budget/` | Annual Budget | II | 7 | 1.000 |
| `sge_food_adaptive/` | Food Safety (degraded) | III | 32 | 0.481 |
| `sge_health/` | Health Statistics | II-T | 13 | 0.786 |

## Baseline Graphs (Paper Table 1)

| Directory | Dataset | Nodes | Paper FC |
|-----------|---------|------:|---------|
| `baseline_who_life/` | WHO | 402 | 0.167 |
| `baseline_wb_child_mortality/` | WB CM | 383 | 0.473 |
| `baseline_wb_population/` | WB Pop | 317 | 0.187 |
| `baseline_wb_maternal/` | WB Maternal | 355 | 0.787 |
| `baseline_inpatient23/` | Inpatient | 450 | 0.438 |
| `baseline_budget/` | Budget | 19 | 1.000 |
| `baseline_food_adaptive/` | Food Safety | 44 | 0.481 |
| `baseline_health/` | Health | 6 | 0.643 |

## Ablation Graphs (Paper Table 4)

| Directory | Condition | Dataset | Nodes |
|-----------|-----------|---------|------:|
| `ablation_schema_only_who/` | Schema-only | WHO | 1296 |
| `ablation_schema_only_wb_cm/` | Schema-only | WB CM | 253 |
| `ablation_schema_only_inpatient/` | Schema-only | Inpatient | 467 |
| `ablation_rule_who/` | Rule SGE | WHO | 322 |
| `ablation_rule_wb_cm/` | Rule SGE | WB CM | 1136 |
| `ablation_c4_serial_only_who/` | C4 Serial-only | WHO | 433 |
| `ablation_c4_serial_only_wb_cm/` | C4 Serial-only | WB CM | 78 |

## Compact Graphs (§4.5)

| Directory | Variant | Nodes |
|-----------|---------|------:|
| `compact_who/` | SGE compact (hash embed) | 201 |
| `compact_who_realembed/` | SGE compact (Ollama embed) | 202 |
| `compact_who_baseline_realembed/` | Baseline compact (Ollama embed) | 220 |
| `compact_wb_cm/` | WB CM compact | 245 |

## Cross-Model (§4.6 Table 9)

| Directory | Dataset | Nodes |
|-----------|---------|------:|
| `crossmodel_gpt_5_mini_who/` | WHO | 4456 |
| `crossmodel_gpt_5_mini_wb_cm/` | WB CM | 5294 |
| `crossmodel_gpt_5_mini_wb_pop/` | WB Pop | 6571 |
| `crossmodel_gpt_5_mini_wb_mat/` | WB Maternal | 3242 |
| `crossmodel_gpt_5_mini_inpatient/` | Inpatient | 975 |

## OOD (§4.3 Table 3)

| Directory | Content |
|-----------|---------|
| `ood/` | 20 OOD blind test graphs (Rule SGE + Baseline) |
| `ood_llm/` | 3 failed OOD datasets with LLM-enhanced Stage 2 |
| `ood_test_gdp/` | Early GDP test (no graph) |

## MS GraphRAG Comparison (Appendix, not in main text)

| Directory | Dataset | Nodes |
|-----------|---------|------:|
| `graphrag_budget/` | Budget | 19 |
| `graphrag_inpatient/` | Inpatient | 160 |
| `graphrag_who/` | WHO | 713 |

## Type-I Sanity Check

| Directory | Dataset |
|-----------|---------|
| `type1_医疗开支_Table_{1,3,5}_{sge,baseline}/` | 3 Type-I tables × SGE/Baseline |
