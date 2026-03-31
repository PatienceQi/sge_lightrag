# Experiment Results

Output JSONs from experiment scripts. These support paper claims but are **NOT** the authoritative evaluation numbers — those live in `evaluation/results/`.

## Ablation & Sensitivity

| File | Script | Paper Section |
|------|--------|--------------|
| `decoupled_ablation_results.json` | ablation/run_decoupled_ablation.py | Table 4 (Schema-only condition) |
| `threshold_sensitivity_results.json` | ablation/threshold_sensitivity.py | §4.3 threshold sweep |
| `type1_sanity_check_results.json` | ablation/run_type1_sanity_check.py | §4.1 Type-I compatibility |

## Statistical Tests

| File | Script | Paper Section |
|------|--------|--------------|
| `entity_level_wilcoxon_results.json` | statistical/entity_level_wilcoxon.py | Appendix C Table C3 (25 countries) |
| `wilcoxon_effect_sizes_results.json` | statistical/wilcoxon_effect_sizes.py | Appendix C Table C2 (50 countries) |
| `paired_mcnemar_results.json` | statistical/paired_mcnemar.py | Appendix C Table C4 |
| `leave_one_domain_out_results.json` | statistical/leave_one_domain_out.py | §4.3 LODO |
| `v3_gold_standard_results.json` | statistical/v3_evaluate_and_stats.py | §4.1 Gold Standard v3 |

## Downstream Probes & E2E

| File | Script | Paper Section |
|------|--------|--------------|
| `graph_native_probe_results.json` | probes/graph_native_probe.py | Table 6 (WHO) |
| `graph_native_probe_wb_pop_results.json` | probes/graph_native_probe_wb_pop.py | Table 6 (WB Pop) |
| `graph_native_probe_inpatient_results.json` | probes/graph_native_probe_inpatient.py | Table 6 (Inpatient) |
| `e2e_qa_results.json` | probes/run_e2e_lightrag_qa.py | Table 7 (hybrid E2E) |
| `e2e_compact_results.json` | probes/run_e2e_compact.py | §4.5 compact E2E |
| `e2e_baseline_compact_results.json` | probes/run_e2e_baseline_compact.py | §4.5 baseline compact |
| `who_fidelity_compact_analysis.json` | probes/who_fidelity_compact_analysis.py | §4.5 FC comparison |

## Cross-Model

| File | Script | Paper Section |
|------|--------|--------------|
| `crossmodel_expansion_results.json` | crossmodel/run_crossmodel_expansion.py | Table 9 |

## OOD

| File | Script | Paper Section |
|------|--------|--------------|
| `blind_test_stage1_results.json` | — (from run_all_tests) | Appendix H |
