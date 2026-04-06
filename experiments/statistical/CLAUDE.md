# Statistical Tests

| Script | Paper Section | Purpose |
|--------|--------------|---------|
| `entity_level_wilcoxon.py` | Appendix C | 25-country Wilcoxon signed-rank (Bonferroni k=4) |
| `wilcoxon_effect_sizes.py` | Appendix C | 50-country expansion + effect sizes |
| `paired_mcnemar.py` | Appendix C | Fact-level McNemar test (local datasets) |
| `leave_one_domain_out.py` | §4.3 | LODO 5-fold consistency check (33 files) |
| `v3_evaluate_and_stats.py` | §4.1 | Gold Standard v3 generation + Bootstrap CI |
| `permutation_test.py` | §4.2 | Permutation test (replaces Wilcoxon for independence violation) |
| `hierarchical_bootstrap.py` | Appendix C | Entity-cluster hierarchical bootstrap CI (addresses within-entity dependence; 1000 resamples at entity level) |

## Note

All statistical tests use entity-level aggregation. Independence assumption is approximate due to same-source extraction — results are reported as auxiliary evidence, not primary claims.
