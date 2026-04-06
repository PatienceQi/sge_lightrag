# Evaluation Results — Authoritative Numbers

These JSONs are the **single source of truth** for all paper numbers. Never guess — always verify against these files.

## Authoritative Files

| File | Paper Section | Content |
|------|--------------|---------|
| `all_results_v2.json` | Table 1, §4.2 | EC/FC for all 8 datasets + Bootstrap 95% CI |
| `qa_results_v3_100q.json` | Appendix B | QA scores: SGE 93/100, Baseline 59/100 |
| `debiased_results.json` | Table 8, §4.6 | Value-first de-biased FC (5 datasets × 2 systems) |
| `baseline_precision_results.json` | §4.4 | Sampled precision: 50/50 = 100% (WHO + Inpatient) |

## Supplementary Files

| File | Paper Section | Content |
|------|--------------|---------|
| `direct_llm_results.json` | — (code release only) | Direct LLM baseline (CSV → triples, no pipeline) |
| `ood_llm_results.json` | Table 3 footnote | LLM-enhanced OOD results (3 failed datasets) |
| `precision_analysis_v1.json` | — (superseded) | Old v1 precision analysis |
| `qa_results_v2.json` | — (superseded by v3) | Old v2 QA results |
| `det_parser_results.json` | §4.3 | Deterministic parser upper bound (WHO FC=0.68, 5/7 ≥0.96) |
| `row_local_baseline_results.json` | §4.3 | Per-row + default prompt (WHO FC=0.167 = Baseline; format alone = 0 gain) |
| `fixed_stv_baseline_results.json` | §4.3 | Fixed S/T/V generic schema (WHO FC≈0.49; dynamic > static) |
| `json_structured_baseline_results.json` | §4.3 | JSON structured output (WHO FC=1.000; confirms coupling) |
| `oecd_type3_ood_results.json` | §4.4 | OECD OOD Type-III evaluation (4 datasets, 83 facts) |
| `oecd_blind_test_results.json` | §4.4 | OECD blind test FC results |
| `csv_verified_precision.json` | §4.4 | CSV-verified precision audit (Type-II 150/150=100%) |
| `error_taxonomy_full.json` | Appendix | Full error taxonomy (7 datasets × 3 systems) |
| `comprehensive_baseline_comparison.json` | Table 1 | All baselines aggregated for comparison table |
| `cost_latency_estimates.json` | Appendix | Cost and latency estimates per system |

## Key Numbers Quick Reference

```
WHO:     SGE FC=1.000, Base FC=0.167, Δ=6.0×
WB CM:   SGE FC=1.000, Base FC=0.473, Δ=2.11×
WB Pop:  SGE FC=1.000, Base FC=0.187, Δ=5.35×
WB Mat:  SGE FC=0.967, Base FC=0.787, Δ=1.23×
Inpat:   SGE FC=0.938, Base FC=0.438, Δ=2.14×
Fort500: SGE FC=1.000, Base FC=0.400, Δ=2.50×
THE:     SGE FC=0.600, Base FC=0.207, Δ=2.90×
QA:      SGE 93/100 (93%), Baseline 59/100 (59%)
StatQ:   SGE 84.8%, Baseline 54.5% (231 questions)
Prec:    CSV-verified Type-II 150/150=100%
Gold:    977 facts total (600 auto + 102 manual + 275 non-gov)
DetPar:  WHO FC=0.68, 5/7 FC≥0.96 (zero LLM)
JSON:    WHO FC=1.000 (multiple coupling mechanisms)
RowLoc:  WHO FC=0.167 = Baseline (format alone = 0)
```
