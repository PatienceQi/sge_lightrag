# Evaluation Framework

## Pipeline

```
Gold Standard (gold/*.jsonl) + Graph (*.graphml)
  → evaluate_coverage.py       → EC/FC (entity-first, 2-hop)
  → evaluate_coverage_debiased.py → FC (value-first, de-biased)
  → run_evaluations_v2.py      → Bootstrap 95% CI
  → run_qa_eval.py             → QA Accuracy (100 questions)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_coverage.py` | EC/FC computation (entity-first: substring match + 2-hop neighbor) |
| `evaluate_coverage_debiased.py` | Value-first de-biased FC (removes entity-naming bias) |
| `run_evaluations_v2.py` | Full v2 evaluation: all datasets + Bootstrap CI (n=1000) |
| `run_qa_eval.py` | Downstream QA via direct graph context (100 questions) |
| `run_precision_analysis.py` | Graph topology + sampled precision analysis |
| `ablation_misclassify.py` | Stage 1 misclassification ablation |
| `baseline_precision_sample.py` | Baseline precision sampling (50 edges, seed=42) |
| `direct_llm_baseline.py` | Direct LLM baseline (CSV → triples, no pipeline) |
| `generate_gold_standards_v3.py` | Gold Standard auto-generation from CSV |
| `generate_gold_ood.py` | OOD dataset Gold Standard generation |
| `graph_loaders.py` | GraphML/JSON graph parsing utilities |
| `run_ood_evaluation.py` | OOD pipeline FC evaluation |
| `generate_gold_non_gov.py` | Non-government Gold Standard generation (Fortune 500 + THE Ranking) |
| `table_aware_baseline.py` | Table-aware prompt baseline (weak + strong variants) |
| `autoschemakg_baseline.py` | AutoSchemaKG end-to-end baseline |
| `nanographrag_baseline.py` | nano-GraphRAG cross-system baseline |
| `graphrag_qa_eval.py` | MS GraphRAG QA evaluation (parquet-based) |
| `run_stratified_precision.py` | Stratified precision sampling (250 edges) |

## Gold Standards (`gold/`)

### Local (manually annotated)

| File | Dataset | Type | Facts |
|------|---------|------|-------|
| `gold_budget.jsonl` | Annual Budget | Type-II | 4e/20f |
| `gold_food_sample.jsonl` | Food Safety | Type-III | 17e/52f |
| `gold_health.jsonl` | Health Stats | Type-II-T | 3e/14f |
| `gold_inpatient_2023.jsonl` | Inpatient | Type-III | 8e/16f |

### International v2 (auto-generated, 25 countries × 6 years)

| File | Dataset | Facts |
|------|---------|-------|
| `gold_who_life_expectancy_v2.jsonl` | WHO Life Expectancy | 150 |
| `gold_wb_child_mortality_v2.jsonl` | WB Child Mortality | 150 |
| `gold_wb_population_v2.jsonl` | WB Population | 150 |
| `gold_wb_maternal_v2.jsonl` | WB Maternal Mortality | 150 |

### Non-Government (auto-generated)

| File | Dataset | Facts |
|------|---------|-------|
| `gold_fortune500_revenue.jsonl` | Fortune 500 Revenue | 125 |
| `gold_the_university_ranking.jsonl` | THE University Ranking | 150 |

## Authoritative Result Files (`results/`)

| File | Content |
|------|---------|
| `all_results_v2.json` | EC/FC for all datasets + Bootstrap CI |
| `qa_results_v3_100q.json` | QA scores v3: 93/100 SGE, 59/100 Baseline |
| `debiased_results.json` | Value-first de-biased FC (5 datasets × 2 systems) |
| `direct_llm_results.json` | Direct LLM baseline numbers |
| `baseline_precision_results.json` | Baseline precision: 50/50 = 100% |
| `non_gov_fc_results.json` | Non-gov FC: Fortune 500 1.0/0.0, THE 0.567/0.200 |
| `table_aware_strong_results.json` | Strong table-aware baseline FC=0.253 |
| `nanographrag_fc_results.json` | nano-GraphRAG FC=0.367 |
| `autoschemakg_results.json` | AutoSchemaKG FC=0.860 |
| `graphrag_qa_results.json` | MS GraphRAG QA 84/88=95.5% |
| `stratified_precision_results.json` | 249/250=99.6% |
| `hipporag_fc_results.json` | HippoRAG FC=1.000 (dense graph) |

## Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **EC** | matched entities / gold entities | Entity identification |
| **FC** | matched facts / gold facts (2-hop substring) | Fact completeness |
| **De-biased FC** | value-first search + subject/year verification | Bias-corrected FC |
| **Bootstrap CI** | 95% CI from 1000 resamples | Statistical significance |

## JSONL Format

```json
{
  "source_file": "filename.csv",
  "row_index": 0,
  "triple": {
    "subject": "Entity name",
    "subject_type": "Policy_Program",
    "relation": "HAS_BUDGET",
    "object": "91.5",
    "object_type": "BudgetAmount",
    "attributes": {"year": "2022-23", "unit": "百万元"}
  },
  "annotator": "name",
  "confidence": "high"
}
```
