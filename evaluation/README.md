# SGE-LightRAG Evaluation Framework

## Overview

This directory contains the complete evaluation framework for SGE-LightRAG, supporting:
- **EC/FC fact coverage evaluation** (substring match + 2-hop neighbor search)
- **CSVFidelity-Bench** — 10 datasets, 3 domains, 977 gold facts + OECD blind test + Type-III OOD
- **Bootstrap 95% confidence intervals** (n=1000 resamples)
- **Wilcoxon signed-rank test** (Bonferroni correction k=4 + effect size r)
- **Downstream QA evaluation** (100 questions: 67 direct + 33 inference)
- **Baseline comparisons** — Row-local, Fixed S/T/V, JSON Structured, Few-shot, Table-aware, AutoSchemaKG, Det Parser
- **OECD blind test** (6 held-out datasets, 83+ gold facts)
- **Type-III OOD evaluation** (Eurostat Crime + US Census)

---

## Evaluation Scripts

### Core Evaluation

| File | Description |
|------|-------------|
| `evaluate_coverage.py` | EC/FC metric computation (entity-first: substring match + 2-hop) |
| `evaluate_coverage_debiased.py` | Value-first de-biased FC (removes entity-naming bias) |
| `run_evaluations_v2.py` | Full v2 evaluation: all datasets + Bootstrap CI (n=1000) |
| `run_qa_eval.py` | Downstream QA via direct graph context (100 questions) |
| `run_all_evaluations.py` | Batch evaluation across all core datasets |
| `batch_runner.py` | Batch execution harness with timeout/retry for baseline runs |
| `graph_loaders.py` | GraphML/JSON graph parsing utilities |
| `generate_gold_standards_v3.py` | Gold Standard auto-generation from CSV |
| `generate_gold_non_gov.py` | Non-government Gold Standard generation (Fortune 500 + THE Ranking) |
| `generate_gold_oecd.py` | Gold Standard generation for OECD blind test (6 datasets, 83 facts) |
| `generate_gold_new_datasets.py` | Gold Standards for Eurostat Crime + US Census (Type-III OOD) |

### Baseline Scripts

| File | Description | Key Finding |
|------|-------------|-------------|
| `row_local_baseline.py` | Per-row chunks + default prompt (format-only control) | WHO FC = 0.167 = Baseline (format alone = zero gain) |
| `fixed_stv_baseline.py` | Fixed generic S/T/V schema (static vs dynamic schema ablation) | WHO FC ≈ 0.66, range 0.12–0.92 (unstable) |
| `json_structured_baseline.py` | JSON structured output (alternative coupling mechanism) | WHO 1.0, Fortune500 1.0; confirms coupling hypothesis |
| `fewshot_baseline.py` | Few-shot structured prompt (3 example triples) | FC ≤ 0.013, 4/5 datasets worse than vanilla |
| `table_aware_baseline.py` | Table-aware prompt (weak + strong variants) | Strong variant FC = 0.253 |
| `autoschemakg_baseline.py` | AutoSchemaKG end-to-end baseline | FC = 0.860 (WHO only) |
| `direct_llm_baseline.py` | Direct LLM baseline (CSV → triples, no pipeline) | Non-comparable (no graph) |
| `baseline_common.py` | Shared LLM/embedding/evaluation utilities for all baselines | — |
| `config.py` | Centralized API key/model config (env vars: SGE_API_KEY, SGE_API_BASE) | — |

### Statistical Analysis

| File | Description |
|------|-------------|
| `run_precision_analysis.py` | Graph topology + sampled precision (50/50 = 100%) |
| `run_stratified_precision.py` | Stratified precision sampling (249/250 = 99.6% SGE) |
| `csv_verified_precision.py` | Deterministic CSV cell lookup precision audit (Type-II 150/150 = 100%) |
| `run_independent_annotation.py` | Dual-LLM annotator precision evaluation |
| `run_stat_question_eval.py` | 231-question answerability evaluation (SGE 84.8% vs Baseline 54.5%) |
| `generate_stat_questions.py` | Statistical analysis question generator (231 questions) |
| `run_error_analysis.py` | Detailed error analysis (missed facts, OOD failures) |
| `run_error_taxonomy.py` | Full error taxonomy across 7 datasets × 3 systems |
| `ablation_misclassify.py` | Stage 1 misclassification ablation |
| `error_analysis_schema_only.py` | Schema-only failure mode analysis |

### OOD and Blind Test Evaluation

| File | Description |
|------|-------------|
| `run_ood_evaluation.py` | OOD pipeline FC evaluation (OECD + Type-III OOD) |
| `generate_gold_oecd.py` | OECD blind test Gold Standard (6 datasets: Education, Labor, Environment, Health, Trade, GDP) |
| `generate_gold_new_datasets.py` | Type-III OOD Gold Standards (Eurostat Crime + US Census) |

### Cross-System Baselines

| File | Description |
|------|-------------|
| `nanographrag_baseline.py` | nano-GraphRAG cross-system baseline (FC = 0.367) |
| `graphrag_qa_eval.py` | MS GraphRAG QA evaluation (parquet-based, 84/88 = 95.5%) |

---

### `batch_runner.py` Usage

The batch runner executes multiple baseline or evaluation scripts with timeout protection and aggregated logging:

```bash
# Run all core baselines
python3 evaluation/batch_runner.py --mode baselines --datasets all

# Run OECD blind test only
python3 evaluation/batch_runner.py --mode oecd --output-dir results/oecd_run

# Run with custom timeout per script (seconds)
python3 evaluation/batch_runner.py --mode baselines --timeout 3600

# Dry run (list scripts without executing)
python3 evaluation/batch_runner.py --mode all --dry-run
```

---

## Gold Standards (`gold/`)

### Local Datasets (manually annotated)

| File | Dataset | Type | Facts |
|------|---------|------|-------|
| `gold_budget.jsonl` | Annual Budget | Type-II | 4e / 20f |
| `gold_food_sample.jsonl` | Food Safety | Type-III | 17e / 52f (14 entity relations + 52 value facts) |
| `gold_health.jsonl` | Health Stats | Type-II-T | 3e / 14f |
| `gold_inpatient_2023.jsonl` | HK Inpatient | Type-III | 8e / 16f |

### International v2 (auto-generated, 25 countries × 6 years × 150 facts)

| File | Dataset | Source | Facts |
|------|---------|--------|-------|
| `gold_who_life_expectancy_v2.jsonl` | WHO Life Expectancy (196 countries) | WHO GHO | 150 |
| `gold_wb_child_mortality_v2.jsonl` | WB Child Mortality (244 countries) | World Bank | 150 |
| `gold_wb_population_v2.jsonl` | WB Population (265 countries) | World Bank | 150 |
| `gold_wb_maternal_v2.jsonl` | WB Maternal Mortality | World Bank | 150 |

v2 Gold Standards are generated directly from experimental CSVs (`generate_gold_standards_v3.py`). 25 target countries selected by GDP rank + geographic balance; 6 target years (2000/2005/2010/2015/2020/2022); values read directly from CSV cells with no truncation or external sources.

### Non-Government (auto-generated)

| File | Dataset | Type | Facts |
|------|---------|------|-------|
| `gold_fortune500_revenue.jsonl` | Fortune 500 Revenue | Type-II | 125 |
| `gold_the_university_ranking.jsonl` | THE University Ranking | Type-III | 150 |

### OECD Blind Test (held-out, auto-generated)

| Dataset | Type | Gold Facts |
|---------|------|------------|
| OECD Education | Type-II | ~15 |
| OECD Labor | Type-II | ~15 |
| OECD Environment | Type-II | ~12 |
| OECD Health | Type-II | ~15 |
| OECD Trade | Type-II | ~13 |
| OECD GDP | Type-II | ~13 |

Total: 83 gold facts across 6 OECD datasets. These are strict blind test — not used in any training or threshold tuning.

### Type-III OOD (held-out, auto-generated)

| Dataset | Type | Domain |
|---------|------|--------|
| Eurostat Crime Statistics | Type-III | EU government |
| US Census Demographics | Type-III | US government |

---

## JSONL Format

```json
{
  "source_file": "dataset_filename.csv",
  "row_index": 0,
  "triple": {
    "subject": "Entity name",
    "subject_type": "EntityType",
    "relation": "HAS_VALUE",
    "object": "42.3",
    "object_type": "StatValue",
    "attributes": {"year": "2020", "unit": "years", "status": "final"}
  },
  "annotator": "auto-v3",
  "confidence": "high",
  "notes": "optional"
}
```

---

## Evaluation Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| **EC (Entity Coverage)** | Gold entities matched in graph via substring | Entity identification ability |
| **FC (Fact Coverage)** | Gold triples found in graph 2-hop neighborhood | Fact completeness |
| **De-biased FC** | Value-first search + subject/year verification | Bias-corrected FC |
| **Bootstrap 95% CI** | 95% CI from n=1000 resamples | Statistical significance |
| **Fisher's exact test** | 2×2 contingency table exact test | Small-sample significance (n<30) |
| **Wilcoxon signed-rank** | Non-parametric paired test + effect size r | Cross-dataset significance |
| **QA Accuracy** | Answer correctness (direct + inference questions) | Downstream task validity |

---

## Running Evaluations

```bash
cd ~/Desktop/SGE/sge_lightrag

# 1. Generate v2 Gold Standards
python3 evaluation/generate_gold_standards_v3.py

# 2. Run full v2 evaluation (EC/FC + Bootstrap CI)
python3 evaluation/run_evaluations_v2.py

# 3. Run QA evaluation
python3 evaluation/run_qa_eval.py

# 4. Run de-biased evaluation
python3 evaluation/evaluate_coverage_debiased.py --batch

# 5. Run baseline comparisons (batch)
python3 evaluation/batch_runner.py --mode baselines

# 6. Run OECD blind test
python3 evaluation/run_ood_evaluation.py --split oecd

# 7. Run Type-III OOD evaluation
python3 evaluation/run_ood_evaluation.py --split type3

# 8. Single-dataset traditional evaluation
python3 evaluation/evaluate.py \
  --graph output/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml \
  --gold evaluation/gold/gold_budget.jsonl
```

---

## Authoritative Result Files (`results/`)

| File | Content |
|------|---------|
| `all_results_v2.json` | EC/FC for all 7 core datasets + Bootstrap CI |
| `qa_results_v3_100q.json` | QA scores v3: SGE 93/100 (93%), Baseline 59/100 (59%) |
| `debiased_results.json` | Value-first de-biased FC (5 datasets × 2 systems) |
| `non_gov_fc_results.json` | Non-gov FC: Fortune500 SGE=1.0/Base=0.4, THE SGE=0.6/Base=0.207 |
| `fewshot_baseline_results.json` | Few-shot results (5 datasets, FC ≤ 0.013, all ≤ baseline) |
| `row_local_baseline_results.json` | Row-local results (WHO FC=0.167 = Baseline) |
| `fixed_stv_baseline_results.json` | Fixed S/T/V results (WHO FC≈0.66, range 0.12–0.92) |
| `json_structured_baseline_results.json` | JSON structured output (WHO=1.0, Fortune500=1.0) |
| `table_aware_strong_results.json` | Strong table-aware baseline (FC=0.253) |
| `autoschemakg_results.json` | AutoSchemaKG (FC=0.860, WHO only) |
| `nanographrag_fc_results.json` | nano-GraphRAG (FC=0.367) |
| `graphrag_qa_results.json` | MS GraphRAG QA (84/88 = 95.5%) |
| `stratified_precision_results.json` | Stratified precision 249/250 = 99.6% SGE |
| `csv_verified_precision.json` | CSV cell lookup precision (Type-II SGE 150/150 = 100%) |
| `direct_llm_results.json` | Direct LLM baseline numbers |
| `stat_question_eval_results.json` | 231-question answerability (SGE 84.8% vs Base 54.5%) |
| `error_analysis_detailed.json` | Detailed error analysis (WB_Mat / THE / OOD failures) |
| `independent_annotation_results.json` | Dual-LLM precision annotation results |

---

## Core Results Summary (CSVFidelity-Bench, 977 gold facts)

| Dataset | Baseline | Row-local | Fixed S/T/V | Det Parser | SGE | Ratio |
|---------|----------|-----------|-------------|------------|-----|-------|
| WHO Life Expectancy | 0.167 | 0.167 | 0.66 | 0.68 | **1.000** | 6.0× |
| WB Child Mortality | 0.473 | — | — | 0.727 | **1.000** | 2.11× |
| WB Population | 0.187 | — | — | 0.960 | **1.000** | 5.35× |
| WB Maternal Mortality | 0.787 | — | — | 0.967 | **0.967** | 1.23× |
| HK Inpatient | 0.438 | — | — | 1.000 | **0.938** | 2.14× |
| Fortune 500 Revenue | 0.400 | — | — | 1.000 | **1.000** | 2.50× |
| THE University Ranking | 0.207 | — | — | 1.000 | **0.600** | 2.90× |

**Key pattern**: Row-local = Baseline (format alone = zero gain). Fixed S/T/V is unstable (format + static constraint insufficient). SGE (format + dynamic constraint) consistently outperforms all single-mechanism baselines.

QA (direct graph context): SGE 93% (93/100) vs Baseline 59% (59/100), trend questions SGE 86% vs Baseline 36%.
E2E LightRAG query: SGE 13% vs Baseline 13% (Δ=0) — vector retrieval bottleneck independent of graph quality.
Wilcoxon (Bonferroni k=4): all 5 international datasets p_Bonf < 0.05, effect size r ≥ 0.80 (large).
