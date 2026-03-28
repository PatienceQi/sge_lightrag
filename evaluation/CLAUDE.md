# Evaluation Framework

## Purpose

Comprehensive evaluation of SGE-LightRAG graph quality: EC/FC information coverage, Bootstrap CI, Fisher's exact test, downstream QA, and baseline comparisons.

## Evaluation Pipeline

```
Gold Standard (JSONL) + Graph (GraphML/JSON) → evaluate_coverage.py → EC/FC
                                              → run_evaluations_v2.py → Bootstrap 95% CI
                                              → run_qa_eval.py → QA Accuracy (100 questions)
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_coverage.py` | EC/FC computation (substring match + 2-hop neighbor search) |
| `run_evaluations_v2.py` | Full v2 evaluation: all datasets + Bootstrap CI (n=1000) |
| `evaluate.py` | Legacy triple matching (P/R/F1, graph topology) |
| `generate_gold_standards.py` | Auto-generate v2 Gold Standard from CSV |
| `run_qa_eval.py` | Downstream QA (100 questions: 67 direct + 33 reasoning) |
| `run_all_evaluations.py` | Batch evaluation across all datasets |
| `direct_llm_baseline.py` | Direct LLM baseline (CSV → triples, no pipeline) |
| `graph_loaders.py` | GraphML/JSON graph parsing utilities |
| `run_precision_analysis.py` | Graph topology + sampled precision analysis |
| `ablation_misclassify.py` | Stage 1 misclassification ablation (dry-run safe) |

## Round 6 Analysis Files

| File | Content |
|------|---------|
| `precision_analysis_v1.json` | Node/edge counts, isolated ratio, entity precision for 5 datasets |
| `precision_sample_who_life_expectancy.jsonl` | 25 sampled WHO edges, annotated (100% correct) |
| `precision_sample_inpatient.jsonl` | 25 sampled Inpatient edges, annotated (100% correct) |
| `qa_independent_10.jsonl` | 10 independently designed reasoning questions (5 comparison + 5 trend) |

## Deprecated Files (v1, superseded by v2)

| File | Superseded by |
|------|--------------|
| `all_results.json` | `all_results_v2.json` |
| `qa_results.json` | `qa_results_v3_100q.json` |
| `gold_who_life_expectancy.jsonl` | `gold_who_life_expectancy_v2.jsonl` |
| `gold_wb_child_mortality.jsonl` | `gold_wb_child_mortality_v2.jsonl` |
| `gold_wb_population.jsonl` | `gold_wb_population_v2.jsonl` |
| `gold_wb_maternal.jsonl` | `gold_wb_maternal_v2.jsonl` |

## Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **EC** | \|matched entities\| / \|gold entities\| | Entity identification |
| **FC** | \|matched facts\| / \|gold facts\| (2-hop substring) | Fact completeness |
| **η** | (FC_SGE/\|V_SGE\|) / (FC_base/\|V_base\|) | Per-node efficiency (small graphs only) |
| **Bootstrap CI** | 95% CI from 1000 resamples | Statistical significance (n≥150) |
| **Fisher's exact** | 2×2 contingency table | Small samples (n<30) |

## Gold Standard Files

### Local (manually annotated, single annotator)

| File | Dataset | Type | Facts |
|------|---------|------|-------|
| `gold_budget.jsonl` | Annual Budget | Type-II | 4e/20f |
| `gold_food_sample.jsonl` | Food Safety | Type-III | 17e/52f |
| `gold_health.jsonl` | Health Stats | Type-II-T | 3e/14f |
| `gold_inpatient_2023.jsonl` | Inpatient | Type-III | 8e/16f |

### International v2 (auto-generated from CSV, 25 countries × 6 years)

| File | Dataset | Facts |
|------|---------|-------|
| `gold_who_life_expectancy_v2.jsonl` | WHO Life Expectancy | 150 |
| `gold_wb_child_mortality_v2.jsonl` | WB Child Mortality | 150 |
| `gold_wb_population_v2.jsonl` | WB Population | 150 |
| `gold_wb_maternal_v2.jsonl` | WB Maternal Mortality | 150 |

**v2 generation**: 25 countries (GDP rank + geographic diversity), 6 years (2000/2005/2010/2015/2020/2022), values read directly from CSV cells (no truncation, no external source).

## Authoritative Result Files

⚠️ **These files are the single source of truth for all paper numbers:**

| File | Content |
|------|---------|
| `all_results_v2.json` | EC/FC for all datasets + Bootstrap CI |
| `qa_results_v2.json` | QA scores v2 (57/60 SGE, 36/60 Baseline) |
| `qa_results_v3_100q.json` | QA scores v3 (93/100 SGE, 59/100 Baseline) — **authoritative** |
| `direct_llm_results.json` | Direct LLM baseline numbers |

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
    "attributes": {"year": "2022-23", "status": "actual", "unit": "百万元"}
  },
  "annotator": "name",
  "confidence": "high"
}
```
