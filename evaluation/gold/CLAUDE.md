# Gold Standards

**DO NOT modify without explicit instruction.** These are the evaluation ground truth.

## v2 International (Paper primary, 25 countries × 6 years = 150 facts each)

| File | Dataset | Facts | Source |
|------|---------|------:|--------|
| `gold_who_life_expectancy_v2.jsonl` | WHO Life Expectancy | 150 | Auto-generated from CSV |
| `gold_wb_child_mortality_v2.jsonl` | WB Child Mortality | 150 | Auto-generated from CSV |
| `gold_wb_population_v2.jsonl` | WB Population | 150 | Auto-generated from CSV |
| `gold_wb_maternal_v2.jsonl` | WB Maternal Mortality | 150 | Auto-generated from CSV |

## v3 International (Extended, same 25 countries, more years)

| File | Dataset |
|------|---------|
| `gold_who_life_expectancy_v3.jsonl` | WHO |
| `gold_wb_child_mortality_v3.jsonl` | WB CM |
| `gold_wb_population_v3.jsonl` | WB Pop |
| `gold_wb_maternal_v3.jsonl` | WB Maternal |

## Local (Manually annotated)

| File | Dataset | Type | Entities | Facts |
|------|---------|------|:--------:|------:|
| `gold_budget.jsonl` | Annual Budget | II | 4 | 20 |
| `gold_food_sample.jsonl` | Food Safety | III | 17 | 52 |
| `gold_health.jsonl` | Health Statistics | II-T | 3 | 14 |
| `gold_inpatient_2023.jsonl` | Inpatient | III | 8 | 16 |

## Non-Government (auto-generated, non-gov domains)

| File | Dataset | Type | Entities/Facts |
|------|---------|------|---------------|
| `gold_fortune500_revenue.jsonl` | Fortune 500 Revenue | Type-II | 25e/125f |
| `gold_the_university_ranking.jsonl` | THE University Ranking | Type-III | 25e/150f |

## OOD (10 blind test datasets, 40 facts each except Education 23, Literacy 10)

| File | Domain |
|------|--------|
| `gold_ood_wb_cereal_production.jsonl` | Agriculture |
| `gold_ood_wb_co2_emissions.jsonl` | Environment |
| `gold_ood_wb_population_growth.jsonl` | Population |
| `gold_ood_wb_education_spending.jsonl` | Education |
| `gold_ood_wb_literacy_rate.jsonl` | Education |
| `gold_ood_wb_health_expenditure.jsonl` | Public Health |
| `gold_ood_wb_gdp_growth.jsonl` | Economy |
| `gold_ood_wb_unemployment.jsonl` | Labor |
| `gold_ood_wb_immunization_dpt.jsonl` | Public Health |
| `gold_ood_wb_immunization_measles.jsonl` | Public Health |

## QA & Precision

| File | Content |
|------|---------|
| `qa_questions.jsonl` | 100 QA questions (67 direct + 33 reasoning) |
| `qa_independent_10.jsonl` | 10 independent reasoning questions (not yet evaluated) |
| `precision_sample_who_life_expectancy.jsonl` | 25 sampled edges (WHO, seed=42) |
| `precision_sample_inpatient.jsonl` | 25 sampled edges (Inpatient, seed=42) |
| `annotation_schema.json` | JSONL format specification |

## Totals

- **Paper primary (v2):** 977 facts (600 international + 102 local + 275 non-gov)
- **OOD:** 353 facts (10 datasets)
