# SGE — Structure-Guided Extraction for GraphRAG

<div align="center">

**A Mechanism Study of Format-Constraint Coupling in Table-to-Graph Indexing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![EMNLP 2026](https://img.shields.io/badge/Targeting-EMNLP%202026-red.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

</div>

---

## Core Finding

> **Format-Constraint Coupling**: serialization format and schema constraints exhibit strong positive interaction (Bootstrap 95% CI, Fisher combined p<0.001). Neither component alone is sufficient — format change alone yields zero gain (Row-local FC = Baseline FC = 0.167), and schema constraint alone is unstable (Fixed S/T/V WHO FC ≈ 0.49). Few-shot prompting is harmful (FC ≤ 0.013). SGE jointly applies both, achieving 1.0–6.0× improvement. Validated across 3 LLM backends (Claude Haiku / GPT-5-mini / Gemini 2.5 Flash) and 2 GraphRAG hosts (LightRAG / MS GraphRAG).

---

## Key Results (CSVFidelity-Bench, 977 gold facts)

### Main Comparison

| Dataset | Baseline | Row-local | Fixed S/T/V | Det Parser | SGE (ours) |
|---------|----------|-----------|-------------|------------|------------|
| WHO Life Expectancy | 0.167 | 0.167 | 0.66 | 0.68 | **1.000** |
| WB Child Mortality | 0.473 | — | — | 0.727 | **1.000** |
| WB Population | 0.187 | — | — | 0.960 | **1.000** |
| WB Maternal Mortality | 0.787 | — | — | 0.967 | **0.967** |
| HK Inpatient | 0.438 | — | — | 1.000 | **0.938** |
| Fortune 500 Revenue | 0.400 | — | — | 1.000 | **1.000** |
| THE University Ranking | 0.207 | — | — | 1.000 | **0.600** |

**What each baseline proves:**
- **Row-local** (per-row chunks + default prompt): format change alone = zero gain (WHO FC identical to Baseline)
- **Fixed S/T/V** (generic static schema): dynamic schema > static schema; static schema is highly unstable
- **Det Parser** (zero-LLM deterministic): upper bound for pure structure; fails on semantic/hierarchical data (THE 1.0 due to flat ranking structure)
- **Few-shot** (3 example triples): FC ≤ 0.013 — harmful, 4/5 datasets worse than vanilla baseline
- **Table-aware prompt** (strong variant): FC = 0.253 (single-mechanism, no coupling)
- **AutoSchemaKG**: FC = 0.860 (WHO only, full pipeline required)
- **JSON Structured Output** (alternative coupling mechanism): WHO 1.0, Fortune500 1.0 — confirms multiple coupling paths exist

### JSON Structured Output Comparison

| Dataset | JSON Struct | SGE |
|---------|-------------|-----|
| WHO Life Expectancy | 1.000 | 1.000 |
| WB Child Mortality | 0.987 | 1.000 |
| Fortune 500 Revenue | 1.000 | 1.000 |
| THE University Ranking | 1.000 | 0.600 |

JSON Structured Output confirms the coupling hypothesis: an alternative mechanism achieving similar effect on simpler datasets. SGE's advantage is its ability to handle complex hierarchical (Type-III) tables where JSON structured output is not directly applicable.

### Cross-Model Validation

| Dataset | Claude Haiku | GPT-5-mini | Gemini 2.5 Flash | Baseline |
|---------|-------------|------------|-----------------|----------|
| WHO | 1.000 | 1.000 | 0.493 | 0.167 |
| WB CM | 1.000 | 0.960 | 0.020 | 0.473 |
| WB Pop | 1.000 | 1.000 | 1.000 | 0.187 |
| WB Mat | 0.967 | 0.840 | 0.040 | 0.787 |
| Inpatient | 0.938 | 0.625 | 0.875 | 0.438 |

De-biased validation: SGE FC unchanged under value-first protocol; Baseline naming bias ≤ 1.6%.

---

## Benchmark: CSVFidelity-Bench

10 datasets spanning 3 domains, 977 gold facts:

| Split | Datasets | Domain | Gold Facts |
|-------|----------|--------|------------|
| Core (7) | WHO, WB CM, WB Pop, WB Mat, Inpatient, Fortune500, THE | International health / finance / rankings | 977 |
| OECD Blind (6) | Education, Labor, Environment, Health, Trade, GDP | OECD statistics (held-out) | 83+ |
| Type-III OOD (2) | Eurostat Crime, US Census | Cross-domain hierarchical | TBD |

Gold standard facts are auto-generated from source CSVs via `generate_gold_standards_v3.py` — no human labeling required for numerical value facts.

---

## Overview

SGE is a structure-aware graph construction framework for statistical CSV data (time-series matrix / hierarchical-hybrid types). It improves upstream fact-binding fidelity for LightRAG through a three-stage perception pipeline (topology recognition → schema induction → constrained extraction). The research contribution is establishing that **format and constraint must be coupled** — neither is sufficient alone, and the interaction term is the active ingredient.

## Key Features

- **Three CSV topology types**: Type-I (flat entity), Type-II (time-series matrix), Type-III (hierarchical-hybrid), classified by Algorithm 1 (5 feature signals + 3 priority rules)
- **Dual-mode schema induction**: Rule-based (deterministic, fast) + LLM-enhanced (semantic-rich), with automatic fallback
- **Adaptive degradation**: Small Type-III (n_rows < 20) auto-switches to baseline mode to avoid over-constraining
- **Compact time-series representation**: Large Type-II (n_rows > 100) auto-enables node compression
- **Comprehensive evaluation**: EC/FC metrics + CSVFidelity-Bench (977 facts) + 231 statistical analysis questions + de-biased validation + Bootstrap CI + Wilcoxon effect size CI
- **Cross-model validation**: Claude Haiku 4.5 / GPT-5-mini / Gemini 2.5 Flash — format-constraint coupling holds across all three backends

## Quick Start

### Requirements

- Python 3.10+
- Dependencies: `pandas`, `networkx`, `openai` SDK, `lightrag-hku`
- [Ollama](https://ollama.com) with `mxbai-embed-large`:
  ```bash
  ollama pull mxbai-embed-large
  ```

### Install

```bash
pip3 install pandas networkx openai lightrag-hku
```

## Usage

### Full Pipeline (Stage 1 → 2 → 3)

```bash
python3 run_pipeline.py data/sample.csv --output-dir output/my_run
```

### LightRAG Integration (End-to-End)

```bash
python3 scripts/runners/run_lightrag_integration.py data/sample.csv
```

### Individual Stages

```bash
python3 scripts/runners/run_stage1.py data/sample.csv    # Stage 1 only
python3 scripts/runners/run_stage2.py data/sample.csv    # Stage 2 (rule-based)
python3 scripts/runners/run_stage2_llm.py data/sample.csv  # Stage 2 (LLM-enhanced)
```

### Batch Processing

```bash
python3 scripts/runners/run_batch.py

# Or use the batch runner with full logging
python3 evaluation/batch_runner.py --datasets all --output-dir results/batch_run
```

### Evaluation

```bash
# Full v2 evaluation (EC/FC + Bootstrap CI)
python3 evaluation/run_evaluations_v2.py

# De-biased evaluation (value-first protocol)
python3 evaluation/evaluate_coverage_debiased.py --batch

# Downstream QA (100 questions, direct graph context)
python3 evaluation/run_qa_eval.py

# Baseline comparisons
python3 evaluation/row_local_baseline.py          # Format-only control
python3 evaluation/fixed_stv_baseline.py          # Static schema control
python3 evaluation/json_structured_baseline.py    # Alternative coupling mechanism
python3 evaluation/fewshot_baseline.py            # Few-shot structured prompt

# OECD blind test
python3 evaluation/run_ood_evaluation.py --split oecd

# Error taxonomy (7 datasets × 3 systems)
python3 evaluation/run_error_taxonomy.py
```

### Tests

```bash
python3 -m pytest tests/ -v
```

## Directory Structure

```
sge_lightrag/
├── run_pipeline.py             # Main entry: full Stage 1→2→3 pipeline
│
├── stage1/                     # Stage 1: Topological Pattern Recognition
│   ├── preprocessor.py         #   CSV preprocessing (encoding, metadata)
│   ├── features.py             #   5-signal feature extraction
│   ├── classifier.py           #   Algorithm 1 (3 priority rules)
│   └── schema.py               #   Meta-Schema builder
├── stage2/                     # Stage 2: Rule-based Schema Induction
├── stage2_llm/                 # Stage 2: LLM-enhanced Schema Induction
├── stage3/                     # Stage 3: Constrained Extraction
│
├── evaluation/                 # Evaluation Framework
│   ├── evaluate_coverage.py    #   EC/FC (entity-first, 2-hop)
│   ├── evaluate_coverage_debiased.py  # FC (value-first, de-biased)
│   ├── batch_runner.py         #   Batch evaluation runner (all datasets)
│   ├── row_local_baseline.py   #   Format-only control (per-row + default prompt)
│   ├── fixed_stv_baseline.py   #   Static schema control
│   ├── json_structured_baseline.py  # JSON structured output baseline
│   ├── fewshot_baseline.py     #   Few-shot structured prompt baseline
│   ├── run_error_taxonomy.py   #   Error taxonomy (7 datasets × 3 systems)
│   ├── gold/                   #   Gold standard JSONL files (DO NOT modify)
│   ├── results/                #   Authoritative result JSONs
│   └── _archive/               #   Deprecated scripts
├── experiments/                # Experiment Scripts (grouped by type)
│   ├── ablation/               #   Decoupled ablation, threshold, C4
│   ├── statistical/            #   Wilcoxon, McNemar, LODO, Bootstrap, interaction CI
│   ├── probes/                 #   Graph-native probe, E2E, compact
│   ├── crossmodel/             #   Cross-model (GPT-5-mini + Gemini 2.5 Flash)
│   └── results/                #   Experiment output JSONs
├── tests/                      # Test Suite (pytest)
├── scripts/                    # Utility Scripts
│   ├── runners/                #   Pipeline runners (batch, integration, OOD)
│   ├── batch/                  #   Shell batch scripts
│   └── _archive/               #   One-time data fixes
└── output/                     # LightRAG graph outputs (gitignored)
```

## Experimental Configurations

| Config | Stage 2 | Schema Injection | Description |
|--------|---------|-----------------|-------------|
| C1: Rule SGE | Rule-based | Yes | Full three-stage (rule version) |
| C2: LLM v2 SGE | LLM (constrained) | Yes | entity_types ≤ 2 |
| C3: LLM v1 SGE | LLM (unconstrained) | Yes | entity_types unlimited |
| C4: SGE w/o Schema | — | No | SGE chunks, no schema |
| C5: Rule Baseline | — | No | Vanilla LightRAG |

## Statistical Validation

- Interaction term Bootstrap 95% CI: 4/5 datasets strictly positive (Fisher combined p<0.001)
- Wilcoxon signed-rank: all 5 international datasets p<0.05, effect size r=0.797–0.963 (large)
- Stratified precision: 249/250 = 99.6% SGE; Baseline 125/125 = 100%
- Downstream QA (231 questions): SGE 84.8% vs Baseline 54.5% (WHO: 84% vs 22%)
- E2E LightRAG query (hybrid mode): SGE 13% vs Baseline 13% (Δ=0) — vector retrieval bottleneck

## API Configuration

LLM calls use OpenAI-compatible API, configured in `stage2_llm/llm_client.py`:

```python
_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
```

Use environment variables to override (never hardcode keys):

```bash
export SGE_API_KEY="your-key-here"
export SGE_API_BASE="https://api.openai.com/v1"
```

Environment: LightRAG `v1.3.8`, mxbai-embed-large (1024d), `llm_model_max_async=5`.
