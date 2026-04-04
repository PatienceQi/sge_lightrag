# SGE — Structure-Guided Extraction for GraphRAG

<div align="center">

**Faithful Graph Construction from Statistical Tables for GraphRAG**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![EMNLP 2026](https://img.shields.io/badge/Targeting-EMNLP%202026-red.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

</div>

---

## Key Results

| Dataset | Baseline FC | SGE FC | Improvement |
|---------|-------------|--------|-------------|
| WHO Life Expectancy | 16.7% | **100%** | **6.0×** |
| WB Population | 18.7% | **100%** | **5.35×** |
| WB Child Mortality | 47.3% | **100%** | **2.11×** |
| HK Inpatient | 43.8% | **93.8%** | **2.14×** |

> **Core Discovery**: **Format-Constraint Coupling** — serialization format and schema constraints exhibit strong positive interaction (Bootstrap 95% CI, Fisher p<0.001). Neither component alone is sufficient; few-shot prompting is even harmful (FC≤0.013). Validated across 3 LLM backends (Claude Haiku / GPT-5-mini / Gemini 2.5 Flash) and 2 GraphRAG hosts (LightRAG / MS GraphRAG).

---

## Overview

SGE is a structure-aware graph construction framework for statistical CSV data (time-series matrix / hierarchical-hybrid types). It improves upstream fact-binding fidelity for LightRAG through a three-stage perception pipeline (topology recognition → schema induction → constrained extraction).

## Key Features

- **Three CSV topology types**: Type-I (flat entity), Type-II (time-series matrix), Type-III (hierarchical-hybrid), classified by Algorithm 1 (5 feature signals + 3 priority rules)
- **Dual-mode schema induction**: Rule-based (deterministic, fast) + LLM-enhanced (semantic-rich), with automatic fallback
- **Adaptive degradation**: Small Type-III (n_rows < 20) auto-switches to baseline mode to avoid over-constraining
- **Compact time-series representation**: Large Type-II (n_rows > 100) auto-enables node compression
- **Comprehensive evaluation**: EC/FC metrics + Gold Standard (977 facts) + 231 statistical analysis questions + de-biased validation + Bootstrap CI + Wilcoxon effect size CI
- **Cross-model validation**: Claude Haiku 4.5 / GPT-5-mini / Gemini 2.5 Flash — format-constraint coupling holds across all three

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
```

### Evaluation

```bash
# Full v2 evaluation (EC/FC + Bootstrap CI)
python3 evaluation/run_evaluations_v2.py

# De-biased evaluation (value-first protocol)
python3 evaluation/evaluate_coverage_debiased.py --batch

# Downstream QA (100 questions, direct graph context)
python3 evaluation/run_qa_eval.py
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
│   ├── gold/                   #   Gold standard JSONL files
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

## Key Results (v2 Gold Standard, 977 facts)

| Dataset | SGE FC | Baseline FC | Ratio |
|---------|--------|-------------|-------|
| WHO Life Expectancy (25 countries × 150 facts) | 1.000 | 0.167 | **6.0×** |
| WB Population (25 countries × 150 facts) | 1.000 | 0.187 | **5.35×** |
| WB Child Mortality (25 countries × 150 facts) | 1.000 | 0.473 | **2.11×** |
| HK Inpatient (318 ICD categories) | 0.938 | 0.438 | **2.14×** |
| WB Maternal Mortality (25 countries × 150 facts) | 0.967 | 0.787 | **1.23×** |
| Fortune 500 Revenue | 1.000 | 0.400 | **2.50×** |
| THE University Ranking | 0.600 | 0.207 | **2.90×** |

### Cross-Model Validation

| Dataset | Claude Haiku | GPT-5-mini | Gemini 2.5 Flash |
|---------|-------------|------------|-----------------|
| WHO | 1.000 | 1.000 | 0.493 |
| WB Pop | 1.000 | 1.000 | 1.000 |
| Inpatient | 0.938 | 0.625 | 0.875 |

De-biased validation: SGE FC unchanged under value-first protocol; Baseline naming bias ≤ 1.6%.

## API Configuration

LLM calls use OpenAI-compatible API, configured in `stage2_llm/llm_client.py`:

```python
_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
```

Environment: LightRAG `v1.3.8`, mxbai-embed-large (1024d), `llm_model_max_async=5`.
