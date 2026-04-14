# SGE — Structure-Guided Extraction for GraphRAG

<div align="center">

**Format-Constraint Coupling in Knowledge Graph Construction from Matrix-Layout Statistical Tables**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Under%20Review-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

</div>

---

## Core Finding

> **Format-Constraint Coupling**: serialization format and schema constraints interact *super-additively* (2×2 factorial, Bootstrap 95% CI strictly positive on 4/6 datasets, Fisher combined p<0.001). Mismatched schemas degrade extraction *below* baseline on 4/6 datasets — entity inflation (3.47×) and extraction refusal (43.6%). Token ablation traces the mechanism to **surface-form anchoring**: the LLM matches schema field names to column-label *tokens* (not positions — shuffling row/field order preserves FC). Validated across 3 format-schema pairings, 3 LLM backends (Claude Haiku / GPT-5-mini / Gemini Flash), and 2 GraphRAG hosts. **Evaluation blindness**: all 3 standard retrieval modes mask fidelity differences (Δ≤1pp); direct graph access exposes +47.6pp gaps.

---

## Key Results (CSVFidelity-Bench, 1,892 gold facts)

### Main Comparison

| Dataset | Baseline | Row-local | Fixed S/T/V | Det Parser | SGE (ours) |
|---------|----------|-----------|-------------|------------|------------|
| WHO Life Expectancy | 0.170 | 0.167 | 0.66 | 0.68 | **1.000** |
| WB Child Mortality | 0.433 | — | — | 0.727 | **1.000** |
| WB Population | 0.133 | — | — | 0.960 | **1.000** |
| WB Maternal Mortality | 0.820 | — | — | 0.967 | **0.973** |
| HK Inpatient | 0.438 | — | — | 1.000 | **0.938** |
| Fortune 500 Revenue | 0.400 | — | — | 1.000 | **1.000** |
| THE University Ranking | 0.207 | — | — | 1.000 | **0.600** |
| *OOD: WB Cereal Prod.* | 0.050 | — | — | — | **0.950** (19×) |
| *OOD: WB CO₂ Emissions* | 0.025 | — | — | — | **0.700** (28×) |
| *OOD: WB Pop. Growth* | 0.075 | — | — | — | **0.625** (8.3×) |
| *Scope: Eurostat Crime* | **0.410** | — | — | — | 0.000 |
| *Scope: US Census Demo.* | 0.022 | — | — | — | **0.244** (11×) |

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

**Schema-only (WHO, descriptive columns):** Claude Sonnet 4.6 FC=0.667, GPT-5-mini FC=0.963, Haiku FC=0.480 — stronger models show better fallback grounding but coupling still adds value (Full SGE=1.000).

De-biased validation: SGE FC unchanged under value-first protocol; Baseline naming bias ≤ 1.6%.

### Token-Level Input Ablation

7 conditions manipulating input tokens while keeping schema fixed (20 gold-filtered chunks, Claude Haiku):

| Condition | WHO FC | WB Pop FC | Interpretation |
|-----------|--------|-----------|----------------|
| M0 Control | 0.400* | 0.220 | Baseline (20/50 entities) |
| M1 Mask labels | 0.400 | 0.183 ↓ | Labels needed only on non-descriptive cols |
| M3 Mask entities | 0.020 ↓↓ | 0.180 ↓ | Entity names necessary for binding |
| M4 Mask values | 0.000† | 0.000† | Values don't participate in anchoring (WHO EC=0.400 intact) |
| M5 Shuffle intra-row | 0.400 | 0.220 | **Field order irrelevant** → surface-form, not positional |
| M6 Shuffle inter-row | 0.400 | 0.240 | **Row order irrelevant** → row-local anchoring |

*Ceiling for 20-chunk subset. †Metric artifact (gold values replaced with 0.000).

**Core finding**: Coupling operates via **surface-form anchoring** — the LLM matches schema field names to column-label tokens regardless of position, moderated by column descriptiveness (CDS).

---

## Benchmark: CSVFidelity-Bench

15 datasets spanning 6 domains, 1,892 gold facts:

| Split | Datasets | Domain | Gold Facts |
|-------|----------|--------|------------|
| Core (7) | WHO, WB CM, WB Pop, WB Mat, Inpatient, Fortune500, THE | Health / finance / rankings | 977 |
| OOD (3) | WB Cereal, WB CO2, WB Pop Growth | Agriculture / environment | 120 |
| Long-format (2) | Eurostat Crime, US Census | Type-III scope boundary | 195 |
| OECD Blind (6) | Education, Labor, Environment, Health, Trade, GDP | OECD statistics (held-out) | 83+ |
| Type-III OOD (2) | Eurostat Crime, US Census | Cross-domain hierarchical | TBD |

Gold standard facts are auto-generated from source CSVs via `generate_gold_standards_v3.py` — no human labeling required for numerical value facts.

---

## Overview

SGE is a *controlled experimental apparatus* (not a production system) for independently manipulating format and schema in knowledge graph construction from statistical CSV data. Its three-stage pipeline (topology recognition → schema induction → constrained extraction) serves as a vehicle to establish that **format and constraint interact super-additively** — the coupling phenomenon, not the pipeline itself, is the contribution. Deterministic parsers already achieve FC≥0.96 on 5/7 well-structured datasets; SGE's value is as a mechanistic probe revealing format-constraint coupling with implications for any LLM-based schema-guided pipeline.

## Key Features

- **Three CSV topology types**: Type-I (flat entity), Type-II (time-series matrix), Type-III (hierarchical-hybrid), classified by Algorithm 1 (5 feature signals + 3 priority rules)
- **Dual-mode schema induction**: Rule-based (deterministic, fast) + LLM-enhanced (semantic-rich), with automatic fallback
- **Adaptive degradation**: Small Type-III (n_rows < 20) auto-switches to baseline mode to avoid over-constraining
- **Compact time-series representation**: Large Type-II (n_rows > 100) auto-enables node compression
- **Comprehensive evaluation**: EC/FC metrics + CSVFidelity-Bench (1,892 facts) + 231 statistical analysis questions + de-biased validation + Bootstrap CI + Wilcoxon effect size CI
- **Cross-model validation**: Claude Haiku 4.5 / Sonnet 4.6 / GPT-5-mini / Gemini 2.5 Flash — format-constraint coupling holds across all four backends (Sonnet Schema-only FC=0.667 confirms coupling value even for stronger models)

## Quick Start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with `mxbai-embed-large`:
  ```bash
  ollama pull mxbai-embed-large
  ```

### Install

```bash
pip3 install -r requirements.txt
```

### Configuration

Copy the environment template and fill in your API key:

```bash
cp .env.example .env
# Edit .env with your API key (OpenAI-compatible endpoint)
source .env
```

### Dataset Setup

All 15 CSVFidelity-Bench datasets are bundled in `dataset/` within this repo (~5 MB):

```
dataset/
├── WHO/                    # WHO Life Expectancy (Type-II)
├── 世界银行数据/             # World Bank: CM, Population, Maternal Mortality
├── 住院病人统计/             # HK Inpatient Statistics (Type-III, 12 yearly CSVs)
├── non_gov/                # Fortune 500 Revenue, THE University Ranking
├── ood_blind_test/         # 15 OOD CSVs (WB indicators + synthetic long-format)
├── OECD_blind_test/        # 4 OECD CSVs (held-out factorial validation)
├── expanded/               # IMF, UN Census, WB long-format
└── README.md               # Data sources, licenses, download URLs
```

No external downloads needed — `git clone` gives you everything.

### Reproducibility

```bash
# Verify all pre-computed results match paper numbers (zero API cost):
./reproduce.sh verify

# Re-run a specific table's experiments (requires API key):
./reproduce.sh regenerate table2
```

See `TABLE_SCRIPT_MAP.md` for the full paper-table-to-script mapping.
For exact version reproduction, use `requirements-pinned.txt`.

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

# Error taxonomy (7 datasets × 3 systems)
python3 evaluation/run_error_taxonomy.py

# Token-level input ablation (7 conditions, requires API key)
python3 experiments/ablation/run_token_ablation.py --max-chunks 20
python3 experiments/ablation/run_token_ablation.py --dry-run  # preview without API calls
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
│   ├── evaluate_coverage.py    #   EC/FC metrics (entity-first, 2-hop)
│   ├── evaluate_coverage_debiased.py  # FC (value-first, de-biased)
│   ├── row_local_baseline.py   #   Format-only control baseline
│   ├── fixed_stv_baseline.py   #   Static schema control baseline
│   ├── json_structured_baseline.py  # Alternative coupling baseline
│   ├── fewshot_baseline.py     #   Few-shot structured prompt baseline
│   ├── table_aware_baseline.py #   Table-aware prompt baseline
│   ├── deterministic_parser_baseline.py  # Zero-LLM deterministic parser
│   ├── gold/                   #   Gold standard JSONL (1,892 facts)
│   └── results/                #   Authoritative result JSONs
├── experiments/                # Reproducibility Scripts
│   ├── ablation/               #   Decoupled ablation, factorial, probing
│   │   ├── run_decoupled_ablation.py     # Schema-only condition (main datasets)
│   │   ├── run_c4_serialization_only.py  # Serial-only condition
│   │   ├── run_ood_schema_only.py        # OOD Schema-only (CDS validation, 3 datasets)
│   │   ├── run_sonnet_factorial.py       # Sonnet 4.6 factorial (model scale test)
│   │   ├── run_token_ablation.py         # Token-level input ablation (M0-M6, 7 conditions)
│   │   └── ...                           # health_exp, probing, 50c, etc.
│   ├── statistical/            #   Interaction CI, Wilcoxon, hierarchical bootstrap
│   ├── crossmodel/             #   Cross-model (GPT-5-mini + Gemini 2.5 Flash)
│   ├── analysis/               #   CDS, TTF, det parser scope, prevalence
│   └── results/                #   Experiment output JSONs
├── tests/                      # Test Suite (pytest, 284 tests)
├── scripts/runners/            #   Pipeline runners (integration, OOD)
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

LLM calls use an OpenAI-compatible API. Configure via environment variables (see `.env.example`):

```bash
export SGE_API_KEY="your-key-here"
export SGE_API_BASE="https://api.openai.com/v1"
export SGE_MODEL="claude-haiku-4-5-20251001"
```

Environment: LightRAG `v1.3.8`, mxbai-embed-large (1024d), `llm_model_max_async=5`.
