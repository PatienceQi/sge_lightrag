# Pipeline Runners

| Script | Purpose |
|--------|---------|
| `run_batch.py` | Batch processing all CSVs under dataset/ |
| `run_all_tests.py` | Stage 1 classification on all 33 files |
| `run_lightrag_integration.py` | SGE → LightRAG end-to-end (rule-based Stage 2) |
| `run_lightrag_integration_llm.py` | SGE → LightRAG end-to-end (LLM Stage 2) |
| `run_ood_llm_sge_only.py` | OOD LLM-enhanced SGE (3 failed datasets) |
| `run_baseline_only.py` | LightRAG baseline (no SGE) |
| `run_lightrag_from_output.py` | Re-run LightRAG from existing SGE output |
| `run_multi_csv_comparison.py` | Multi-CSV SGE vs Baseline comparison |
| `run_stage1.py` | Stage 1 only (classification) |
| `run_stage2.py` | Stage 2 only (rule-based schema induction) |
| `run_stage2_llm.py` | Stage 2 only (LLM-enhanced schema induction) |
| `run_non_gov_experiments.py` | Non-government dataset experiments (Fortune 500, THE Ranking) |

## Path Convention

All scripts use `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` to locate `sge_lightrag/`.
