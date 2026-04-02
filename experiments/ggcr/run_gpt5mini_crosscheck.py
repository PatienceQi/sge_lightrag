#!/usr/bin/env python3
"""
run_gpt5mini_crosscheck.py — GGCR cross-model robustness check with GPT-5-mini.

Same 97 questions, same 3 LLM systems (sge_ggcr, pure_compact, concat_all),
but using gpt-5-mini instead of Claude Haiku. Results saved separately.

Usage:
    python3 experiments/ggcr/run_gpt5mini_crosscheck.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override MODEL before importing retriever
import experiments.ggcr.graph_guided_retriever as retriever
retriever.MODEL = "gpt-5-mini"
print(f"Model overridden to: {retriever.MODEL}")

# Override results path
import experiments.ggcr.run_ggcr_eval as eval_module
eval_module.RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ggcr_results_gpt5mini.json"

# Run evaluation (LLM systems only, graph_native is model-independent)
eval_module.run_evaluation(
    systems=["sge_ggcr", "pure_compact", "concat_all"],
    verbose=False,
)
