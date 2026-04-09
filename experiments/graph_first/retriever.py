#!/usr/bin/env python3
"""
retriever.py — Graph-first retrieval pipeline for statistical QA.

Pipeline:
  1. parse_statistical_query(question) → structured query dict
  2. build_graph_context(graph_path, parsed_query) → context string
  3. If context is empty → fallback to compact chunk vector retrieval
  4. _call_llm(context, question) → answer string

Supports both SGE and Baseline graph paths across all 7 datasets.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests as _requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.graph_first.query_parser import parse_statistical_query
from experiments.graph_first.graph_context import build_graph_context

# ── LLM config ────────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("SGE_API_KEY", "")
_BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT_EN = (
    "You are a precise QA assistant for statistical data. "
    "Answer based ONLY on the provided graph context. "
    "When asked for a ranking, list entities in order. "
    "When asked for a count or average, compute precisely from the data. "
    "Be concise. If the data is insufficient, say so."
)

_SYSTEM_PROMPT_ZH = (
    "你是一个统计数据问答助手。只根据提供的图谱上下文回答。"
    "如果问题要求排名，请按顺序列出。"
    "如果问题要求数量或均值，请精确计算。"
    "回答简洁明了。如果数据不足，请说明。"
)

# ── Graph path registry (all 7 main datasets) ─────────────────────────────────
SGE_GRAPH_PATHS: dict[str, str] = {
    "who":        "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_cm":      "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_pop":     "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_mat":     "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
    "inpatient":  "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
    "fortune500": "output/fortune500_revenue/lightrag_storage/graph_chunk_entity_relation.graphml",
    "the":        "output/the_university_ranking/lightrag_storage/graph_chunk_entity_relation.graphml",
}

BASELINE_GRAPH_PATHS: dict[str, str] = {
    "who":        "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_cm":      "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_pop":     "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_mat":     "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
    "inpatient":  "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
    "fortune500": "output/baseline_fortune500_revenue/lightrag_storage/graph_chunk_entity_relation.graphml",
    "the":        "output/baseline_the_university_ranking/lightrag_storage/graph_chunk_entity_relation.graphml",
}

# ── Compact chunk fallback (reuse existing GGCR compact index for 3 datasets) ──
_COMPACT_DATASETS = {"who", "wb_cm", "inpatient"}


def _call_llm(
    context: str,
    question: str,
    language: str = "en",
    max_retries: int = 3,
) -> str:
    """Call Claude Haiku via wolfai proxy with retry."""
    system = _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT_EN
    for attempt in range(max_retries):
        try:
            resp = _requests.post(
                f"{_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Graph context:\n{context}\n\nQuestion: {question}"},
                    ],
                    "max_tokens": 512,
                    "temperature": 0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return f"[LLM Error: {exc}]"


def _fallback_compact_context(question: str, dataset: str) -> str:
    """
    Fallback: vector retrieval on compact chunks when graph match fails.
    Only available for datasets that have compact indexes (who, wb_cm, inpatient).
    """
    if dataset not in _COMPACT_DATASETS:
        return f"[No graph match found for dataset={dataset}]"

    try:
        from experiments.ggcr.compact_chunks import build_compact_index
        index = build_compact_index(dataset, embed=True)
        chunks = index.vector_retrieve(question, top_k=5)
        return "\n---\n".join(chunks)
    except Exception as exc:
        return f"[Compact fallback failed: {exc}]"


def retrieve_graph_first(
    question: str,
    dataset: str,
    system: str = "sge",
    max_context_tokens: int = 3000,
) -> dict:
    """
    Graph-first retrieval pipeline.

    Args:
        question: Natural language question (English or Chinese).
        dataset: Dataset key, one of: who, wb_cm, wb_pop, wb_mat,
                 inpatient, fortune500, the.
        system: "sge" or "baseline" — which graph to query.
        max_context_tokens: Token budget for graph context.

    Returns:
        Dict with keys:
          answer (str): LLM-generated answer
          context_source (str): "graph" or "compact_fallback" or "no_context"
          parsed_query (dict): Structured query from parser
          context_length (int): Character count of context used
    """
    path_registry = SGE_GRAPH_PATHS if system == "sge" else BASELINE_GRAPH_PATHS
    graph_rel_path = path_registry.get(dataset)
    graph_abs_path = str(PROJECT_ROOT / graph_rel_path) if graph_rel_path else None

    # Step 1: Parse question
    parsed = parse_statistical_query(question)

    # Step 2: Build graph context
    context = ""
    context_source = "no_context"

    if graph_abs_path:
        context = build_graph_context(graph_abs_path, parsed, max_context_tokens)
        if context:
            context_source = "graph"

    # Step 3: Fallback to compact chunks if no graph match
    if not context:
        context = _fallback_compact_context(question, dataset)
        context_source = "compact_fallback" if not context.startswith("[") else "no_context"

    # Step 4: LLM answer generation
    language = "zh" if any(
        "\u4e00" <= ch <= "\u9fff" for ch in question
    ) else "en"
    answer = _call_llm(context, question, language)

    return {
        "answer": answer,
        "context_source": context_source,
        "parsed_query": parsed,
        "context_length": len(context),
    }


if __name__ == "__main__":
    result = retrieve_graph_first(
        "What was China's life expectancy in 2020?",
        dataset="who",
        system="sge",
    )
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['context_source']}")
    print(f"Parsed: {result['parsed_query']}")
