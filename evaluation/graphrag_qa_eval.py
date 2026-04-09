#!/usr/bin/env python3
"""
graphrag_qa_eval.py — MS GraphRAG QA Evaluation

Strategy: Direct parquet-based context retrieval + LLM answering.
  1. Load entities, relationships, text_units parquet files from graphrag output
  2. For each question, find matching entities by title substring search
  3. Collect context from related text_units and relationship descriptions
  4. Call LLM to answer the question
  5. Score: answer contains expected_value (using same logic as run_qa_eval.py)

Note: The CLI-based graphrag query is not usable because:
  - local/global search requires community_reports.parquet (not generated)
  - basic search requires Ollama embeddings (not running)
  This script bypasses the CLI and reads parquet files directly.

Usage:
    python3 evaluation/graphrag_qa_eval.py \
        --questions evaluation/gold/qa_questions.jsonl \
        --output evaluation/results/graphrag_qa_results.json \
        --verbose
"""

from __future__ import annotations

import re
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pandas or pyarrow not installed", file=sys.stderr)
    sys.exit(1)

from lightrag.llm.openai import openai_complete_if_cache

# ── Constants ─────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

MAX_CONTEXT_CHARS = 8000

DATASET_TO_GRAPHRAG_DIR: dict[str, str] = {
    "who":               "output/graphrag_who",
    "inpatient":         "output/graphrag_inpatient_full",
    "wb_child_mortality":"output/graphrag_wb_cm",
    "wb_population":     "output/graphrag_wb_pop",
    "wb_maternal":       "output/graphrag_wb_mat",
    "budget":            "output/graphrag_budget",
}

SYSTEM_ZH = (
    "你是一个知识图谱问答助手。根据下面提供的文本上下文，"
    "用简洁的中文回答问题。如果上下文包含具体数值，请直接引用该数值。"
    "只根据给定上下文回答，不要编造信息。"
)

SYSTEM_EN = (
    "You are a knowledge graph QA assistant. Answer the question concisely "
    "based on the provided context. If the context contains specific numeric "
    "values, quote them directly. Only use the provided context."
)


# ── Parquet loading ───────────────────────────────────────────────────────────

def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file and return as DataFrame."""
    return pq.read_table(str(path)).to_pandas()


def load_graphrag_index(graphrag_dir: str) -> dict[str, pd.DataFrame]:
    """Load entities, relationships, and text_units from graphrag output dir."""
    root = PROJECT_ROOT / graphrag_dir / "output"
    return {
        "entities":      _load_parquet(root / "entities.parquet"),
        "relationships": _load_parquet(root / "relationships.parquet"),
        "text_units":    _load_parquet(root / "text_units.parquet"),
    }


# ── Context building ──────────────────────────────────────────────────────────

def _find_matching_entities(entities: pd.DataFrame, entity_query: str) -> list[str]:
    """
    Return entity IDs matching entity_query with these priority passes:
    1. Exact title match (case-insensitive)
    2. query is a substring of title (case-insensitive)
    3. title (>=4 chars) is a substring of query — only when title is a prefix of query
    4. Description contains query (for country-code entities like CHN → "China...")
    """
    query_lower = entity_query.lower()
    matched_ids = []
    seen: set[str] = set()

    # Pass 1: exact title equals query
    for _, row in entities.iterrows():
        eid = str(row["id"])
        if str(row["title"]).lower() == query_lower:
            if eid not in seen:
                seen.add(eid)
                matched_ids.append(eid)

    if matched_ids:
        return matched_ids

    # Pass 2: query is a substring of title
    for _, row in entities.iterrows():
        eid = str(row["id"])
        title_lower = str(row["title"]).lower()
        if query_lower in title_lower:
            if eid not in seen:
                seen.add(eid)
                matched_ids.append(eid)

    if matched_ids:
        return matched_ids

    # Pass 3: description contains query (country-code entities with full name in description)
    for _, row in entities.iterrows():
        eid = str(row["id"])
        if eid in seen:
            continue
        desc_lower = str(row.get("description", "")).lower()
        if query_lower in desc_lower:
            seen.add(eid)
            matched_ids.append(eid)

    return matched_ids


def _extract_ids_from_field(value: object) -> list[str]:
    """Extract string IDs from a field that may be a list, numpy array, or None."""
    import numpy as np
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value if v is not None]
    return []


def _get_entity_text_unit_ids(entities: pd.DataFrame, entity_ids: list[str]) -> set[str]:
    """Collect all text_unit_ids referenced by the given entity IDs."""
    text_unit_ids: set[str] = set()
    for _, row in entities[entities["id"].isin(entity_ids)].iterrows():
        text_unit_ids.update(_extract_ids_from_field(row.get("text_unit_ids")))
    return text_unit_ids


def _get_relationship_text_unit_ids(
    relationships: pd.DataFrame, entity_ids: list[str]
) -> set[str]:
    """Collect text_unit_ids from relationships where source or target is in entity_ids."""
    id_set = set(entity_ids)
    mask = relationships["source"].isin(id_set) | relationships["target"].isin(id_set)
    text_unit_ids: set[str] = set()
    for _, row in relationships[mask].iterrows():
        text_unit_ids.update(_extract_ids_from_field(row.get("text_unit_ids")))
    return text_unit_ids


def _get_relationship_descriptions(
    relationships: pd.DataFrame, entity_ids: list[str]
) -> list[str]:
    """Get relationship description strings for context."""
    id_set = set(entity_ids)
    mask = relationships["source"].isin(id_set) | relationships["target"].isin(id_set)
    descs = []
    for _, row in relationships[mask].iterrows():
        src = str(row.get("source", ""))
        tgt = str(row.get("target", ""))
        desc = str(row.get("description", "")).strip()
        if desc:
            descs.append(f"[Relation] {src} → {tgt}: {desc}")
    return descs


def _search_text_units_direct(text_units: pd.DataFrame, query: str) -> list[str]:
    """
    Fallback: search text_unit content directly for query string.
    Returns matching text strings.
    """
    matches = []
    for _, row in text_units.iterrows():
        text = str(row.get("text", ""))
        if query in text or query.lower() in text.lower():
            matches.append(f"[Data] {text.strip()}")
    return matches


def build_context(
    index: dict[str, pd.DataFrame],
    entity_query: str,
) -> str:
    """
    Build text context for a question by:
    1. Finding entities matching entity_query in entity titles
    2. Collecting their descriptions and related text_units
    3. Fallback: search text_units directly for the query string
    4. Collecting relationship descriptions
    Returns concatenated context string (truncated to MAX_CONTEXT_CHARS).
    """
    entities = index["entities"]
    relationships = index["relationships"]
    text_units = index["text_units"]

    # Find matching entities by title
    entity_ids = _find_matching_entities(entities, entity_query)

    data_parts: list[str] = []     # raw data chunks (text_units)
    entity_descs: list[str] = []   # entity descriptions
    rel_descs: list[str] = []      # relationship descriptions

    if entity_ids:
        # Collect text_unit IDs from entities and relationships
        tu_ids = _get_entity_text_unit_ids(entities, entity_ids)
        tu_ids |= _get_relationship_text_unit_ids(relationships, entity_ids)

        # Add text_unit content first (raw CSV data with actual values)
        tu_id_set = set(str(i) for i in tu_ids)
        for _, row in text_units.iterrows():
            if str(row["id"]) in tu_id_set:
                text = str(row.get("text", "")).strip()
                if text:
                    data_parts.append(f"[Data] {text}")

        # Add entity descriptions for primary entities only (non-STAT_VALUE)
        for _, row in entities[entities["id"].isin(entity_ids)].iterrows():
            etype = str(row.get("type", "")).upper()
            if etype in ("STAT_VALUE",):
                continue  # skip stat entities — their data is already in text_units
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            if desc:
                entity_descs.append(f"[Entity] {title}: {desc}")

        # Collect relationship descriptions
        rel_descs = _get_relationship_descriptions(relationships, entity_ids)[:10]

    # Build context: data first (most important), then entity descriptions, then relations
    context_parts = data_parts + entity_descs + rel_descs

    # Fallback: search text_units directly for the query string
    if not context_parts:
        direct_matches = _search_text_units_direct(text_units, entity_query)
        context_parts.extend(direct_matches[:10])  # limit to 10 matching chunks

    if not context_parts:
        return f"[No context found for '{entity_query}']"

    full_context = "\n".join(context_parts)
    return full_context[:MAX_CONTEXT_CHARS]


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalize_number(s: str) -> str:
    """Normalize number strings: remove thousands separators, trailing .0"""
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\.0+\b", "", s)
    return s


def _match(answer: str, value: str) -> bool:
    """Case-insensitive substring match with number normalization."""
    if not value:
        return False
    al, vl = answer.lower(), value.lower()
    if vl in al:
        return True
    return _normalize_number(vl) in _normalize_number(al)


def score_answer(answer: str, q: dict) -> bool:
    """Return True if answer correctly addresses the question."""
    expected = q.get("expected_value", "")
    if _match(answer, expected):
        return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


# ── LLM ──────────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system_prompt: str) -> str:
    return await openai_complete_if_cache(
        MODEL, prompt,
        system_prompt=system_prompt,
        api_key=API_KEY, base_url=BASE_URL,
        max_tokens=256,
        timeout=120,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

_index_cache: dict[str, dict[str, pd.DataFrame]] = {}


def get_index(graphrag_dir: str) -> dict[str, pd.DataFrame]:
    """Load and cache graphrag index for a dataset directory."""
    if graphrag_dir not in _index_cache:
        _index_cache[graphrag_dir] = load_graphrag_index(graphrag_dir)
    return _index_cache[graphrag_dir]


async def eval_question(
    q: dict, graphrag_dir: str, language: str, verbose: bool
) -> dict:
    """Evaluate a single question against a graphrag index."""
    index = get_index(graphrag_dir)
    entity_query = q.get("entity", q["question"][:20])
    context = build_context(index, entity_query)

    system_prompt = SYSTEM_ZH if language == "zh" else SYSTEM_EN
    if language == "zh":
        prompt = f"文本上下文：\n{context}\n\n问题：{q['question']}"
    else:
        prompt = f"Context:\n{context}\n\nQuestion: {q['question']}"

    try:
        answer = await call_llm(prompt, system_prompt)
    except Exception as e:
        answer = f"[LLM Error: {e}]"

    correct = score_answer(answer, q)

    if verbose:
        print(f"    Context ({len(context)} chars): {context[:200]}...")
        print(f"    Answer [{'OK' if correct else 'FAIL'}]: {answer[:200]}")

    return {
        "answer": answer,
        "correct": correct,
        "context_length": len(context),
        "context_found": not context.startswith("[Entity") and not context.startswith("[No"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_eval(
    questions_path: str,
    output_path: str | None,
    verbose: bool,
    datasets_filter: list[str] | None,
) -> None:
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Filter to datasets that have a graphrag directory
    supported = set(DATASET_TO_GRAPHRAG_DIR.keys())
    questions = [
        q for q in questions
        if q["dataset"] in supported
        and (datasets_filter is None or q["dataset"] in datasets_filter)
    ]

    print(f"Evaluating {len(questions)} questions across MS GraphRAG\n")

    results = []
    dataset_names = list(dict.fromkeys(q["dataset"] for q in questions))

    for ds in dataset_names:
        graphrag_dir = DATASET_TO_GRAPHRAG_DIR[ds]
        graphrag_path = PROJECT_ROOT / graphrag_dir
        if not graphrag_path.exists():
            print(f"  SKIP {ds}: directory not found ({graphrag_path})")
            continue

        ds_qs = [q for q in questions if q["dataset"] == ds]
        lang = ds_qs[0]["language"]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()} ({lang}) — {len(ds_qs)} questions")
        print(f"GraphRAG dir: {graphrag_dir}")
        print(f"{'='*60}")

        for q in ds_qs:
            print(f"\n  [{q['id']}] {q['question']}")
            print(f"  Expected: {q['expected_value']}")

            result = await eval_question(q, graphrag_dir, lang, verbose)

            mark = "OK" if result["correct"] else "FAIL"
            if not verbose:
                print(f"  [{mark}]: {result['answer'][:120]}")

            results.append({
                "id":              q["id"],
                "dataset":         ds,
                "question":        q["question"],
                "expected_value":  q["expected_value"],
                "entity":          q.get("entity", ""),
                "type":            q.get("type", "direct"),
                "answer":          result["answer"],
                "correct":         result["correct"],
                "context_length":  result["context_length"],
                "context_found":   result["context_found"],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MS GRAPHRAG QA EVALUATION RESULTS")
    print(f"{'='*60}")

    datasets_summary: dict[str, dict] = {}
    header = f"{'Dataset':<20} | {'Correct':^10} | {'Accuracy':^10}"
    print(header)
    print("-" * len(header))

    for ds in dataset_names:
        ds_r = [r for r in results if r["dataset"] == ds]
        if not ds_r:
            continue
        n = len(ds_r)
        correct = sum(r["correct"] for r in ds_r)
        accuracy = correct / n
        print(f"{ds:<20} | {correct}/{n}      | {accuracy:.1%}")
        datasets_summary[ds] = {
            "total":    n,
            "correct":  correct,
            "accuracy": accuracy,
        }

    total = len(results)
    if total > 0:
        total_correct = sum(r["correct"] for r in results)
        total_accuracy = total_correct / total
        print("-" * len(header))
        print(f"{'OVERALL':<20} | {total_correct}/{total}      | {total_accuracy:.1%}")
    print(f"{'='*60}")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "system":    "ms_graphrag_v3.0.8_direct_parquet",
        "note":      (
            "Downgrade to v1.0.0 not possible (not on PyPI). "
            "Using v3.0.8 with direct parquet-based retrieval "
            "(CLI query unavailable: community_reports.parquet not generated, "
            "Ollama embeddings not running)."
        ),
        "datasets": datasets_summary,
        "results":  results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MS GraphRAG QA evaluation using direct parquet retrieval"
    )
    parser.add_argument(
        "--questions",
        default="evaluation/gold/qa_questions.jsonl",
        help="Path to QA questions JSONL file",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/graphrag_qa_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Limit to specific datasets (e.g. who inpatient wb_population)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full context and answer for each question",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_eval(
            questions_path=args.questions,
            output_path=args.output,
            verbose=args.verbose,
            datasets_filter=args.datasets,
        )
    )
