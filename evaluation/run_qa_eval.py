#!/usr/bin/env python3
"""
run_qa_eval.py — Downstream QA Evaluation for SGE-LightRAG

Strategy: Direct graph context retrieval + LLM answering.
  1. Load graphml
  2. Find entities matching the question's key entity (substring match)
  3. Collect 1-2 hop context (entity description + connected edge texts)
  4. Call LLM: "Given this KG context, answer the question"
  5. Score: answer contains expected_value (case-insensitive substring)

This approach directly tests whether the graph has the information needed,
independent of LightRAG's internal query pipeline quirks.

Usage:
    python3 evaluation/run_qa_eval.py \
        --questions evaluation/gold/qa_questions.jsonl \
        --output evaluation/results/qa_results.json \
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
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed", file=sys.stderr)
    sys.exit(1)

from lightrag.llm.openai import openai_complete_if_cache

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

SYSTEM_ZH = (
    "你是一个知识图谱问答助手。根据下面提供的知识图谱上下文，"
    "用简洁的中文回答问题。如果上下文包含具体数值，请直接引用该数值。"
    "只根据给定上下文回答，不要编造信息。"
)

SYSTEM_EN = (
    "You are a knowledge graph QA assistant. Answer the question concisely "
    "based on the provided knowledge graph context. If the context contains "
    "specific numeric values, quote them directly. Only use the provided context."
)


async def call_llm(prompt: str, system_prompt: str) -> str:
    return await openai_complete_if_cache(
        MODEL, prompt,
        system_prompt=system_prompt,
        api_key=API_KEY, base_url=BASE_URL,
        max_tokens=256,
        timeout=120,
    )


def load_graph_context(graphml_path: str, entity_query: str, max_hops: int = 2) -> str:
    """
    Load a graphml and extract text context for the given entity query.
    Returns concatenated entity descriptions + edge keywords/descriptions.
    """
    try:
        G = nx.read_graphml(graphml_path)
    except Exception as e:
        return f"[Graph load error: {e}]"

    # Build name → node_id index (case-insensitive)
    name_to_id: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        name = str(data.get("entity_name") or data.get("name") or nid).strip()
        name_to_id[name.lower()] = nid

    # Find matching node(s) via substring match
    entity_lower = entity_query.lower()
    matched_ids = []
    for name_l, nid in name_to_id.items():
        if entity_lower in name_l or name_l in entity_lower:
            matched_ids.append(nid)

    if not matched_ids:
        return f"[Entity '{entity_query}' not found in graph]"

    # Collect context: node description + edge texts up to max_hops
    context_parts = []
    visited = set(matched_ids)

    frontier = list(matched_ids)
    for hop in range(max_hops):
        next_frontier = []
        for nid in frontier:
            node_data = G.nodes[nid]
            name = str(node_data.get("entity_name") or node_data.get("name") or nid).strip()
            desc = str(node_data.get("description", "")).strip()
            if desc:
                context_parts.append(f"[Entity] {name}: {desc}")

            for nb_id in G.neighbors(nid):
                edge_data = G.edges[nid, nb_id] if G.has_edge(nid, nb_id) else {}
                nb_name = str(G.nodes[nb_id].get("entity_name") or
                              G.nodes[nb_id].get("name") or nb_id).strip()
                kw   = str(edge_data.get("keywords", "")).strip()
                edesc = str(edge_data.get("description", "")).strip()
                if kw or edesc:
                    context_parts.append(
                        f"[Edge] {name} → {nb_name}: {kw} {edesc}".strip()
                    )
                if nb_id not in visited:
                    visited.add(nb_id)
                    next_frontier.append(nb_id)

        frontier = next_frontier
        if not frontier:
            break

    if not context_parts:
        return f"[No context found for '{entity_query}']"

    # Limit context to ~3000 chars to stay within token budget
    full_context = "\n".join(context_parts)
    return full_context[:3000]


def _normalize_number(s: str) -> str:
    """Normalize number strings for comparison.
    Removes thousands separators (comma/Chinese fullwidth comma) and trailing .0
    so that '196,617.0' matches '196617', '77.6162' matches '77.61', etc.
    """
    s = s.replace(',', '').replace('，', '')
    # Remove trailing .0 / .00 (whole numbers stored as floats)
    s = re.sub(r'\.0+\b', '', s)
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
    """Return True if the answer correctly addresses the question.
    - direct: expected_value must appear in answer
    - comparison/trend: expected_value OR secondary_value (numeric evidence) suffices
    """
    expected = q.get("expected_value", "")
    if _match(answer, expected):
        return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


async def eval_question(q: dict, graph_dir: str, language: str, verbose: bool) -> dict:
    """Evaluate a single question against a graph."""
    graphml_path = str(PROJECT_ROOT / graph_dir / "graph_chunk_entity_relation.graphml")
    entity_query = q.get("entity", q["question"][:20])

    context = load_graph_context(graphml_path, entity_query)
    system_prompt = SYSTEM_ZH if language == "zh" else SYSTEM_EN

    prompt = (
        f"知识图谱上下文：\n{context}\n\n问题：{q['question']}"
        if language == "zh"
        else f"Knowledge Graph Context:\n{context}\n\nQuestion: {q['question']}"
    )

    try:
        answer = await call_llm(prompt, system_prompt)
    except Exception as e:
        answer = f"[LLM Error: {e}]"

    correct = score_answer(answer, q)

    if verbose:
        print(f"    Context ({len(context)} chars): {context[:150]}...")
        print(f"    Answer [{'✓' if correct else '✗'}]: {answer[:200]}")

    return {
        "answer": answer,
        "correct": correct,
        "context_length": len(context),
        "context_found": not context.startswith("[Entity") and not context.startswith("[Graph"),
    }


async def run_eval(questions_path: str, output_path: str | None, verbose: bool):
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions\n")

    results = []

    # Process questions grouped by dataset for cleaner output
    datasets = list(dict.fromkeys(q["dataset"] for q in questions))

    for ds in datasets:
        ds_qs = [q for q in questions if q["dataset"] == ds]
        lang  = ds_qs[0]["language"]
        sge_path  = ds_qs[0]["sge_graph"]
        base_path = ds_qs[0]["baseline_graph"]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()} ({lang}) — {len(ds_qs)} questions")
        print(f"{'='*60}")

        for q in ds_qs:
            print(f"\n  [{q['id']}] {q['question']}")
            print(f"  Expected: {q['expected_value']}")

            sge_result  = await eval_question(q, sge_path,  lang, verbose)
            base_result = await eval_question(q, base_path, lang, verbose)

            sge_mark  = '✓' if sge_result['correct']  else '✗'
            base_mark = '✓' if base_result['correct'] else '✗'

            if not verbose:
                print(f"  SGE [{sge_mark}]: {sge_result['answer'][:120]}")
                print(f"  Base[{base_mark}]: {base_result['answer'][:120]}")

            results.append({
                "id":             q["id"],
                "dataset":        ds,
                "question":       q["question"],
                "expected_value": q["expected_value"],
                "entity":         q.get("entity", ""),
                "sge_answer":     sge_result["answer"],
                "sge_correct":    sge_result["correct"],
                "sge_context_found": sge_result["context_found"],
                "baseline_answer":   base_result["answer"],
                "baseline_correct":  base_result["correct"],
                "baseline_context_found": base_result["context_found"],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("QA EVALUATION RESULTS")
    print(f"{'='*60}")

    header = f"{'Dataset':<12} | {'SGE':^9} | {'Baseline':^9} | Δ"
    print(header)
    print("-" * len(header))

    summary = {}
    for ds in datasets:
        ds_r = [r for r in results if r["dataset"] == ds]
        n   = len(ds_r)
        sge = sum(r["sge_correct"]      for r in ds_r)
        base = sum(r["baseline_correct"] for r in ds_r)
        delta = sge - base
        sign  = "+" if delta > 0 else ""
        print(f"{ds:<12} | {sge}/{n} ({sge/n:.0%})  | {base}/{n} ({base/n:.0%}) | {sign}{delta}")
        summary[ds] = {"n": n, "sge_correct": sge, "baseline_correct": base,
                       "sge_accuracy": sge / n, "baseline_accuracy": base / n}

    total = len(results)
    sge_t  = sum(r["sge_correct"]      for r in results)
    base_t = sum(r["baseline_correct"] for r in results)
    delta_t = sge_t - base_t
    sign_t = "+" if delta_t > 0 else ""
    print("-" * len(header))
    print(f"{'OVERALL':<12} | {sge_t}/{total} ({sge_t/total:.0%}) | {base_t}/{total} ({base_t/total:.0%}) | {sign_t}{delta_t}")
    print(f"{'='*60}")

    output_data = {
        "timestamp":      datetime.now().isoformat(),
        "questions_file": questions_path,
        "total_questions": total,
        "summary": summary,
        "overall": {
            "sge_correct": sge_t, "baseline_correct": base_t,
            "sge_accuracy": sge_t / total, "baseline_accuracy": base_t / total,
        },
        "results": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n[JSON]")
        print(json.dumps(output_data, ensure_ascii=False, indent=2))

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Downstream QA evaluation for SGE-LightRAG")
    parser.add_argument("--questions", default="evaluation/gold/qa_questions.jsonl")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_eval(args.questions, args.output, args.verbose))


if __name__ == "__main__":
    main()
