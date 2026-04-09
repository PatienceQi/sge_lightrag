#!/usr/bin/env python3
"""
hipporag_baseline.py — HippoRAG v2 baseline for multi-system comparison.

Runs HippoRAG indexing + QA on CSV datasets using:
  - LLM: Claude Haiku via wolfai proxy (OpenAI-compatible)
  - Embedding: mxbai-embed-large via Ollama (local)

HippoRAG v2 uses OpenIE to extract triples → builds an igraph KG
→ retrieves via Personalized PageRank → LLM QA.

Usage (must use hipporag conda env):
    python evaluation/hipporag_baseline.py \
        --dataset who \
        --output output/hipporag_who

    python evaluation/hipporag_baseline.py \
        --dataset who --mode qa \
        --questions evaluation/gold/qa_questions.jsonl
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
# LLM: wolfai proxy (OpenAI-compatible)
os.environ["OPENAI_API_KEY"] = os.environ.get("SGE_API_KEY", "")

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_BASE_URL = "https://wolfai.top/v1"

# Embedding: Ollama local
EMBED_MODEL = "mxbai-embed-large"
EMBED_BASE_URL = "http://127.0.0.1:11434/v1"

PROJECT_ROOT = Path(__file__).parent.parent

# ── Dataset → chunk directory mapping (same naive serialization as other baselines)
DATASET_CHUNKS = {
    "who": "output/who_life_expectancy/chunks",
    "wb_cm": "output/wb_child_mortality/chunks",
    "wb_pop": "output/wb_population/chunks",
    "wb_mat": "output/wb_maternal/chunks",
    "inpatient": "output/inpatient_2023/chunks",
}


def load_chunks(dataset: str) -> list[str]:
    """Load text chunks for a dataset."""
    chunks_dir = PROJECT_ROOT / DATASET_CHUNKS[dataset]
    chunk_files = sorted(chunks_dir.glob("*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")
    return chunks


def run_index(dataset: str, output_dir: str):
    """Run HippoRAG indexing on a dataset."""
    from hipporag import HippoRAG

    chunks = load_chunks(dataset)
    save_dir = str(PROJECT_ROOT / output_dir)

    print(f"\nInitializing HippoRAG...")
    print(f"  LLM: {LLM_MODEL} via {LLM_BASE_URL}")
    print(f"  Embedding: {EMBED_MODEL} via {EMBED_BASE_URL}")
    print(f"  Save dir: {save_dir}")

    # Use "text-embedding-xxx" to trigger OpenAI embedding path,
    # then override the actual model name to match Ollama's model
    hippo = HippoRAG(
        save_dir=save_dir,
        llm_model_name=LLM_MODEL,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name="text-embedding-mxbai",
        embedding_base_url=EMBED_BASE_URL,
    )
    # Override to use actual Ollama model name
    hippo.embedding_model.embedding_model_name = EMBED_MODEL

    print(f"\nIndexing {len(chunks)} documents...")
    hippo.index(docs=chunks)
    print(f"Indexing complete. Output saved to {save_dir}")

    # Export graph stats
    try:
        import igraph
        graph_path = Path(save_dir) / "graph.pickle"
        if graph_path.exists():
            g = igraph.Graph.Read_Pickle(str(graph_path))
            print(f"\nGraph stats: {g.vcount()} nodes, {g.ecount()} edges")
    except Exception as e:
        print(f"Could not read graph stats: {e}")


def run_qa(dataset: str, output_dir: str, questions_path: str, results_path: str | None):
    """Run QA evaluation using HippoRAG retrieval."""
    from hipporag import HippoRAG

    save_dir = str(PROJECT_ROOT / output_dir)

    hippo = HippoRAG(
        save_dir=save_dir,
        llm_model_name=LLM_MODEL,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name="text-embedding-mxbai",
        embedding_base_url=EMBED_BASE_URL,
    )
    hippo.embedding_model.embedding_model_name = EMBED_MODEL

    # Load questions for this dataset
    with open(PROJECT_ROOT / questions_path, encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f if line.strip()]

    ds_questions = [q for q in all_questions if q["dataset"] == dataset]
    if not ds_questions:
        # Try matching by prefix
        ds_map = {"who": "who", "wb_cm": "wb_child_mortality", "wb_pop": "wb_population",
                  "wb_mat": "wb_maternal", "inpatient": "inpatient"}
        ds_questions = [q for q in all_questions if q["dataset"] == ds_map.get(dataset, dataset)]

    if not ds_questions:
        print(f"No questions found for dataset '{dataset}'")
        return

    print(f"\nRunning QA on {len(ds_questions)} questions for {dataset}...")

    queries = [q["question"] for q in ds_questions]

    # Use HippoRAG's built-in QA
    try:
        solutions, answers, qa_details = hippo.rag_qa(queries=queries)
    except Exception as e:
        print(f"HippoRAG QA failed: {e}")
        print("Falling back to retrieve + manual QA...")
        answers = _fallback_qa(hippo, queries, ds_questions)
        solutions = None
        qa_details = None

    # Score answers
    results = []
    correct_count = 0
    for i, q in enumerate(ds_questions):
        answer = answers[i] if i < len(answers) else "[NO ANSWER]"
        is_correct = _score_answer(answer, q)
        if is_correct:
            correct_count += 1

        mark = '✓' if is_correct else '✗'
        print(f"  [{q['id']}] [{mark}] {q['question']}")
        print(f"    Expected: {q['expected_value']}")
        print(f"    Answer: {answer[:150]}")

        results.append({
            "id": q["id"],
            "dataset": dataset,
            "question": q["question"],
            "expected_value": q["expected_value"],
            "hipporag_answer": answer,
            "hipporag_correct": is_correct,
        })

    n = len(ds_questions)
    print(f"\n{'=' * 40}")
    print(f"HippoRAG QA: {correct_count}/{n} ({correct_count / n:.0%})")
    print(f"{'=' * 40}")

    if results_path:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "system": "hipporag_v2",
            "dataset": dataset,
            "config": {
                "llm_model": LLM_MODEL,
                "embedding_model": EMBED_MODEL,
            },
            "total_questions": n,
            "correct": correct_count,
            "accuracy": correct_count / n,
            "results": results,
        }
        rp = PROJECT_ROOT / results_path
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {rp}")


def _fallback_qa(hippo, queries, questions):
    """Fallback: retrieve passages then call LLM manually."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=LLM_BASE_URL)
    answers = []

    for i, query in enumerate(queries):
        try:
            solutions = hippo.retrieve(queries=[query], num_to_retrieve=5)
            passages = []
            if solutions and solutions[0].retrieved_passages:
                passages = [p.text if hasattr(p, 'text') else str(p)
                            for p in solutions[0].retrieved_passages[:5]]
            context = "\n---\n".join(passages) if passages else "[No context retrieved]"
        except Exception:
            context = "[Retrieval failed]"

        lang = questions[i].get("language", "en")
        system_prompt = (
            "你是一个问答助手。根据下面提供的上下文，用简洁的中文回答问题。"
            "如果上下文包含具体数值，请直接引用该数值。"
            if lang == "zh" else
            "You are a QA assistant. Answer concisely based on the context. "
            "Quote specific values if present."
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context[:3000]}\n\nQuestion: {query}"},
                ],
                max_tokens=256,
                temperature=0,
            )
            answers.append(resp.choices[0].message.content)
        except Exception as e:
            answers.append(f"[Error: {e}]")

    return answers


def _normalize_number(s: str) -> str:
    s = s.replace(',', '').replace('，', '')
    s = re.sub(r'\.0+\b', '', s)
    return s


def _match(answer: str, value: str) -> bool:
    if not value:
        return False
    al, vl = answer.lower(), value.lower()
    if vl in al:
        return True
    return _normalize_number(vl) in _normalize_number(al)


def _score_answer(answer: str, q: dict) -> bool:
    expected = q.get("expected_value", "")
    if _match(answer, expected):
        return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


def main():
    parser = argparse.ArgumentParser(description="HippoRAG baseline evaluation")
    parser.add_argument("--dataset", required=True,
                        choices=["who", "wb_cm", "wb_pop", "wb_mat", "inpatient"])
    parser.add_argument("--output", default=None, help="Output directory for HippoRAG index")
    parser.add_argument("--mode", choices=["index", "qa", "both"], default="both",
                        help="Run indexing, QA, or both")
    parser.add_argument("--questions", default="evaluation/gold/qa_questions.jsonl")
    parser.add_argument("--results", default=None, help="QA results output path")
    args = parser.parse_args()

    output_dir = args.output or f"output/hipporag_{args.dataset}"
    results_path = args.results or f"evaluation/results/hipporag_{args.dataset}_qa.json"

    if args.mode in ("index", "both"):
        run_index(args.dataset, output_dir)

    if args.mode in ("qa", "both"):
        run_qa(args.dataset, output_dir, args.questions, results_path)


if __name__ == "__main__":
    main()
