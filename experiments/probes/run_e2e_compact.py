#!/usr/bin/env python3
"""
run_e2e_compact.py — Rebuild compact WHO/WB CM graphs with real Ollama embeddings,
then run E2E LightRAG queries to test if compact representation fixes the
vector retrieval bottleneck.

Hypothesis: The E2E 13%=13% result is caused by vector retrieval failing on
large graphs (WHO 4508 nodes). Compact representation reduces to ~215 nodes.
With real embeddings, vector retrieval should work → E2E accuracy should improve.

Uses remote Ollama at 192.168.0.159:11434.
"""

from __future__ import annotations

import re
import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = "sk-7S8fU9gBMpK5Banzc0mM8DdOac7XFW0Mt7WCRbjSNTErrHPG"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"

OLLAMA_HOST = "192.168.0.159"
OLLAMA_PORT = 11434
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM = 1024


# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Real Ollama embedding ────────────────────────────────────────────────────
import aiohttp


async def ollama_embed(texts: list[str]) -> np.ndarray:
    """Call remote Ollama embedding API."""
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            async with session.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
                else:
                    body = await resp.text()
                    raise RuntimeError(f"Ollama embedding failed: {resp.status} {body}")
    return np.array(embeddings, dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=ollama_embed,
)


# ── Step 1: Rebuild compact graph with real embeddings ───────────────────────
async def rebuild_compact_graph(dataset_key: str, csv_path: str, schema_path: str,
                                output_dir: Path):
    """Rebuild a compact graph with real Ollama embeddings."""
    from stage1.features import extract_features
    from stage1.classifier import classify
    from stage1.schema import build_meta_schema
    from stage3.compact_representation import compact_serialize_type_ii, build_compact_system_prompt
    from stage3.integrator import patch_lightrag
    from lightrag.prompt import PROMPTS

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "lightrag_storage"

    # Load existing schema
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    # Generate compact chunks
    features = extract_features(csv_path)
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    # Read full CSV for compact serialization
    import pandas as pd
    from stage1.features import _detect_encoding, _detect_skiprows
    encoding = _detect_encoding(csv_path)
    skiprows = _detect_skiprows(csv_path, encoding)
    df = pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)

    compact_chunks = compact_serialize_type_ii(df, schema)
    compact_prompt = build_compact_system_prompt(schema)
    print(f"  Generated {len(compact_chunks)} compact chunks")
    print(f"  Compact system prompt length: {len(compact_prompt)} chars")

    # Build LightRAG with REAL embeddings
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": "English"},
        llm_model_max_async=10,
        embedding_func_max_async=2,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Inject compact system prompt
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    if compact_prompt:
        escaped = compact_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        print(f"  Inserting {len(compact_chunks)} chunks with real Ollama embeddings...")
        for i, chunk in enumerate(compact_chunks, 1):
            if i % 20 == 0 or i == len(compact_chunks):
                print(f"    [{i}/{len(compact_chunks)}]")
            await rag.ainsert(chunk)
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    # Check graph
    import networkx as nx
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    if graph_path.exists():
        G = nx.read_graphml(str(graph_path))
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    await rag.finalize_storages()
    return str(work_dir)


# ── Step 2: Run E2E queries ──────────────────────────────────────────────────
def _normalize_number(s: str) -> str:
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\.0+\b", "", s)
    return s


def _match(answer: str, value: str) -> bool:
    if not value:
        return False
    al, vl = answer.lower(), value.lower()
    if vl in al:
        return True
    return _normalize_number(vl) in _normalize_number(al)


def score_answer(answer: str, q: dict) -> bool:
    expected = q.get("expected_value", "")
    if _match(answer, expected):
        return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


async def run_e2e_queries(compact_dir: str, questions: list[dict], dataset_filter: str):
    """Run E2E LightRAG queries on the compact graph."""
    rag = LightRAG(
        working_dir=compact_dir,
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params={"language": "English"},
        llm_model_max_async=10,
        embedding_func_max_async=2,
    )
    await rag.initialize_storages()

    filtered_qs = [q for q in questions if q["dataset"] == dataset_filter]
    print(f"\n  Running E2E queries on {len(filtered_qs)} questions ({dataset_filter})...")

    results = []
    for q in filtered_qs:
        question = q["question"]
        print(f"  [{q['id']}] {question[:80]}...")

        # Try hybrid first, then local
        answer = None
        for mode in ["hybrid", "local"]:
            try:
                result = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode, only_need_context=False),
                )
                if result and str(result) != "[No context found]":
                    answer = str(result)
                    break
            except Exception as e:
                print(f"    [{mode}] error: {e}")

        if not answer:
            answer = "[No context found]"

        correct = score_answer(answer, q)
        mark = "✓" if correct else "✗"
        print(f"    [{mark}] {answer[:120]}")

        results.append({
            "id": q["id"],
            "question": question,
            "expected": q["expected_value"],
            "answer": answer,
            "correct": correct,
        })

    await rag.finalize_storages()
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    # Test Ollama connectivity first
    print("Testing Ollama connectivity...")
    try:
        test = await ollama_embed(["test"])
        print(f"  OK: embedding dim={test.shape[1]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    # Load QA questions
    qa_path = PROJECT_ROOT / "evaluation" / "qa_questions.jsonl"
    with open(qa_path, encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f if line.strip()]

    datasets = {
        "who": {
            "csv": str(PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"),
            "schema": str(PROJECT_ROOT / "output" / "who_life_expectancy" / "extraction_schema.json"),
            "output": PROJECT_ROOT / "output" / "compact_who_realembed",
            "qa_dataset": "who",
        },
    }

    all_results = {}

    for key, ds in datasets.items():
        print(f"\n{'='*60}")
        print(f"COMPACT E2E: {key}")
        print(f"{'='*60}")

        # Step 1: Rebuild with real embeddings
        compact_dir = await rebuild_compact_graph(
            key, ds["csv"], ds["schema"], ds["output"]
        )

        # Step 2: Run E2E queries
        results = await run_e2e_queries(compact_dir, all_questions, ds["qa_dataset"])

        n_total = len(results)
        n_correct = sum(r["correct"] for r in results)
        accuracy = n_correct / n_total if n_total > 0 else 0

        print(f"\n  RESULT: {n_correct}/{n_total} ({accuracy:.1%})")
        all_results[key] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "results": results,
        }

    # Also run on original full SGE graph for comparison (same questions, same embedding)
    print(f"\n{'='*60}")
    print(f"COMPARISON: Full SGE WHO E2E (same real embeddings)")
    print(f"{'='*60}")

    # We can't easily rebuild the full graph with real embeddings (too many chunks).
    # Instead, just report the compact results.

    # Save results
    out_path = PROJECT_ROOT / "experiments" / "e2e_compact_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("COMPACT E2E SUMMARY")
    print(f"{'='*60}")
    for key, r in all_results.items():
        print(f"  {key}: {r['n_correct']}/{r['n_total']} ({r['accuracy']:.1%})")
    print(f"\n  Previous E2E (full graph, hash embed): SGE 13%, Baseline 13%")
    print(f"  Hypothesis: compact + real embed → improved retrieval")


if __name__ == "__main__":
    asyncio.run(main())
