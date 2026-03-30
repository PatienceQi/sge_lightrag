#!/usr/bin/env python3
"""Quick E2E query on already-built compact WHO graph with real embeddings."""

from __future__ import annotations
import re, sys, json, asyncio, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
import aiohttp

API_KEY = "sk-7S8fU9gBMpK5Banzc0mM8DdOac7XFW0Mt7WCRbjSNTErrHPG"
BASE_URL = "https://wolfai.top/v1"
MODEL = "gpt-5-mini"  # GPT for query compatibility (LightRAG keyword extraction needs raw JSON)
OLLAMA_HOST = "192.168.0.159"
EMBED_DIM = 1024

async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs)

async def ollama_embed(texts: list[str]) -> np.ndarray:
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            async with session.post(
                f"http://{OLLAMA_HOST}:11434/api/embeddings",
                json={"model": "mxbai-embed-large", "prompt": text},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
                else:
                    raise RuntimeError(f"Ollama failed: {resp.status}")
    return np.array(embeddings, dtype=np.float32)

EMBED_FUNC = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=512, func=ollama_embed)

def _norm(s): return re.sub(r"\.0+\b", "", s.replace(",", ""))
def _match(answer, value):
    if not value: return False
    return value.lower() in answer.lower() or _norm(value.lower()) in _norm(answer.lower())
def score(answer, q):
    if _match(answer, q.get("expected_value", "")): return True
    if q.get("type") in ("comparison","trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False

async def main():
    # Load questions
    with open(PROJECT_ROOT / "evaluation/gold/qa_questions.jsonl") as f:
        all_qs = [json.loads(l) for l in f if l.strip()]
    who_qs = [q for q in all_qs if q["dataset"] == "who"]
    print(f"WHO questions: {len(who_qs)}")

    # Init compact graph RAG
    compact_dir = str(PROJECT_ROOT / "output/compact_who_realembed/lightrag_storage")
    rag = LightRAG(
        working_dir=compact_dir, llm_model_func=llm_func,
        embedding_func=EMBED_FUNC, addon_params={"language": "English"},
        llm_model_max_async=10, embedding_func_max_async=2)
    await rag.initialize_storages()

    results = []
    for q in who_qs:
        print(f"\n[{q['id']}] {q['question'][:80]}")
        answer = None
        for mode in ["hybrid", "local"]:
            try:
                r = await rag.aquery(q["question"], param=QueryParam(mode=mode, only_need_context=False))
                if r and "[No context" not in str(r):
                    answer = str(r)
                    print(f"  [{mode}] got response ({len(answer)} chars)")
                    break
            except Exception as e:
                print(f"  [{mode}] error: {e}")
        if not answer:
            answer = "[No context found]"

        correct = score(answer, q)
        mark = "✓" if correct else "✗"
        print(f"  [{mark}] Expected: {q['expected_value']}")
        print(f"  Answer: {answer[:150]}")
        results.append({"id": q["id"], "correct": correct, "answer": answer[:200]})

    await rag.finalize_storages()

    n = len(results)
    c = sum(r["correct"] for r in results)
    print(f"\n{'='*60}")
    print(f"COMPACT WHO E2E RESULTS: {c}/{n} ({c/n:.1%})")
    print(f"Previous full graph E2E: 0/24 (0%) for WHO")
    print(f"Direct graph access: 23/24 (96%) for WHO")
    print(f"{'='*60}")

    with open(PROJECT_ROOT / "experiments/results/e2e_compact_results.json", "w") as f:
        json.dump({"who_compact_e2e": {"correct": c, "total": n, "accuracy": c/n, "details": results}}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
