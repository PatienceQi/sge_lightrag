#!/usr/bin/env python3
"""
run_e2e_baseline_compact.py — Baseline compact WHO control experiment.

Builds a compact WHO graph using LightRAG DEFAULT extraction (NO SGE schema),
with real Ollama embeddings, then runs the same 24 WHO E2E queries.

Purpose: Control for SGE compact 95.8% — if Baseline compact also scores high,
the gain comes from compression alone; if low, it confirms SGE schema is needed.
"""

from __future__ import annotations

import re, sys, json, asyncio, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
import aiohttp
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = "sk-7S8fU9gBMpK5Banzc0mM8DdOac7XFW0Mt7WCRbjSNTErrHPG"
BASE_URL = "https://wolfai.top/v1"
BUILD_MODEL = "claude-haiku-4-5-20251001"  # Same as SGE compact build
QUERY_MODEL = "gpt-5-mini"  # Same as SGE compact query (JSON compatible)
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
EMBED_DIM = 1024

OUTPUT_DIR = PROJECT_ROOT / "output" / "compact_who_baseline_realembed"


# ── LLM functions ───────────────────────────────────────────────────────────
async def build_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        BUILD_MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs)


async def query_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        QUERY_MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs)


# ── Ollama embedding (truncate to 450 tokens to avoid context length error) ─
MAX_EMBED_CHARS = 1500  # ~375 tokens, safely within mxbai-embed-large's 512 limit

async def ollama_embed(texts: list[str]) -> np.ndarray:
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            truncated = text[:MAX_EMBED_CHARS] if len(text) > MAX_EMBED_CHARS else text
            async with session.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
                json={"model": "mxbai-embed-large", "prompt": truncated},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
                else:
                    body = await resp.text()
                    raise RuntimeError(f"Ollama {resp.status}: {body[:200]}")
    return np.array(embeddings, dtype=np.float32)

EMBED_FUNC = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=512, func=ollama_embed)


# ── Build baseline compact chunks (NO schema, just row-per-entity format) ──
def build_baseline_compact_chunks(csv_path: str) -> list[str]:
    """
    Build compact chunks WITHOUT SGE schema — just group all years per entity
    into a single text block using plain natural language.
    This mimics what a naive user might do: one chunk per country, all data inline.
    """
    from stage1.features import _detect_encoding, _detect_skiprows

    encoding = _detect_encoding(csv_path)
    skiprows = _detect_skiprows(csv_path, encoding)
    df = pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)

    # WHO CSV structure: first col is country, rest are year columns
    # Detect structure minimally (no SGE schema)
    subject_col = df.columns[0]
    year_cols = [c for c in df.columns[1:] if re.search(r'\d{4}', str(c))]

    if not year_cols:
        # Fallback: treat all numeric columns as data
        year_cols = [c for c in df.columns[1:] if df[c].dtype in ('float64', 'int64')]

    chunks = []
    for _, row in df.iterrows():
        entity = str(row[subject_col]).strip()
        if not entity or entity.lower() == 'nan':
            continue

        # Build plain text chunk — NO schema guidance, just data
        parts = [f"{entity}:"]
        for yc in year_cols:
            val = row[yc]
            if pd.notna(val):
                parts.append(f"  {yc}: {val}")

        chunk = "\n".join(parts)
        chunks.append(chunk)

    return chunks


# ── Scoring ─────────────────────────────────────────────────────────────────
def _norm(s): return re.sub(r"\.0+\b", "", s.replace(",", ""))
def _match(answer, value):
    if not value: return False
    return value.lower() in answer.lower() or _norm(value.lower()) in _norm(answer.lower())
def score(answer, q):
    if _match(answer, q.get("expected_value", "")): return True
    if q.get("type") in ("comparison", "trend") and q.get("secondary_value"):
        return _match(answer, q["secondary_value"])
    return False


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("BASELINE COMPACT WHO — Control Experiment")
    print("=" * 60)

    # Test Ollama
    print("\nTesting Ollama connectivity...")
    try:
        test = await ollama_embed(["test"])
        print(f"  OK: dim={test.shape[1]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    # ── Step 1: Build baseline compact graph ────────────────────────────────
    csv_path = str(PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv")
    # Verify file exists
    if not Path(csv_path).exists():
        # Try alternate paths
        for alt in [
            "/Users/qipatience/Desktop/SGE/dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        ]:
            if Path(alt).exists():
                csv_path = alt
                break

    print(f"\nCSV: {csv_path}")
    chunks = build_baseline_compact_chunks(csv_path)
    print(f"Generated {len(chunks)} baseline compact chunks")
    print(f"Sample chunk:\n{chunks[0][:300]}\n")

    work_dir = OUTPUT_DIR / "lightrag_storage"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build with Claude Haiku (same as SGE compact build) — DEFAULT prompts, no schema
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=build_llm_func,
        embedding_func=EMBED_FUNC,
        addon_params={"language": "English"},
        llm_model_max_async=3,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    print(f"\nInserting {len(chunks)} chunks (NO schema injection, default LightRAG prompts)...")
    print("  (local Ollama with OLLAMA_NUM_PARALLEL=4)")
    for i, chunk in enumerate(chunks, 1):
        if i % 20 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}]")
        await rag.ainsert(chunk)

    # Check graph
    import networkx as nx
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    if graph_path.exists():
        G = nx.read_graphml(str(graph_path))
        print(f"\nBaseline compact graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    await rag.finalize_storages()

    # ── Step 2: Run E2E queries with GPT-5-mini ─────────────────────────────
    print(f"\n{'='*60}")
    print("QUERYING baseline compact graph...")
    print(f"{'='*60}")

    # Re-init with query model
    rag2 = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=query_llm_func,
        embedding_func=EMBED_FUNC,
        addon_params={"language": "English"},
        llm_model_max_async=3,
        embedding_func_max_async=1,
    )
    await rag2.initialize_storages()

    # Load WHO questions
    with open(PROJECT_ROOT / "evaluation" / "qa_questions.jsonl") as f:
        all_qs = [json.loads(l) for l in f if l.strip()]
    who_qs = [q for q in all_qs if q["dataset"] == "who"]
    print(f"WHO questions: {len(who_qs)}")

    results = []
    for q in who_qs:
        print(f"\n[{q['id']}] {q['question'][:80]}")
        answer = None
        for mode in ["hybrid", "local"]:
            try:
                r = await rag2.aquery(q["question"], param=QueryParam(mode=mode, only_need_context=False))
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

    await rag2.finalize_storages()

    # ── Results ──────────────────────────────────────────────────────────────
    n = len(results)
    c = sum(r["correct"] for r in results)
    print(f"\n{'='*60}")
    print(f"BASELINE COMPACT WHO E2E: {c}/{n} ({c/n:.1%})")
    print(f"SGE compact WHO E2E:      23/24 (95.8%)")
    print(f"Full graph E2E:           0/24 (0%) — both SGE and Baseline")
    print(f"{'='*60}")

    # Save
    out_path = PROJECT_ROOT / "experiments" / "e2e_baseline_compact_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "who_baseline_compact_e2e": {
                "correct": c, "total": n, "accuracy": c / n if n > 0 else 0,
                "details": results,
                "graph_nodes": G.number_of_nodes() if graph_path.exists() else None,
            },
            "comparison": {
                "sge_compact": "23/24 (95.8%)",
                "baseline_compact": f"{c}/{n} ({c/n:.1%})" if n > 0 else "N/A",
                "full_graph_e2e": "0/24 (0%)",
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
