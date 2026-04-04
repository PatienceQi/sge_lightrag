"""THE Ranking ablation: remove Country from Type-III serialization hierarchy.

Verifies that removing the intermediate Country level restores
University→YearValue direct binding. Original FC=0.567.
"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests as _requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

# ── Config ──
API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"
OLLAMA_URL = "http://localhost:11434/v1"
EMBED_DIM = 1024

WORKING_DIR = str(PROJECT_ROOT / "output" / "the_ablation_no_country")
CHUNKS_DIR = PROJECT_ROOT / "output" / "the_ablation_no_country" / "chunks"
GOLD_FILE = PROJECT_ROOT / "evaluation" / "gold" / "gold_the_university_ranking.jsonl"
SCHEMA_FILE = PROJECT_ROOT / "output" / "the_university_ranking" / "extraction_schema.json"


# ── LLM ──
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding (Ollama with hash fallback) ──
def _hash_embed(text):
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


def _ollama_sync(texts):
    resp = _requests.post(
        f"{OLLAMA_URL}/embeddings",
        json={"model": "mxbai-embed-large", "input": texts}, timeout=60,
    )
    resp.raise_for_status()
    return np.array([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)


async def embed_func(texts):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _ollama_sync, texts)
    except Exception:
        return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBED = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=512, func=embed_func)


async def main():
    from stage3.prompt_injector import generate_system_prompt

    with open(SCHEMA_FILE) as f:
        schema = json.load(f)
    prompt = generate_system_prompt(schema)

    os.makedirs(WORKING_DIR, exist_ok=True)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        embedding_func=EMBED,
        addon_params={"entity_extraction_system_prompt": prompt},
        llm_model_max_async=2,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    chunks = sorted(CHUNKS_DIR.glob("chunk_*.txt"))
    print(f"Inserting {len(chunks)} modified chunks (no Country hierarchy)")
    print(f"Sample: {chunks[0].read_text().strip()[:100]}...")

    for i, f in enumerate(chunks, 1):
        await rag.ainsert(f.read_text().strip())
        if i % 10 == 0:
            print(f"  {i}/{len(chunks)}")

    print("Insertion complete. Evaluating...")

    # ── Evaluate FC ──
    import networkx as nx
    from collections import defaultdict

    gpath = Path(WORKING_DIR) / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    G = nx.read_graphml(str(gpath))
    nodes = set(G.nodes())
    print(f"Graph: {len(nodes)} nodes, {G.number_of_edges()} edges")

    adj = defaultdict(set)
    for u, v in G.edges():
        adj[u].add(v)
        adj[v].add(u)

    gold = [json.loads(l) for l in open(GOLD_FILE)]
    covered = 0
    uncovered = []

    for fact in gold:
        subj, val = fact["triple"]["subject"], fact["triple"]["object"]
        matches = [n for n in nodes if subj.lower() in n.lower()]
        if not matches:
            uncovered.append(f"{subj} -> {val} (not found)")
            continue
        found = False
        for s in matches:
            for n1 in adj.get(s, set()):
                if val in str(n1):
                    found = True
                    break
                for n2 in adj.get(n1, set()):
                    if val in str(n2):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            covered += 1
        else:
            uncovered.append(f"{subj} -> {val}")

    fc = covered / len(gold)
    print(f"\nFC = {covered}/{len(gold)} = {fc:.3f}")
    print(f"Original THE FC = 0.567 → Ablation FC = {fc:.3f}")
    if uncovered:
        print(f"Uncovered ({len(uncovered)}):")
        for u in uncovered[:10]:
            print(f"  {u}")

    result = {
        "experiment": "THE_ablation_no_country",
        "original_fc": 0.567,
        "ablation_fc": fc,
        "gold_facts": len(gold),
        "covered": covered,
        "nodes": len(nodes),
        "edges": G.number_of_edges(),
    }
    out = PROJECT_ROOT / "experiments" / "results" / "the_ablation_no_country.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
