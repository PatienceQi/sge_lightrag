#!/usr/bin/env python3
"""
run_type1_sanity_check.py — Type-I sanity check experiment.

Runs SGE and LightRAG Baseline on 3 Type-I files (medical expenditure tables),
compares EC/FC to confirm "Type-I is a compatibility fallback, not a gain scenario".

Type-I files are flat entity-attribute tables with no time dimension or hierarchy.
SGE treats them as default (batch=5 merge), so SGE ≈ Baseline is expected.
"""

from __future__ import annotations

import os
import sys, json, asyncio, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL = "claude-haiku-4-5-20251001"

# Use remote Ollama embedding
import numpy as np
import aiohttp

OLLAMA_HOST = "192.168.0.159"
OLLAMA_PORT = 11434

async def ollama_embed(texts: list[str]) -> np.ndarray:
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            async with session.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
                json={"model": "mxbai-embed-large", "prompt": text},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
                else:
                    raise RuntimeError(f"Ollama failed: {resp.status}")
    return np.array(embeddings, dtype=np.float32)

EMBED_FUNC = EmbeddingFunc(embedding_dim=1024, max_token_size=512, func=ollama_embed)


async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs)


# ── Type-I files to test ────────────────────────────────────────────────────
_DATASET_BASE = PROJECT_ROOT.parent / "dataset"
TYPE1_FILES = [
    {
        "name": "医疗开支 Table_1",
        "csv": str(_DATASET_BASE / "香港本地医疗卫生总开支账目 " / "Table_1.csv"),
    },
    {
        "name": "医疗开支 Table_3",
        "csv": str(_DATASET_BASE / "香港本地医疗卫生总开支账目 " / "Table_3.csv"),
    },
    {
        "name": "医疗开支 Table_5",
        "csv": str(_DATASET_BASE / "香港本地医疗卫生总开支账目 " / "Table_5.csv"),
    },
]


# ── Build Gold Standard for Type-I (simple: extract all entity-value pairs) ─
def build_type1_gold(csv_path: str) -> list[dict]:
    """
    Build a minimal Gold Standard for Type-I files.
    Type-I = flat entity-attribute table. Each row is an entity with attribute values.
    Gold facts = (entity_name, attribute_name, value) for numeric columns.
    """
    from stage1.features import _detect_encoding, _detect_skiprows
    import pandas as pd

    encoding = _detect_encoding(csv_path)
    skiprows = _detect_skiprows(csv_path, encoding)
    df = pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)

    # First column is typically the entity/label column
    subject_col = df.columns[0]
    # Find numeric columns
    numeric_cols = [c for c in df.columns[1:] if df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    # Also check string columns that might be numeric
    for c in df.columns[1:]:
        if c not in numeric_cols:
            try:
                vals = pd.to_numeric(df[c].dropna().astype(str).str.replace(',', ''), errors='coerce')
                if vals.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(c)
            except:
                pass

    facts = []
    for _, row in df.iterrows():
        entity = str(row[subject_col]).strip()
        if not entity or entity.lower() == 'nan' or entity == '':
            continue
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns for manageability
            val = row[col]
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str and val_str.lower() != 'nan':
                    facts.append({
                        "subject": entity,
                        "relation": col.strip(),
                        "object": val_str,
                    })

    # Limit to 20 facts per file for sanity check
    return facts[:20]


# ── Evaluate EC/FC against graph ────────────────────────────────────────────
def evaluate_graph(graph_path: str, facts: list[dict]) -> dict:
    """Evaluate EC and FC using substring matching (same as main evaluation)."""
    import networkx as nx

    if not Path(graph_path).exists():
        return {"ec": 0.0, "fc": 0.0, "nodes": 0, "edges": 0}

    G = nx.read_graphml(graph_path)
    nodes = {n.lower(): n for n in G.nodes()}
    node_data = {n: G.nodes[n] for n in G.nodes()}

    # All text in graph (node names + descriptions + edge descriptions)
    all_text = ""
    for n in G.nodes():
        all_text += f" {n} "
        desc = G.nodes[n].get("description", "")
        if desc:
            all_text += f" {desc} "
    for u, v, d in G.edges(data=True):
        desc = d.get("description", "")
        if desc:
            all_text += f" {desc} "
    all_text = all_text.lower()

    # EC: subject substring match
    entities = set(f["subject"] for f in facts)
    matched_entities = sum(1 for e in entities if e.lower() in all_text)
    ec = matched_entities / len(entities) if entities else 0.0

    # FC: fact substring match (subject + object both in graph text)
    matched_facts = 0
    for f in facts:
        subj = f["subject"].lower()
        obj_val = f["object"].lower().replace(",", "")
        # Check if both subject and value appear in graph
        if subj in all_text and obj_val in all_text:
            matched_facts += 1
    fc = matched_facts / len(facts) if facts else 0.0

    return {
        "ec": round(ec, 3),
        "fc": round(fc, 3),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "n_entities": len(entities),
        "n_facts": len(facts),
        "matched_entities": matched_entities,
        "matched_facts": matched_facts,
    }


# ── Run pipeline ────────────────────────────────────────────────────────────
async def run_one(file_info: dict, mode: str) -> dict:
    """Run either 'sge' or 'baseline' on a single Type-I file."""
    csv_path = file_info["csv"]
    name = file_info["name"]

    if not Path(csv_path).exists():
        print(f"  SKIP: {csv_path} not found")
        return {"error": "file not found"}

    tag = f"type1_{name.replace(' ', '_')}_{mode}"
    work_dir = PROJECT_ROOT / "output" / tag / "lightrag_storage"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build chunks
    if mode == "sge":
        # Use SGE pipeline
        from stage1.features import extract_features
        from stage1.classifier import classify
        from stage1.schema import build_meta_schema
        from stage2.inductor import induce_schema_from_meta
        from stage3.serializer import serialize_csv
        from stage3.prompt_injector import generate_system_prompt

        features = extract_features(csv_path)
        table_type = classify(features)
        meta_schema = build_meta_schema(features, table_type)
        extraction_schema = induce_schema_from_meta(features, table_type, meta_schema)
        chunks = serialize_csv(csv_path, extraction_schema)
        system_prompt = generate_system_prompt(extraction_schema)

        print(f"  SGE: type={table_type}, {len(chunks)} chunks")
    else:
        # Baseline: just read CSV and serialize as plain text
        from stage1.features import _detect_encoding, _detect_skiprows
        import pandas as pd

        encoding = _detect_encoding(csv_path)
        skiprows = _detect_skiprows(csv_path, encoding)
        df = pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)

        # Default LightRAG behavior: just dump rows as text
        chunks = []
        batch = []
        for _, row in df.iterrows():
            line = "; ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            batch.append(line)
            if len(batch) >= 5:
                chunks.append("\n".join(batch))
                batch = []
        if batch:
            chunks.append("\n".join(batch))

        system_prompt = None
        print(f"  Baseline: {len(chunks)} chunks")

    # Build graph
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=llm_func,
        embedding_func=EMBED_FUNC,
        addon_params={"language": "Chinese"},
        llm_model_max_async=5,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Inject system prompt if SGE
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    if mode == "sge" and system_prompt:
        escaped = system_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        for i, chunk in enumerate(chunks, 1):
            await rag.ainsert(chunk)
            if i % 5 == 0 or i == len(chunks):
                print(f"    [{i}/{len(chunks)}]")
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    await rag.finalize_storages()

    # Evaluate
    graph_path = str(work_dir / "graph_chunk_entity_relation.graphml")
    facts = build_type1_gold(csv_path)
    result = evaluate_graph(graph_path, facts)
    result["mode"] = mode
    result["name"] = name

    print(f"  Result: EC={result['ec']}, FC={result['fc']}, nodes={result['nodes']}")
    return result


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("TYPE-I SANITY CHECK — SGE vs Baseline on flat entity tables")
    print("=" * 60)

    all_results = []

    for file_info in TYPE1_FILES:
        print(f"\n--- {file_info['name']} ---")

        # Run SGE
        sge_result = await run_one(file_info, "sge")
        all_results.append(sge_result)

        # Run Baseline
        base_result = await run_one(file_info, "baseline")
        all_results.append(base_result)

        if "error" not in sge_result and "error" not in base_result:
            print(f"\n  Summary: SGE EC={sge_result['ec']} FC={sge_result['fc']} | "
                  f"Base EC={base_result['ec']} FC={base_result['fc']}")

    # Summary table
    print(f"\n{'='*60}")
    print("TYPE-I SANITY CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':<25} {'Mode':<10} {'EC':<8} {'FC':<8} {'Nodes':<8}")
    print("-" * 59)
    for r in all_results:
        if "error" not in r:
            print(f"{r['name']:<25} {r['mode']:<10} {r['ec']:<8} {r['fc']:<8} {r['nodes']:<8}")

    # Save
    out_path = PROJECT_ROOT / "experiments" / "type1_sanity_check_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
