#!/usr/bin/env python3
"""
run_crossmodel_expansion.py — Expand cross-model validation to 5 datasets.

Runs GPT-5-mini on WB Population and WB Maternal (the 2 datasets missing from
the original 3-dataset cross-model test), then evaluates EC/FC.

Reuses existing SGE output (chunks + extraction_schema.json) from each dataset.
Only the LightRAG extraction LLM backend is changed.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.integrator import patch_lightrag

API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
TARGET_MODEL = "gpt-5-mini"
EMBED_DIM = 1024

DATASETS = {
    "wb_pop": {
        "label": "WB Population",
        "existing_sge": PROJECT_ROOT / "output" / "wb_population",
        "gold": PROJECT_ROOT / "evaluation" / "gold_wb_population_v2.jsonl",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "existing_sge": PROJECT_ROOT / "output" / "wb_maternal",
        "gold": PROJECT_ROOT / "evaluation" / "gold_wb_maternal_v2.jsonl",
    },
}


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


async def run_crossmodel(dataset_key):
    """Run SGE with GPT-5-mini on a dataset."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.prompt import PROMPTS

    ds = DATASETS[dataset_key]
    existing_dir = ds["existing_sge"]
    output_dir = PROJECT_ROOT / "output" / f"crossmodel_gpt_5_mini_{dataset_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CROSS-MODEL: {TARGET_MODEL} on {ds['label']}")
    print("=" * 60)

    # Load existing SGE schema and chunks
    schema_path = existing_dir / "extraction_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(f"  Schema from: {schema_path}")
    print(f"  Entity types: {schema.get('entity_types', 'N/A')}")

    chunks_dir = existing_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"  Chunks: {len(chunks)}")

    # Build payload
    payload = patch_lightrag(schema)

    # LLM function
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            TARGET_MODEL, prompt, system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=API_KEY, base_url=BASE_URL, **kwargs,
        )

    # Embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
    )

    work_dir = output_dir / "lightrag_storage"
    work_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        addon_params=payload["addon_params"],
        llm_model_max_async=10,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Override system prompt if needed
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    if payload.get("system_prompt") and not payload.get("use_baseline_mode"):
        sp = payload["system_prompt"]
        escaped = sp.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        print(f"\n  Inserting {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            if i % 50 == 0 or i == len(chunks):
                print(f"    [{i}/{len(chunks)}]")
            await rag.ainsert(chunk)
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    stats = {
        "label": f"{TARGET_MODEL}-{dataset_key}",
        "model": TARGET_MODEL,
        "dataset": ds["label"],
        "chunks": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    await rag.finalize_storages()

    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Evaluate EC/FC
    print(f"\n  Evaluating against {ds['gold']}...")
    evaluate_graph(graph_path, ds["gold"], stats, output_dir)

    return stats


def evaluate_graph(graph_path, gold_path, stats, output_dir):
    """Evaluate EC/FC using the coverage evaluator."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
        from evaluate_coverage import evaluate_coverage

        results = evaluate_coverage(str(graph_path), str(gold_path))
        stats["EC"] = results.get("entity_coverage", 0)
        stats["FC"] = results.get("fact_coverage", 0)
        print(f"  EC={stats['EC']:.3f}, FC={stats['FC']:.3f}")
    except Exception as e:
        print(f"  Evaluation error: {e}")
        # Try manual evaluation
        try:
            from graph_loaders import load_graph_entities
            import networkx as nx

            G = nx.read_graphml(str(graph_path))
            nodes = [G.nodes[n].get("label", n).lower() for n in G.nodes()]

            gold_facts = []
            with open(gold_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        gold_facts.append(json.loads(line))

            # EC: substring match
            entities = set()
            for fact in gold_facts:
                sub = fact.get("triple", {}).get("subject", "")
                if sub:
                    entities.add(sub)

            matched_entities = 0
            for entity in entities:
                entity_lower = entity.lower()
                if any(entity_lower in node for node in nodes):
                    matched_entities += 1

            ec = matched_entities / len(entities) if entities else 0
            stats["EC"] = ec
            stats["FC"] = "needs_manual_eval"
            print(f"  Manual EC={ec:.3f}, FC=needs further evaluation")
        except Exception as e2:
            print(f"  Manual evaluation also failed: {e2}")

    # Save updated stats
    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def main():
    results = {}
    for key in DATASETS:
        try:
            stats = await run_crossmodel(key)
            results[key] = stats
        except Exception as e:
            print(f"\nERROR on {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}

    # Save combined results
    out_path = PROJECT_ROOT / "experiments" / "crossmodel_expansion_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll results saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-MODEL EXPANSION SUMMARY")
    print("=" * 60)
    for key, r in results.items():
        if "error" in r:
            print(f"  {key}: ERROR - {r['error']}")
        else:
            ec = r.get("EC", "N/A")
            fc = r.get("FC", "N/A")
            nodes = r.get("nodes", "N/A")
            print(f"  {r.get('dataset', key)}: EC={ec}, FC={fc}, nodes={nodes}")


if __name__ == "__main__":
    asyncio.run(main())
