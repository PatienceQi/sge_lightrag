#!/usr/bin/env python3
"""
lightrag_latest_baseline.py — Test latest LightRAG (v1.4.x) on CSV input.

Demonstrates that the current official LightRAG pipeline still exhibits
structural blindness on statistical CSV data: no topology recognition,
no schema induction, no constrained extraction.

Usage:
    # Run in the Python 3.11 venv with LightRAG v1.4.x installed:
    /tmp/raganything_env/bin/python3 lightrag_latest_baseline.py \
        --csv <csv_path> --output-dir <dir> --gold <gold_jsonl>

Requirements:
    pip install lightrag-hku networkx
    ANTHROPIC_API_KEY env var set (for Claude Haiku extraction)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Bypass macOS proxy for localhost (Ollama)
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Ensure the project root is importable (for evaluation tools)
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_lightrag_baseline(csv_path: str, output_dir: str):
    """Ingest a CSV file through LightRAG's default pipeline (no SGE)."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    os.makedirs(output_dir, exist_ok=True)

    # Read CSV as plain text (the "naive serialization" approach)
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        csv_text = f.read()

    # Use local Ollama embedding via aiohttp (same as paper config)
    import numpy as np
    import aiohttp

    async def ollama_embed(texts: list[str]) -> np.ndarray:
        embeddings = []
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=False,  # bypass system proxy
        ) as session:
            for text in texts:
                truncated = text[:2000] if len(text) > 2000 else text
                async with session.post(
                    "http://127.0.0.1:11434/api/embeddings",
                    json={"model": "mxbai-embed-large", "prompt": truncated},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()
                    if "embedding" in data:
                        embeddings.append(data["embedding"])
                    elif "embeddings" in data:
                        embeddings.append(data["embeddings"][0])
                    else:
                        # Fallback: zero vector
                        embeddings.append([0.0] * 1024)
        return np.array(embeddings)

    # Use OpenAI-compatible proxy (same as project's existing config)
    _API_KEY = os.environ.get("WOLFAI_API_KEY", "")
    _BASE_URL = "https://wolfai.top/v1"

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        return await openai_complete_if_cache(
            "claude-haiku-4-5-20251001",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=_BASE_URL,
            api_key=_API_KEY,
            **kwargs,
        )

    rag = LightRAG(
        working_dir=output_dir,
        llm_model_func=llm_func,
        llm_model_name="claude-haiku-4-5-20251001",
        llm_model_kwargs={"temperature": 0, "max_tokens": 8192},
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=ollama_embed,
        ),
    )

    # v1.4.x requires explicit storage initialization
    if hasattr(rag, 'initialize_storages'):
        await rag.initialize_storages()

    print(f"Ingesting CSV ({len(csv_text)} chars) into LightRAG v1.4.x...")
    await rag.ainsert(csv_text)
    print("Ingestion complete.")

    # Check output graph
    graphml_path = Path(output_dir) / "graph_chunk_entity_relation.graphml"
    if graphml_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graphml_path))
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print(f"Graph file not found at {graphml_path}")

    return str(graphml_path)


def evaluate_graph(graphml_path: str, gold_path: str):
    """Run EC/FC evaluation on the output graph."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_entity_coverage,
        check_fact_coverage, compute_structural_quality,
    )

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graphml_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0

    structure = compute_structural_quality(G, graph_nodes)

    results = {
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "gold_entities": len(gold_entities),
        "gold_facts": len(facts),
        "matched_entities": len(matched_entities),
        "covered_facts": len(covered),
        "graph_nodes": structure["num_nodes"],
        "graph_edges": structure["num_edges"],
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test latest LightRAG on CSV (structural blindness verification)"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--output-dir", required=True, help="LightRAG working dir")
    parser.add_argument("--gold", default=None, help="Gold JSONL for evaluation")
    args = parser.parse_args()

    graphml_path = asyncio.run(run_lightrag_baseline(args.csv, args.output_dir))

    if args.gold and Path(graphml_path).exists():
        results = evaluate_graph(graphml_path, args.gold)
        print(f"\n{'='*55}")
        print("EVALUATION RESULTS (LightRAG v1.4.x Baseline)")
        print(f"{'='*55}")
        print(f"  EC: {results['ec']:.4f}")
        print(f"  FC: {results['fc']:.4f}")
        print(f"  Graph: {results['graph_nodes']} nodes, {results['graph_edges']} edges")
        print(json.dumps(results, indent=2))

        # Save results
        out_path = Path(args.output_dir) / "evaluation_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
