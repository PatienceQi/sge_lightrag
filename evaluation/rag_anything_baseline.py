#!/usr/bin/env python3
"""
rag_anything_baseline.py — RAG-Anything baseline on CSV statistical data.

Tests two paths:
  Path A: RAG-Anything insert() with CSV text (equivalent to LightRAG ainsert)
  Path B: Direct TableModalProcessor analysis of CSV table content

Usage:
    /tmp/raganything_env/bin/python3 rag_anything_baseline.py \
        --csv <csv_path> --output-dir <dir> --gold <gold_jsonl>
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

sys.path.insert(0, str(Path(__file__).parent.parent))

_API_KEY = os.environ.get("WOLFAI_API_KEY", "")
_BASE_URL = "https://wolfai.top/v1"


async def _make_llm_func():
    """Create OpenAI-compatible LLM function for LightRAG."""
    from lightrag.llm.openai import openai_complete_if_cache

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

    return llm_func


async def _make_embed_func():
    """Create Ollama embedding function."""
    import numpy as np
    import aiohttp

    async def ollama_embed(texts: list[str]) -> np.ndarray:
        embeddings = []
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=False
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
                    else:
                        embeddings.append([0.0] * 1024)
        return np.array(embeddings)

    return ollama_embed


async def run_path_a(csv_path: str, output_dir: str):
    """Path A: RAG-Anything insert() with CSV as text."""
    from raganything import RAGAnything
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    print("\n" + "=" * 60)
    print("PATH A: RAG-Anything insert() with CSV text")
    print("=" * 60)

    work_dir = os.path.join(output_dir, "path_a")
    os.makedirs(work_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        csv_text = f.read()

    llm_func = await _make_llm_func()
    embed_func = await _make_embed_func()

    rag = LightRAG(
        working_dir=work_dir,
        llm_model_func=llm_func,
        llm_model_name="claude-haiku-4-5-20251001",
        llm_model_kwargs={"temperature": 0, "max_tokens": 8192},
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=embed_func,
        ),
    )

    raga = RAGAnything(lightrag=rag)

    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()

    print(f"Ingesting CSV ({len(csv_text)} chars) via RAG-Anything insert()...")
    await rag.ainsert(csv_text)
    print("Path A complete.")

    graphml = os.path.join(work_dir, "graph_chunk_entity_relation.graphml")
    return graphml if os.path.exists(graphml) else None


async def run_path_b(csv_path: str, output_dir: str):
    """Path B: Direct TableModalProcessor on CSV table content."""
    from raganything.modalprocessors import TableModalProcessor

    print("\n" + "=" * 60)
    print("PATH B: TableModalProcessor direct analysis")
    print("=" * 60)

    work_dir = os.path.join(output_dir, "path_b")
    os.makedirs(work_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        csv_text = f.read()

    # Create the LLM caption function for TableModalProcessor
    async def modal_caption_func(prompt, system_prompt=None, **kwargs):
        from lightrag.llm.openai import openai_complete_if_cache
        result = await openai_complete_if_cache(
            "claude-haiku-4-5-20251001",
            prompt,
            system_prompt=system_prompt,
            base_url=_BASE_URL,
            api_key=_API_KEY,
            max_tokens=8192,
            temperature=0,
        )
        # Handle streaming response
        if hasattr(result, '__aiter__'):
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            return "".join(chunks)
        return result

    # TableModalProcessor requires a lightrag instance
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    embed_func = await _make_embed_func()
    llm_func_inner = await _make_llm_func()

    rag_b = LightRAG(
        working_dir=work_dir,
        llm_model_func=llm_func_inner,
        llm_model_name="claude-haiku-4-5-20251001",
        llm_model_kwargs={"temperature": 0, "max_tokens": 8192},
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=embed_func,
        ),
    )
    if hasattr(rag_b, "initialize_storages"):
        await rag_b.initialize_storages()

    processor = TableModalProcessor(
        lightrag=rag_b,
        modal_caption_func=modal_caption_func,
    )

    # Package CSV as table content (RAG-Anything's expected format)
    table_content = {
        "table_body": csv_text[:5000],  # First 5000 chars (representative sample)
        "table_caption": ["WHO Life Expectancy Statistical Data"],
        "img_path": None,
        "table_footnote": [],
    }

    print("Processing CSV via TableModalProcessor...")
    try:
        result = await processor.process_multimodal_content(
            modal_content=table_content,
            content_type="table",
            file_path=csv_path,
        )
        # raganything returns 3 values: (summary, metadata, chunk_results)
        if isinstance(result, tuple) and len(result) >= 2:
            text_desc = result[0]
            metadata = result[1]
        else:
            text_desc = str(result)
            metadata = {}
        print(f"Description length: {len(text_desc)} chars")
        print(f"Metadata keys: {list(metadata.keys())}")

        # Save the analysis
        result_path = os.path.join(work_dir, "table_analysis.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "text_description": text_desc[:2000],
                "metadata": {k: str(v)[:500] for k, v in metadata.items()},
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved analysis to {result_path}")

        # _create_entity_and_chunk already inserted into LightRAG's graph
        # Finalize storage to ensure GraphML is written
        if hasattr(rag_b, 'finalize_storages'):
            await rag_b.finalize_storages()
        print("Path B ingestion complete.")

        graphml = os.path.join(work_dir, "graph_chunk_entity_relation.graphml")
        return graphml if os.path.exists(graphml) else None

    except Exception as e:
        print(f"Path B failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_graph(graphml_path: str, gold_path: str, label: str):
    """Run EC/FC evaluation."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_entity_coverage, check_fact_coverage,
    )

    if not os.path.exists(graphml_path):
        print(f"  [{label}] Graph not found: {graphml_path}")
        return None

    gold_entities, facts = load_gold(gold_path)

    import networkx as nx
    G = nx.read_graphml(graphml_path)
    if G.number_of_nodes() == 0:
        print(f"  [{label}] Empty graph")
        return {"ec": 0, "fc": 0, "nodes": 0, "edges": 0}

    _, graph_nodes, entity_text = load_graph(graphml_path)
    matched = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched) / len(gold_entities) if gold_entities else 0
    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0

    result = {
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "matched_entities": len(matched),
        "covered_facts": len(covered),
        "total_entities": len(gold_entities),
        "total_facts": len(facts),
    }

    print(f"\n  [{label}] EC={result['ec']:.4f}  FC={result['fc']:.4f}  "
          f"({result['nodes']} nodes, {result['edges']} edges)")
    return result


async def main_async(args):
    results = {}

    # Path A
    graphml_a = await run_path_a(args.csv, args.output_dir)
    if graphml_a and args.gold:
        results["path_a"] = evaluate_graph(graphml_a, args.gold, "Path A")

    # Path B
    graphml_b = await run_path_b(args.csv, args.output_dir)
    if graphml_b and args.gold:
        results["path_b"] = evaluate_graph(graphml_b, args.gold, "Path B")

    # Save combined results
    results["metadata"] = {
        "csv": args.csv,
        "raganything_version": "1.2.10",
        "lightrag_version": "1.4.12",
        "llm": "claude-haiku-4-5-20251001",
    }
    out_path = os.path.join(args.output_dir, "rag_anything_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Anything baseline on CSV statistical data"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gold", default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
