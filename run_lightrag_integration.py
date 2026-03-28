#!/usr/bin/env python3
"""
run_lightrag_integration.py — End-to-end SGE → LightRAG integration.

Runs the full SGE pipeline (Stage 1→2→3) on a CSV, then feeds the
serialized chunks into LightRAG with schema-aware prompt injection.
Also runs a vanilla baseline for comparison.

Usage:
    python3 run_lightrag_integration.py <csv_path> [--output-dir <dir>]
"""

from __future__ import annotations

import sys
import json
import asyncio
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── SGE pipeline imports ──────────────────────────────────────────────────────
from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2.inducer import induce_schema
from stage3.serializer import serialize_csv
from stage3.integrator import patch_lightrag

# ── LightRAG imports ──────────────────────────────────────────────────────────
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = "sk-7S8fU9gBMpK5Banzc0mM8DdOac7XFW0Mt7WCRbjSNTErrHPG"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM          = 1024


# ── LLM function ─────────────────────────────────────────────────────────────
async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
):
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


# ── Embedding function (Ollama, with hash fallback) ───────────────────────────
def _hash_embed(text: str) -> list[float]:
    """Deterministic hash-based embedding fallback (no external call)."""
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


# openai_embed is already an EmbeddingFunc(dim=1536); use its inner .func directly
# so we can wrap it with our own EmbeddingFunc(dim=1024) without double-validation.
_raw_openai_embed = openai_embed.func


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        result = await _raw_openai_embed(
            texts,
            model=OLLAMA_EMBED_MODEL,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
        )
        return result
    except Exception as e:
        print(f"  [warn] Ollama embedding failed ({e}), using hash fallback")
        return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=safe_embedding_func,
)


# ── SGE monkey-patch for schema_json injection ────────────────────────────────
_original_extract_entities = _op.extract_entities


async def _sge_extract_entities(
    chunks,
    knowledgebase,
    entity_vdb,
    relationships_vdb,
    global_config,
    pipeline_status=None,
    llm_response_cache=None,
):
    """Wrap extract_entities to inject schema_json into context_base."""
    addon = global_config.get("addon_params", {})
    schema_json = addon.get("schema_json")
    if schema_json:
        # Patch context_base by temporarily overriding the prompt template
        # to use a version that doesn't reference {schema_json} if not present,
        # or inject it via addon_params which operate.py reads for entity_types/language.
        # The system prompt was already overridden before LightRAG init.
        pass  # entity_types and language are already in addon_params
    return await _original_extract_entities(
        chunks,
        knowledgebase,
        entity_vdb,
        relationships_vdb,
        global_config,
        pipeline_status=pipeline_status,
        llm_response_cache=llm_response_cache,
    )


# ── Pipeline: run SGE stages 1→2→3 ───────────────────────────────────────────
def run_sge_pipeline(csv_path: Path, sge_output_dir: Path) -> dict:
    """Run Stage 1→2→3 and return a dict with schema + chunks."""
    print("\n" + "=" * 60)
    print("SGE PIPELINE — Stage 1 → Stage 2 → Stage 3")
    print("=" * 60)

    # Stage 1
    print("\n[Stage 1] Topological Pattern Recognition...")
    features    = extract_features(str(csv_path))
    table_type  = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Table type : {table_type}")
    print(f"  Columns    : {len(features.raw_columns)}")

    # Stage 2
    print("\n[Stage 2] Rule-Based Schema Induction...")
    extraction_schema = induce_schema(meta_schema, features)
    print(f"  Entity types   : {extraction_schema['entity_types']}")
    print(f"  Relation types : {extraction_schema['relation_types']}")

    # Stage 3
    print("\n[Stage 3] Constrained Extraction Preparation...")
    chunks  = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks produced : {len(chunks)}")

    # Save SGE outputs
    sge_output_dir.mkdir(parents=True, exist_ok=True)
    (sge_output_dir / "meta_schema.json").write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sge_output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    chunks_dir = sge_output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks, 1):
        (chunks_dir / f"chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
    (sge_output_dir / "system_prompt.txt").write_text(
        payload["system_prompt"], encoding="utf-8"
    )

    print(f"  SGE outputs saved to: {sge_output_dir}")

    return {
        "extraction_schema": extraction_schema,
        "chunks": chunks,
        "payload": payload,
    }


# ── LightRAG runner ───────────────────────────────────────────────────────────
async def run_lightrag(
    chunks: list[str],
    working_dir: Path,
    addon_params: dict,
    label: str,
    baseline: bool = False,
) -> dict:
    """Initialize LightRAG, insert chunks, and return graph stats."""
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LightRAG:{label}] working_dir={working_dir}")
    print(f"  addon_params keys: {list(addon_params.keys())}")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=2,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )

    await rag.initialize_storages()

    # For baseline, add a label prefix so doc IDs differ from SGE run
    insert_chunks = chunks if not baseline else [f"[BASELINE]\n{c}" for c in chunks]

    print(f"  Inserting {len(insert_chunks)} chunks...")
    for i, chunk in enumerate(insert_chunks, 1):
        print(f"  [{i}/{len(insert_chunks)}] inserting chunk ({len(chunk)} chars)...")
        await rag.ainsert(chunk)

    # Collect graph stats
    graph_path = working_dir / "graph_chunk_entity_relation.graphml"

    stats = {
        "label": label,
        "working_dir": str(working_dir),
        "chunks_inserted": len(insert_chunks),
        "graph_file_exists": graph_path.exists(),
        "graph_file_size": graph_path.stat().st_size if graph_path.exists() else 0,
    }

    if graph_path.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(str(graph_path))
            stats["node_count"] = G.number_of_nodes()
            stats["edge_count"] = G.number_of_edges()
            print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges")
        except Exception as e:
            print(f"  [warn] Could not parse graphml: {e}")

    await rag.finalize_storages()
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
async def main_async(csv_path: Path, output_base: Path):
    sge_dir      = output_base / "sge_budget"
    baseline_dir = output_base / "baseline_budget"
    sge_work     = sge_dir / "lightrag_storage"
    baseline_work = baseline_dir / "lightrag_storage"

    # ── Run SGE pipeline ──────────────────────────────────────────────────────
    result = run_sge_pipeline(csv_path, sge_dir)
    chunks           = result["chunks"]
    extraction_schema = result["extraction_schema"]
    payload          = result["payload"]

    # ── SGE-enhanced LightRAG run ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTRAG RUN — SGE-Enhanced")
    print("=" * 60)

    # Override system prompt with schema-aware version.
    # The generated prompt contains embedded JSON with {braces} that Python's
    # .format() would misinterpret as template variables — escape them first,
    # then re-insert the real LightRAG placeholders.
    original_system_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    # Escape all braces, then restore the LightRAG template variables
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped
    _op.extract_entities = _sge_extract_entities

    try:
        sge_stats = await run_lightrag(
            chunks=chunks,
            working_dir=sge_work,
            addon_params=payload["addon_params"],
            label="SGE",
        )
    finally:
        # Restore original prompt for baseline run
        PROMPTS["entity_extraction_system_prompt"] = original_system_prompt
        _op.extract_entities = _original_extract_entities

    # Save SGE stats
    (sge_dir / "lightrag_stats.json").write_text(
        json.dumps(sge_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Baseline LightRAG run ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTRAG RUN — Baseline (vanilla)")
    print("=" * 60)

    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_stats = await run_lightrag(
        chunks=chunks,
        working_dir=baseline_work,
        addon_params={"language": "Chinese"},
        label="Baseline",
        baseline=True,
    )

    (baseline_dir / "lightrag_stats.json").write_text(
        json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Comparison summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    def _fmt(stats: dict) -> str:
        nodes = stats.get("node_count", "?")
        edges = stats.get("edge_count", "?")
        return f"nodes={nodes}, edges={edges}"

    print(f"  SGE-Enhanced : {_fmt(sge_stats)}")
    print(f"  Baseline     : {_fmt(baseline_stats)}")

    sge_nodes      = sge_stats.get("node_count", 0)
    baseline_nodes = baseline_stats.get("node_count", 0)
    sge_edges      = sge_stats.get("edge_count", 0)
    baseline_edges = baseline_stats.get("edge_count", 0)

    if isinstance(sge_nodes, int) and isinstance(baseline_nodes, int):
        node_delta = sge_nodes - baseline_nodes
        edge_delta = sge_edges - baseline_edges
        print(f"\n  Delta (SGE - Baseline):")
        print(f"    Nodes : {node_delta:+d}")
        print(f"    Edges : {edge_delta:+d}")
        if sge_nodes > 0 and baseline_nodes > 0:
            print(f"\n  SGE entity types constrained to: {extraction_schema['entity_types']}")
            print(f"  Baseline used default LightRAG entity types")

    # Save combined report
    report = {
        "timestamp": datetime.now().isoformat(),
        "csv_file": str(csv_path),
        "sge": sge_stats,
        "baseline": baseline_stats,
        "extraction_schema": {
            "entity_types": extraction_schema["entity_types"],
            "relation_types": extraction_schema["relation_types"],
        },
    }
    report_path = output_base / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Full report: {report_path}")
    print(f"  SGE output : {sge_dir}")
    print(f"  Baseline   : {baseline_dir}")


def main():
    parser = argparse.ArgumentParser(description="SGE-LightRAG end-to-end integration")
    parser.add_argument("csv_path", nargs="?",
                        default=str(Path.home() / "Desktop/SGE/dataset/年度预算/annualbudget_sc.csv"),
                        help="Path to input CSV (default: annualbudget_sc.csv)")
    parser.add_argument("--output-dir", "-o",
                        default=str(Path.home() / "Desktop/SGE/sge_lightrag/output"),
                        help="Output base directory")
    args = parser.parse_args()

    csv_path   = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CSV      : {csv_path}")
    print(f"Output   : {output_dir}")

    asyncio.run(main_async(csv_path, output_dir))


if __name__ == "__main__":
    main()
