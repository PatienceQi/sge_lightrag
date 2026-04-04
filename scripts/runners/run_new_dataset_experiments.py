#!/usr/bin/env python3
"""
Run SGE + Baseline experiments on two new OOD datasets and evaluate FC.

Datasets:
  1. Eurostat Crime Statistics (long-format Type-III)
  2. US Census Demographics (long-format Type-III)

For each dataset:
  - SGE Pipeline: Stage 1 → Stage 2 (RULE mode) → Stage 3 → LightRAG → FC
  - Baseline: raw CSV text → LightRAG → FC

Usage:
    python3 scripts/runners/run_new_dataset_experiments.py
    python3 scripts/runners/run_new_dataset_experiments.py eurostat
    python3 scripts/runners/run_new_dataset_experiments.py us_census
    python3 scripts/runners/run_new_dataset_experiments.py all
"""
from __future__ import annotations

import sys
import json
import shutil
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2.inducer import induce_schema as induce_schema_rule
from stage3.serializer import serialize_csv
from stage3.integrator import patch_lightrag

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "gpt-4o-mini"
EMBED_DIM = 1024

DATASET_DIR = PROJECT_ROOT.parent / "dataset" / "ood_blind_test"
OUTPUT_DIR  = PROJECT_ROOT / "output"
GOLD_DIR    = PROJECT_ROOT / "evaluation" / "gold"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DATASETS = {
    "eurostat": {
        "csv":          DATASET_DIR / "synthetic_eurostat_crime_statistics.csv",
        "gold":         GOLD_DIR / "gold_eurostat_crime.jsonl",
        "sge_output":   OUTPUT_DIR / "eurostat_crime",
        "base_output":  OUTPUT_DIR / "baseline_eurostat_crime",
        "language":     "English",
        "label":        "Eurostat Crime Statistics",
    },
    "us_census": {
        "csv":          DATASET_DIR / "synthetic_us_census_population_by_demographics.csv",
        "gold":         GOLD_DIR / "gold_us_census_demographics.jsonl",
        "sge_output":   OUTPUT_DIR / "us_census_demographics",
        "base_output":  OUTPUT_DIR / "baseline_us_census_demographics",
        "language":     "English",
        "label":        "US Census Demographics",
    },
}


# ── LLM function (copied exactly from run_non_gov_experiments.py) ─────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding via urllib3 pool to bypass macOS proxy ─────────────────────────
import urllib3 as _urllib3
_pool = _urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)


def _ollama_embed_sync(texts: list[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        if len(text) > 1000:
            text = text[:1000]
        body = json.dumps({"model": "mxbai-embed-large", "prompt": text}).encode()
        resp = _pool.urlopen(
            "POST", "/api/embeddings", body=body,
            headers={"Content-Type": "application/json"}, timeout=120.0,
        )
        emb = json.loads(resp.data)["embedding"]
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    import asyncio as _aio
    loop = _aio.get_event_loop()
    for attempt in range(3):
        try:
            return await loop.run_in_executor(None, _ollama_embed_sync, texts)
        except Exception as e:
            if attempt < 2:
                print(f"  [warn] Embed attempt {attempt+1} failed: {e}, retrying...")
                await _aio.sleep(2)
            else:
                print(f"  [warn] Embed failed 3x, using hash fallback")
                return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
)


# ── Naive baseline: raw CSV text ──────────────────────────────────────────────
def read_csv_as_text(csv_path: Path) -> str:
    """Read raw CSV into plain text string using pandas."""
    import pandas as pd
    from stage1.features import _detect_encoding

    encoding = _detect_encoding(str(csv_path))
    df = None
    for enc in [encoding, "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        raise ValueError(f"Cannot read CSV: {csv_path}")

    print(f"  CSV shape (baseline): {df.shape}")
    return df.to_string()


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks at line boundaries, respecting max_chars."""
    lines = text.split("\n")
    chunks, current_lines, current_size = [], [], 0

    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines, current_size = [], 0
        current_lines.append(line)
        current_size += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


# ── Monkey-patch for SGE schema injection ─────────────────────────────────────
_original_extract_entities = _op.extract_entities


async def _sge_extract_entities(
    chunks, knowledgebase, entity_vdb, relationships_vdb,
    global_config, pipeline_status=None, llm_response_cache=None,
):
    return await _original_extract_entities(
        chunks, knowledgebase, entity_vdb, relationships_vdb,
        global_config, pipeline_status=pipeline_status,
        llm_response_cache=llm_response_cache,
    )


# ── SGE Pipeline (rule-based Stage 2) ─────────────────────────────────────────
def run_sge_pipeline(csv_path: Path, output_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"SGE PIPELINE: {csv_path.name}")
    print(f"{'='*60}")

    features = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Classified as: {table_type}")

    # Stage 2: rule-based only (as specified)
    extraction_schema = induce_schema_rule(meta_schema, features)
    print(f"  Entity types: {extraction_schema.get('entity_types', [])}")

    # Stage 3: serialize CSV into chunks
    chunks = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks generated: {len(chunks)}")

    # Persist artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_schema.json").write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2))
    (output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2))
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, c in enumerate(chunks, 1):
        (chunks_dir / f"chunk_{i:04d}.txt").write_text(c, encoding="utf-8")
    (output_dir / "system_prompt.txt").write_text(payload["system_prompt"])

    return {"chunks": chunks, "payload": payload, "schema": extraction_schema}


# ── LightRAG runner ───────────────────────────────────────────────────────────
def _count_processed_docs(storage_dir: Path) -> int:
    """Count how many docs are in 'processed' state in the kv_store."""
    kv_path = storage_dir / "kv_store_doc_status.json"
    if not kv_path.exists():
        return 0
    data = json.loads(kv_path.read_text())
    return sum(
        1 for v in data.values()
        if isinstance(v, dict) and v.get("status") == "processed"
    )


async def run_lightrag(
    chunks: list[str], working_dir: Path, addon_params: dict, label: str,
) -> dict:
    storage_dir = working_dir / "lightrag_storage"
    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"

    # Skip if already fully processed (all chunks done)
    processed = _count_processed_docs(storage_dir)
    if processed >= len(chunks) and graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats = {
            "label": label, "chunks": len(chunks),
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        }
        print(f"\n[LightRAG:{label}] Already complete — "
              f"{stats['nodes']} nodes, {stats['edges']} edges (skipping)")
        return stats

    # Clean only if no useful partial state exists
    if storage_dir.exists() and processed == 0:
        print(f"  Cleaning empty/broken storage: {storage_dir}")
        shutil.rmtree(str(storage_dir))
    storage_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[LightRAG:{label}] {storage_dir} (resume from {processed}/{len(chunks)})")

    rag = LightRAG(
        working_dir=str(storage_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=5,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    print(f"  Inserting {len(chunks)} chunks (batch, max_async=5)...")
    await rag.ainsert(chunks)

    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"
    stats = {"label": label, "chunks": len(chunks)}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    await rag.finalize_storages()
    return stats


# ── FC evaluation ─────────────────────────────────────────────────────────────
def evaluate_fc(graphml_path: Path, gold_path: Path) -> dict:
    """Compute EC and FC by importing evaluation functions directly."""
    sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
    from evaluate_coverage import load_gold, load_graph, check_entity_coverage, check_fact_coverage

    if not graphml_path.exists():
        print(f"  [warn] Graph not found: {graphml_path}")
        return {"ec": 0.0, "fc": 0.0, "covered": 0, "total": 0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(str(gold_path))
    G, graph_nodes, entity_text = load_graph(str(graphml_path))

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    return {
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "covered": len(covered),
        "total": len(facts),
        "gold_entities": len(gold_entities),
        "matched_entities": len(matched_entities),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }


# ── Dataset runner ────────────────────────────────────────────────────────────
async def run_dataset(name: str, config: dict) -> dict:
    csv_path = config["csv"]
    gold_path = config["gold"]
    label = config["label"]

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return {}
    if not gold_path.exists():
        print(f"ERROR: Gold not found: {gold_path}")
        return {}

    print(f"\n{'#'*60}")
    print(f"DATASET: {label}")
    print(f"{'#'*60}")

    # ── 1. SGE Pipeline ──
    sge_result = run_sge_pipeline(csv_path, config["sge_output"])
    chunks = sge_result["chunks"]
    payload = sge_result["payload"]

    # ── 2. SGE-enhanced LightRAG ──
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped
    _op.extract_entities = _sge_extract_entities

    try:
        sge_stats = await run_lightrag(
            chunks, config["sge_output"],
            payload["addon_params"], f"SGE-{name}",
        )
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt
        _op.extract_entities = _original_extract_entities

    # ── 3. SGE FC evaluation ──
    sge_graph = config["sge_output"] / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    sge_fc = evaluate_fc(sge_graph, gold_path)
    sge_stats.update(sge_fc)
    print(f"\n  SGE FC = {sge_fc['fc']:.4f}  (EC={sge_fc['ec']:.4f}, "
          f"{sge_fc['covered']}/{sge_fc['total']} facts)")

    # ── 4. Baseline (raw CSV → LightRAG) ──
    csv_text = read_csv_as_text(csv_path)
    baseline_chunks = chunk_text(csv_text)
    print(f"\n  Baseline: {len(baseline_chunks)} chunks from raw CSV text")

    baseline_stats = await run_lightrag(
        baseline_chunks, config["base_output"],
        {"language": config["language"]}, f"Baseline-{name}",
    )

    # ── 5. Baseline FC evaluation ──
    base_graph = config["base_output"] / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    base_fc = evaluate_fc(base_graph, gold_path)
    baseline_stats.update(base_fc)
    print(f"\n  Baseline FC = {base_fc['fc']:.4f}  (EC={base_fc['ec']:.4f}, "
          f"{base_fc['covered']}/{base_fc['total']} facts)")

    ratio = sge_fc["fc"] / base_fc["fc"] if base_fc["fc"] > 0 else float("inf")
    print(f"\n  FC Ratio (SGE/Baseline) = {ratio:.2f}x")

    return {
        "dataset": name,
        "label": label,
        "csv": str(csv_path),
        "gold": str(gold_path),
        "sge": sge_stats,
        "baseline": baseline_stats,
        "fc_ratio": round(ratio, 4),
        "timestamp": datetime.now().isoformat(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main_async(targets: list[str]) -> None:
    all_results = {}

    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        result = await run_dataset(name, DATASETS[name])
        if result:
            all_results[name] = result

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY — FC Results")
    print(f"{'='*60}")
    print(f"{'Dataset':<30} {'SGE FC':>8} {'Base FC':>8} {'Ratio':>8}")
    print(f"{'-'*56}")
    for name, res in all_results.items():
        sge_fc  = res["sge"].get("fc", 0.0)
        base_fc = res["baseline"].get("fc", 0.0)
        ratio   = res.get("fc_ratio", 0.0)
        print(f"{res['label']:<30} {sge_fc:>8.4f} {base_fc:>8.4f} {ratio:>7.2f}x")
    print(f"{'='*60}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "new_dataset_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run SGE + Baseline experiments on new OOD datasets."
    )
    parser.add_argument("datasets", nargs="*", default=["all"],
                        help="Dataset names to run (eurostat, us_census, all)")
    args = parser.parse_args()

    if "all" in args.datasets:
        targets = list(DATASETS.keys())
    else:
        targets = args.datasets

    print(f"Running experiments on: {targets}")
    asyncio.run(main_async(targets))


if __name__ == "__main__":
    main()
