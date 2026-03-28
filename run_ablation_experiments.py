#!/usr/bin/env python3
"""
run_ablation_experiments.py — Run ablation and cross-model experiments.

Supports three experiment types:
  1. rule_ablation: Run Rule SGE (no LLM) on a dataset, compare with LLM SGE
  2. cross_model: Run SGE with a different LLM (e.g., gpt-5-mini)
  3. compact: Run compact representation mode on large Type-II datasets

Usage:
    python3 run_ablation_experiments.py --experiment rule_ablation --dataset who
    python3 run_ablation_experiments.py --experiment cross_model --dataset who --model gpt-5-mini
    python3 run_ablation_experiments.py --experiment compact --dataset wb_cm
"""

from __future__ import annotations

import sys
import json
import asyncio
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.integrator import patch_lightrag

API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

# Dataset configurations
DATASETS = {
    "who": {
        "csv": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold_who_life_expectancy_v2.jsonl",
        "existing_sge": "output/who_life_expectancy",
        "existing_baseline": "output/baseline_who_life",
    },
    "wb_cm": {
        "csv": "../dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold_wb_child_mortality_v2.jsonl",
        "existing_sge": "output/wb_child_mortality",
        "existing_baseline": "output/baseline_wb_child_mortality",
    },
    "inpatient": {
        "csv": "../dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold_inpatient_2023.jsonl",
        "existing_sge": "output/inpatient_2023",
        "existing_baseline": "output/baseline_inpatient23",
    },
}


async def llm_model_func_factory(model_name):
    """Create an LLM function for a specific model."""
    from lightrag.llm.openai import openai_complete_if_cache

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            model_name, prompt, system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=API_KEY, base_url=BASE_URL, **kwargs,
        )
    return llm_func


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


async def run_lightrag_experiment(chunks, working_dir, addon_params, system_prompt_override,
                                  model_name, label):
    """Run LightRAG with given chunks and configuration."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.prompt import PROMPTS

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM, max_token_size=512, func=safe_embedding_func,
    )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    llm_func = await llm_model_func_factory(model_name)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        addon_params=addon_params,
        llm_model_max_async=10,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Override system prompt if provided
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    if system_prompt_override:
        escaped = system_prompt_override.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        PROMPTS["entity_extraction_system_prompt"] = escaped

    try:
        print(f"\n[{label}] Inserting {len(chunks)} chunks into {working_dir}...")
        for i, chunk in enumerate(chunks, 1):
            if i % 50 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}]")
            await rag.ainsert(chunk)
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats = {"label": label, "chunks": len(chunks), "timestamp": datetime.now().isoformat()}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    await rag.finalize_storages()
    return stats


async def run_rule_ablation(dataset_key):
    """Run Rule SGE ablation: compare Rule SGE vs existing LLM SGE."""
    ds = DATASETS[dataset_key]
    csv_path = str(PROJECT_ROOT / ds["csv"])
    output_dir = PROJECT_ROOT / "output" / f"ablation_rule_{dataset_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run pipeline with rule mode
    print("=" * 60)
    print(f"ABLATION: Rule SGE on {dataset_key}")
    print("=" * 60)

    from stage1.features import extract_features
    from stage1.classifier import classify
    from stage1.schema import build_meta_schema
    from stage2.inducer import induce_schema as rule_induce
    from stage3.serializer import serialize_csv

    features = extract_features(csv_path)
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    # Use inducer.py directly (produces column_roles needed by serializer)
    schema = rule_induce(meta_schema, features)
    schema["_n_rows"] = features.n_rows
    if "time_dimension" not in schema:
        schema["time_dimension"] = meta_schema.get("time_dimension", {})

    (output_dir / "extraction_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Entity types: {schema['entity_types']}")
    print(f"  Table type: {schema['table_type']}")

    # Step 2: Serialize chunks
    chunks = serialize_csv(csv_path, schema)
    print(f"  Chunks: {len(chunks)}")

    # Step 3: Run LightRAG
    payload = patch_lightrag(schema)
    work_dir = output_dir / "lightrag_storage"

    stats = await run_lightrag_experiment(
        chunks=chunks,
        working_dir=work_dir,
        addon_params=payload["addon_params"],
        system_prompt_override=payload["system_prompt"] if not payload.get("use_baseline_mode") else None,
        model_name=DEFAULT_MODEL,
        label=f"Rule-SGE-{dataset_key}",
    )

    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Results in: {output_dir}")


async def run_cross_model(dataset_key, model_name):
    """Run SGE with a different LLM model."""
    ds = DATASETS[dataset_key]
    existing_dir = PROJECT_ROOT / ds["existing_sge"]
    output_dir = PROJECT_ROOT / "output" / f"crossmodel_{model_name.replace('-','_')}_{dataset_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CROSS-MODEL: {model_name} on {dataset_key}")
    print("=" * 60)

    # Load existing SGE schema and chunks
    schema_path = existing_dir / "extraction_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(f"  Schema from: {schema_path}")
    print(f"  Entity types: {schema['entity_types']}")

    chunks_dir = existing_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    chunks = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"  Chunks: {len(chunks)}")

    # Build payload
    payload = patch_lightrag(schema)
    work_dir = output_dir / "lightrag_storage"

    stats = await run_lightrag_experiment(
        chunks=chunks,
        working_dir=work_dir,
        addon_params=payload["addon_params"],
        system_prompt_override=payload["system_prompt"] if not payload.get("use_baseline_mode") else None,
        model_name=model_name,
        label=f"{model_name}-{dataset_key}",
    )

    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Results in: {output_dir}")


async def run_compact(dataset_key):
    """Run compact representation mode."""
    ds = DATASETS[dataset_key]
    csv_path = str(PROJECT_ROOT / ds["csv"])
    output_dir = PROJECT_ROOT / "output" / f"compact_{dataset_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"COMPACT MODE: {dataset_key}")
    print("=" * 60)

    from stage1.features import extract_features
    from stage1.classifier import classify
    from stage1.schema import build_meta_schema
    from stage2.inducer import induce_schema as rule_induce
    from stage3.serializer import serialize_csv
    from stage3.compact_representation import should_use_compact

    features = extract_features(csv_path)
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    schema = rule_induce(meta_schema, features)
    if "time_dimension" not in schema:
        schema["time_dimension"] = meta_schema.get("time_dimension", {})

    # Use actual CSV row count (not feature sample) for compact mode threshold
    import pandas as pd
    try:
        df_full = pd.read_csv(csv_path, skiprows=4 if "World Bank" in csv_path or "API_" in csv_path else 0,
                              encoding="utf-8", on_bad_lines="skip")
        actual_n_rows = len(df_full)
    except Exception:
        actual_n_rows = features.n_rows
    schema["_n_rows"] = actual_n_rows

    print(f"  actual_n_rows={actual_n_rows}, should_use_compact={should_use_compact(schema, actual_n_rows)}")

    # Serialize with compact mode
    chunks = serialize_csv(csv_path, schema)
    print(f"  Chunks: {len(chunks)} (compact)")

    (output_dir / "extraction_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Run LightRAG
    payload = patch_lightrag(schema)
    work_dir = output_dir / "lightrag_storage"

    stats = await run_lightrag_experiment(
        chunks=chunks,
        working_dir=work_dir,
        addon_params=payload["addon_params"],
        system_prompt_override=payload["system_prompt"] if not payload.get("use_baseline_mode") else None,
        model_name=DEFAULT_MODEL,
        label=f"Compact-{dataset_key}",
    )

    (output_dir / "experiment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Results in: {output_dir}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["rule_ablation", "cross_model", "compact"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--model", default="gpt-5-mini", help="Model for cross_model experiment")
    args = parser.parse_args()

    if args.experiment == "rule_ablation":
        await run_rule_ablation(args.dataset)
    elif args.experiment == "cross_model":
        await run_cross_model(args.dataset, args.model)
    elif args.experiment == "compact":
        await run_compact(args.dataset)


if __name__ == "__main__":
    asyncio.run(main())
