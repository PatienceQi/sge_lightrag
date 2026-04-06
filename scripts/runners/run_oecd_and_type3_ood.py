#!/usr/bin/env python3
"""
run_oecd_and_type3_ood.py — Run SGE + Baseline on OECD blind test + Type-III OOD datasets.

Uses a SEPARATE API key to run in parallel with other experiments.

Datasets:
  OECD blind test (4 datasets):
    1. oecd_gdp_usd_millions.csv (Type-II)
    2. oecd_hospital_beds_per_1000.csv (Type-II)
    3. oecd_discharge_by_country_disease.csv (Type-III)
    4. oecd_germany_hospital_discharge.csv (Type-III)

  Type-III OOD (2 datasets):
    5. wb_health_expenditure_by_category.csv (Type-III)
    6. wb_immunization_multi_vaccine.csv (Type-III)

Usage:
    python3 scripts/runners/run_oecd_and_type3_ood.py
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

# ── Config (KEY 2 — separate from main experiments) ──────────────────────────
API_KEY  = "sk-ihcMkgpox4BU2E0Rg8D016v2k6UZgy2pvui4AX2UHwvg4GB4"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

OECD_DIR    = Path("/Users/qipatience/Desktop/SGE/dataset/OECD_blind_test")
OOD_DIR     = Path("/Users/qipatience/Desktop/SGE/dataset/ood_blind_test")
OUTPUT_DIR  = PROJECT_ROOT / "output"
GOLD_DIR    = PROJECT_ROOT / "evaluation" / "gold"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DATASETS = {
    "oecd_gdp": {
        "csv":  OECD_DIR / "oecd_gdp_usd_millions.csv",
        "gold": GOLD_DIR / "gold_oecd_gdp.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_oecd_gdp",
        "baseline_output":  OUTPUT_DIR / "baseline_oecd_gdp",
    },
    "oecd_beds": {
        "csv":  OECD_DIR / "oecd_hospital_beds_per_1000.csv",
        "gold": GOLD_DIR / "gold_oecd_hospital_beds.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_oecd_beds",
        "baseline_output":  OUTPUT_DIR / "baseline_oecd_beds",
    },
    "oecd_discharge": {
        "csv":  OECD_DIR / "oecd_discharge_by_country_disease.csv",
        "gold": GOLD_DIR / "gold_oecd_discharge_country.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_oecd_discharge",
        "baseline_output":  OUTPUT_DIR / "baseline_oecd_discharge",
    },
    "oecd_germany": {
        "csv":  OECD_DIR / "oecd_germany_hospital_discharge.csv",
        "gold": GOLD_DIR / "gold_oecd_germany_discharge.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_oecd_germany",
        "baseline_output":  OUTPUT_DIR / "baseline_oecd_germany",
    },
    "type3_health_exp": {
        "csv":  OOD_DIR / "wb_health_expenditure_by_category.csv",
        "gold": GOLD_DIR / "gold_ood_wb_health_exp_category.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_type3_health_exp",
        "baseline_output":  OUTPUT_DIR / "baseline_type3_health_exp",
    },
    "type3_immun_multi": {
        "csv":  OOD_DIR / "wb_immunization_multi_vaccine.csv",
        "gold": GOLD_DIR / "gold_ood_wb_immunization_multi.jsonl",
        "sge_output":      OUTPUT_DIR / "sge_type3_immun_multi",
        "baseline_output":  OUTPUT_DIR / "baseline_type3_immun_multi",
    },
}

# ── LLM & Embedding ─────────────────────────────────────────────────────────

async def llm_func(prompt, system_prompt=None, history_messages=[], **kw):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kw)

def _hash_embed(t):
    v = [0.0]*EMBED_DIM
    h = hashlib.sha256(t.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        v[i] = (h[i]-128)/128.0
    return v

async def embed_func(texts):
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)

EMBED = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=512, func=embed_func)

# ── SGE Pipeline ─────────────────────────────────────────────────────────────

async def run_sge(csv_path: Path, output_dir: Path) -> dict:
    """Run full SGE pipeline: Stage 1 → 2 → 3 → LightRAG."""
    print(f"  [SGE] Stage 1: classify")
    features = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"    Type: {table_type}")

    print(f"  [SGE] Stage 2: induce schema (rule)")
    schema = induce_schema_rule(meta_schema, features)

    print(f"  [SGE] Stage 3: serialize + inject")
    chunks = serialize_csv(str(csv_path), schema)
    print(f"    Chunks: {len(chunks)}")

    storage_dir = output_dir / "lightrag_storage"
    if storage_dir.exists():
        shutil.rmtree(str(storage_dir))
    storage_dir.mkdir(parents=True, exist_ok=True)

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    patch_lightrag(schema, language="English")

    try:
        rag = LightRAG(
            working_dir=str(storage_dir),
            llm_model_func=llm_func,
            embedding_func=EMBED,
            addon_params={"language": "English"},
            llm_model_max_async=4,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
            force_llm_summary_on_merge=999,
        )
        await rag.initialize_storages()
        for i, chunk in enumerate(chunks, 1):
            if i <= 3 or i % 50 == 0 or i == len(chunks):
                print(f"    [{i}/{len(chunks)}]")
            await rag.ainsert(chunk)
        await rag.finalize_storages()
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"
    return {"graph": str(graph_path), "chunks": len(chunks), "exists": graph_path.exists()}

# ── Baseline ─────────────────────────────────────────────────────────────────

async def run_baseline(csv_path: Path, output_dir: Path) -> dict:
    """Run vanilla LightRAG baseline (naive CSV serialization)."""
    import pandas as pd
    from stage1.preprocessor import preprocess_csv

    print(f"  [Baseline] Naive serialization")
    df, _ = preprocess_csv(str(csv_path))
    text = df.to_string(index=False)
    # Split into chunks of ~2000 chars
    chunk_size = 2000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"    Chunks: {len(chunks)}")

    storage_dir = output_dir / "lightrag_storage"
    if storage_dir.exists():
        shutil.rmtree(str(storage_dir))
    storage_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(storage_dir),
        llm_model_func=llm_func,
        embedding_func=EMBED,
        addon_params={"language": "English"},
        llm_model_max_async=4,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,
        force_llm_summary_on_merge=999,
    )
    await rag.initialize_storages()
    for i, chunk in enumerate(chunks, 1):
        if i <= 3 or i % 20 == 0 or i == len(chunks):
            print(f"    [{i}/{len(chunks)}]")
        await rag.ainsert(chunk)
    await rag.finalize_storages()

    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"
    return {"graph": str(graph_path), "chunks": len(chunks), "exists": graph_path.exists()}

# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_fc(graph_path: str, gold_path: str) -> dict:
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_entity_coverage, check_fact_coverage)
    if not Path(graph_path).exists():
        return {"ec": 0, "fc": 0, "error": "graph_not_found"}
    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)
    matched = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched) / len(gold_entities) if gold_entities else 0
    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0
    return {"ec": round(ec, 4), "fc": round(fc, 4),
            "matched_entities": len(matched), "total_entities": len(gold_entities),
            "covered_facts": len(covered), "total_facts": len(facts)}

# ── Main ─────────────────────────────────────────────────────────────────────

async def run_dataset(name: str, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"CSV: {cfg['csv']}")
    print(f"{'='*60}")

    result = {"dataset": name, "timestamp": datetime.now().isoformat()}

    # SGE
    try:
        sge_stats = await run_sge(cfg["csv"], cfg["sge_output"])
        sge_eval = evaluate_fc(sge_stats["graph"], str(cfg["gold"]))
        result["sge"] = {**sge_stats, **sge_eval}
        print(f"  SGE: EC={sge_eval['ec']:.4f} FC={sge_eval['fc']:.4f}")
    except Exception as e:
        print(f"  SGE FAILED: {e}")
        result["sge"] = {"error": str(e)}

    # Baseline
    try:
        base_stats = await run_baseline(cfg["csv"], cfg["baseline_output"])
        base_eval = evaluate_fc(base_stats["graph"], str(cfg["gold"]))
        result["baseline"] = {**base_stats, **base_eval}
        print(f"  Baseline: EC={base_eval['ec']:.4f} FC={base_eval['fc']:.4f}")
    except Exception as e:
        print(f"  Baseline FAILED: {e}")
        result["baseline"] = {"error": str(e)}

    return result


async def main():
    results = {}
    # Load existing results if any
    result_path = RESULTS_DIR / "oecd_type3_ood_results.json"
    if result_path.exists():
        with open(result_path) as f:
            results = json.load(f)

    for name, cfg in DATASETS.items():
        if name in results and "sge" in results[name] and "baseline" in results[name]:
            sge_fc = results[name]["sge"].get("fc", -1)
            base_fc = results[name]["baseline"].get("fc", -1)
            if sge_fc >= 0 and base_fc >= 0:
                print(f"\nSKIP {name} (already done: SGE FC={sge_fc}, Base FC={base_fc})")
                continue

        result = await run_dataset(name, cfg)
        results[name] = result

        # Save after each dataset
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        sge_fc = r.get("sge", {}).get("fc", "ERR")
        base_fc = r.get("baseline", {}).get("fc", "ERR")
        print(f"  {name}: SGE FC={sge_fc} | Base FC={base_fc}")


if __name__ == "__main__":
    asyncio.run(main())
