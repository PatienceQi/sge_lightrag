#!/usr/bin/env python3
"""
run_crossmodel_gemini.py — Cross-model validation with Gemini 2.5 Flash.

Runs the full SGE pipeline (Stage 1→2→3) on 5 datasets using Gemini 2.5 Flash
as the LightRAG extraction backend (via OpenAI-compatible API), then evaluates
EC/FC against v2 gold standards.

Validates that format-constraint coupling holds across a third LLM backend.

Usage:
    python3 experiments/crossmodel/run_crossmodel_gemini.py
    python3 experiments/crossmodel/run_crossmodel_gemini.py --datasets WHO WB_CM
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import asyncio
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2_llm.inductor import induce_schema as induce_schema_llm
from stage2.inducer import induce_schema as induce_schema_rule
from stage3.serializer import serialize_csv
from stage3.integrator import patch_lightrag

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS
import lightrag.operate as _op

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL    = "gemini-2.5-flash"
EMBED_DIM = 1024

DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR   = PROJECT_ROOT / "output"
GOLD_DIR     = PROJECT_ROOT / "evaluation" / "gold"
EVAL_SCRIPT  = PROJECT_ROOT / "evaluation" / "evaluate_coverage.py"
RESULTS_DIR  = PROJECT_ROOT / "experiments" / "results"

# Dataset-specific WB data CSVs start with "API_" prefix
DATASETS = {
    "WHO": {
        "label": "WHO Life Expectancy",
        "csv": DATASET_ROOT / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv",
        "output": OUTPUT_DIR / "crossmodel_gemini_who",
        "gold": GOLD_DIR / "gold_who_life_expectancy_v2.jsonl",
        "language": "English",
        "wb_skip_rows": False,
        "encoding": None,
    },
    "WB_CM": {
        "label": "WB Child Mortality",
        "csv": DATASET_ROOT / "世界银行数据" / "child_mortality" / "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "output": OUTPUT_DIR / "crossmodel_gemini_wb_cm",
        "gold": GOLD_DIR / "gold_wb_child_mortality_v2.jsonl",
        "language": "English",
        "wb_skip_rows": True,
        "encoding": None,
    },
    "WB_Pop": {
        "label": "WB Population",
        "csv": DATASET_ROOT / "世界银行数据" / "population" / "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "output": OUTPUT_DIR / "crossmodel_gemini_wb_pop",
        "gold": GOLD_DIR / "gold_wb_population_v2.jsonl",
        "language": "English",
        "wb_skip_rows": True,
        "encoding": None,
    },
    "WB_Mat": {
        "label": "WB Maternal Mortality",
        "csv": DATASET_ROOT / "世界银行数据" / "maternal_mortality" / "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "output": OUTPUT_DIR / "crossmodel_gemini_wb_mat",
        "gold": GOLD_DIR / "gold_wb_maternal_v2.jsonl",
        "language": "English",
        "wb_skip_rows": True,
        "encoding": None,
    },
    "Inpatient": {
        "label": "Inpatient HK 2023",
        "csv": DATASET_ROOT / "住院病人统计" / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "output": OUTPUT_DIR / "crossmodel_gemini_inpatient",
        "gold": GOLD_DIR / "gold_inpatient_2023.jsonl",
        "language": "Chinese",
        "wb_skip_rows": False,
        "encoding": "big5hkscs",
    },
}


# ── LLM function ──────────────────────────────────────────────────────────────
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    kwargs.setdefault("timeout", 120)
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs,
    )


# ── Embedding: Ollama via urllib3 (bypass macOS proxy) ───────────────────────
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


# ── SGE pipeline ──────────────────────────────────────────────────────────────
def run_sge_pipeline(csv_path: Path, output_dir: Path, ds_config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"SGE PIPELINE: {csv_path.name}")
    print(f"{'='*60}")

    features    = extract_features(str(csv_path))
    table_type  = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    print(f"  Type: {table_type}")

    # Stage 2: LLM with rule fallback
    rule_schema = induce_schema_rule(meta_schema, features)
    try:
        llm_schema = induce_schema_llm(str(csv_path))
        extraction_schema = {
            **rule_schema,
            "entity_types": llm_schema["entity_types"],
            "relation_types": llm_schema["relation_types"],
            "prompt_context": llm_schema.get("prompt_context", ""),
            "extraction_rules": llm_schema.get("extraction_rules", {}),
        }
        stage2_mode = "llm_enhanced"
        print(f"  LLM Stage 2 OK: {extraction_schema['entity_types']}")
    except Exception as e:
        print(f"  LLM failed ({e}), using rule-based fallback")
        extraction_schema = rule_schema
        stage2_mode = "rule_fallback"

    # Stage 3
    chunks  = serialize_csv(str(csv_path), extraction_schema)
    payload = patch_lightrag(extraction_schema)
    print(f"  Chunks: {len(chunks)}")

    # Persist SGE artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_schema.json").write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2))
    (output_dir / "extraction_schema.json").write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2))
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks, 1):
        (chunks_dir / f"chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
    (output_dir / "system_prompt.txt").write_text(payload["system_prompt"])
    (output_dir / "stage2_mode.txt").write_text(stage2_mode)

    return {"chunks": chunks, "payload": payload, "schema": extraction_schema,
            "stage2_mode": stage2_mode}


# ── LightRAG runner ───────────────────────────────────────────────────────────
_original_extract_entities = _op.extract_entities


async def _sge_passthrough(chunks, knowledgebase, entity_vdb, relationships_vdb,
                            global_config, pipeline_status=None,
                            llm_response_cache=None):
    return await _original_extract_entities(
        chunks, knowledgebase, entity_vdb, relationships_vdb,
        global_config, pipeline_status=pipeline_status,
        llm_response_cache=llm_response_cache,
    )


async def run_lightrag(chunks: list[str], working_dir: Path,
                       payload: dict, label: str) -> dict:
    # Preserve LLM response cache across retries; only clean graph + doc state
    llm_cache_backup: dict | None = None
    cache_path = working_dir / "kv_store_llm_response_cache.json"
    if working_dir.exists():
        if cache_path.exists():
            try:
                llm_cache_backup = json.loads(cache_path.read_text(encoding="utf-8"))
                print(f"  [cache] Preserving {len(llm_cache_backup)} LLM cache entries")
            except Exception:
                llm_cache_backup = None
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    # Restore LLM cache so the run can resume faster
    if llm_cache_backup:
        cache_path.write_text(
            json.dumps(llm_cache_backup, ensure_ascii=False), encoding="utf-8"
        )
    print(f"\n[LightRAG:{label}] {working_dir}")

    addon_params = payload["addon_params"]

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EMBEDDING_FUNC,
        addon_params=addon_params,
        llm_model_max_async=3,
        embedding_func_max_async=1,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Inject SGE system prompt
    # The raw_prompt is fully rendered (all placeholders filled by prompt_injector).
    # LightRAG calls .format(**context_base) on the system prompt, so we must
    # escape all literal braces ({} from JSON schema) to avoid KeyError.
    # The restore loop below is a no-op (vars already filled), kept for clarity.
    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    raw_prompt = payload["system_prompt"]
    escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
    for var in ("tuple_delimiter", "completion_delimiter", "entity_types",
                "examples", "language"):
        escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
    PROMPTS["entity_extraction_system_prompt"] = escaped
    # Diagnostic: verify JSON schema survives escaping after .format()
    try:
        test_fmt = escaped.format(
            tuple_delimiter="<|#|>", completion_delimiter="<|COMPLETE|>",
            entity_types="test", examples="test", language="test",
        )
        if '"table_type"' in test_fmt or '"entity_types"' in test_fmt:
            print("  [DIAG] Schema JSON intact after .format() ✓")
        else:
            print("  [DIAG] WARNING: Schema JSON may be malformed after .format()")
    except (KeyError, IndexError) as e:
        print(f"  [DIAG] WARNING: .format() failed: {e}")
    _op.extract_entities = _sge_passthrough

    try:
        print(f"  Inserting {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            if i % 10 == 0 or i == len(chunks):
                print(f"  [{i}/{len(chunks)}]")
            await rag.ainsert(chunk)
    finally:
        PROMPTS["entity_extraction_system_prompt"] = original_prompt
        _op.extract_entities = _original_extract_entities

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats: dict = {"label": label, "chunks": len(chunks)}
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        stats["nodes"] = G.number_of_nodes()
        stats["edges"] = G.number_of_edges()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    else:
        print(f"  [warn] Graph file not found: {graph_path}")
        stats["nodes"] = 0
        stats["edges"] = 0

    await rag.finalize_storages()
    return stats


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_fc(graph_path: Path, gold_path: Path) -> dict:
    """Run evaluate_coverage.py as subprocess and parse JSON results."""
    if not graph_path.exists():
        print(f"  [eval] Graph not found: {graph_path}")
        return {"ec": 0.0, "fc": 0.0, "error": "graph_missing"}
    if not gold_path.exists():
        print(f"  [eval] Gold not found: {gold_path}")
        return {"ec": 0.0, "fc": 0.0, "error": "gold_missing"}

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--graph", str(graph_path),
         "--gold", str(gold_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        err_tail = result.stderr[-300:] if result.stderr else "no stderr"
        print(f"  [eval] evaluate_coverage.py failed: {err_tail}")
        return {"ec": 0.0, "fc": 0.0, "error": err_tail}

    output = result.stdout
    try:
        marker = output.find("[JSON]")
        search_text = output[marker + 6:] if marker >= 0 else output
        j_start = search_text.find("{")
        j_end   = search_text.rfind("}") + 1
        data    = json.loads(search_text[j_start:j_end])
        ec = data["entity_coverage"]["coverage"]
        fc = data["fact_coverage"]["coverage"]
        print(f"  EC={ec:.3f}, FC={fc:.3f}")
        return {"ec": ec, "fc": fc,
                "ec_matched": data["entity_coverage"]["matched"],
                "ec_total":   data["entity_coverage"]["total"],
                "fc_covered": data["fact_coverage"]["covered"],
                "fc_total":   data["fact_coverage"]["total"]}
    except Exception as parse_err:
        print(f"  [eval] JSON parse error: {parse_err}")
        return {"ec": 0.0, "fc": 0.0, "error": f"parse_failed: {parse_err}"}


# ── Per-dataset runner ────────────────────────────────────────────────────────
async def run_dataset(key: str, ds: dict) -> dict:
    csv_path   = ds["csv"]
    output_dir = ds["output"]
    gold_path  = ds["gold"]

    print(f"\n{'#'*60}")
    print(f"DATASET: {ds['label']} ({key})")
    print(f"CSV: {csv_path}")
    print(f"{'#'*60}")

    if not csv_path.exists():
        msg = f"CSV not found: {csv_path}"
        print(f"  ERROR: {msg}")
        return {"error": msg}

    if not gold_path.exists():
        msg = f"Gold not found: {gold_path}"
        print(f"  ERROR: {msg}")
        return {"error": msg}

    # Clean output dir to ensure fresh run
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1→2→3
    sge_result = run_sge_pipeline(csv_path, output_dir, ds)
    chunks  = sge_result["chunks"]
    payload = sge_result["payload"]

    # LightRAG injection
    lightrag_dir = output_dir / "lightrag_storage"
    graph_stats  = await run_lightrag(chunks, lightrag_dir, payload, f"Gemini-{key}")

    # Evaluate
    graph_path = lightrag_dir / "graph_chunk_entity_relation.graphml"
    eval_result = evaluate_fc(graph_path, gold_path)

    return {
        "label": ds["label"],
        "model": MODEL,
        "stage2_mode": sge_result["stage2_mode"],
        "chunks": len(chunks),
        "nodes": graph_stats.get("nodes", 0),
        "edges": graph_stats.get("edges", 0),
        "ec": eval_result.get("ec", 0.0),
        "fc": eval_result.get("fc", 0.0),
        "ec_matched": eval_result.get("ec_matched"),
        "ec_total":   eval_result.get("ec_total"),
        "fc_covered": eval_result.get("fc_covered"),
        "fc_total":   eval_result.get("fc_total"),
        "timestamp": datetime.now().isoformat(),
    }


# ── Comparison table ──────────────────────────────────────────────────────────
def load_gpt5_mini_results() -> dict:
    """Load existing GPT-5-mini results for comparison."""
    path = RESULTS_DIR / "crossmodel_expansion_results.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_haiku_results() -> dict:
    """Load Claude Haiku baseline results from authoritative all_results_v2.json."""
    path = PROJECT_ROOT / "evaluation" / "results" / "all_results_v2.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def print_comparison_table(gemini_results: dict) -> None:
    """Print a three-way comparison table: Claude Haiku vs GPT-5-mini vs Gemini."""
    gpt_data   = load_gpt5_mini_results()
    haiku_data = load_haiku_results()

    # Dataset key mapping to haiku all_results_v2.json keys
    haiku_key_map = {
        "WHO":      "WHO Life Expectancy (v2)",
        "WB_CM":    "WB Child Mortality (v2)",
        "WB_Pop":   "WB Population (v2)",
        "WB_Mat":   "WB Maternal Mortality (v2)",
        "Inpatient": "Inpatient 2023",
    }
    # Dataset key mapping to gpt-5-mini keys
    gpt_key_map = {
        "WB_Mat": "wb_mat",
        "WB_Pop": "wb_pop",
    }

    print("\n" + "=" * 90)
    print("CROSS-MODEL COMPARISON: Claude Haiku vs GPT-5-mini vs Gemini 2.5 Flash")
    print("=" * 90)
    header = f"{'Dataset':<28} | {'Haiku FC':>9} | {'GPT-5-mini FC':>14} | {'Gemini FC':>10} | {'Nodes':>7} | {'Edges':>7}"
    print(header)
    print("-" * 90)

    for ds_key, r in gemini_results.items():
        if "error" in r:
            print(f"  {ds_key:<26} | ERROR: {r['error']}")
            continue

        gemini_fc = r.get("fc", "N/A")
        if isinstance(gemini_fc, float):
            gemini_fc_str = f"{gemini_fc:.3f}"
        else:
            gemini_fc_str = str(gemini_fc)

        # Haiku FC from all_results_v2.json
        haiku_fc_str = "N/A"
        hk = haiku_key_map.get(ds_key)
        if hk and hk in haiku_data:
            hfc = haiku_data[hk].get("sge_fc") or haiku_data[hk].get("fc")
            if hfc is not None:
                haiku_fc_str = f"{hfc:.3f}"

        # GPT-5-mini FC from crossmodel_expansion_results.json
        gpt_fc_str = "N/A"
        gk = gpt_key_map.get(ds_key)
        if gk and gk in gpt_data:
            gfc = gpt_data[gk].get("FC") or gpt_data[gk].get("fc")
            if gfc is not None and isinstance(gfc, float):
                gpt_fc_str = f"{gfc:.3f}"

        label  = r.get("label", ds_key)
        nodes  = r.get("nodes", "?")
        edges  = r.get("edges", "?")
        print(f"  {label:<26} | {haiku_fc_str:>9} | {gpt_fc_str:>14} | {gemini_fc_str:>10} | {str(nodes):>7} | {str(edges):>7}")

    print("=" * 90)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main_async(target_keys: list[str]) -> None:
    all_results: dict = {}

    for key in target_keys:
        if key not in DATASETS:
            print(f"Unknown dataset: {key}. Available: {list(DATASETS.keys())}")
            continue
        try:
            result = await run_dataset(key, DATASETS[key])
            all_results[key] = result
        except Exception as exc:
            import traceback
            print(f"\nERROR on {key}: {exc}")
            traceback.print_exc()
            all_results[key] = {"error": str(exc)}

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "crossmodel_gemini_results.json"
    output_payload = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "datasets": {
            k: {
                "ec":    v.get("ec"),
                "fc":    v.get("fc"),
                "nodes": v.get("nodes"),
                "edges": v.get("edges"),
            }
            for k, v in all_results.items()
            if "error" not in v
        },
        "full_results": all_results,
    }
    out_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    print_comparison_table(all_results)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-model validation with Gemini 2.5 Flash")
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS.keys()),
        help=f"Datasets to run (default: all). Choices: {list(DATASETS.keys())}")
    args = parser.parse_args()

    targets = args.datasets
    print(f"Model  : {MODEL}")
    print(f"Targets: {targets}")
    asyncio.run(main_async(targets))


if __name__ == "__main__":
    main()
