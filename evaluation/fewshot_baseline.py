#!/usr/bin/env python3
"""
fewshot_baseline.py — Few-Shot Structured Prompt Baseline for SGE-LightRAG.

Tests the simplest practitioner approach: provide a few example triples in the
prompt, then ask LightRAG to extract ALL entities and relations from the CSV.
No SGE pipeline (Stage 1/2/3) is used — only the few-shot prompt hint.

This baseline answers: "Would a careful practitioner with example triples match
SGE's performance, without understanding the underlying structure?"

Methodology:
  1. Load raw CSV for each dataset
  2. Build a few-shot prompt with 3 example triples showing the desired format
  3. Feed naive-serialized CSV chunks to LightRAG with the enhanced system prompt
  4. Evaluate FC against v2 gold standards
  5. Save results to evaluation/results/fewshot_baseline_results.json

Usage:
    python3 evaluation/fewshot_baseline.py
    python3 evaluation/fewshot_baseline.py --dataset who
    python3 evaluation/fewshot_baseline.py --dataset who --fresh
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS

from evaluation.evaluate_coverage import (
    load_gold,
    load_graph,
    check_entity_coverage,
    check_fact_coverage,
)

# ---------------------------------------------------------------------------
# API / embedding config
# ---------------------------------------------------------------------------

API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL = "claude-haiku-4-5-20251001"
EMBED_DIM = 1024

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# World Bank CSV files have 4 metadata rows before the actual header.
# WHO file has a clean header (no skip rows needed).

DATASETS: dict[str, dict] = {
    "who": {
        "label": "WHO Life Expectancy (Type-II)",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "skiprows": 0,
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "sge_graph": (
            "output/who_life_expectancy/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline_graph": (
            "output/baseline_who_life/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "output_dir": "output/fewshot_who",
        "language": "English",
        "fewshot_examples": (
            "From 'China,CHN,Life expectancy at birth (years),WHOSIS_000001,"
            "70.83,...' extract:\n"
            "  entity<|#|>CHN<|#|>Country_Code<|#|>A country in the dataset\n"
            "  relation<|#|>CHN<|#|>70.83<|#|>LIFE_EXPECTANCY<|#|>"
            "CHN had life expectancy of 70.83 in 2000\n"
            "  relation<|#|>CHN<|#|>72.44<|#|>LIFE_EXPECTANCY<|#|>"
            "CHN had life expectancy of 72.44 in 2005\n"
        ),
    },
    "wb_cm": {
        "label": "WB Child Mortality (Type-II)",
        "csv_path": (
            "dataset/世界银行数据/child_mortality/"
            "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv"
        ),
        "skiprows": 4,
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "sge_graph": (
            "output/wb_child_mortality/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline_graph": (
            "output/baseline_wb_child_mortality/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "output_dir": "output/fewshot_wb_cm",
        "language": "English",
        "fewshot_examples": (
            "From 'Argentina,ARG,Mortality rate under-5,SH.DYN.MORT,"
            "19.4,...' extract:\n"
            "  entity<|#|>ARG<|#|>Country_Code<|#|>A country in the dataset\n"
            "  relation<|#|>ARG<|#|>19.4<|#|>UNDER5_MORTALITY_RATE<|#|>"
            "ARG had under-5 mortality rate of 19.4 in 2000\n"
            "  relation<|#|>ARG<|#|>16.7<|#|>UNDER5_MORTALITY_RATE<|#|>"
            "ARG had under-5 mortality rate of 16.7 in 2005\n"
        ),
    },
    "wb_pop": {
        "label": "WB Population (Type-II)",
        "csv_path": (
            "dataset/世界银行数据/population/"
            "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"
        ),
        "skiprows": 4,
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "sge_graph": (
            "output/wb_population/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline_graph": (
            "output/baseline_wb_population/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "output_dir": "output/fewshot_wb_pop",
        "language": "English",
        "fewshot_examples": (
            "From 'China,CHN,Population total,SP.POP.TOTL,"
            "1411778724,...' extract:\n"
            "  entity<|#|>CHN<|#|>Country_Code<|#|>A country in the dataset\n"
            "  relation<|#|>CHN<|#|>1411778724<|#|>HAS_POPULATION<|#|>"
            "CHN had population of 1411778724 in 2020\n"
            "  relation<|#|>CHN<|#|>1412360325<|#|>HAS_POPULATION<|#|>"
            "CHN had population of 1412360325 in 2021\n"
        ),
    },
    "wb_mat": {
        "label": "WB Maternal Mortality (Type-II)",
        "csv_path": (
            "dataset/世界银行数据/maternal_mortality/"
            "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv"
        ),
        "skiprows": 4,
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "sge_graph": (
            "output/wb_maternal/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline_graph": (
            "output/baseline_wb_maternal/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "output_dir": "output/fewshot_wb_mat",
        "language": "English",
        "fewshot_examples": (
            "From 'Brazil,BRA,Maternal mortality ratio,SH.STA.MMRT,"
            "77,...' extract:\n"
            "  entity<|#|>BRA<|#|>Country_Code<|#|>A country in the dataset\n"
            "  relation<|#|>BRA<|#|>77<|#|>MATERNAL_MORTALITY_RATE<|#|>"
            "BRA had maternal mortality ratio of 77 in 2000\n"
            "  relation<|#|>BRA<|#|>68<|#|>MATERNAL_MORTALITY_RATE<|#|>"
            "BRA had maternal mortality ratio of 68 in 2005\n"
        ),
    },
    "inpatient": {
        "label": "HK Inpatient 2023 (Type-III)",
        "csv_path": (
            "dataset/住院病人统计/"
            "Inpatient Discharges and Deaths in Hospitals and "
            "Registered Deaths in Hong Kong by Disease 2023 (SC).csv"
        ),
        "skiprows": 0,
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "sge_graph": (
            "output/inpatient_2023/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "baseline_graph": (
            "output/baseline_inpatient23/lightrag_storage/"
            "graph_chunk_entity_relation.graphml"
        ),
        "output_dir": "output/fewshot_inpatient",
        "language": "Chinese",
        "fewshot_examples": (
            "From '肺炎,J12-J18,60614,55487,...' extract:\n"
            "  entity<|#|>肺炎<|#|>Disease_Category<|#|>A disease in the dataset\n"
            "  relation<|#|>肺炎<|#|>60614<|#|>INPATIENT_TOTAL<|#|>"
            "肺炎 had total inpatient count of 60614\n"
            "  relation<|#|>肺炎<|#|>55487<|#|>INPATIENT_HA_HOSPITAL<|#|>"
            "肺炎 had HA hospital inpatient count of 55487\n"
        ),
    },
}

# ---------------------------------------------------------------------------
# LLM function
# ---------------------------------------------------------------------------


async def llm_model_func(
    prompt,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedding function (Ollama via urllib3 to bypass macOS proxy)
# ---------------------------------------------------------------------------

import urllib3 as _urllib3

_pool = _urllib3.HTTPConnectionPool(OLLAMA_HOST, port=OLLAMA_PORT, maxsize=4)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


def _ollama_embed_sync(texts: list[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        clipped = text[:1000] if len(text) > 1000 else text
        body = json.dumps({"model": "mxbai-embed-large", "prompt": clipped}).encode()
        resp = _pool.urlopen(
            "POST",
            "/api/embeddings",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        emb = json.loads(resp.data)["embedding"]
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    import asyncio as _aio

    loop = _aio.get_event_loop()
    for attempt in range(3):
        try:
            return await loop.run_in_executor(None, _ollama_embed_sync, texts)
        except Exception as exc:
            if attempt < 2:
                print(f"  [warn] Embed attempt {attempt + 1} failed: {exc}, retrying...")
                await _aio.sleep(2)
            else:
                print(f"  [warn] Embed failed 3x, using hash fallback")
                return np.array([_hash_embed(t) for t in texts], dtype=np.float32)
    # Should not be reached, but satisfies type checker
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=safe_embedding_func,
)

# ---------------------------------------------------------------------------
# Few-shot prompt construction
# ---------------------------------------------------------------------------

FEW_SHOT_WRAPPER = """\
---FEW-SHOT STRUCTURED EXTRACTION---

You are extracting entities and relations from statistical CSV data.
Follow the exact output format shown in the examples below.

EXAMPLES:
{fewshot_examples}

INSTRUCTIONS:
- Extract ALL entities and relations from the following CSV data in this exact format
- Use entity<|#|>NAME<|#|>TYPE<|#|>DESCRIPTION for entities
- Use relation<|#|>SUBJECT<|#|>OBJECT<|#|>RELATION_TYPE<|#|>DESCRIPTION for relations
- Preserve ALL numeric values with their associated entities and time periods
- Do not skip any rows or values

---END FEW-SHOT EXAMPLES---

{original_prompt}"""


def build_fewshot_prompt(original_prompt: str, fewshot_examples: str) -> str:
    """Build a few-shot enhanced system prompt.

    Prepends example triples to the original LightRAG system prompt.
    Only the system prompt is modified — no Stage 1/2/3 is invoked.
    """
    return FEW_SHOT_WRAPPER.format(
        fewshot_examples=fewshot_examples.strip(),
        original_prompt=original_prompt,
    )


# ---------------------------------------------------------------------------
# CSV loading and naive serialization
# ---------------------------------------------------------------------------


def load_and_serialize_csv(csv_path: Path, skiprows: int) -> list[str]:
    """Load CSV and produce naive text chunks (same as LightRAG baseline).

    Uses pandas to_string() for naive serialization, then splits into
    chunks at line boundaries (~4000 chars each).
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed.", file=sys.stderr)
        sys.exit(1)

    from stage1.features import _detect_encoding

    encoding = _detect_encoding(str(csv_path))
    df = None
    for enc in [encoding, "utf-8-sig", "utf-8", "latin-1"]:
        try:
            if skiprows > 0:
                df = pd.read_csv(csv_path, encoding=enc, skiprows=skiprows)
            else:
                df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError(f"Cannot read CSV with any encoding: {csv_path}")

    print(f"  CSV shape: {df.shape}")

    # Show first 3 rows as context
    sample = df.head(3).to_string()
    print(f"  First 3 rows:\n{sample}\n")

    text = df.to_string()
    return _chunk_text(text, max_chars=4000)


def _chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks at line boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_size = 0
        current_lines.append(line)
        current_size += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


# ---------------------------------------------------------------------------
# LightRAG runner
# ---------------------------------------------------------------------------


async def run_fewshot_lightrag(
    chunks: list[str],
    working_dir: Path,
    fewshot_examples: str,
    language: str,
    fresh: bool,
) -> dict:
    """Run LightRAG with few-shot enhanced system prompt.

    The system prompt is overridden with few-shot examples prepended.
    No SGE stages are invoked — this is a pure prompt-engineering baseline.

    Parameters
    ----------
    chunks         : list of naive CSV text chunks
    working_dir    : LightRAG storage directory
    fewshot_examples : example triples string
    language       : LLM response language
    fresh          : if True, remove existing graph before running
    """
    working_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        graph_file = working_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            graph_file.unlink()
            print("  Removed existing graph for fresh run.")

    original_prompt = PROMPTS["entity_extraction_system_prompt"]
    enhanced_prompt = build_fewshot_prompt(original_prompt, fewshot_examples)

    print(f"  Enhanced prompt length: {len(enhanced_prompt)} chars "
          f"(original: {len(original_prompt)})")

    PROMPTS["entity_extraction_system_prompt"] = enhanced_prompt

    try:
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EMBEDDING_FUNC,
            addon_params={"language": language},
            llm_model_max_async=5,
            embedding_func_max_async=4,
            entity_extract_max_gleaning=0,
        )
        await rag.initialize_storages()

        insert_chunks = [f"[FEWSHOT_BASELINE]\n{c}" for c in chunks]
        print(f"  Inserting {len(insert_chunks)} chunks...")

        for i, chunk in enumerate(insert_chunks, 1):
            print(f"  [{i}/{len(insert_chunks)}] ({len(chunk)} chars)")
            await rag.ainsert(chunk)

        await rag.finalize_storages()

    finally:
        # Always restore original prompt
        PROMPTS["entity_extraction_system_prompt"] = original_prompt

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    stats: dict = {
        "working_dir": str(working_dir),
        "chunks_inserted": len(insert_chunks),
        "graph_file_exists": graph_path.exists(),
    }

    if graph_path.exists():
        try:
            import networkx as nx

            G = nx.read_graphml(str(graph_path))
            stats["node_count"] = G.number_of_nodes()
            stats["edge_count"] = G.number_of_edges()
            print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges")
        except Exception as exc:
            print(f"  [warn] Could not parse graphml: {exc}", file=sys.stderr)

    return stats


# ---------------------------------------------------------------------------
# FC evaluation
# ---------------------------------------------------------------------------


def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    """Run FC evaluation on a graph against a gold standard.

    Parameters
    ----------
    graph_path : path to the .graphml file
    gold_path  : path to the gold .jsonl file
    label      : human-readable label for logging

    Returns
    -------
    dict with ec, fc, matched/total counts
    """
    if not Path(graph_path).exists():
        print(f"  [warn] Graph not found: {graph_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "graph_not_found"}
    if not Path(gold_path).exists():
        print(f"  [warn] Gold not found: {gold_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "gold_not_found"}

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(
        f"  [{label}] EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
        f"FC={fc:.4f} ({len(covered)}/{len(facts)})"
    )

    return {
        "label": label,
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "ec_matched": len(matched_entities),
        "ec_total": len(gold_entities),
        "fc_covered": len(covered),
        "fc_total": len(facts),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# Main pipeline per dataset
# ---------------------------------------------------------------------------


async def run_dataset(dataset_key: str, fresh: bool) -> dict:
    """Run few-shot baseline and evaluate FC for one dataset."""
    if dataset_key not in DATASETS:
        print(
            f"ERROR: unknown dataset '{dataset_key}'. "
            f"Available: {list(DATASETS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"FEW-SHOT BASELINE: {cfg['label']}")
    print(f"{'='*60}")

    csv_path = PROJECT_ROOT / cfg["csv_path"]
    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    lightrag_storage = output_dir / "lightrag_storage"

    sge_graph_path = str(PROJECT_ROOT / cfg["sge_graph"])
    baseline_graph_path = str(PROJECT_ROOT / cfg["baseline_graph"])
    fewshot_graph_path = str(lightrag_storage / "graph_chunk_entity_relation.graphml")

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Build naive chunks from raw CSV
    print(f"\n[Step 1] Loading and serializing CSV: {csv_path.name}")
    chunks = load_and_serialize_csv(csv_path, cfg["skiprows"])
    print(f"  Produced {len(chunks)} chunks")

    # Build few-shot graph
    print(f"\n[Step 2] Building Few-Shot LightRAG Graph...")
    print(f"  Output: {lightrag_storage}")
    stats = await run_fewshot_lightrag(
        chunks=chunks,
        working_dir=lightrag_storage,
        fewshot_examples=cfg["fewshot_examples"],
        language=cfg["language"],
        fresh=fresh,
    )

    # Save run stats
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "run_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Evaluate all three systems
    print(f"\n[Step 3] FC Evaluation...")
    fewshot_result = evaluate_graph_fc(fewshot_graph_path, gold_path, "Few-Shot")
    sge_result = evaluate_graph_fc(sge_graph_path, gold_path, "SGE")
    baseline_result = evaluate_graph_fc(baseline_graph_path, gold_path, "Baseline")

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "timestamp": datetime.now().isoformat(),
        "fewshot": fewshot_result,
        "sge": sge_result,
        "baseline": baseline_result,
    }


def print_comparison(result: dict) -> None:
    """Print a formatted three-way comparison table."""
    fs = result["fewshot"]
    sge = result["sge"]
    bl = result["baseline"]

    print(f"\n{'='*62}")
    print(f"THREE-WAY COMPARISON: {result['label']}")
    print(f"{'='*62}")
    print(f"{'Method':<20} | {'EC':>6} | {'FC':>6} | {'Nodes':>6} | {'Edges':>6}")
    print(f"{'-'*62}")

    for r, name in [(sge, "SGE"), (fs, "Few-Shot"), (bl, "Baseline")]:
        if r.get("error"):
            print(
                f"  {name:<18} | {'N/A':>6} | {'N/A':>6} | "
                f"{'N/A':>6} | {'N/A':>6}  [{r['error']}]"
            )
        else:
            print(
                f"  {name:<18} | {r['ec']:>6.4f} | {r['fc']:>6.4f} | "
                f"{r.get('node_count', '?'):>6} | {r.get('edge_count', '?'):>6}"
            )

    print(f"{'='*62}")

    sge_fc = sge.get("fc", 0.0)
    fs_fc = fs.get("fc", 0.0)
    bl_fc = bl.get("fc", 0.0)

    if not fs.get("error"):
        fs_vs_base = fs_fc - bl_fc
        sge_vs_fs = sge_fc - fs_fc
        print(f"\n[Interpretation]")
        print(
            f"  Few-Shot vs Baseline: {fs_vs_base:+.4f} "
            f"({'improvement' if fs_vs_base > 0 else 'no improvement'})"
        )
        print(
            f"  SGE vs Few-Shot:      {sge_vs_fs:+.4f} "
            f"({'SGE adds value' if sge_vs_fs > 0.05 else 'marginal difference'})"
        )

        if sge_vs_fs > 0.05:
            print(
                f"\n  Conclusion: Few-shot examples alone are insufficient "
                f"(Δ FC={sge_vs_fs:.4f}).\n"
                f"  SGE's three-stage pipeline provides structural gains "
                f"beyond few-shot prompting."
            )
        elif fs_vs_base > 0.05:
            print(
                f"\n  Conclusion: Few-shot examples help but SGE achieves "
                f"higher coverage.\n  Both approaches improve over baseline."
            )
        else:
            print(
                f"\n  Conclusion: Few-shot examples provide no significant "
                f"improvement over baseline."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot structured prompt baseline for SGE-LightRAG."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing few-shot graph before running (force rebuild)",
    )
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if (args.dataset is None or args.dataset == "all") else [args.dataset]

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "fewshot_baseline_results.json"

    # Load existing results
    existing: list = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, IOError):
            existing = []

    for key in keys:
        result = asyncio.run(run_dataset(key, args.fresh))
        print_comparison(result)

        # Merge — replace existing entry for this dataset if present
        updated = [r for r in existing if r.get("dataset") != key]
        updated.append(result)
        existing = updated

        out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
