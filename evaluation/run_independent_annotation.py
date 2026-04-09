#!/usr/bin/env python3
"""
run_independent_annotation.py — Independent Precision Annotation for SGE vs Baseline

Simulates two independent annotators using LLM calls to verify knowledge graph
edges against source CSV data across 3 datasets: WHO, Inpatient, THE University Ranking.

Two annotators:
  - Annotator A: claude-haiku-4-5-20251001
  - Annotator B: gpt-5-mini

Outputs to evaluation/results/independent_annotation_results.json.

Usage:
    python3 evaluation/run_independent_annotation.py

Run from sge_lightrag/ directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed. pip install networkx", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai not installed. pip install openai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent   # sge_lightrag/
SGE_ROOT = BASE_DIR.parent                          # SGE/
DATASET_DIR = BASE_DIR / "dataset"                  # sge_lightrag/dataset (symlink)
NON_GOV_DIR = SGE_ROOT / "dataset" / "non_gov"     # SGE/dataset/non_gov (actual location)
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SAMPLE_SIZE = 50
RANDOM_SEED = 42
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://wolfai.top/v1"
API_KEY = os.environ.get("SGE_API_KEY", "")
MODEL_A = "claude-haiku-4-5-20251001"
MODEL_B = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "who": {
        "label": "WHO Life Expectancy",
        "sge_graph": OUTPUT_DIR / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline_graph": OUTPUT_DIR / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "csv": DATASET_DIR / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv",
        "csv_type": "world_bank",  # skip 0 rows (WHO uses standard header)
        "csv_skiprows": 0,
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "sge_graph": OUTPUT_DIR / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline_graph": OUTPUT_DIR / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "csv": DATASET_DIR / "住院病人统计" / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "csv_type": "inpatient",
        "csv_skiprows": 0,
    },
    "the": {
        "label": "THE University Ranking",
        "sge_graph": OUTPUT_DIR / "the_university_ranking" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline_graph": OUTPUT_DIR / "baseline_the_university_ranking" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "csv": NON_GOV_DIR / "the_university_ranking.csv",
        "csv_type": "the_ranking",
        "csv_skiprows": 0,
    },
}


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph_edges(graphml_path: Path) -> list[dict]:
    """Load a LightRAG GraphML and return list of edge records."""
    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML not found: {graphml_path}")

    G = nx.read_graphml(str(graphml_path))

    edges = []
    for u, v, data in G.edges(data=True):
        src_data = G.nodes[u]
        tgt_data = G.nodes[v]

        src_name = str(
            src_data.get("entity_name") or src_data.get("name") or u
        ).strip()
        tgt_name = str(
            tgt_data.get("entity_name") or tgt_data.get("name") or v
        ).strip()

        edges.append({
            "source_node": src_name,
            "target_node": tgt_name,
            "keywords": str(data.get("keywords", "")),
            "description": str(data.get("description", "")),
        })

    return edges


def sample_edges(edges: list[dict], n: int, seed: int) -> list[dict]:
    """Randomly sample n edges with a fixed seed for reproducibility."""
    rng = random.Random(seed)
    pool = list(edges)
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_for_dataset(dataset_key: str) -> pd.DataFrame:
    """Load the source CSV for a given dataset key."""
    cfg = DATASETS[dataset_key]
    csv_path = cfg["csv"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    skiprows = cfg["csv_skiprows"]

    # Try UTF-8 first, fall back to GBK for Chinese files
    for encoding in ("utf-8", "utf-8-sig", "gbk", "big5hkscs"):
        try:
            df = pd.read_csv(str(csv_path), skiprows=skiprows, encoding=encoding)
            return df
        except (UnicodeDecodeError, Exception):
            continue

    raise RuntimeError(f"Failed to read CSV with any encoding: {csv_path}")


def get_csv_context(df: pd.DataFrame, max_rows: int = 5) -> tuple[list[str], str]:
    """Return (column_names, rows_text) for the LLM prompt."""
    columns = list(df.columns)
    sample = df.head(max_rows)
    rows_text = sample.to_string(index=False, max_cols=20)
    return columns, rows_text


# ---------------------------------------------------------------------------
# LLM annotation
# ---------------------------------------------------------------------------

ANNOTATION_SYSTEM_PROMPT = (
    "You are verifying whether a knowledge graph edge is factually correct "
    "based on the source CSV data."
)

ANNOTATION_USER_TEMPLATE = """\
Source CSV columns: {columns}
Relevant CSV rows (first 5 rows):
{rows}

Edge to verify:
- Source entity: {source}
- Target entity: {target}
- Relation: {keywords}
- Description: {description}

Is this edge factually correct based on the CSV data? Answer with:
- "correct" if the relationship accurately reflects data in the CSV
- "incorrect" if the relationship contradicts or misrepresents the CSV data
- "unverifiable" if the CSV doesn't contain enough information to verify

Respond with ONLY a JSON object: {{"verdict": "correct/incorrect/unverifiable", "reason": "brief explanation"}}"""


async def call_llm_async(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    attempt: int = 0,
) -> dict:
    """Call LLM and parse JSON response. Returns dict with verdict and reason."""
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_verdict(raw)
    except Exception as exc:
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_DELAY_BASE * (attempt + 1)
            print(f"    [RETRY {attempt + 1}/{MAX_RETRIES}] {model}: {exc} — waiting {wait}s")
            await asyncio.sleep(wait)
            return await call_llm_async(client, model, system_prompt, user_prompt, attempt + 1)
        print(f"    [ERROR] {model} failed after {MAX_RETRIES} attempts: {exc}")
        return {"verdict": "unverifiable", "reason": f"API error: {exc}"}


def _parse_verdict(raw: str) -> dict:
    """Parse the LLM JSON response into a dict with verdict and reason."""
    # Try direct JSON parse
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        verdict = str(obj.get("verdict", "")).lower().strip()
        if verdict not in ("correct", "incorrect", "unverifiable"):
            verdict = "unverifiable"
        return {"verdict": verdict, "reason": str(obj.get("reason", ""))}
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON block from markdown code fence
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).lower().strip()
            if verdict not in ("correct", "incorrect", "unverifiable"):
                verdict = "unverifiable"
            return {"verdict": verdict, "reason": str(obj.get("reason", ""))}
        except json.JSONDecodeError:
            pass

    # Fallback: scan for verdict keyword in plain text
    lower = raw.lower()
    if "incorrect" in lower:
        return {"verdict": "incorrect", "reason": raw[:200]}
    if "correct" in lower:
        return {"verdict": "correct", "reason": raw[:200]}
    return {"verdict": "unverifiable", "reason": raw[:200]}


async def annotate_edge(
    client: AsyncOpenAI,
    model: str,
    edge: dict,
    columns: list[str],
    rows_text: str,
) -> dict:
    """Annotate a single edge and return the verdict dict."""
    user_prompt = ANNOTATION_USER_TEMPLATE.format(
        columns=", ".join(str(c) for c in columns),
        rows=rows_text,
        source=edge["source_node"],
        target=edge["target_node"],
        keywords=edge["keywords"],
        description=edge["description"],
    )
    result = await call_llm_async(
        client, model, ANNOTATION_SYSTEM_PROMPT, user_prompt
    )
    return result


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two annotators with identical label sets."""
    assert len(labels_a) == len(labels_b), "Annotator label lists must have equal length."
    n = len(labels_a)
    if n == 0:
        return 0.0

    all_labels = sorted({"correct", "incorrect", "unverifiable"})

    # Observed agreement
    observed_agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement
    expected_agree = 0.0
    for label in all_labels:
        p_a = sum(1 for a in labels_a if a == label) / n
        p_b = sum(1 for b in labels_b if b == label) / n
        expected_agree += p_a * p_b

    if expected_agree >= 1.0:
        return 1.0

    kappa = (observed_agree - expected_agree) / (1.0 - expected_agree)
    return round(kappa, 4)


# ---------------------------------------------------------------------------
# Precision calculation
# ---------------------------------------------------------------------------

def compute_precision(verdicts: list[str]) -> Optional[float]:
    """Compute precision: correct / (correct + incorrect). Returns None if denominator is 0."""
    n_correct = sum(1 for v in verdicts if v == "correct")
    n_incorrect = sum(1 for v in verdicts if v == "incorrect")
    denom = n_correct + n_incorrect
    if denom == 0:
        return None
    return round(n_correct / denom, 4)


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def _find_relevant_rows(df: pd.DataFrame, edge: dict, max_rows: int = 10) -> str:
    """Search the CSV for rows relevant to this edge's source/target entities."""
    source = str(edge["source_node"]).lower()
    target = str(edge["target_node"]).lower()
    matched = []
    for _, row in df.iterrows():
        row_str = " ".join(str(v).lower() for v in row.values)
        if source in row_str or target in row_str:
            matched.append(row)
            if len(matched) >= max_rows:
                break
    if not matched:
        # Fallback: return first 5 rows
        return df.head(5).to_string(index=False)
    import pandas as _pd
    return _pd.DataFrame(matched).to_string(index=False)


async def annotate_graph(
    client: AsyncOpenAI,
    dataset_key: str,
    graph_label: str,
    graphml_path: Path,
    df: pd.DataFrame,
    columns: list[str],
    rows_text: str,
) -> list[dict]:
    """
    Sample 50 edges from the graph and annotate each with both models.
    Returns list of annotation records.
    """
    print(f"\n  Loading graph: {graphml_path.name}")
    edges = load_graph_edges(graphml_path)
    print(f"  Total edges: {len(edges)}")

    sampled = sample_edges(edges, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  Sampled {len(sampled)} edges (seed={RANDOM_SEED})")

    # Process edges in batches of 10 for speed
    BATCH_SIZE = 10

    async def _annotate_one(i: int, edge: dict) -> dict:
        # Per-edge relevant CSV context instead of static first-5 rows
        edge_rows_text = _find_relevant_rows(df, edge)
        result_a, result_b = await asyncio.gather(
            annotate_edge(client, MODEL_A, edge, columns, edge_rows_text),
            annotate_edge(client, MODEL_B, edge, columns, edge_rows_text),
        )
        return {
            "dataset": dataset_key,
            "graph": graph_label,
            "edge_index": i,
            "source_node": edge["source_node"],
            "target_node": edge["target_node"],
            "keywords": edge["keywords"],
            "description": edge["description"],
            "annotator_a": {
                "model": MODEL_A,
                "verdict": result_a["verdict"],
                "reason": result_a["reason"],
            },
            "annotator_b": {
                "model": MODEL_B,
                "verdict": result_b["verdict"],
                "reason": result_b["reason"],
            },
        }

    records = []
    for batch_start in range(0, len(sampled), BATCH_SIZE):
        batch = sampled[batch_start:batch_start + BATCH_SIZE]
        print(f"    Batch {batch_start // BATCH_SIZE + 1}/{(len(sampled) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} edges)")
        batch_results = await asyncio.gather(
            *[_annotate_one(batch_start + j, edge) for j, edge in enumerate(batch)]
        )
        records.extend(batch_results)

    # Legacy variable bindings removed — records already built above
    if False:  # dead code block for diff compatibility
        record = {
            "dataset": dataset_key,
            "graph": graph_label,
            "edge_index": 0,
            "source_node": "",
            "target_node": "",
            "keywords": "",
            "description": "",
            "annotator_a": {
                "model": MODEL_A,
                "verdict": result_b["verdict"],
                "reason": result_b["reason"],
            },
        }
        records.append(record)

    return records


async def run_annotation_pipeline() -> dict:
    """Run the full annotation pipeline across all datasets and graphs."""
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_records: list[dict] = []

    for dataset_key, cfg in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {cfg['label']}")
        print(f"{'=' * 60}")

        # Load CSV once per dataset (shared between SGE and Baseline)
        print(f"  Loading CSV: {cfg['csv'].name}")
        try:
            df = load_csv_for_dataset(dataset_key)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        columns, rows_text = get_csv_context(df)
        print(f"  CSV columns: {len(columns)}, sample rows loaded")

        for graph_label, graphml_path in [
            ("sge", cfg["sge_graph"]),
            ("baseline", cfg["baseline_graph"]),
        ]:
            print(f"\n  --- Annotating {graph_label.upper()} graph ---")
            try:
                records = await annotate_graph(
                    client,
                    dataset_key,
                    graph_label,
                    graphml_path,
                    df,
                    columns,
                    rows_text,
                )
                all_records.extend(records)
            except FileNotFoundError as exc:
                print(f"  [SKIP] {exc}")

    return _compute_results(all_records)


def _compute_results(records: list[dict]) -> dict:
    """Compute precision, kappa, and per-dataset breakdowns from annotation records."""
    print("\n\nComputing results...")

    # Group records by dataset + graph
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        key = (rec["dataset"], rec["graph"])
        groups.setdefault(key, []).append(rec)

    # Per-dataset-graph stats
    per_dataset: dict[str, dict] = {}
    for (dataset_key, graph_label), recs in groups.items():
        verdicts_a = [r["annotator_a"]["verdict"] for r in recs]
        verdicts_b = [r["annotator_b"]["verdict"] for r in recs]

        prec_a = compute_precision(verdicts_a)
        prec_b = compute_precision(verdicts_b)
        kappa = cohen_kappa(verdicts_a, verdicts_b)

        stats = {
            "n_edges": len(recs),
            "annotator_a": {
                "model": MODEL_A,
                "precision": prec_a,
                "correct": verdicts_a.count("correct"),
                "incorrect": verdicts_a.count("incorrect"),
                "unverifiable": verdicts_a.count("unverifiable"),
            },
            "annotator_b": {
                "model": MODEL_B,
                "precision": prec_b,
                "correct": verdicts_b.count("correct"),
                "incorrect": verdicts_b.count("incorrect"),
                "unverifiable": verdicts_b.count("unverifiable"),
            },
            "cohens_kappa": kappa,
        }

        if dataset_key not in per_dataset:
            per_dataset[dataset_key] = {
                "label": DATASETS[dataset_key]["label"],
                "sge": {},
                "baseline": {},
            }
        per_dataset[dataset_key][graph_label] = stats

    # Overall SGE vs Baseline precision
    sge_records_a = [r["annotator_a"]["verdict"] for r in records if r["graph"] == "sge"]
    sge_records_b = [r["annotator_b"]["verdict"] for r in records if r["graph"] == "sge"]
    base_records_a = [r["annotator_a"]["verdict"] for r in records if r["graph"] == "baseline"]
    base_records_b = [r["annotator_b"]["verdict"] for r in records if r["graph"] == "baseline"]

    overall = {
        "sge": {
            "annotator_a_precision": compute_precision(sge_records_a),
            "annotator_b_precision": compute_precision(sge_records_b),
            "cohens_kappa": cohen_kappa(sge_records_a, sge_records_b),
            "n_edges": len(sge_records_a),
        },
        "baseline": {
            "annotator_a_precision": compute_precision(base_records_a),
            "annotator_b_precision": compute_precision(base_records_b),
            "cohens_kappa": cohen_kappa(base_records_a, base_records_b),
            "n_edges": len(base_records_a),
        },
    }

    # Print summary
    print("\n--- SUMMARY ---")
    for dataset_key, ds_stats in per_dataset.items():
        print(f"\n{ds_stats['label']}:")
        for graph_label in ("sge", "baseline"):
            gs = ds_stats.get(graph_label, {})
            if not gs:
                continue
            pa = gs["annotator_a"].get("precision")
            pb = gs["annotator_b"].get("precision")
            kappa = gs.get("cohens_kappa")
            pa_s = f"{pa:.3f}" if pa is not None else "N/A"
            pb_s = f"{pb:.3f}" if pb is not None else "N/A"
            k_s = f"{kappa:.3f}" if kappa is not None else "N/A"
            print(f"  {graph_label.upper():8s}: A={pa_s}, B={pb_s}, kappa={k_s}")

    print("\nOverall:")
    for graph_label, gs in overall.items():
        pa = gs["annotator_a_precision"]
        pb = gs["annotator_b_precision"]
        kappa = gs["cohens_kappa"]
        pa_s = f"{pa:.3f}" if pa is not None else "N/A"
        pb_s = f"{pb:.3f}" if pb is not None else "N/A"
        k_s = f"{kappa:.3f}" if kappa is not None else "N/A"
        print(f"  {graph_label.upper():8s}: A={pa_s}, B={pb_s}, kappa={k_s}, n={gs['n_edges']}")

    return {
        "config": {
            "sample_size_per_graph": SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "datasets": list(DATASETS.keys()),
        },
        "per_dataset": per_dataset,
        "overall": overall,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "independent_annotation_results.json"

    print("SGE-LightRAG Independent Precision Annotation")
    print(f"Models: A={MODEL_A}, B={MODEL_B}")
    print(f"Sample size: {SAMPLE_SIZE} edges per graph, seed={RANDOM_SEED}")
    print(f"Output: {output_path}")

    results = asyncio.run(run_annotation_pipeline())

    with open(str(output_path), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
