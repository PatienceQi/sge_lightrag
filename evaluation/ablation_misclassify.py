#!/usr/bin/env python3
"""
ablation_misclassify.py — Ablation study: effect of deliberate Stage 1 misclassification.

PURPOSE
-------
Isolates the contribution of correct topology classification (Stage 1) by
forcing a dataset to be treated as the wrong type, then measuring how much
FC (Fact Coverage) degrades.

EXPERIMENT DESIGN
-----------------
For each (dataset, forced-type) combination:

  1. Load the CSV.
  2. Run Stage 1 normally → real τ_real and Meta-Schema S_real.
  3. Override τ to --force-type and rebuild S with the wrong type → S_wrong.
  4. Run Stage 2 on (features, τ_wrong, S_wrong) → Σ_wrong (extraction schema).
  5. Run Stage 3 serialization with Σ_wrong → text chunks (wrong-type format).
  6. [--dry-run skips] Run LightRAG ainsert() with wrong-type chunks and
     the wrong-type PROMPTS override.
  7. Evaluate FC from the resulting graph against the gold standard.
  8. Compare: real-type FC (from all_results_v2.json) vs wrong-type FC → delta.

OUTPUT
------
A comparison table:

  Dataset    | Real τ | Forced τ | FC (real) | FC (wrong) | ΔFC
  -----------+--------+----------+-----------+------------+------
  who        | II     | III      | 1.000     | ???        | ???

WARNING
-------
  Running WITHOUT --dry-run will call the Claude API (claude-haiku-4-5-20251001)
  and rebuild LightRAG knowledge graphs. This costs real money (~$0.50–$2.00 per
  run) and takes 10–30 minutes. Always test with --dry-run first.

USAGE EXAMPLES
--------------
  # Dry run: inspect chunks without calling the API
  python3 evaluation/ablation_misclassify.py --dataset who --force-type III --dry-run

  # Forced misclassification for WB Child Mortality
  python3 evaluation/ablation_misclassify.py --dataset wb_cm --force-type I --dry-run

  # Full run (calls API, rebuilds graph)
  python3 evaluation/ablation_misclassify.py --dataset who --force-type III

  # Show all chunks and schema in dry-run mode
  python3 evaluation/ablation_misclassify.py --dataset inpatient --force-type II --dry-run --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root setup — allows running from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical type strings (match stage1/classifier.py)
TYPE_I   = "Flat-Entity"
TYPE_II  = "Time-Series-Matrix"
TYPE_III = "Hierarchical-Hybrid"

TYPE_LABEL_MAP = {
    "I":   TYPE_I,
    "II":  TYPE_II,
    "III": TYPE_III,
}

# Dataset registry: name → CSV path, gold JSONL, and the real-type FC from
# evaluation/all_results_v2.json (kept in sync with the authoritative file).
# Paths are relative to PROJECT_ROOT.
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "who": {
        "csv": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold_who_life_expectancy_v2.jsonl",
        "real_type": TYPE_II,
        "real_fc_key": "WHO",   # key in all_results_v2.json
        "output_dir": "output/ablation_who_{forced_type}",
    },
    "wb_cm": {
        "csv": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold_wb_child_mortality_v2.jsonl",
        "real_type": TYPE_II,
        "real_fc_key": "WB_CM",
        "output_dir": "output/ablation_wb_cm_{forced_type}",
    },
    "inpatient": {
        "csv": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and "
               "Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold_inpatient_2023.jsonl",
        "real_type": TYPE_III,
        "real_fc_key": "HK_Inpatient",
        "output_dir": "output/ablation_inpatient_{forced_type}",
    },
}

# LightRAG model configuration (matches main pipeline)
LLM_MODEL   = "claude-haiku-4-5-20251001"
EMBED_MODEL = "mxbai-embed-large-v1"
LLM_MAX_ASYNC = 5


# ---------------------------------------------------------------------------
# Stage 1: run normally, then override the type
# ---------------------------------------------------------------------------

def run_stage1_with_override(
    csv_path: str,
    forced_type: str,
) -> tuple[Any, str, dict, str, dict]:
    """
    Run Stage 1 normally, then substitute τ with forced_type.

    Returns
    -------
    (features, real_type, real_meta_schema, forced_type, forced_meta_schema)

    The forced_meta_schema is rebuilt from the same FeatureSet but with the
    wrong table_type, so all downstream stages receive the wrong structural
    interpretation.
    """
    from stage1.features import extract_features
    from stage1.classifier import classify
    from stage1.schema import build_meta_schema

    features = extract_features(csv_path)
    real_type = classify(features)
    real_meta_schema = build_meta_schema(features, real_type)

    # Rebuild Meta-Schema with the wrong type — this changes time_dimension,
    # composite_key presentation, and extraction_rule_summary.
    forced_meta_schema = build_meta_schema(features, forced_type)

    return features, real_type, real_meta_schema, forced_type, forced_meta_schema


# ---------------------------------------------------------------------------
# Stage 2: induce schema from the (wrong) meta-schema
# ---------------------------------------------------------------------------

def run_stage2_with_wrong_type(
    features: Any,
    forced_type: str,
    forced_meta_schema: dict,
) -> dict:
    """
    Run Stage 2 schema induction using the forced (wrong) type.

    Delegates to stage2.inductor.induce_schema_from_meta, which dispatches
    to the type-specific handler for forced_type — so a Type-II dataset
    gets the Type-III handler logic, etc.
    """
    from stage2.inductor import induce_schema_from_meta

    schema = induce_schema_from_meta(features, forced_type, forced_meta_schema)
    # Propagate row count so compact-mode check works correctly in Stage 3.
    schema["_n_rows"] = features.n_rows
    return schema


# ---------------------------------------------------------------------------
# Stage 3: serialise with the wrong schema
# ---------------------------------------------------------------------------

def run_stage3_serialization(
    csv_path: str,
    wrong_schema: dict,
) -> list[str]:
    """
    Serialize the CSV using the wrong-type extraction schema.

    Returns text chunks as they would be fed to LightRAG ainsert().
    """
    from stage3.integrator import prepare_chunks

    return prepare_chunks(csv_path, wrong_schema)


def build_lightrag_payload(wrong_schema: dict) -> dict:
    """
    Build the PROMPTS override payload for the wrong-type schema.

    Returns the dict from stage3.integrator.patch_lightrag().
    """
    from stage3.integrator import patch_lightrag

    return patch_lightrag(wrong_schema, language="English")


# ---------------------------------------------------------------------------
# LightRAG insertion (only run when not --dry-run)
# ---------------------------------------------------------------------------

async def _ainsert_chunks(
    working_dir: str,
    chunks: list[str],
    payload: dict,
) -> None:
    """
    Asynchronously insert chunks into a fresh LightRAG instance.

    WARNING: Makes real API calls to claude-haiku-4-5-20251001.
    """
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.anthropic import anthropic_complete_if_cache
        from lightrag.llm.ollama import ollama_embed
        from lightrag.utils import EmbeddingFunc
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG is not importable. Cannot run without --dry-run. "
            f"Import error: {exc}"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running without --dry-run."
        )

    system_prompt = payload.get("system_prompt")
    addon_params  = payload.get("addon_params", {})

    async def llm_func(prompt, system_prompt=system_prompt, **kwargs):
        return await anthropic_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            **kwargs,
        )

    async def embed_func(texts):
        return await ollama_embed(
            texts,
            embed_model=EMBED_MODEL,
        )

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embed_func,
        ),
        llm_model_max_async=LLM_MAX_ASYNC,
        addon_params=addon_params,
    )

    # Override system prompt if not using baseline mode
    if system_prompt and not payload.get("use_baseline_mode", False):
        try:
            from lightrag.prompt import PROMPTS
            PROMPTS["entity_extraction_system_prompt"] = system_prompt
        except ImportError:
            pass  # LightRAG version without PROMPTS dict — skip override

    await rag.ainsert(chunks)


# ---------------------------------------------------------------------------
# Evaluation: compute FC against gold standard
# ---------------------------------------------------------------------------

def evaluate_fc(graph_path: str, gold_path: str) -> dict:
    """
    Compute EC and FC from a graphml file against a gold JSONL.

    Imports evaluate_coverage directly to avoid subprocess overhead.
    Returns a dict with entity_coverage and fact_coverage sub-dicts.
    """
    from evaluation.evaluate_coverage import load_gold, load_graph
    from evaluation.evaluate_coverage import check_entity_coverage, check_fact_coverage

    gold_entities, facts = load_gold(gold_path)
    _G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, not_covered = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    return {
        "entity_coverage": {
            "matched": len(matched_entities),
            "total": len(gold_entities),
            "coverage": round(ec, 4),
        },
        "fact_coverage": {
            "covered": len(covered),
            "total": len(facts),
            "coverage": round(fc, 4),
        },
    }


# ---------------------------------------------------------------------------
# Load real-type FC from authoritative results file
# ---------------------------------------------------------------------------

def load_real_fc(dataset_key: str, results_file: Path) -> float | None:
    """
    Read the authoritative FC for the given dataset from all_results_v2.json.

    Returns None if the file does not exist or the key is missing.
    """
    if not results_file.exists():
        return None

    with open(results_file, encoding="utf-8") as fh:
        results = json.load(fh)

    real_fc_key = DATASET_REGISTRY[dataset_key]["real_fc_key"]

    # Navigate the nested structure: results[short_name]["sge"]["fact_coverage"]
    entry = results.get(real_fc_key) or results.get(dataset_key)
    if not entry:
        return None

    sge_entry = entry.get("sge") or entry.get("SGE") or entry
    fc_entry = sge_entry.get("fact_coverage")
    if isinstance(fc_entry, dict):
        return fc_entry.get("coverage")
    if isinstance(fc_entry, (int, float)):
        return float(fc_entry)
    return None


# ---------------------------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------------------------

def print_dry_run_report(
    dataset: str,
    csv_path: str,
    real_type: str,
    forced_type: str,
    real_meta_schema: dict,
    forced_meta_schema: dict,
    wrong_schema: dict,
    chunks: list[str],
    payload: dict,
    verbose: bool,
) -> None:
    """Print a detailed dry-run report without making any API calls."""
    sep = "=" * 70

    print(sep)
    print("ABLATION DRY-RUN REPORT")
    print(sep)
    print(f"  Dataset    : {dataset}")
    print(f"  CSV        : {csv_path}")
    print(f"  Real type  : {real_type}")
    print(f"  Forced type: {forced_type}")
    print()

    print("[Stage 1 — Real Meta-Schema (excerpt)]")
    print(f"  table_type           : {real_meta_schema.get('table_type')}")
    print(f"  extraction_rule_summary: {real_meta_schema.get('extraction_rule_summary','')[:80]}")
    print(f"  time_dimension       : {real_meta_schema.get('time_dimension')}")
    print()

    print("[Stage 1 — Forced Meta-Schema (excerpt)]")
    print(f"  table_type           : {forced_meta_schema.get('table_type')}")
    print(f"  extraction_rule_summary: {forced_meta_schema.get('extraction_rule_summary','')[:80]}")
    print(f"  time_dimension       : {forced_meta_schema.get('time_dimension')}")
    print()

    print("[Stage 2 — Wrong-Type Extraction Schema (excerpt)]")
    print(f"  table_type    : {wrong_schema.get('table_type')}")
    print(f"  entity_types  : {wrong_schema.get('entity_types')}")
    print(f"  relation_types: {wrong_schema.get('relation_types')}")
    bm = wrong_schema.get("use_baseline_mode", False)
    print(f"  baseline_mode : {bm}"
          + (f" (reason: {wrong_schema.get('adaptive_reason')})" if bm else ""))
    print()

    print(f"[Stage 3 — Serialization]")
    print(f"  Chunks produced: {len(chunks)}")
    print(f"  Baseline mode  : {payload.get('use_baseline_mode', False)}")
    print(f"  Compact mode   : {payload.get('use_compact_mode', False)}")
    print()

    if verbose and chunks:
        print("[Sample Chunks (first 3)]")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  --- Chunk {i+1} ---")
            # Indent each line
            for line in chunk.splitlines():
                print(f"    {line}")
        print()

    print("[NOTE] No API calls made. Re-run without --dry-run to build the graph.")
    print(sep)


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    dataset: str,
    real_type: str,
    forced_type: str,
    real_fc: float | None,
    wrong_fc: float,
    wrong_ec: float,
) -> None:
    """Print the final comparison table of real-type vs wrong-type FC."""
    real_fc_str = f"{real_fc:.4f}" if real_fc is not None else "N/A (no results file)"
    wrong_fc_str = f"{wrong_fc:.4f}"

    if real_fc is not None:
        delta = wrong_fc - real_fc
        delta_str = f"{delta:+.4f}"
    else:
        delta_str = "N/A"

    sep = "=" * 70
    print()
    print(sep)
    print("ABLATION RESULT: MISCLASSIFICATION IMPACT ON FC")
    print(sep)
    header = f"{'Dataset':<12} {'Real τ':<18} {'Forced τ':<18} {'FC (real)':<12} {'FC (wrong)':<12} {'ΔFC':<8}"
    print(header)
    print("-" * 70)
    row = (
        f"{dataset:<12} "
        f"{real_type:<18} "
        f"{forced_type:<18} "
        f"{real_fc_str:<12} "
        f"{wrong_fc_str:<12} "
        f"{delta_str:<8}"
    )
    print(row)
    print(sep)

    if real_fc is not None and wrong_fc < real_fc:
        print(f"\nInterpretation: Correct classification IMPROVES FC by {real_fc - wrong_fc:.4f} "
              f"({(real_fc - wrong_fc) * 100:.1f} percentage points).")
    elif real_fc is not None and wrong_fc >= real_fc:
        print(f"\nInterpretation: Wrong classification does NOT degrade FC for this dataset "
              f"(ΔFC = {delta_str}). Consider whether the forced type is actually equivalent.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study: force Stage 1 to produce the wrong type and measure FC impact.\n\n"
            "WARNING: running without --dry-run makes real API calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to run the ablation on.",
    )
    parser.add_argument(
        "--force-type",
        required=True,
        choices=["I", "II", "III"],
        dest="force_type",
        help="Topology type to force (must differ from the real type for a meaningful ablation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Print the chunks and schema without calling the API. "
            "No LightRAG graph will be built and no evaluation can be done."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print sample chunks and full schema details in dry-run mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON results to this file path (only used without --dry-run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_key  = args.dataset
    forced_label = args.force_type
    forced_type  = TYPE_LABEL_MAP[forced_label]
    dry_run      = args.dry_run

    registry    = DATASET_REGISTRY[dataset_key]
    csv_path    = str(PROJECT_ROOT / registry["csv"])
    gold_path   = str(PROJECT_ROOT / registry["gold"])
    results_file = PROJECT_ROOT / "evaluation" / "all_results_v2.json"

    # Validate CSV exists
    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Warn if forcing the same type (no meaningful ablation)
    real_type_expected = registry["real_type"]
    if forced_type == real_type_expected:
        print(
            f"WARNING: --force-type {forced_label} is the SAME as the real type "
            f"({real_type_expected}). This is not a misclassification ablation — "
            "results will match the normal pipeline.",
            file=sys.stderr,
        )

    if not dry_run:
        print(
            "\n" + "!" * 70 + "\n"
            "  WARNING: --dry-run NOT set. This will call the Anthropic API\n"
            "  (claude-haiku-4-5-20251001) and rebuild a LightRAG graph.\n"
            "  Estimated cost: $0.50–$2.00  |  Estimated time: 10–30 min\n"
            "  Set ANTHROPIC_API_KEY before proceeding.\n"
            "  Press Ctrl+C within 5 seconds to abort.\n"
            + "!" * 70 + "\n",
            file=sys.stderr,
        )
        import time
        time.sleep(5)

    # -----------------------------------------------------------------
    # Stage 1: extract features + override type
    # -----------------------------------------------------------------
    print(f"[1/5] Running Stage 1 on '{dataset_key}' CSV …")
    features, real_type, real_meta_schema, forced_type_str, forced_meta_schema = \
        run_stage1_with_override(csv_path, forced_type)

    print(f"      Real type  : {real_type}")
    print(f"      Forced type: {forced_type_str}")

    # -----------------------------------------------------------------
    # Stage 2: induce schema with wrong type
    # -----------------------------------------------------------------
    print("[2/5] Running Stage 2 with wrong type …")
    wrong_schema = run_stage2_with_wrong_type(features, forced_type_str, forced_meta_schema)
    print(f"      entity_types  : {wrong_schema.get('entity_types')}")
    bm = wrong_schema.get("use_baseline_mode", False)
    print(f"      baseline_mode : {bm}")

    # -----------------------------------------------------------------
    # Stage 3: serialize with wrong schema
    # -----------------------------------------------------------------
    print("[3/5] Serializing CSV with wrong-type schema …")
    chunks = run_stage3_serialization(csv_path, wrong_schema)
    print(f"      Chunks produced: {len(chunks)}")

    payload = build_lightrag_payload(wrong_schema)
    print(f"      LightRAG payload: system_prompt={'set' if payload.get('system_prompt') else 'None (baseline)'}")

    # -----------------------------------------------------------------
    # Dry-run path: report and exit
    # -----------------------------------------------------------------
    if dry_run:
        print_dry_run_report(
            dataset=dataset_key,
            csv_path=csv_path,
            real_type=real_type,
            forced_type=forced_type_str,
            real_meta_schema=real_meta_schema,
            forced_meta_schema=forced_meta_schema,
            wrong_schema=wrong_schema,
            chunks=chunks,
            payload=payload,
            verbose=args.verbose,
        )
        return

    # -----------------------------------------------------------------
    # Full run: build LightRAG graph, then evaluate
    # -----------------------------------------------------------------
    working_dir = str(
        PROJECT_ROOT / registry["output_dir"].format(forced_type=forced_label)
    )
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    print(f"[4/5] Inserting {len(chunks)} chunks into LightRAG at: {working_dir}")
    print("      (this makes real API calls — Ctrl+C to abort)")

    asyncio.run(_ainsert_chunks(working_dir, chunks, payload))
    print("      LightRAG insertion complete.")

    # -----------------------------------------------------------------
    # Evaluate FC
    # -----------------------------------------------------------------
    graph_path = str(
        Path(working_dir) / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
    )

    if not Path(graph_path).exists():
        print(f"ERROR: Graph file not found after insertion: {graph_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[5/5] Evaluating FC against gold standard …")
    eval_results = evaluate_fc(graph_path, gold_path)
    wrong_fc = eval_results["fact_coverage"]["coverage"]
    wrong_ec = eval_results["entity_coverage"]["coverage"]
    print(f"      EC (wrong): {wrong_ec:.4f}")
    print(f"      FC (wrong): {wrong_fc:.4f}")

    # Load authoritative real-type FC
    real_fc = load_real_fc(dataset_key, results_file)

    print_comparison_table(
        dataset=dataset_key,
        real_type=real_type,
        forced_type=forced_type_str,
        real_fc=real_fc,
        wrong_fc=wrong_fc,
        wrong_ec=wrong_ec,
    )

    # -----------------------------------------------------------------
    # Save JSON output
    # -----------------------------------------------------------------
    output_data = {
        "dataset": dataset_key,
        "csv_path": csv_path,
        "real_type": real_type,
        "forced_type": forced_type_str,
        "working_dir": working_dir,
        "graph_path": graph_path,
        "evaluation": eval_results,
        "real_fc": real_fc,
        "delta_fc": (wrong_fc - real_fc) if real_fc is not None else None,
    }

    output_path = args.output or str(
        PROJECT_ROOT / f"evaluation/ablation_{dataset_key}_forced{forced_label}.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
