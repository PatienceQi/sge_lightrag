#!/usr/bin/env python3
"""
hybrid_pipeline.py — Det Parser First, SGE Fallback pipeline.

Steps:
  1. Stage 1 classifier → topology type
  2. Deterministic parser → GraphML (zero LLM calls)
  3. Quality signal on Det Parser output (no gold needed)
  4. "accept" → use Det Parser graph (0 LLM calls)
     "fallback" → run full SGE pipeline (LLM calls used)

Usage (as module):
    from evaluation.hybrid_pipeline import run_hybrid
    result = run_hybrid("who", csv_path, output_dir, gold_path=..., force_sge=True)

Usage (CLI):
    python3 evaluation/hybrid_pipeline.py --dataset who \\
        --csv dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv \\
        --output-dir output/hybrid_who [--force-sge]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify
from evaluation.det_parser_quality import compute_quality_signal

_DET_PARSER_SCRIPT = _REPO_ROOT / "evaluation" / "deterministic_parser_baseline.py"
_EVAL_SCRIPT       = _REPO_ROOT / "evaluation" / "evaluate_coverage.py"
_RUN_PIPELINE      = _REPO_ROOT / "run_pipeline.py"


def _run_stage1(csv_path: str) -> dict:
    """Run Stage 1 classifier; return table_type and type_short."""
    features = extract_features(csv_path)
    table_type = classify(features)
    type_map = {
        "Time-Series-Matrix": "type-ii",
        "Hierarchical-Hybrid": "type-iii",
        "Flat-Entity": "type-i",
    }
    return {"table_type": table_type, "type_short": type_map.get(table_type, "type-ii")}


def _run_subprocess(cmd: list, timeout: int, error_prefix: str) -> dict:
    """Run a subprocess and return a result dict with success/elapsed/error."""
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        if r.returncode != 0:
            return {"success": False, "elapsed_s": elapsed,
                    "error": r.stderr[-400:] or f"{error_prefix}: non-zero exit"}
        return {"success": True, "elapsed_s": elapsed, "error": None}
    except subprocess.TimeoutExpired:
        return {"success": False, "elapsed_s": round(time.time() - t0, 2),
                "error": f"{error_prefix}: timed out ({timeout}s)"}
    except Exception as exc:
        return {"success": False, "elapsed_s": round(time.time() - t0, 2),
                "error": str(exc)}


def _run_det_parser(csv_path: str, output_graphml: str, table_type: str) -> dict:
    """Run deterministic_parser_baseline.py → GraphML."""
    cmd = [sys.executable, str(_DET_PARSER_SCRIPT),
           "--csv", csv_path, "--output", output_graphml, "--type", table_type]
    r = _run_subprocess(cmd, timeout=120, error_prefix="Det parser")
    return {**r, "graphml_path": output_graphml}


def _run_sge_pipeline(csv_path: str, output_dir: str) -> dict:
    """
    Run run_pipeline.py (Stage 1→2→3) in auto mode.

    Produces LightRAG-ready chunks and prompts but does NOT run LightRAG
    insertion — that requires an async LightRAG instance and the generated
    assets in output_dir/chunks/ and output_dir/prompts/.
    """
    cmd = [sys.executable, str(_RUN_PIPELINE), csv_path,
           "--output-dir", output_dir, "--stage2-mode", "auto"]
    r = _run_subprocess(cmd, timeout=600, error_prefix="SGE pipeline")
    report_path = str(Path(output_dir) / "pipeline_report.json")
    return {**r, "output_dir": output_dir,
            "report_path": report_path if Path(report_path).exists() else None}


def _run_fc_eval(graphml_path: str, gold_path: str) -> Optional[dict]:
    """Run evaluate_coverage.py and parse the JSON result, or return None."""
    if not Path(graphml_path).exists() or not Path(gold_path).exists():
        return None
    r = subprocess.run(
        [sys.executable, str(_EVAL_SCRIPT), "--graph", graphml_path, "--gold", gold_path],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return {"error": r.stderr[-200:]}
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_hybrid(
    dataset_name: str,
    csv_path: str,
    output_dir: str,
    gold_path: Optional[str] = None,
    force_sge: bool = False,
    force_sge_reason: Optional[str] = None,
    force_det_parser: bool = False,
    force_det_reason: Optional[str] = None,
) -> dict:
    """
    Run the Det Parser First → SGE Fallback hybrid pipeline.

    Parameters
    ----------
    dataset_name     : short identifier (e.g. "who", "wb_cm")
    csv_path         : path to source CSV
    output_dir       : directory to write all pipeline outputs
    gold_path        : optional gold JSONL for FC evaluation
    force_sge        : override quality signal to always use SGE fallback
    force_sge_reason : human-readable reason for the force override

    Returns dict with: dataset, method_used, stage1_type, quality_signal,
    det_parser, sge_pipeline, fc_result, llm_calls, total_elapsed_s,
    output_graphml, force_sge, force_sge_reason, error.
    """
    t_total = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "dataset": dataset_name, "csv_path": csv_path, "method_used": "error",
        "stage1_type": None, "quality_signal": None, "det_parser": None,
        "sge_pipeline": None, "fc_result": None, "llm_calls": 0,
        "total_elapsed_s": 0.0, "output_graphml": None,
        "force_sge": force_sge, "force_sge_reason": force_sge_reason,
        "force_det_parser": force_det_parser, "force_det_reason": force_det_reason,
        "error": None,
    }

    # Step 1: Stage 1 Classification
    try:
        stage1 = _run_stage1(csv_path)
    except Exception as exc:
        result["error"] = f"Stage 1 failed: {exc}"
        result["total_elapsed_s"] = round(time.time() - t_total, 2)
        return result

    result["stage1_type"] = stage1["table_type"]
    print(f"[{dataset_name}] Stage 1: {stage1['table_type']}")

    # Step 2: Run Det Parser
    det_graphml_path = str(out_dir / f"{dataset_name}_det_parser.graphml")
    det_result = _run_det_parser(csv_path, det_graphml_path, stage1["type_short"])
    result["det_parser"] = det_result

    if not det_result["success"]:
        print(f"[{dataset_name}] Det Parser FAILED: {det_result['error']}")
        recommendation = "fallback"
        result["quality_signal"] = {"recommendation": "fallback", "error": det_result["error"]}
    else:
        print(f"[{dataset_name}] Det Parser done in {det_result['elapsed_s']}s")
        # Step 3: Compute Quality Signal
        signal = compute_quality_signal(det_graphml_path, csv_path)
        result["quality_signal"] = signal
        recommendation = signal["recommendation"]
        print(
            f"[{dataset_name}] Quality: entity_cov={signal['entity_coverage']}, "
            f"val_complete={signal['value_completeness']}, "
            f"e/n={signal['edge_node_ratio']} → {recommendation.upper()}"
        )

    # force_det_parser overrides quality signal to always accept Det Parser
    if force_det_parser and recommendation == "fallback":
        recommendation = "accept"
        override_msg = "force_det_parser=True"
        if force_det_reason:
            override_msg += f" ({force_det_reason})"
        print(f"[{dataset_name}] Quality overridden: {override_msg}")

    # force_sge overrides quality signal to always use SGE fallback
    if force_sge and recommendation == "accept":
        recommendation = "fallback"
        override_msg = "force_sge=True"
        if force_sge_reason:
            override_msg += f" ({force_sge_reason})"
        print(f"[{dataset_name}] Quality overridden: {override_msg}")

    # Step 4: Accept or Fallback
    if recommendation == "accept":
        result["method_used"] = "det_parser"
        result["llm_calls"] = 0
        result["output_graphml"] = det_graphml_path
        print(f"[{dataset_name}] Decision: ACCEPT Det Parser (0 LLM calls)")
    else:
        print(f"[{dataset_name}] Decision: FALLBACK to SGE pipeline")
        sge_out_dir = str(out_dir / f"{dataset_name}_sge")
        sge_result = _run_sge_pipeline(csv_path, sge_out_dir)
        result["sge_pipeline"] = sge_result
        result["method_used"] = "sge" if sge_result["success"] else "error"
        result["llm_calls"] = -1  # -1 = varies by dataset

        sge_graphml = (
            Path(sge_out_dir) / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
        )
        result["output_graphml"] = str(sge_graphml) if sge_graphml.exists() else None

        if not sge_result["success"]:
            result["error"] = f"SGE pipeline failed: {sge_result['error']}"
            print(f"[{dataset_name}] SGE FAILED: {sge_result['error']}")
        else:
            print(f"[{dataset_name}] SGE pipeline ready in {sge_result['elapsed_s']}s")

    # Optional FC Evaluation (only if gold_path provided and graphml exists)
    if gold_path and result["output_graphml"]:
        try:
            fc = _run_fc_eval(result["output_graphml"], gold_path)
            result["fc_result"] = fc
            if fc and "fact_coverage" in fc:
                print(f"[{dataset_name}] FC = {fc['fact_coverage'].get('coverage', '?')}")
        except Exception as exc:
            result["fc_result"] = {"error": str(exc)}

    result["total_elapsed_s"] = round(time.time() - t_total, 2)

    # Persist per-dataset result
    result_path = out_dir / f"{dataset_name}_hybrid_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _cli() -> None:
    """CLI: run hybrid pipeline for a single dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Det Parser First → SGE Fallback pipeline")
    parser.add_argument("--dataset",    required=True, help="Dataset short name")
    parser.add_argument("--csv",        required=True, help="Path to CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--gold", default=None, help="Optional gold JSONL for FC evaluation")
    parser.add_argument("--force-sge", action="store_true",
                        help="Force SGE fallback regardless of quality signal")
    args = parser.parse_args()

    result = run_hybrid(
        dataset_name=args.dataset, csv_path=args.csv,
        output_dir=args.output_dir, gold_path=args.gold, force_sge=args.force_sge,
    )
    print("\n" + "=" * 60)
    print(f"HYBRID RESULT: {result['dataset']}")
    print(f"  method_used  : {result['method_used']}")
    print(f"  stage1_type  : {result['stage1_type']}")
    print(f"  llm_calls    : {result['llm_calls']}")
    print(f"  elapsed      : {result['total_elapsed_s']}s")
    if result.get("fc_result") and "fact_coverage" in result.get("fc_result", {}):
        print(f"  FC           : {result['fc_result']['fact_coverage']['coverage']}")
    if result.get("error"):
        print(f"  ERROR        : {result['error']}")
    print("=" * 60)


if __name__ == "__main__":
    _cli()
