#!/usr/bin/env python3
"""
run_multi_csv_comparison.py — Wrapper to run SGE vs Baseline on multiple CSV types.

Handles:
- Type III CSV (food safety): standard UTF-8 comma-separated
- Type II-transposed CSV (health stats): UTF-16LE tab-separated, needs preprocessing

Does NOT modify run_lightrag_integration.py.
"""

from __future__ import annotations

import sys
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the integration module's async main and helpers
from run_lightrag_integration import run_sge_pipeline, run_lightrag, PROMPTS, _op
from run_lightrag_integration import _original_extract_entities, _sge_extract_entities
import lightrag.operate as _op_mod
from lightrag.prompt import PROMPTS as _PROMPTS


def preprocess_healthstat_csv(src: Path, dst: Path) -> None:
    """Convert UTF-16LE tab-separated healthstat CSV to UTF-8 comma-separated.
    
    The file is transposed (years as columns, metrics as rows).
    We convert it to a standard row-per-metric format suitable for SGE.
    """
    with open(src, encoding="utf-16-le") as f:
        content = f.read()

    # Strip BOM if present
    content = content.lstrip("\ufeff")

    lines = content.strip().split("\n")

    # Find the header row (contains year numbers like 2024, 2023...)
    header_row_idx = None
    years = []
    for i, line in enumerate(lines):
        cols = [c.strip() for c in line.split("\t")]
        year_cols = [c for c in cols if c.isdigit() and 2000 <= int(c) <= 2030]
        if len(year_cols) >= 3:
            header_row_idx = i
            years = year_cols
            break

    if header_row_idx is None:
        # Fallback: just write as-is with tab→comma conversion
        with open(dst, "w", encoding="utf-8", newline="") as f:
            for line in lines:
                f.write(line.replace("\t", ",") + "\n")
        return

    # Build normalized CSV: metric, year, value
    out_rows = [["指标", "年份", "数值"]]

    for line in lines[header_row_idx + 1:]:
        cols = [c.strip() for c in line.split("\t")]
        if not cols or not any(cols):
            continue

        # Get metric name (first non-empty cell, possibly merged from prev row)
        metric = ""
        sub_metric = ""
        for c in cols[:3]:
            if c and not c.startswith("注") and not c.isdigit():
                if not metric:
                    metric = c
                elif not sub_metric:
                    sub_metric = c

        if not metric:
            continue

        full_metric = f"{metric} {sub_metric}".strip() if sub_metric else metric

        # Skip footnote/source rows
        if any(kw in full_metric for kw in ["注释", "资料来源", "注："]):
            continue

        # Extract year values (starting from col index where years begin)
        # Find where numeric data starts
        data_start = 3  # typically col 3 onwards
        for j, year in enumerate(years):
            col_idx = data_start + j
            if col_idx < len(cols):
                val = cols[col_idx].strip()
                if val and val not in ("", "-", "N/A"):
                    out_rows.append([full_metric, year, val])

    import csv
    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    print(f"  [preprocess] Converted {len(out_rows)-1} data rows → {dst}")


async def run_one_csv(
    csv_path: Path,
    output_base: Path,
    sge_subdir: str,
    baseline_subdir: str,
    label: str,
) -> dict:
    """Run SGE + Baseline on a single CSV, return comparison stats."""
    sge_dir = output_base / sge_subdir
    baseline_dir = output_base / baseline_subdir
    sge_work = sge_dir / "lightrag_storage"
    baseline_work = baseline_dir / "lightrag_storage"

    result = {
        "csv": str(csv_path),
        "label": label,
        "error": None,
        "sge": {},
        "baseline": {},
        "extraction_schema": {},
    }

    try:
        # Run SGE pipeline
        sge_result = run_sge_pipeline(csv_path, sge_dir)
        chunks = sge_result["chunks"]
        extraction_schema = sge_result["extraction_schema"]
        payload = sge_result["payload"]
        result["extraction_schema"] = {
            "entity_types": extraction_schema["entity_types"],
            "relation_types": extraction_schema["relation_types"],
        }

        # SGE-enhanced LightRAG run
        print("\n" + "=" * 60)
        print(f"LIGHTRAG RUN — SGE-Enhanced [{label}]")
        print("=" * 60)

        original_system_prompt = _PROMPTS["entity_extraction_system_prompt"]
        raw_prompt = payload["system_prompt"]
        escaped = raw_prompt.replace("{", "{{").replace("}", "}}")
        for var in ("tuple_delimiter", "completion_delimiter", "entity_types", "examples", "language"):
            escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
        _PROMPTS["entity_extraction_system_prompt"] = escaped
        _op_mod.extract_entities = _sge_extract_entities

        try:
            sge_stats = await run_lightrag(
                chunks=chunks,
                working_dir=sge_work,
                addon_params=payload["addon_params"],
                label=f"SGE-{label}",
            )
        finally:
            _PROMPTS["entity_extraction_system_prompt"] = original_system_prompt
            _op_mod.extract_entities = _original_extract_entities

        (sge_dir / "lightrag_stats.json").write_text(
            json.dumps(sge_stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["sge"] = sge_stats

        # Baseline LightRAG run
        print("\n" + "=" * 60)
        print(f"LIGHTRAG RUN — Baseline [{label}]")
        print("=" * 60)

        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_stats = await run_lightrag(
            chunks=chunks,
            working_dir=baseline_work,
            addon_params={"language": "Chinese"},
            label=f"Baseline-{label}",
            baseline=True,
        )
        (baseline_dir / "lightrag_stats.json").write_text(
            json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["baseline"] = baseline_stats

    except Exception as e:
        import traceback
        result["error"] = traceback.format_exc()
        print(f"\n[ERROR] {label}: {e}")
        traceback.print_exc()

    return result


async def main_async():
    output_base = Path.home() / "Desktop/SGE/sge_lightrag/output"
    output_base.mkdir(parents=True, exist_ok=True)

    food_csv = Path.home() / "Desktop/SGE/dataset/食物安全及公众卫生统计数字/stat_foodSafty_publicHealth.csv"
    health_csv_orig = Path.home() / "Desktop/SGE/dataset/香港主要医疗卫生统计数字/healthstat_table1.csv"

    # Preprocess healthstat CSV
    health_csv_converted = output_base / "healthstat_table1_converted.csv"
    print("\n[Preprocessing] Converting healthstat CSV (UTF-16LE → UTF-8)...")
    preprocess_healthstat_csv(health_csv_orig, health_csv_converted)

    results = []

    # ── Run 1: Food Safety CSV (Type III) ─────────────────────────────────────
    print("\n" + "#" * 70)
    print("# RUN 1: Food Safety CSV (Type III)")
    print("#" * 70)
    food_result = await run_one_csv(
        csv_path=food_csv,
        output_base=output_base,
        sge_subdir="sge_food",
        baseline_subdir="baseline_food",
        label="food",
    )
    results.append(food_result)

    # ── Run 2: Health Stats CSV (Type II transposed) ──────────────────────────
    print("\n" + "#" * 70)
    print("# RUN 2: Health Stats CSV (Type II transposed)")
    print("#" * 70)
    health_result = await run_one_csv(
        csv_path=health_csv_converted,
        output_base=output_base,
        sge_subdir="sge_health",
        baseline_subdir="baseline_health",
        label="health",
    )
    results.append(health_result)

    # ── Load budget results ───────────────────────────────────────────────────
    budget_report_path = output_base / "comparison_report.json"
    budget_data = {}
    if budget_report_path.exists():
        budget_data = json.loads(budget_report_path.read_text(encoding="utf-8"))

    # ── Save combined results ─────────────────────────────────────────────────
    combined = {
        "timestamp": datetime.now().isoformat(),
        "runs": results,
        "budget_reference": budget_data,
    }
    combined_path = output_base / "multi_csv_comparison.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Combined results: {combined_path}")

    return combined


def main():
    combined = asyncio.run(main_async())

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-CSV COMPARISON SUMMARY")
    print("=" * 70)
    for run in combined.get("runs", []):
        label = run["label"]
        sge = run.get("sge", {})
        baseline = run.get("baseline", {})
        err = run.get("error")
        if err:
            print(f"  [{label}] ERROR: {err[:200]}")
        else:
            print(f"  [{label}] SGE: nodes={sge.get('node_count','?')}, edges={sge.get('edge_count','?')}")
            print(f"  [{label}] Baseline: nodes={baseline.get('node_count','?')}, edges={baseline.get('edge_count','?')}")


if __name__ == "__main__":
    main()
