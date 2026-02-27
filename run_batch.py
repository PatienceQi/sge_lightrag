#!/usr/bin/env python3
"""
run_batch.py — Batch runner for SGE-LightRAG pipeline across all CSV files.

Finds all CSV files under ~/Desktop/SGE/dataset/ recursively,
runs run_pipeline.py on each, and produces a summary report.
"""

import sys
import os
import json
import subprocess
import re
import unicodedata
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("~/Desktop/SGE/dataset/").expanduser()
OUTPUT_BASE  = Path("~/Desktop/SGE/sge_lightrag/output/").expanduser()
PIPELINE_SCRIPT = Path(__file__).parent / "run_pipeline.py"
PLAN_DIR     = Path("~/Desktop/SGE/plan/").expanduser()

# Chinese folder → short ASCII prefix for output dir naming
FOLDER_PREFIX = {
    "住院病人统计":                          "inpatient",
    "年度预算":                              "annualbudget",
    "食物安全及公众卫生统计数字":              "foodsafety",
    "香港主要医疗卫生统计数字":               "healthstat",
    "香港本地医疗卫生总开支账目":             "hkhealth_expenditure",
    "有关人口的专题文章 - 住户开支统计调查结果": "hes",
}


def sanitize_name(csv_path: Path) -> str:
    """
    Build a clean ASCII output-dir name from the CSV path.
    Strategy: <folder_prefix>__<stem_ascii>
    """
    # Determine folder prefix
    parts = csv_path.parts
    prefix = "misc"
    for part in parts:
        for cn, en in FOLDER_PREFIX.items():
            if cn in part:
                prefix = en
                break

    stem = csv_path.stem
    # Normalize unicode → ASCII-safe (strip accents, keep ASCII)
    stem_ascii = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    # Replace spaces/parens/special chars with underscores
    stem_ascii = re.sub(r"[^\w]", "_", stem_ascii)
    stem_ascii = re.sub(r"_+", "_", stem_ascii).strip("_")

    return f"{prefix}__{stem_ascii}"


def _needs_preprocessing(csv_path: Path) -> bool:
    """Return True if the file needs preprocessing (UTF-16 tab-sep, etc.)."""
    try:
        from stage1.features import _detect_encoding
        enc = _detect_encoding(str(csv_path))
        return "utf-16" in enc
    except Exception:
        return False


def run_one(csv_path: Path) -> dict:
    """Run the pipeline on a single CSV. Returns a result dict."""
    out_name = sanitize_name(csv_path)
    out_dir  = OUTPUT_BASE / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "csv_file":   str(csv_path),
        "filename":   csv_path.name,
        "folder":     csv_path.parent.name,
        "output_dir": str(out_dir),
        "out_name":   out_name,
        "status":     "pending",
        "error":      None,
        "table_type": None,
        "entity_types":   [],
        "relation_types": [],
        "chunks":     None,
        "preprocessed": False,
    }

    # Preprocess UTF-16 tab-separated files to a temp UTF-8 CSV
    temp_path = None
    pipeline_input = csv_path
    if _needs_preprocessing(csv_path):
        try:
            from preprocessor import preprocess_to_tempfile
            temp_path, _ = preprocess_to_tempfile(str(csv_path))
            pipeline_input = Path(temp_path)
            result["preprocessed"] = True
        except Exception as exc:
            result["status"] = "error"
            result["error"]  = f"Preprocessing failed: {exc}"
            return result

    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        str(pipeline_input),
        "--output-dir", str(out_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PIPELINE_SCRIPT.parent),
        )
        if proc.returncode != 0:
            result["status"] = "error"
            result["error"]  = (proc.stderr or proc.stdout or "non-zero exit").strip()[-800:]
        else:
            result["status"] = "ok"
            # Parse pipeline_report.json for details
            report_path = out_dir / "pipeline_report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                s1 = report.get("stages", {}).get("stage1", {})
                s2 = report.get("stages", {}).get("stage2", {})
                s3 = report.get("stages", {}).get("stage3", {})
                result["table_type"]     = s1.get("table_type")
                result["entity_types"]   = s2.get("entity_types", [])
                result["relation_types"] = s2.get("relation_types", [])
                result["chunks"]         = s3.get("chunk_count")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"]  = "Pipeline timed out after 120s"
    except Exception as exc:
        result["status"] = "error"
        result["error"]  = str(exc)
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return result


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)

    # Find all CSV files (case-insensitive extension)
    csv_files = sorted(
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".csv"
    )

    print(f"Found {len(csv_files)} CSV files under {DATASET_DIR}")
    print("=" * 70)

    results = []
    for i, csv_path in enumerate(csv_files, 1):
        rel = csv_path.relative_to(DATASET_DIR)
        print(f"\n[{i:02d}/{len(csv_files)}] {rel}")
        r = run_one(csv_path)
        results.append(r)
        status_icon = "✓" if r["status"] == "ok" else "✗"
        pre_flag = " [preprocessed]" if r.get("preprocessed") else ""
        print(f"  {status_icon} status={r['status']}{pre_flag}  type={r['table_type']}  "
              f"chunks={r['chunks']}  entities={r['entity_types']}")
        if r["error"]:
            print(f"  ERROR: {r['error'][:200]}")

    # ── Save batch report JSON ────────────────────────────────────────────────
    batch_report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "ok":    sum(1 for r in results if r["status"] == "ok"),
        "error": sum(1 for r in results if r["status"] != "ok"),
        "results": results,
    }
    report_path = OUTPUT_BASE / "batch_report.json"
    report_path.write_text(
        json.dumps(batch_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nBatch report saved: {report_path}")

    # ── Write Markdown summary ────────────────────────────────────────────────
    write_markdown_summary(results, batch_report)

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {batch_report['ok']}/{batch_report['total']} succeeded, "
          f"{batch_report['error']} failed")


def write_markdown_summary(results: list, batch_report: dict):
    lines = []
    lines.append("# Batch Pipeline Results\n")
    lines.append(f"Generated: {batch_report['timestamp']}  ")
    lines.append(f"Total: **{batch_report['total']}** files | "
                 f"✓ {batch_report['ok']} succeeded | "
                 f"✗ {batch_report['error']} failed\n")

    # ── Main table ────────────────────────────────────────────────────────────
    lines.append("## Results Table\n")
    lines.append("| # | Filename | Folder | Type | Entity Types | Relation Types | Chunks | Status |")
    lines.append("|---|----------|--------|------|--------------|----------------|--------|--------|")

    for i, r in enumerate(results, 1):
        fname   = r["filename"]
        folder  = r["folder"]
        ttype   = r["table_type"] or "—"
        ents    = ", ".join(r["entity_types"]) if r["entity_types"] else "—"
        rels    = ", ".join(r["relation_types"]) if r["relation_types"] else "—"
        chunks  = str(r["chunks"]) if r["chunks"] is not None else "—"
        status  = "✓ ok" if r["status"] == "ok" else f"✗ {r['status']}"
        lines.append(f"| {i} | `{fname}` | {folder} | {ttype} | {ents} | {rels} | {chunks} | {status} |")

    # ── Errors section ────────────────────────────────────────────────────────
    errors = [r for r in results if r["status"] != "ok"]
    if errors:
        lines.append("\n## Errors\n")
        for r in errors:
            lines.append(f"### `{r['filename']}`")
            lines.append(f"- Status: `{r['status']}`")
            lines.append(f"- Error: {r['error']}\n")
    else:
        lines.append("\n## Errors\n\nNone — all files processed successfully.\n")

    # ── Distribution stats ────────────────────────────────────────────────────
    lines.append("## Distribution Stats\n")

    # Table type distribution
    type_counts: dict = {}
    for r in results:
        t = r["table_type"] or "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1
    lines.append("### Table Type Distribution\n")
    lines.append("| Table Type | Count |")
    lines.append("|------------|-------|")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {t} | {c} |")

    # Chunk distribution
    ok_results = [r for r in results if r["chunks"] is not None]
    if ok_results:
        chunk_counts = [r["chunks"] for r in ok_results]
        lines.append(f"\n### Chunk Counts (successful files)\n")
        lines.append(f"- Min: {min(chunk_counts)}")
        lines.append(f"- Max: {max(chunk_counts)}")
        lines.append(f"- Avg: {sum(chunk_counts)/len(chunk_counts):.1f}")
        lines.append(f"- Total: {sum(chunk_counts)}")

    # Entity type frequency
    ent_freq: dict = {}
    for r in results:
        for e in r["entity_types"]:
            ent_freq[e] = ent_freq.get(e, 0) + 1
    if ent_freq:
        lines.append("\n### Entity Type Frequency\n")
        lines.append("| Entity Type | Files |")
        lines.append("|-------------|-------|")
        for e, c in sorted(ent_freq.items(), key=lambda x: -x[1]):
            lines.append(f"| {e} | {c} |")

    # Relation type frequency
    rel_freq: dict = {}
    for r in results:
        for rel in r["relation_types"]:
            rel_freq[rel] = rel_freq.get(rel, 0) + 1
    if rel_freq:
        lines.append("\n### Relation Type Frequency\n")
        lines.append("| Relation Type | Files |")
        lines.append("|---------------|-------|")
        for rel, c in sorted(rel_freq.items(), key=lambda x: -x[1]):
            lines.append(f"| {rel} | {c} |")

    md_path = PLAN_DIR / "batch_pipeline_results.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown summary saved: {md_path}")


if __name__ == "__main__":
    main()
