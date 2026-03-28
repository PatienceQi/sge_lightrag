#!/usr/bin/env python3
"""
run_pipeline.py — Full Stage 1 → Stage 2 → Stage 3 pipeline CLI.

Usage:
    python3 run_pipeline.py <csv_path> [--output-dir <dir>] [--stage2-mode {llm,rule,auto}]

Outputs:
    <output-dir>/
        meta_schema.json              — Stage 1 output
        extraction_schema.json        — Stage 2 output
        chunks/
            chunk_0001.txt            — serialized text chunks
            chunk_0002.txt
            ...
        prompts/
            system_prompt.txt         — schema-aware system prompt for LightRAG
            user_prompt_template.txt  — user prompt template ({input_text} placeholder)
        pipeline_report.json          — summary of all stages
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage3.serializer import serialize_csv
from stage3.prompt_injector import generate_system_prompt, generate_user_prompt_template
from stage3.integrator import patch_lightrag


def _run_stage2(meta_schema, features, csv_path: str, mode: str, table_type: str):
    """
    Run Stage 2 schema induction with the specified mode.

    Args:
        meta_schema: Stage 1 meta-schema dict
        features: Stage 1 FeatureSet
        csv_path: path to the original CSV (needed by LLM inducer)
        mode: "llm"  — use LLM-enhanced induction only (fail if error)
              "rule" — use rule-based induction only
              "auto" — try LLM first, fall back to rule-based on any error
        table_type: Stage 1 classification result

    Returns:
        (extraction_schema, used_mode)  where used_mode is "llm" or "rule"
    """
    if mode == "rule":
        from stage2.inductor import induce_schema_from_meta
        return induce_schema_from_meta(features, table_type, meta_schema), "rule"

    # mode == "llm" or "auto"
    try:
        from stage2_llm.inductor import induce_schema as llm_induce
        schema = llm_induce(csv_path)
        # LLM schema lacks column_roles needed by Stage 3 serializer.
        # Merge rule-based column_roles into the LLM schema.
        if "column_roles" not in schema:
            from stage2.inducer import induce_schema as rule_induce
            rule_schema = rule_induce(meta_schema, features)
            schema["column_roles"] = rule_schema["column_roles"]
            if "parsed_time_headers" not in schema:
                schema["parsed_time_headers"] = rule_schema.get("parsed_time_headers", [])
            if "entity_extraction_template" not in schema:
                schema["entity_extraction_template"] = rule_schema.get("entity_extraction_template", "")
            if "relation_extraction_template" not in schema:
                schema["relation_extraction_template"] = rule_schema.get("relation_extraction_template", "")
        return schema, "llm"
    except Exception as exc:
        if mode == "llm":
            # Hard failure — propagate
            raise RuntimeError(f"LLM Stage 2 failed (use --stage2-mode auto for fallback): {exc}") from exc
        # mode == "auto" — fall back to rule-based (through inductor for adaptive mode)
        print(f"  [WARN] LLM Stage 2 failed ({type(exc).__name__}: {exc}), falling back to rule-based.")
        from stage2.inductor import induce_schema_from_meta
        return induce_schema_from_meta(features, table_type, meta_schema), "rule"


def main():
    parser = argparse.ArgumentParser(
        description="SGE-LightRAG full pipeline: Stage 1 → Stage 2 → Stage 3"
    )
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: ./sge_output/<csv_stem>)",
    )
    parser.add_argument(
        "--stage2-mode",
        choices=["llm", "rule", "auto"],
        default="llm",
        help=(
            "Stage 2 induction mode: "
            "'llm' (default) — LLM-enhanced only; "
            "'rule' — rule-based only; "
            "'auto' — try LLM, fall back to rule-based on error"
        ),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        Path.cwd() / "sge_output" / csv_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks").mkdir(exist_ok=True)
    (output_dir / "prompts").mkdir(exist_ok=True)

    report = {
        "csv_file": str(csv_path),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        "stages": {},
    }

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 1 — Topological Pattern Recognition")
    print("=" * 60)

    features   = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    print(f"Table Type : {table_type}")
    print(f"Columns    : {len(features.raw_columns)}")

    meta_schema_path = output_dir / "meta_schema.json"
    meta_schema_path.write_text(
        json.dumps(meta_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Written    : {meta_schema_path}")

    report["stages"]["stage1"] = {
        "table_type": table_type,
        "column_count": len(features.raw_columns),
        "output_file": str(meta_schema_path),
    }

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    mode_label = {"llm": "LLM-Enhanced", "rule": "Rule-Based", "auto": "Auto (LLM → Rule fallback)"}
    print(f"STAGE 2 — Schema Induction [{mode_label[args.stage2_mode]}]")
    print("=" * 60)

    try:
        extraction_schema, used_mode = _run_stage2(meta_schema, features, str(csv_path), args.stage2_mode, table_type)

        # Propagate time_dimension from meta_schema so Stage 3 serializer
        # can detect transposed tables.
        if 'time_dimension' not in extraction_schema:
            extraction_schema['time_dimension'] = meta_schema.get('time_dimension', {})
        # Pass actual row count so Stage 3 compact-mode check works correctly.
        extraction_schema['_n_rows'] = features.n_rows
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Mode used      : {used_mode}")
    if extraction_schema.get("use_baseline_mode"):
        print(f"[Adaptive]     : BASELINE MODE (n_rows={features.n_rows} < 20, type={table_type})")
        print(f"                 Reason: {extraction_schema.get('adaptive_reason','')}")
        print(f"                 Schema injection skipped — LightRAG default prompts will be used")
    print(f"Entity Types   : {extraction_schema['entity_types']}")
    print(f"Relation Types : {extraction_schema['relation_types']}")

    schema_path = output_dir / "extraction_schema.json"
    schema_path.write_text(
        json.dumps(extraction_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Written        : {schema_path}")

    report["stages"]["stage2"] = {
        "mode": used_mode,
        "entity_types": extraction_schema["entity_types"],
        "relation_types": extraction_schema["relation_types"],
        "output_file": str(schema_path),
    }

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STAGE 3 — Constrained Extraction Preparation")
    print("=" * 60)

    # 3a. Serialize CSV into text chunks
    chunks = serialize_csv(str(csv_path), extraction_schema)
    print(f"Chunks produced : {len(chunks)}")

    chunks_dir = output_dir / "chunks"
    for i, chunk in enumerate(chunks, start=1):
        chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
        chunk_file.write_text(chunk, encoding="utf-8")

    print(f"Written to      : {chunks_dir}/")

    # 3b. Generate prompts (skip if adaptive baseline mode)
    if extraction_schema.get("use_baseline_mode"):
        system_prompt = "[BASELINE MODE — using LightRAG default prompts, no schema injection]"
        user_prompt_tmpl = "[BASELINE MODE — using LightRAG default prompts]"
    else:
        system_prompt = generate_system_prompt(extraction_schema)
        user_prompt_tmpl = generate_user_prompt_template(extraction_schema)

    sys_prompt_path = output_dir / "prompts" / "system_prompt.txt"
    usr_prompt_path = output_dir / "prompts" / "user_prompt_template.txt"

    sys_prompt_path.write_text(system_prompt, encoding="utf-8")
    usr_prompt_path.write_text(user_prompt_tmpl, encoding="utf-8")

    print(f"System prompt   : {sys_prompt_path}")
    print(f"User template   : {usr_prompt_path}")

    # 3c. LightRAG integration payload
    payload = patch_lightrag(extraction_schema)
    if payload.get("use_compact_mode"):
        print(f"[Compact Mode]  : COMPACT REPRESENTATION (n_rows={features.n_rows} > 100)")
        print(f"                  Entity types: {payload['entity_types']}")

    report["stages"]["stage3"] = {
        "chunk_count": len(chunks),
        "chunks_dir": str(chunks_dir),
        "system_prompt_file": str(sys_prompt_path),
        "user_prompt_template_file": str(usr_prompt_path),
        "lightrag_entity_types": payload["entity_types"],
        "lightrag_addon_params_keys": list(payload["addon_params"].keys()),
    }

    # ── Pipeline report ───────────────────────────────────────────────────────
    report_path = output_dir / "pipeline_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output dir      : {output_dir}")
    print(f"Report          : {report_path}")
    print()
    print("Output files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(output_dir)}  ({size} bytes)")


if __name__ == "__main__":
    main()
