"""
inductor.py — Main orchestration for LLM-enhanced schema induction.

Pipeline:
  1. Run Stage 1 (feature extraction → classification → meta-schema)
  2. Build a CSV snippet (header + first 5 data rows)
  3. Call Claude Haiku via llm_client to generate the rich extraction schema
  4. Parse and validate the JSON response
  5. Return the final schema dict
"""

import json
import sys
import os

# Allow running from the sge_lightrag root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema

from .llm_client import call_llm
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _build_csv_snippet(csv_path: str, n_rows: int = 5) -> str:
    """
    Return a plain-text CSV snippet: header line + first n_rows data rows.
    Uses the same encoding detection as Stage 1 (via pandas).
    """
    import pandas as pd
    from stage1.features import _detect_encoding

    encoding = _detect_encoding(csv_path)
    if "utf-16" in encoding:
        df = pd.read_csv(csv_path, encoding="utf-16", sep="\t",
                         header=None, nrows=n_rows + 5)
    else:
        df = None
        for enc in [encoding, "utf-8-sig", "utf-8", "gbk", "big5hkscs"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc, nrows=n_rows + 5)
                break
            except Exception:
                continue
        if df is None:
            raise ValueError(f"Cannot read CSV for snippet: {csv_path}")

    # Limit to header + first n_rows
    snippet_df = df.head(n_rows)
    return snippet_df.to_csv(index=False)


def _parse_schema_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response. Strips markdown fences if present.
    Raises ValueError if the JSON is invalid or missing required keys.
    """
    text = raw.strip()

    # Strip markdown code fences if the model added them anyway
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner).strip()

    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\n\nRaw response:\n{raw[:500]}") from exc

    # Validate required keys
    required = {"table_type", "entity_types", "relation_types", "extraction_rules", "prompt_context"}
    missing = required - set(schema.keys())
    if missing:
        raise ValueError(f"LLM schema missing required keys: {missing}")

    if not schema.get("entity_types"):
        raise ValueError("entity_types is empty")
    if not schema.get("relation_types"):
        raise ValueError("relation_types is empty")
    if not schema.get("prompt_context"):
        raise ValueError("prompt_context is empty")

    return schema


def induce_schema(csv_path: str) -> dict:
    """
    Full Stage 2 pipeline: run Stage 1 then call LLM to produce a rich schema.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file to process.

    Returns
    -------
    dict
        The rich extraction schema with keys: table_type, entity_types,
        relation_types, extraction_rules, prompt_context.

    Raises
    ------
    RuntimeError  if the LLM API call fails
    ValueError    if the LLM response cannot be parsed
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    features    = extract_features(csv_path)
    table_type  = classify(features)
    meta_schema = build_meta_schema(features, table_type)

    # ── CSV snippet ───────────────────────────────────────────────────────────
    csv_snippet = _build_csv_snippet(csv_path, n_rows=5)

    # ── Build prompts ─────────────────────────────────────────────────────────
    meta_schema_json = json.dumps(meta_schema, ensure_ascii=False, indent=2)
    user_prompt = build_user_prompt(table_type, meta_schema_json, csv_snippet)

    # ── LLM call with retry on parse failure ──────────────────────────────────
    max_attempts = 3
    last_error = None
    for attempt in range(1, max_attempts + 1):
        raw_response = call_llm(SYSTEM_PROMPT, user_prompt)
        try:
            schema = _parse_schema_response(raw_response)
            break
        except ValueError as exc:
            last_error = exc
            if attempt < max_attempts:
                import time
                time.sleep(1)
                continue
            raise ValueError(f"Failed to parse LLM response after {max_attempts} attempts: {last_error}") from exc

    # Attach Stage 1 meta-schema for downstream reference
    schema["_meta_schema"] = meta_schema

    return schema
