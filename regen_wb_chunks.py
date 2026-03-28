#!/usr/bin/env python3
"""
regen_wb_chunks.py — Regenerate wb_child_mortality chunks with explicit triple format.

Problem: Current format "Year: X | Value: Y" causes LLM to write summaries
instead of creating HAS_VALUE edges. ~20% of country nodes end up isolated.

Fix: Use "ENTITY, YEAR, VALUE" format on every line so each line is an
unambiguous fact triple that LLM cannot "summarize away".

Also filters to years 2000–2022 (matching gold standard) to keep chunks short.
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNKS_DIR   = PROJECT_ROOT / "output" / "wb_child_mortality" / "chunks"
SCHEMA_PATH  = PROJECT_ROOT / "output" / "wb_child_mortality" / "extraction_schema.json"
CSV_PATH     = Path("/Users/qipatience/Desktop/SGE/dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv")

YEAR_FROM = 2000
YEAR_TO   = 2022


def read_wb_csv(path: Path) -> pd.DataFrame:
    """Read World Bank CSV (4 metadata rows before header)."""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(str(path), encoding=enc, skiprows=4, header=0)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def generate_chunks(df: pd.DataFrame, schema: dict) -> list[str]:
    column_roles = schema.get("column_roles", {})

    # Identify columns
    country_code_col = next((c for c, r in column_roles.items() if r == "subject"
                             and "Code" in str(c)), "Country Code")
    country_name_col = next((c for c, r in column_roles.items()
                             if r == "metadata" and "Name" in str(c)), "Country Name")
    indicator_name_col = next((c for c, r in column_roles.items()
                               if r == "metadata" and "Indicator Name" in str(c)),
                              "Indicator Name")

    year_cols = [c for c, r in column_roles.items() if r == "time_value"
                 and str(c).isdigit() and YEAR_FROM <= int(c) <= YEAR_TO]
    year_cols.sort(key=int)

    chunks = []
    for _, row in df.iterrows():
        code = str(row.get(country_code_col, "")).strip()
        if not code or code.lower() in ("", "nan"):
            continue

        name = str(row.get(country_name_col, "")).strip()
        indicator = str(row.get(indicator_name_col, "")).strip()
        if indicator.lower() in ("", "nan"):
            indicator = "Under-5 Mortality Rate (per 1,000 live births)"

        # Collect non-null year-value pairs
        facts = []
        for yr_col in year_cols:
            if yr_col not in row:
                continue
            val = row[yr_col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            try:
                v = float(str(val))
                if v > 0:
                    # Round to 1 decimal to avoid floating point noise
                    facts.append((str(yr_col), f"{v:.1f}"))
            except ValueError:
                pass

        if not facts:
            # No valid data in range — skip this country
            continue

        # Build chunk: header + natural-language observation sentences.
        # Format: "CODE observed mortality_rate_yearYYYY=VALUE per 1,000 live births."
        # This format causes the LLM to:
        #   1. Use "CODE" as primary entity identifier (appears in every sentence)
        #   2. Write edge descriptions that include year=VALUE strings
        #   3. Include the country code in the entity description (for fuzzy matching)
        header = (f"Record: {code} ({name}), "
                  f"indicator={indicator.split('(')[0].strip().replace(' ', '_')}")
        fact_lines = [
            f"{code} observed mortality_rate_year{yr}={val} per 1,000 live births."
            for yr, val in facts
        ]

        chunk = header + "\n" + "\n".join(fact_lines)
        chunks.append(chunk)

    return chunks


def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = read_wb_csv(CSV_PATH)
    print(f"  Shape: {df.shape}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    chunks = generate_chunks(df, schema)
    print(f"Generated {len(chunks)} chunks (years {YEAR_FROM}–{YEAR_TO})")

    # Preview first 2 chunks
    for i, c in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(c[:300])

    # Overwrite existing chunks
    # Remove old chunk files first
    for old in CHUNKS_DIR.glob("chunk_*.txt"):
        old.unlink()

    for i, chunk in enumerate(chunks, 1):
        fname = CHUNKS_DIR / f"chunk_{i:04d}.txt"
        fname.write_text(chunk, encoding="utf-8")

    print(f"\nWrote {len(chunks)} chunks to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
