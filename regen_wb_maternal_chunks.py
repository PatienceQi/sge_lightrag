#!/usr/bin/env python3
"""
regen_wb_maternal_chunks.py — Regenerate wb_maternal chunks with country-name format.

Problem: Current chunks use country codes as entity identifiers,
but gold standard uses full country names (China, India, etc.).

Fix: Use country name as entity identifier in each observation sentence.

Format: "China observed maternal_mortality_year2021=21 per 100,000 live births."
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNKS_DIR  = PROJECT_ROOT / "output" / "wb_maternal" / "chunks"
SCHEMA_PATH = PROJECT_ROOT / "output" / "wb_maternal" / "extraction_schema.json"
CSV_PATH    = Path("/Users/qipatience/Desktop/SGE/dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv")

YEAR_FROM = 2000
YEAR_TO   = 2021  # Maternal mortality data goes up to 2021


def read_wb_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(str(path), encoding=enc, skiprows=4, header=0)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def generate_chunks(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    name_col = next((c for c in cols if "Country Name" in str(c) or c == "Country Name"), cols[0])
    code_col = next((c for c in cols if "Country Code" in str(c) or c == "Country Code"), cols[1])

    year_cols = [c for c in cols if str(c).isdigit() and YEAR_FROM <= int(c) <= YEAR_TO]
    year_cols.sort(key=int)

    chunks = []
    for _, row in df.iterrows():
        code = str(row.get(code_col, "")).strip()
        name = str(row.get(name_col, "")).strip()
        if not code or code.lower() in ("", "nan") or not name or name.lower() in ("", "nan"):
            continue

        facts = []
        for yr_col in year_cols:
            val = row.get(yr_col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            try:
                v = float(str(val))
                if v >= 0:
                    # Maternal mortality is typically an integer
                    facts.append((str(yr_col), f"{v:.0f}"))
            except ValueError:
                pass

        if not facts:
            continue

        header = f"Record: {name} ({code}), indicator=Maternal_Mortality_Ratio"
        fact_lines = [
            f"{name} observed maternal_mortality_year{yr}={val} per 100,000 live births."
            for yr, val in facts
        ]

        chunk = header + "\n" + "\n".join(fact_lines)
        chunks.append(chunk)

    return chunks


def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = read_wb_csv(CSV_PATH)
    print(f"  Shape: {df.shape}")

    chunks = generate_chunks(df)
    print(f"Generated {len(chunks)} chunks (years {YEAR_FROM}–{YEAR_TO})")

    # Preview first 2 chunks
    for i, c in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(c[:400])

    # Overwrite existing chunks
    for old in CHUNKS_DIR.glob("chunk_*.txt"):
        old.unlink()

    for i, chunk in enumerate(chunks, 1):
        fname = CHUNKS_DIR / f"chunk_{i:04d}.txt"
        fname.write_text(chunk, encoding="utf-8")

    print(f"\nWrote {len(chunks)} chunks to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
