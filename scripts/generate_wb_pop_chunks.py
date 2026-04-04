#!/usr/bin/env python3
"""
generate_wb_pop_chunks.py — Generate SGE-serialized text chunks for WB Population dataset.

Produces one chunk per country in the format:
    Entity: COUNTRY_CODE / SP.POP.TOTL
    Year: YYYY | Value: VVVVVVV
    ...
    Remark: Country Name
    Remark: Population, total

Output: graphrag_sge_wb_pop/input/chunk_NNNN.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(
    "/Users/qipatience/Desktop/SGE/dataset/世界银行数据/population"
    "/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"
)
OUTPUT_DIR = Path(
    "/Users/qipatience/Desktop/SGE/sge_lightrag/output/graphrag_sge_wb_pop/input"
)


def _format_value(raw_val) -> str | None:
    """Convert raw CSV value to clean integer string (no .0 suffix)."""
    if pd.isna(raw_val):
        return None
    try:
        int_val = int(float(str(raw_val)))
        return str(int_val)
    except (ValueError, TypeError):
        return None


def generate_chunks(csv_path: Path, output_dir: Path) -> int:
    """Generate one SGE-serialized chunk per country row."""
    df = pd.read_csv(csv_path, skiprows=4, encoding="utf-8-sig")

    year_cols = [c for c in df.columns if c.isdigit()]

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = 0

    for _, row in df.iterrows():
        country_name = str(row.get("Country Name", "")).strip()
        country_code = str(row.get("Country Code", "")).strip()
        indicator_code = str(row.get("Indicator Code", "")).strip()
        indicator_name = str(row.get("Indicator Name", "")).strip()

        if not country_code or country_code.lower() == "nan":
            continue

        lines = [f"Entity: {country_code} / {indicator_code}"]
        has_data = False

        for year in year_cols:
            val_str = _format_value(row.get(year))
            if val_str is None:
                continue
            lines.append(f"Year: {year} | Value: {val_str}")
            has_data = True

        if not has_data:
            continue

        if country_name and country_name.lower() != "nan":
            lines.append(f"Remark: {country_name}")
        lines.append(f"Remark: {indicator_name}")

        chunk_idx += 1
        chunk_text = "\n".join(lines)
        out_path = output_dir / f"chunk_{chunk_idx:04d}.txt"
        out_path.write_text(chunk_text, encoding="utf-8")

    return chunk_idx


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {CSV_PATH}")
    count = generate_chunks(CSV_PATH, OUTPUT_DIR)
    print(f"Generated {count} chunks → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
