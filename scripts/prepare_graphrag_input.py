#!/usr/bin/env python3
"""
prepare_graphrag_input.py — Generate naive-serialized text chunks for MS GraphRAG.

Reads source CSV files and creates one text file per entity (row) in the
GraphRAG input directory. Uses naive serialization (no SGE structure awareness)
to serve as a fair baseline comparison.

Usage:
    python3 scripts/prepare_graphrag_input.py \
        --csv dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv \
        --output output/graphrag_who/input \
        --skip-rows 0

    python3 scripts/prepare_graphrag_input.py \
        --csv "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv" \
        --output output/graphrag_wb_cm/input \
        --skip-rows 4
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


def read_csv_with_skip(csv_path: Path, skip_rows: int) -> list[list[str]]:
    """Read CSV, skipping metadata rows (common in WB format)."""
    text = csv_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    reader = csv.reader(lines[skip_rows:])
    return list(reader)


def serialize_type2_row(header: list[str], row: list[str]) -> str:
    """Naive serialization for Type-II (temporal matrix) rows.

    Format matches existing graphrag_who input:
        Entity: CODE / INDICATOR_CODE
        Year: YYYY | Value: X.XX
        ...
        Remark: Country Name
        Remark: Indicator Name
    """
    country_name = row[0].strip()
    country_code = row[1].strip()
    indicator_name = row[2].strip()
    indicator_code = row[3].strip()

    lines = [f"Entity: {country_code} / {indicator_code}"]

    for i in range(4, len(header)):
        year = header[i].strip()
        value = row[i].strip() if i < len(row) else ""
        if value:
            lines.append(f"Year: {year} | Value: {value}")

    if country_name:
        lines.append(f"Remark: {country_name}")
    if indicator_name:
        lines.append(f"Remark: {indicator_name}")

    return "\n".join(lines)


def serialize_type3_row(header: list[str], row: list[str]) -> str:
    """Naive serialization for Type-III (hierarchical) rows.

    Format: key-value pairs separated by pipes.
    """
    parts = []
    for i, col in enumerate(header):
        value = row[i].strip() if i < len(row) else ""
        if value:
            parts.append(f"{col.strip()}: {value}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Prepare GraphRAG input chunks from CSV")
    parser.add_argument("--csv", required=True, help="Source CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Number of metadata rows to skip (WB format: 4)")
    parser.add_argument("--type", choices=["II", "III"], default="II",
                        help="CSV topology type (II=temporal matrix, III=hierarchical)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_with_skip(csv_path, args.skip_rows)
    if not rows:
        print("ERROR: No data rows found")
        return

    header = rows[0]
    data_rows = rows[1:]

    # Filter out empty rows
    data_rows = [r for r in data_rows if any(c.strip() for c in r)]

    print(f"CSV: {csv_path.name}")
    print(f"Header columns: {len(header)}")
    print(f"Data rows: {len(data_rows)}")

    serialize_fn = serialize_type2_row if args.type == "II" else serialize_type3_row

    chunk_count = 0
    for i, row in enumerate(data_rows):
        text = serialize_fn(header, row)
        if not text.strip():
            continue
        chunk_count += 1
        chunk_file = output_dir / f"chunk_{chunk_count:04d}.txt"
        chunk_file.write_text(text, encoding="utf-8")

    print(f"Generated {chunk_count} chunks in {output_dir}")


if __name__ == "__main__":
    main()
