#!/usr/bin/env python3
"""
generate_inpatient_chunks.py — Generate SGE-serialized text chunks for Inpatient dataset.

Produces one chunk per disease row in the format:
    Disease: ICD_CODE / DISEASE_NAME
    INPATIENT_HA_HOSPITAL: VALUE
    INPATIENT_TOTAL: VALUE
    REGISTERED_DEATHS: VALUE
    Remark: Chinese disease name

The chunk format mirrors the gold standard relations:
  - INPATIENT_TOTAL → 住院病人出院及死亡人次*-合计
  - INPATIENT_HA_HOSPITAL → 住院病人出院及死亡人次*-医院管理局辖下医院
  - REGISTERED_DEATHS → 全港登记死亡人数-合计

Output: graphrag_sge_inpatient/input/chunk_NNNN.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计"
    "/Inpatient Discharges and Deaths in Hospitals and Registered Deaths"
    " in Hong Kong by Disease 2023 (SC).csv"
)
OUTPUT_DIR = Path(
    "/Users/qipatience/Desktop/SGE/sge_lightrag/output/graphrag_sge_inpatient/input"
)

# Column aliases (detected from CSV)
COL_ICD = "《疾病和有关健康问题的国际统计分类》第十次修订本的详细序号"
COL_DISEASE = "疾病类别"
COL_HA = "住院病人出院及死亡人次*-医院管理局辖下医院"
COL_PRISON = "住院病人出院及死亡人次*-惩教署辖下医院"
COL_PRIVATE = "住院病人出院及死亡人次*-私家医院"
COL_TOTAL = "住院病人出院及死亡人次*-合计"
COL_DEATH_M = "全港登记死亡人数-男性"
COL_DEATH_F = "全港登记死亡人数-女性"
COL_DEATH_UK = "全港登记死亡人数-性别不详"
COL_DEATH_TOTAL = "全港登记死亡人数-合计"


def _format_int(raw_val) -> str | None:
    """Convert raw CSV value to clean integer string (no .0 suffix)."""
    if pd.isna(raw_val):
        return None
    try:
        int_val = int(float(str(raw_val)))
        return str(int_val)
    except (ValueError, TypeError):
        return None


def generate_chunks(csv_path: Path, output_dir: Path) -> int:
    """Generate one SGE-serialized chunk per disease row."""
    df = pd.read_csv(csv_path, skiprows=2, encoding="utf-8-sig")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = 0

    for _, row in df.iterrows():
        icd_code = str(row.get(COL_ICD, "")).strip()
        disease_name = str(row.get(COL_DISEASE, "")).strip()

        if not icd_code or icd_code.lower() == "nan":
            continue
        if not disease_name or disease_name.lower() == "nan":
            continue

        ha_val = _format_int(row.get(COL_HA))
        total_val = _format_int(row.get(COL_TOTAL))
        death_total_val = _format_int(row.get(COL_DEATH_TOTAL))
        death_m_val = _format_int(row.get(COL_DEATH_M))
        death_f_val = _format_int(row.get(COL_DEATH_F))
        prison_val = _format_int(row.get(COL_PRISON))
        private_val = _format_int(row.get(COL_PRIVATE))

        lines = [f"Disease: {icd_code} / {disease_name}"]

        if ha_val is not None:
            lines.append(f"INPATIENT_HA_HOSPITAL: {ha_val}")
        if prison_val is not None and prison_val != "0":
            lines.append(f"INPATIENT_PRISON_HOSPITAL: {prison_val}")
        if private_val is not None and private_val != "0":
            lines.append(f"INPATIENT_PRIVATE_HOSPITAL: {private_val}")
        if total_val is not None:
            lines.append(f"INPATIENT_TOTAL: {total_val}")
        if death_m_val is not None and death_m_val != "0":
            lines.append(f"REGISTERED_DEATHS_MALE: {death_m_val}")
        if death_f_val is not None and death_f_val != "0":
            lines.append(f"REGISTERED_DEATHS_FEMALE: {death_f_val}")
        if death_total_val is not None:
            lines.append(f"REGISTERED_DEATHS: {death_total_val}")

        lines.append(f"Remark: ICD {icd_code}")
        lines.append(f"Remark: {disease_name}")

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
    print(f"Generated {count} chunks -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
