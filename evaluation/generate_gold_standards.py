#!/usr/bin/env python3
"""
Gold Standard Generator — Phase 1 Expansion (B区 COLING 2027 target)

Expands gold standards from ~156 facts to 600+ facts by:
- 25 diverse countries across regions (G20 + emerging markets)
- 5-6 years per country per dataset
- Values read DIRECTLY from source CSVs (fixes WB Population precision issue)

Datasets expanded:
  - WHO Life Expectancy   (Country_Code, 2 dec):  8→25 countries × 6 years = ~150 facts
  - WB Child Mortality    (Country_Code, 1 dec):  7→25 countries × 6 years = ~150 facts
  - WB Population         (Country name, integer): 7→25 countries × 6 years = ~150 facts
  - WB Maternal Mortality (Country name, integer): 7→25 countries × 5 years = ~125 facts
"""

import csv
import io
import json
import os
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATASET_DIR = Path("/Users/qipatience/Desktop/SGE/dataset/世界银行数据")
WHO_CSV = Path("/Users/qipatience/Desktop/SGE/sge_lightrag/dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv")

# ── 目标国家 (25 个：G20 + 区域代表) ───────────────────────────────────────
TARGET_CODES = [
    "CHN", "IND", "USA", "GBR", "DEU", "FRA", "JPN", "BRA", "CAN", "AUS",
    "KOR", "MEX", "RUS", "SAU", "ZAF", "TUR", "IDN", "ARG", "ITA", "ESP",
    "EGY", "NGA", "PAK", "BGD", "THA",
]


def read_wb_csv(filepath):
    """读取 World Bank CSV（前4行为 metadata，第5行为 header）"""
    with open(filepath, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_line = lines[4]
    headers = next(csv.reader(io.StringIO(header_line)))
    rows = {}
    for line in lines[5:]:
        line = line.strip()
        if not line:
            continue
        row = next(csv.reader(io.StringIO(line)))
        if len(row) < 4:
            continue
        code = row[1].strip().strip('"')
        name = row[0].strip().strip('"')
        rows[code] = {"name": name, "code": code, "headers": headers, "values": row}
    return rows


def read_who_csv(filepath):
    """读取 WHO CSV（标准 header，第1行）"""
    rows = {}
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for i, row in enumerate(reader, start=1):
            if len(row) < 4:
                continue
            code = row[1].strip()
            name = row[0].strip()
            rows[code] = {"name": name, "code": code, "headers": headers,
                          "values": row, "row_index": i}
    return rows


def get_value(row_data, year):
    """从行数据中获取指定年份的值，返回 None 如果不存在或为空"""
    headers = row_data["headers"]
    values = row_data["values"]
    try:
        idx = headers.index(str(year))
        val = values[idx].strip().strip('"')
        return val if val and val not in ("", "N/A", "..") else None
    except (ValueError, IndexError):
        return None


def fmt_value(raw, decimals):
    """
    格式化数值：
    - decimals=0: 整数（如人口、产妇死亡率）
    - decimals>0: 截断（truncate）到 N 位小数，而非四舍五入
      原因：图谱存储完整精度值（如 75.25530861），截断后的 "75.25"
      是图谱字符串的子串；四舍五入得到 "75.26" 则不是。
    """
    import math
    try:
        f = float(raw)
        if decimals == 0:
            return str(int(round(f)))
        # 截断（不进位）
        factor = 10 ** decimals
        truncated = math.floor(f * factor) / factor
        return f"{truncated:.{decimals}f}"
    except (ValueError, TypeError):
        return None


# ── 1. WHO Life Expectancy ───────────────────────────────────────────────────
def gen_who(output_path):
    rows = read_who_csv(WHO_CSV)
    years = [2000, 2005, 2010, 2015, 2019, 2021]
    triples = []
    for code in TARGET_CODES:
        if code not in rows:
            continue
        r = rows[code]
        for yr in years:
            raw = get_value(r, yr)
            if raw is None:
                continue
            val = fmt_value(raw, 2)
            if val is None:
                continue
            triples.append({
                "source_file": "API_WHO_WHOSIS_000001_life_expectancy.csv",
                "row_index": r["row_index"],
                "triple": {
                    "subject": code,
                    "subject_type": "Country_Code",
                    "relation": "LIFE_EXPECTANCY",
                    "object": val,
                    "object_type": "StatValue",
                    "attributes": {
                        "year": str(yr),
                        "unit": "years",
                        "indicator": "WHOSIS_000001",
                        "country_name": r["name"],
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
                "notes": f"Generated from CSV row {r['row_index']}",
            })
    _write(triples, output_path, "WHO Life Expectancy")
    return len(triples)


# ── 2. WB Child Mortality ─────────────────────────────────────────────────────
def gen_wb_child_mortality(output_path):
    csv_path = DATASET_DIR / "child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv"
    rows = read_wb_csv(csv_path)
    years = [2000, 2005, 2010, 2015, 2020, 2022]
    triples = []
    for i, (code, r) in enumerate(rows.items()):
        if code not in TARGET_CODES:
            continue
        for yr in years:
            raw = get_value(r, yr)
            if raw is None:
                continue
            val = fmt_value(raw, 1)
            if val is None:
                continue
            triples.append({
                "source_file": "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
                "row_index": i + 5,
                "triple": {
                    "subject": code,
                    "subject_type": "Country_Code",
                    "relation": "UNDER5_MORTALITY_RATE",
                    "object": val,
                    "object_type": "StatValue",
                    "attributes": {
                        "year": str(yr),
                        "unit": "per 1,000 live births",
                        "indicator": "SH.DYN.MORT",
                        "country_name": r["name"],
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
                "notes": "",
            })
    _write(triples, output_path, "WB Child Mortality")
    return len(triples)


# ── 3. WB Population ──────────────────────────────────────────────────────────
def gen_wb_population(output_path):
    csv_path = DATASET_DIR / "population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"
    rows = read_wb_csv(csv_path)
    years = [2000, 2005, 2010, 2015, 2020, 2023]
    triples = []
    for i, (code, r) in enumerate(rows.items()):
        if code not in TARGET_CODES:
            continue
        for yr in years:
            raw = get_value(r, yr)
            if raw is None:
                continue
            val = fmt_value(raw, 0)   # integer
            if val is None:
                continue
            triples.append({
                "source_file": "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
                "row_index": i + 5,
                "triple": {
                    "subject": r["name"],            # full country name (schema uses "Country")
                    "subject_type": "Country",
                    "relation": "POPULATION",
                    "object": val,
                    "object_type": "StatValue",
                    "attributes": {
                        "year": str(yr),
                        "unit": "persons",
                        "indicator": "SP.POP.TOTL",
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
                "notes": "Value read directly from source CSV to avoid version precision mismatch",
            })
    _write(triples, output_path, "WB Population")
    return len(triples)


# ── 4. WB Maternal Mortality ──────────────────────────────────────────────────
def gen_wb_maternal(output_path):
    csv_path = DATASET_DIR / "maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv"
    rows = read_wb_csv(csv_path)
    years = [2000, 2005, 2010, 2015, 2019, 2021]
    triples = []
    for i, (code, r) in enumerate(rows.items()):
        if code not in TARGET_CODES:
            continue
        for yr in years:
            raw = get_value(r, yr)
            if raw is None:
                continue
            val = fmt_value(raw, 0)   # integer
            if val is None:
                continue
            triples.append({
                "source_file": "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
                "row_index": i + 5,
                "triple": {
                    "subject": r["name"],
                    "subject_type": "Country",
                    "relation": "MATERNAL_MORTALITY_RATE",
                    "object": val,
                    "object_type": "StatValue",
                    "attributes": {
                        "year": str(yr),
                        "unit": "per 100,000 live births",
                        "indicator": "SH.STA.MMRT",
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
                "notes": "",
            })
    _write(triples, output_path, "WB Maternal Mortality")
    return len(triples)


def _write(triples, output_path, label):
    with open(output_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  [{label}] {len(triples)} facts → {output_path.name}")


if __name__ == "__main__":
    print("Generating expanded gold standards...")
    print()
    totals = {}

    totals["WHO"] = gen_who(
        EVAL_DIR / "gold_who_life_expectancy_v2.jsonl"
    )
    totals["WB_CM"] = gen_wb_child_mortality(
        EVAL_DIR / "gold_wb_child_mortality_v2.jsonl"
    )
    totals["WB_Pop"] = gen_wb_population(
        EVAL_DIR / "gold_wb_population_v2.jsonl"
    )
    totals["WB_Mat"] = gen_wb_maternal(
        EVAL_DIR / "gold_wb_maternal_v2.jsonl"
    )

    print()
    print("=" * 50)
    print("Summary:")
    grand = 0
    for k, v in totals.items():
        print(f"  {k}: {v} facts")
        grand += v
    print(f"  TOTAL NEW: {grand} facts")

    # 加上现有 HK Gov 数据集不变的部分
    existing = {
        "Budget": 20, "Food Safety": 66, "Health": 14, "Inpatient": 16
    }
    hk_total = sum(existing.values())
    print(f"  Existing HK Gov: {hk_total} facts (unchanged)")
    print(f"  GRAND TOTAL: {grand + hk_total} facts")
