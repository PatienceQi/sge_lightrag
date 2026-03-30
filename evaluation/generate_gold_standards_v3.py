#!/usr/bin/env python3
"""
Gold Standard v3 — Expand from 25 to 50 countries.

Same logic as v2 but with 50 countries for stronger statistical power.
Outputs v3 JSONL files (does NOT overwrite v2).
"""

import sys
from pathlib import Path

# Reuse all v2 logic
sys.path.insert(0, str(Path(__file__).parent))
from generate_gold_standards import (
    read_who_csv, read_wb_csv, get_value, fmt_value, _write,
    WHO_CSV, DATASET_DIR, EVAL_DIR,
)

# Original 25 + 25 new countries (geographic + economic diversity)
TARGET_CODES_V3 = [
    # Original 25 (G20 + emerging)
    "CHN", "IND", "USA", "GBR", "DEU", "FRA", "JPN", "BRA", "CAN", "AUS",
    "KOR", "MEX", "RUS", "SAU", "ZAF", "TUR", "IDN", "ARG", "ITA", "ESP",
    "EGY", "NGA", "PAK", "BGD", "THA",
    # New 25 (Europe, Middle East, SE Asia, Africa, Latin America, Oceania)
    "NLD", "CHE", "SWE", "NOR", "POL", "CZE", "AUT", "BEL", "GRC", "ISR",
    "IRN", "IRQ", "MYS", "PHL", "VNM", "KEN", "ETH", "GHA", "TZA", "COD",
    "UKR", "COL", "PER", "CHL", "NZL",
]


def gen_who_v3(output_path):
    rows = read_who_csv(WHO_CSV)
    years = [2000, 2005, 2010, 2015, 2019, 2021]
    triples = []
    for code in TARGET_CODES_V3:
        if code not in rows:
            print(f"    WARN: {code} not in WHO CSV")
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
                    "attributes": {"year": str(yr), "unit": "years",
                                   "indicator": "WHOSIS_000001", "country_name": r["name"]},
                },
                "annotator": "gold_v3", "confidence": "high",
            })
    _write(triples, output_path, "WHO v3")
    return len(triples)


def gen_wb_v3(output_path, csv_path, years, decimals, relation, unit, indicator, subject_type="Country_Code", label="WB"):
    rows = read_wb_csv(csv_path)
    triples = []
    for i, (code, r) in enumerate(rows.items()):
        if code not in TARGET_CODES_V3:
            continue
        subj = code if subject_type == "Country_Code" else r["name"]
        for yr in years:
            raw = get_value(r, yr)
            if raw is None:
                continue
            val = fmt_value(raw, decimals)
            if val is None:
                continue
            triples.append({
                "source_file": csv_path.name,
                "row_index": i + 5,
                "triple": {
                    "subject": subj,
                    "subject_type": subject_type,
                    "relation": relation,
                    "object": val,
                    "object_type": "StatValue",
                    "attributes": {"year": str(yr), "unit": unit, "indicator": indicator},
                },
                "annotator": "gold_v3", "confidence": "high",
            })
    _write(triples, output_path, label)
    return len(triples)


if __name__ == "__main__":
    print(f"Generating v3 Gold Standards (50 countries)...\n")
    totals = {}

    totals["WHO"] = gen_who_v3(EVAL_DIR / "gold_who_life_expectancy_v3.jsonl")

    totals["WB_CM"] = gen_wb_v3(
        EVAL_DIR / "gold_wb_child_mortality_v3.jsonl",
        DATASET_DIR / "child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        [2000, 2005, 2010, 2015, 2020, 2022], 1,
        "UNDER5_MORTALITY_RATE", "per 1,000 live births", "SH.DYN.MORT",
        label="WB Child Mortality v3")

    totals["WB_Pop"] = gen_wb_v3(
        EVAL_DIR / "gold_wb_population_v3.jsonl",
        DATASET_DIR / "population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        [2000, 2005, 2010, 2015, 2020, 2023], 0,
        "POPULATION", "persons", "SP.POP.TOTL",
        subject_type="Country", label="WB Population v3")

    totals["WB_Mat"] = gen_wb_v3(
        EVAL_DIR / "gold_wb_maternal_v3.jsonl",
        DATASET_DIR / "maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        [2000, 2005, 2010, 2015, 2019, 2021], 0,
        "MATERNAL_MORTALITY_RATE", "per 100,000 live births", "SH.STA.MMRT",
        subject_type="Country", label="WB Maternal v3")

    print(f"\n{'='*50}")
    grand = sum(totals.values())
    for k, v in totals.items():
        print(f"  {k}: {v} facts")
    print(f"  TOTAL v3: {grand} facts (v2 was 600)")
