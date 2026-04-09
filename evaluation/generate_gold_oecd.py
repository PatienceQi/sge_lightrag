#!/usr/bin/env python3
"""
generate_gold_oecd.py — Generate Gold Standard JSONL for OECD blind test datasets.

These datasets were NOT visible during SGE pipeline development (Stage 1 rules,
Stage 2 schema templates, Stage 3 serialization). They serve as a blind test set.

Usage:
    python3 evaluation/generate_gold_oecd.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "evaluation" / "gold"
OECD_DIR = BASE_DIR.parent / "dataset" / "OECD_blind_test"

# Sample countries for Type-II datasets (diverse, not in training set)
SAMPLE_COUNTRIES_GDP = ["Australia", "Canada", "Germany", "Japan", "Korea"]
SAMPLE_COUNTRIES_BEDS = ["Austria", "Canada", "Germany", "Japan", "Korea"]
SAMPLE_YEARS = ["2015", "2017", "2019", "2021", "2023"]


def generate_type_ii_gold(
    csv_path: Path,
    output_path: Path,
    countries: list[str],
    years: list[str],
    subject_col: str,
    indicator: str,
    unit: str,
) -> int:
    """Generate Gold JSONL for Type-II (Country × Year) dataset."""
    df = pd.read_csv(csv_path)
    facts = []

    for _, row in df.iterrows():
        country = str(row[subject_col]).strip()
        if country not in countries:
            continue

        for year in years:
            if year not in df.columns:
                continue
            val = row.get(year)
            if pd.isna(val) or str(val).strip() in ("", "nan", ".."):
                continue

            facts.append({
                "source_file": csv_path.name,
                "row_index": int(row.name),
                "triple": {
                    "subject": country,
                    "subject_type": "Country",
                    "relation": f"HAS_{indicator.upper()}_IN_YEAR",
                    "object": str(val).strip(),
                    "object_type": "StatValue",
                    "attributes": {"year": year, "unit": unit},
                },
                "annotator": "auto",
                "confidence": "high",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"  {output_path.name}: {len(facts)} facts ({len(countries)} countries × {len(years)} years)")
    return len(facts)


def generate_discharge_gold(csv_path: Path, output_path: Path) -> int:
    """Generate Gold JSONL for OECD discharge by country/disease (Type-III)."""
    df = pd.read_csv(csv_path)
    facts = []

    # Sample: 3 countries × 2 diseases × 3 years = 18 facts
    countries = ["Australia", "Canada", "Germany"]
    diseases = ["DICDA000", "DICDB000"]
    years = ["2017", "2019", "2021"]

    for _, row in df.iterrows():
        country = str(row.get("Country", "")).strip()
        icd = str(row.get("ICD_Chapter", "")).strip()
        disease = str(row.get("Disease_Group", "")).strip()

        if country not in countries or icd not in diseases:
            continue

        for year in years:
            if year not in df.columns:
                continue
            val = row.get(year)
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                continue

            facts.append({
                "source_file": csv_path.name,
                "row_index": int(row.name),
                "triple": {
                    "subject": country,
                    "subject_type": "Country",
                    "relation": "HAS_DISCHARGE_COUNT",
                    "object": str(int(float(str(val).replace(",", "")))),
                    "object_type": "StatValue",
                    "attributes": {
                        "year": year,
                        "icd_chapter": icd,
                        "disease_group": disease,
                        "unit": "discharges",
                    },
                },
                "annotator": "auto",
                "confidence": "high",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"  {output_path.name}: {len(facts)} facts")
    return len(facts)


def generate_germany_discharge_gold(csv_path: Path, output_path: Path) -> int:
    """Generate Gold JSONL for Germany hospital discharge (Type-III, single country)."""
    df = pd.read_csv(csv_path)
    facts = []

    # Sample: 5 diseases × 3 years = 15 facts
    icd_codes = ["DICDA000", "DICDB000", "DICDA100", "DICDA101", "DICDA102"]
    years = ["2017", "2019", "2021"]

    for _, row in df.iterrows():
        icd = str(row.get("ICD_Code", "")).strip()
        disease = str(row.get("Disease_Category", "")).strip()

        if icd not in icd_codes:
            continue

        for year in years:
            if year not in df.columns:
                continue
            val = row.get(year)
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                continue

            facts.append({
                "source_file": csv_path.name,
                "row_index": int(row.name),
                "triple": {
                    "subject": disease if disease != icd else icd,
                    "subject_type": "Disease_Category",
                    "relation": "HAS_DISCHARGE_COUNT",
                    "object": str(int(float(str(val).replace(",", "")))),
                    "object_type": "StatValue",
                    "attributes": {
                        "year": year,
                        "icd_code": icd,
                        "country": "Germany",
                        "unit": "discharges",
                    },
                },
                "annotator": "auto",
                "confidence": "high",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"  {output_path.name}: {len(facts)} facts")
    return len(facts)


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    print("Generating OECD blind test Gold Standards...")

    # 1. GDP
    total += generate_type_ii_gold(
        OECD_DIR / "oecd_gdp_usd_millions.csv",
        GOLD_DIR / "gold_oecd_gdp.jsonl",
        SAMPLE_COUNTRIES_GDP, SAMPLE_YEARS,
        "Country", "GDP", "USD millions",
    )

    # 2. Hospital beds
    total += generate_type_ii_gold(
        OECD_DIR / "oecd_hospital_beds_per_1000.csv",
        GOLD_DIR / "gold_oecd_hospital_beds.jsonl",
        SAMPLE_COUNTRIES_BEDS, SAMPLE_YEARS,
        "Country", "HOSPITAL_BEDS", "per 1000 population",
    )

    # 3. Discharge by country/disease (Type-III)
    total += generate_discharge_gold(
        OECD_DIR / "oecd_discharge_by_country_disease.csv",
        GOLD_DIR / "gold_oecd_discharge_country.jsonl",
    )

    # 4. Germany discharge (Type-III)
    total += generate_germany_discharge_gold(
        OECD_DIR / "oecd_germany_hospital_discharge.csv",
        GOLD_DIR / "gold_oecd_germany_discharge.jsonl",
    )

    print(f"\nTotal: {total} gold facts across 4 OECD datasets")


if __name__ == "__main__":
    main()
