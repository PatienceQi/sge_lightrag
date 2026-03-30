#!/usr/bin/env python3
"""
Generate Gold Standards for OOD Type-II datasets.

OOD Type-II CSVs have simple Country×Year format (no WB metadata header).
Samples 10 countries × 4 years = up to 40 facts per dataset.
"""

import csv
import json
import re
from pathlib import Path

OOD_DIR = Path(__file__).parent.parent.parent / "dataset" / "ood_blind_test"
GOLD_DIR = Path(__file__).parent / "gold"

# Sample 10 diverse countries (subset of v2's 25)
SAMPLE_COUNTRIES = [
    "Argentina", "Australia", "Brazil", "China", "France",
    "Germany", "India", "Japan", "Mexico", "United States",
]

# Also try 3-letter codes
SAMPLE_CODES = [
    "ARG", "AUS", "BRA", "CHN", "FRA",
    "DEU", "IND", "JPN", "MEX", "USA",
]


def _detect_year_columns(headers):
    """Find columns that look like years (4-digit numbers 1900-2099)."""
    year_cols = []
    for h in headers:
        h_stripped = h.strip()
        if re.match(r'^(19|20)\d{2}$', h_stripped):
            year_cols.append(h_stripped)
    return year_cols


def _pick_sample_years(year_cols, n=4):
    """Pick n representative years spread across the range."""
    years = sorted(int(y) for y in year_cols)
    if len(years) <= n:
        return [str(y) for y in years]
    # Pick first, last, and 2 evenly spaced middle years
    indices = [0, len(years)//3, 2*len(years)//3, len(years)-1]
    return [str(years[i]) for i in indices[:n]]


def generate_gold_for_type2(csv_path, relation_name, unit, decimals=2):
    """Generate Gold Standard for a Type-II Country×Year CSV."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    year_cols = _detect_year_columns(headers)
    if not year_cols:
        print(f"  SKIP {csv_path.name}: no year columns found")
        return []

    sample_years = _pick_sample_years(year_cols)
    first_col = headers[0]  # Country column

    triples = []
    for row_idx, row in enumerate(rows):
        country = row[first_col].strip()
        # Match against sample countries (full name or code)
        matched = False
        for sc, code in zip(SAMPLE_COUNTRIES, SAMPLE_CODES):
            if country == sc or country == code or country.startswith(sc):
                matched = True
                subject = country
                break
        if not matched:
            continue

        for yr in sample_years:
            raw = row.get(yr, '').strip()
            if not raw or raw in ('', '..', 'NA', 'N/A'):
                continue
            try:
                val = float(raw)
                if decimals == 0:
                    formatted = str(int(val))
                else:
                    formatted = f"{val:.{decimals}f}"
                    # Remove trailing zeros after decimal
                    if '.' in formatted:
                        formatted = formatted.rstrip('0').rstrip('.')
            except (ValueError, TypeError):
                continue

            triples.append({
                "source_file": csv_path.name,
                "row_index": row_idx,
                "triple": {
                    "subject": subject,
                    "subject_type": "Country",
                    "relation": relation_name,
                    "object": formatted,
                    "object_type": "StatValue",
                    "attributes": {"year": yr, "unit": unit},
                },
                "annotator": "gold_ood_auto",
                "confidence": "high",
            })

    return triples


# OOD Type-II dataset configs
OOD_TYPE2_CONFIGS = {
    "wb_gdp_growth.csv": {
        "relation": "GDP_GROWTH_ANNUAL_PCT",
        "unit": "%",
        "decimals": 2,
    },
    "wb_unemployment.csv": {
        "relation": "UNEMPLOYMENT_RATE",
        "unit": "%",
        "decimals": 2,
    },
    "wb_education_spending.csv": {
        "relation": "EDUCATION_SPENDING_PCT_GDP",
        "unit": "% of GDP",
        "decimals": 2,
    },
    "wb_health_expenditure.csv": {
        "relation": "HEALTH_EXPENDITURE_PCT_GDP",
        "unit": "% of GDP",
        "decimals": 2,
    },
    "wb_co2_emissions.csv": {
        "relation": "CO2_EMISSIONS_PER_CAPITA",
        "unit": "metric tons per capita",
        "decimals": 2,
    },
    "wb_cereal_production.csv": {
        "relation": "CEREAL_PRODUCTION",
        "unit": "metric tons",
        "decimals": 0,
    },
    "wb_literacy_rate.csv": {
        "relation": "ADULT_LITERACY_RATE",
        "unit": "%",
        "decimals": 2,
    },
    "wb_population_growth.csv": {
        "relation": "POPULATION_GROWTH_ANNUAL_PCT",
        "unit": "%",
        "decimals": 2,
    },
    "wb_immunization_dpt.csv": {
        "relation": "DPT_IMMUNIZATION_COVERAGE",
        "unit": "%",
        "decimals": 0,
    },
    "wb_immunization_measles.csv": {
        "relation": "MEASLES_IMMUNIZATION_COVERAGE",
        "unit": "%",
        "decimals": 0,
    },
}


def main():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    total_facts = 0

    for filename, cfg in OOD_TYPE2_CONFIGS.items():
        csv_path = OOD_DIR / filename
        if not csv_path.exists():
            print(f"  MISSING: {filename}")
            continue

        triples = generate_gold_for_type2(
            csv_path,
            relation_name=cfg["relation"],
            unit=cfg["unit"],
            decimals=cfg["decimals"],
        )

        if not triples:
            print(f"  EMPTY: {filename} (0 facts)")
            continue

        out_path = GOLD_DIR / f"gold_ood_{csv_path.stem}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for t in triples:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        print(f"  {filename}: {len(triples)} facts → {out_path.name}")
        total_facts += len(triples)

    print(f"\nTotal OOD Gold Standard facts: {total_facts}")


if __name__ == "__main__":
    main()
