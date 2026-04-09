#!/usr/bin/env python3
"""
generate_gold_expanded.py — Gold Standard generation for expanded evaluation datasets.

Generates gold standards for three new datasets covering different table topologies:
  A. IMF Fiscal Cross-Tabulation (Type-III) — financial hierarchical cross-tab
     Structure: Metric_Category × Sub_Category × Country + year-value columns
  B. UN Census Cross-Tabulation (Type-III) — demographic snapshot, no time dimension
     Structure: Region × Sub_Region × Age_Group × Sex + numeric value columns
  C. WB Indicators Long-Format (Type-II-Long) — melted/tidy economic indicators
     Structure: Country × Indicator × Year × Value (long/tidy format)

Output:
  evaluation/gold/gold_imf_fiscal_cross_tab.jsonl    (~15 entities × 5 facts = 75 facts)
  evaluation/gold/gold_un_census_cross_tab.jsonl     (~20 entities × 3 facts = 60 facts)
  evaluation/gold/gold_wb_indicators_long.jsonl      (~10 entities × 9 facts = 90 facts)

Usage:
    python3 evaluation/generate_gold_expanded.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPANDED_DIR = PROJECT_ROOT.parent / "dataset" / "expanded"
GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"


def _write_jsonl(triples: list[dict], output_path: Path, label: str) -> int:
    """Write triples to JSONL file, one JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")
    print(f"  [{label}] wrote {len(triples)} facts → {output_path.name}")
    return len(triples)


def _verify_output(path: Path, n_facts: int) -> None:
    """Read back the JSONL and verify line count and format."""
    if not path.exists():
        print(f"  [ERROR] Output not found: {path}", file=sys.stderr)
        return

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) != n_facts:
        print(
            f"  [WARN] Line count mismatch: expected {n_facts}, got {len(lines)}",
            file=sys.stderr,
        )

    try:
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        assert "triple" in first
        assert "triple" in last
        assert first["triple"]["object_type"] == "StatValue"
        print(f"  [OK] Verified {len(lines)} facts in {path.name}")
        print(f"       First: {first['triple']['subject']} | "
              f"{first['triple']['relation']} | {first['triple']['object']}")
    except (json.JSONDecodeError, KeyError, AssertionError) as exc:
        print(f"  [ERROR] Format validation failed: {exc}", file=sys.stderr)


def gen_imf_fiscal(
    n_entities: int = 15,
    target_years: tuple[str, ...] = ("2018", "2019", "2020", "2021", "2022"),
) -> int:
    """Generate Gold Standard for IMF Fiscal Cross-Tabulation dataset.

    Dataset structure: Metric_Category × Sub_Category × Country + year-value columns
    Each entity is identified by (Metric_Category, Sub_Category, Country).
    Each fact is (entity, HAS_FISCAL_VALUE, year_value, year).

    Selects first n_entities rows to keep gold standard manageable.
    """
    csv_path = EXPANDED_DIR / "imf_fiscal_cross_tab.csv"
    output_path = GOLD_DIR / "gold_imf_fiscal_cross_tab.jsonl"

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    triples: list[dict] = []
    year_cols = [y for y in target_years if y in (rows[0].keys() if rows else [])]

    for row_index, row in enumerate(rows[:n_entities]):
        category = row.get("Metric_Category", "").strip()
        sub_category = row.get("Sub_Category", "").strip()
        country = row.get("Country", "").strip()
        if not category or not sub_category or not country:
            continue

        subject = f"{category} > {sub_category} > {country}"

        for year in year_cols:
            raw_val = row.get(year, "").strip()
            if not raw_val:
                continue
            try:
                float(raw_val)
            except ValueError:
                continue

            triples.append({
                "source_file": "imf_fiscal_cross_tab.csv",
                "row_index": row_index,
                "triple": {
                    "subject": subject,
                    "subject_type": "FiscalMetric",
                    "relation": "HAS_FISCAL_VALUE",
                    "object": raw_val,
                    "object_type": "StatValue",
                    "attributes": {
                        "year": year,
                        "metric_category": category,
                        "sub_category": sub_category,
                        "country": country,
                        "unit": "USD billions",
                        "domain": "fiscal_statistics",
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
            })

    return _write_jsonl(triples, output_path, "IMF Fiscal Cross-Tab")


def gen_un_census(n_entities: int = 20) -> int:
    """Generate Gold Standard for UN Census Cross-Tabulation dataset.

    Dataset structure: Region × Sub_Region × Age_Group × Sex + numeric columns
    This is a snapshot dataset with NO time dimension.
    Each entity is identified by (Region, Sub_Region, Age_Group, Sex).
    Each fact is (entity, HAS_POPULATION, value) or
                 (entity, HAS_DEPENDENCY_RATIO, value) etc.

    Selects first n_entities rows.
    """
    csv_path = EXPANDED_DIR / "un_census_cross_tab.csv"
    output_path = GOLD_DIR / "gold_un_census_cross_tab.jsonl"

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Define which columns to extract as facts (relation, unit)
    value_col_map = {
        "Population_Thousands": ("HAS_POPULATION", "thousands"),
        "Dependency_Ratio": ("HAS_DEPENDENCY_RATIO", "ratio"),
        "Labor_Force_Participation": ("HAS_LABOR_PARTICIPATION", "ratio"),
    }

    triples: list[dict] = []
    for row_index, row in enumerate(rows[:n_entities]):
        region = row.get("Region", "").strip()
        sub_region = row.get("Sub_Region", "").strip()
        age_group = row.get("Age_Group", "").strip()
        sex = row.get("Sex", "").strip()

        if not region or not age_group or not sex:
            continue

        subject = f"{region} | {age_group} | {sex}"
        if sub_region:
            subject = f"{region} | {sub_region} | {age_group} | {sex}"

        for col_name, (relation, unit) in value_col_map.items():
            raw_val = row.get(col_name, "").strip()
            if not raw_val:
                continue
            try:
                float(raw_val)
            except ValueError:
                continue

            triples.append({
                "source_file": "un_census_cross_tab.csv",
                "row_index": row_index,
                "triple": {
                    "subject": subject,
                    "subject_type": "DemographicGroup",
                    "relation": relation,
                    "object": raw_val,
                    "object_type": "StatValue",
                    "attributes": {
                        "region": region,
                        "sub_region": sub_region,
                        "age_group": age_group,
                        "sex": sex,
                        "unit": unit,
                        "domain": "demographics",
                        "year": "2020",  # snapshot year
                    },
                },
                "annotator": "gold_expanded",
                "confidence": "high",
            })

    return _write_jsonl(triples, output_path, "UN Census Cross-Tab")


def gen_wb_long_format(
    n_countries: int = 10,
    target_indicators: tuple[str, ...] = (
        "GDP_per_capita", "Inflation_Rate", "Unemployment_Rate"
    ),
    target_years: tuple[str, ...] = ("2015", "2018", "2021"),
) -> int:
    """Generate Gold Standard for WB Indicators Long-Format dataset.

    Dataset structure: Country × Indicator × Year × Value (long/tidy format)
    This is the format that previously BROKE SGE (misclassified as Type-III).
    Each entity is identified by Country.
    Each fact is (country, HAS_INDICATOR_VALUE, value, {indicator, year}).

    Selection: first n_countries × target_indicators × target_years
    """
    csv_path = EXPANDED_DIR / "wb_indicators_long_format.csv"
    output_path = GOLD_DIR / "gold_wb_indicators_long.jsonl"

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_countries = list(dict.fromkeys(r["Country"] for r in rows))
    selected_countries = all_countries[:n_countries]
    target_ind_set = set(target_indicators)
    target_yr_set = set(target_years)

    print(f"  Countries selected ({len(selected_countries)}): {selected_countries}")
    print(f"  Indicators: {sorted(target_ind_set)}")
    print(f"  Target years: {sorted(target_yr_set)}")

    # Build lookup: (country, indicator, year) -> (value, unit)
    lookup: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row["Country"], row["Indicator"], row["Year"])
        lookup[key] = (row["Value"], row.get("Unit", ""))

    triples: list[dict] = []
    row_index = 0

    for country in selected_countries:
        for indicator in sorted(target_ind_set):
            for year in sorted(target_yr_set):
                result = lookup.get((country, indicator, year))
                if result is None:
                    continue
                value, unit = result
                if not value.strip():
                    continue

                subject = country
                relation = f"HAS_{indicator.upper()}"

                triples.append({
                    "source_file": "wb_indicators_long_format.csv",
                    "row_index": row_index,
                    "triple": {
                        "subject": subject,
                        "subject_type": "Country",
                        "relation": relation,
                        "object": value.strip(),
                        "object_type": "StatValue",
                        "attributes": {
                            "year": year,
                            "indicator": indicator,
                            "unit": unit.strip(),
                            "domain": "macroeconomics",
                        },
                    },
                    "annotator": "gold_expanded",
                    "confidence": "high",
                    "notes": (
                        f"Long-format row: {country}, {indicator}, {year}"
                    ),
                })
                row_index += 1

    return _write_jsonl(triples, output_path, "WB Indicators Long-Format")


def main() -> None:
    """Generate gold standards for all three expanded datasets."""
    print("=" * 64)
    print("GOLD STANDARD GENERATION — Expanded Evaluation Datasets")
    print("=" * 64)

    if not EXPANDED_DIR.exists():
        print(f"ERROR: Expanded dataset directory not found: {EXPANDED_DIR}",
              file=sys.stderr)
        sys.exit(1)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Dataset A] IMF Fiscal Cross-Tabulation (Type-III)")
    print(f"  Source: {EXPANDED_DIR / 'imf_fiscal_cross_tab.csv'}")
    n_imf = gen_imf_fiscal(n_entities=15)
    _verify_output(GOLD_DIR / "gold_imf_fiscal_cross_tab.jsonl", n_imf)

    print("\n[Dataset B] UN Census Cross-Tabulation (Type-III, no time dimension)")
    print(f"  Source: {EXPANDED_DIR / 'un_census_cross_tab.csv'}")
    n_census = gen_un_census(n_entities=20)
    _verify_output(GOLD_DIR / "gold_un_census_cross_tab.jsonl", n_census)

    print("\n[Dataset C] WB Indicators Long-Format (Type-II-Long, melted format)")
    print(f"  Source: {EXPANDED_DIR / 'wb_indicators_long_format.csv'}")
    n_wb_long = gen_wb_long_format(
        n_countries=10,
        target_indicators=("GDP_per_capita", "Inflation_Rate", "Unemployment_Rate"),
        target_years=("2015", "2018", "2021"),
    )
    _verify_output(GOLD_DIR / "gold_wb_indicators_long.jsonl", n_wb_long)

    total = n_imf + n_census + n_wb_long
    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"  IMF Fiscal Cross-Tab:        {n_imf} facts")
    print(f"  UN Census Cross-Tab:         {n_census} facts")
    print(f"  WB Indicators Long-Format:   {n_wb_long} facts")
    print(f"  Total:                       {total} facts")
    print(f"  Output directory:            {GOLD_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
