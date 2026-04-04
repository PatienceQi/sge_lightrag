#!/usr/bin/env python3
"""
generate_gold_new_datasets.py — Gold Standard generation for new OOD datasets.

Generates gold standards for two Type-III long-format datasets:
  A. Eurostat Crime Statistics (1365 rows, long format: Category × Country × Year)
  B. US Census Population by Demographics (840 rows, long format: State × Age × Year)

These datasets extend the OOD evaluation coverage for the EMNLP paper by
providing additional Type-III cross-domain benchmarks beyond the WB OOD set.

Output:
  evaluation/gold/gold_eurostat_crime.jsonl      (up to 10×5×3 = 150 facts)
  evaluation/gold/gold_us_census_demographics.jsonl (10×3×3 = 90 facts)

Usage:
    python3 evaluation/generate_gold_new_datasets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset" / "ood_blind_test"
GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"


def _write_jsonl(triples: list[dict], output_path: Path, label: str) -> int:
    """Write triples to JSONL file, one JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")
    print(f"  [{label}] wrote {len(triples)} facts → {output_path.name}")
    return len(triples)


def gen_eurostat_crime(
    n_categories: int = 10,
    n_countries: int = 5,
    target_years: tuple[str, ...] = ("2015", "2018", "2021"),
) -> int:
    """Generate Gold Standard for Eurostat Crime Statistics dataset.

    Dataset format: Crime_Category, Country, Year, Rate_per_100k
    Selection: first n_categories unique Crime_Categories ×
               first n_countries unique Countries ×
               specified target_years

    Each fact: {"subject": "Assault - Austria", "relation": "HAS_CRIME_RATE",
                "object": "315.7", "attributes": {...}}

    Parameters
    ----------
    n_categories : number of unique crime categories to include (max available)
    n_countries  : number of unique countries to include
    target_years : year values to sample

    Returns
    -------
    int — number of facts written
    """
    csv_path = DATASET_DIR / "synthetic_eurostat_crime_statistics.csv"
    output_path = GOLD_DIR / "gold_eurostat_crime.jsonl"

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Preserve insertion order for reproducible selection
    all_categories = list(dict.fromkeys(r["Crime_Category"] for r in rows))
    all_countries = list(dict.fromkeys(r["Country"] for r in rows))

    selected_categories = all_categories[:n_categories]
    selected_countries = all_countries[:n_countries]
    target_years_set = set(target_years)

    print(f"  Categories selected ({len(selected_categories)}): {selected_categories}")
    print(f"  Countries selected ({len(selected_countries)}): {selected_countries}")
    print(f"  Target years: {sorted(target_years_set)}")

    # Build lookup for fast access: (category, country, year) -> rate
    lookup: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (row["Crime_Category"], row["Country"], row["Year"])
        lookup[key] = row["Rate_per_100k"]

    triples: list[dict] = []
    row_index = 0

    for category in selected_categories:
        for country in selected_countries:
            for year in sorted(target_years_set):
                rate = lookup.get((category, country, year))
                if rate is None or rate.strip() == "":
                    continue

                subject = f"{category} - {country}"
                triples.append({
                    "source_file": "synthetic_eurostat_crime_statistics.csv",
                    "row_index": row_index,
                    "triple": {
                        "subject": subject,
                        "subject_type": "CrimeCategory_Country",
                        "relation": "HAS_CRIME_RATE",
                        "object": rate.strip(),
                        "object_type": "StatValue",
                        "attributes": {
                            "year": year,
                            "crime_category": category,
                            "country": country,
                            "unit": "per 100k population",
                            "domain": "crime_statistics",
                        },
                    },
                    "annotator": "gold_new_datasets",
                    "confidence": "high",
                    "notes": f"Generated from long-format CSV row for {subject}, year {year}",
                })
                row_index += 1

    return _write_jsonl(triples, output_path, "Eurostat Crime")


def gen_us_census_demographics(
    n_states: int = 10,
    n_age_groups: int = 3,
    target_years: tuple[str, ...] = ("2012", "2016", "2020"),
) -> int:
    """Generate Gold Standard for US Census Population by Demographics dataset.

    Dataset format: State, Age_Group, Year, Population_Thousands
    Selection: first n_states unique States ×
               first n_age_groups unique Age_Groups ×
               specified target_years

    Each fact: {"subject": "Arizona - 18_to_24", "relation": "HAS_POPULATION",
                "object": "655", "attributes": {...}}

    Parameters
    ----------
    n_states     : number of unique states to include
    n_age_groups : number of unique age groups to include
    target_years : year values to sample

    Returns
    -------
    int — number of facts written
    """
    csv_path = DATASET_DIR / "synthetic_us_census_population_by_demographics.csv"
    output_path = GOLD_DIR / "gold_us_census_demographics.jsonl"

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Preserve insertion order for reproducible selection
    all_states = list(dict.fromkeys(r["State"] for r in rows))
    all_age_groups = list(dict.fromkeys(r["Age_Group"] for r in rows))

    selected_states = all_states[:n_states]
    selected_age_groups = all_age_groups[:n_age_groups]
    target_years_set = set(target_years)

    print(f"  States selected ({len(selected_states)}): {selected_states}")
    print(f"  Age groups selected ({len(selected_age_groups)}): {selected_age_groups}")
    print(f"  Target years: {sorted(target_years_set)}")

    # Build lookup for fast access: (state, age_group, year) -> population
    lookup: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (row["State"], row["Age_Group"], row["Year"])
        lookup[key] = row["Population_Thousands"]

    triples: list[dict] = []
    row_index = 0

    for state in selected_states:
        for age_group in selected_age_groups:
            for year in sorted(target_years_set):
                population = lookup.get((state, age_group, year))
                if population is None or population.strip() == "":
                    continue

                subject = f"{state} - {age_group}"
                triples.append({
                    "source_file": "synthetic_us_census_population_by_demographics.csv",
                    "row_index": row_index,
                    "triple": {
                        "subject": subject,
                        "subject_type": "State_AgeGroup",
                        "relation": "HAS_POPULATION",
                        "object": population.strip(),
                        "object_type": "StatValue",
                        "attributes": {
                            "year": year,
                            "state": state,
                            "age_group": age_group,
                            "unit": "thousands",
                            "domain": "demographics",
                        },
                    },
                    "annotator": "gold_new_datasets",
                    "confidence": "high",
                    "notes": (
                        f"Generated from long-format CSV row for {subject}, year {year}"
                    ),
                })
                row_index += 1

    return _write_jsonl(triples, output_path, "US Census Demographics")


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
        return

    # Parse first and last lines for format validation
    try:
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        assert "triple" in first
        assert "triple" in last
        assert first["triple"]["object_type"] == "StatValue"
        print(f"  [OK] Verified {n_facts} facts in {path.name}")
        print(f"       First: {first['triple']['subject']} | "
              f"{first['triple']['relation']} | {first['triple']['object']} "
              f"(year={first['triple']['attributes'].get('year', '?')})")
        print(f"       Last:  {last['triple']['subject']} | "
              f"{last['triple']['relation']} | {last['triple']['object']} "
              f"(year={last['triple']['attributes'].get('year', '?')})")
    except (json.JSONDecodeError, KeyError, AssertionError) as exc:
        print(f"  [ERROR] Format validation failed: {exc}", file=sys.stderr)


def main() -> None:
    print("=" * 62)
    print("GOLD STANDARD GENERATION — New OOD Datasets")
    print("=" * 62)

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Dataset A] Eurostat Crime Statistics (long format, Type-III)")
    print(f"  Source: {DATASET_DIR / 'synthetic_eurostat_crime_statistics.csv'}")
    n_crime = gen_eurostat_crime(
        n_categories=10,  # will use all 7 if fewer available
        n_countries=5,
        target_years=("2015", "2018", "2021"),
    )
    _verify_output(GOLD_DIR / "gold_eurostat_crime.jsonl", n_crime)

    print("\n[Dataset B] US Census Population by Demographics (long format, Type-III)")
    print(f"  Source: {DATASET_DIR / 'synthetic_us_census_population_by_demographics.csv'}")
    n_census = gen_us_census_demographics(
        n_states=10,
        n_age_groups=3,
        target_years=("2012", "2016", "2020"),
    )
    _verify_output(GOLD_DIR / "gold_us_census_demographics.jsonl", n_census)

    print("\n" + "=" * 62)
    print(f"SUMMARY")
    print(f"  Eurostat Crime:        {n_crime} facts")
    print(f"  US Census Demographics:{n_census} facts")
    print(f"  Total:                 {n_crime + n_census} facts")
    print(f"  Output directory:      {GOLD_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
