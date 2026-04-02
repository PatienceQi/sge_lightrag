#!/usr/bin/env python3
"""
generate_full_gold.py — Generate full-scale WHO Gold Standard (all ~190 countries).

For Track B: large-scale experiment to counter "just concatenate all" argument.
Uses same 6 years as v2 (2000, 2005, 2010, 2015, 2019, 2021).

Usage:
    python3 experiments/ggcr/generate_full_gold.py
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WHO_CSV = PROJECT_ROOT / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"
OUTPUT_GOLD = PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_full.jsonl"
OUTPUT_BENCHMARK = PROJECT_ROOT / "evaluation" / "gold" / "ggcr_benchmark_full.jsonl"

YEARS = ["2000", "2005", "2010", "2015", "2019", "2021"]


def read_who_csv():
    """Read WHO CSV → {country_code: {name, years: {year: value}}}."""
    countries = {}
    with open(WHO_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("Country Code", "").strip()
            name = row.get("Country Name", "").strip()
            if not code or len(code) != 3:
                continue
            years_data = {}
            for yr in YEARS:
                val = row.get(yr, "").strip()
                if val:
                    try:
                        years_data[yr] = round(float(val), 2)
                    except ValueError:
                        continue
            if years_data:
                countries[code] = {"name": name, "years": years_data}
    return countries


def generate_gold(countries):
    """Generate Gold Standard JSONL for all countries."""
    triples = []
    for code, info in sorted(countries.items()):
        for yr, val in sorted(info["years"].items()):
            triples.append({
                "source_file": "API_WHO_WHOSIS_000001_life_expectancy.csv",
                "row_index": 0,
                "triple": {
                    "subject": code,
                    "subject_type": "Country_Code",
                    "relation": "LIFE_EXPECTANCY",
                    "object": str(val),
                    "object_type": "StatValue",
                    "attributes": {
                        "year": yr,
                        "unit": "years",
                        "indicator": "WHOSIS_000001",
                        "country_name": info["name"],
                    },
                },
                "annotator": "gold_full",
                "confidence": "high",
            })

    with open(OUTPUT_GOLD, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Generated {len(triples)} facts for {len(countries)} countries → {OUTPUT_GOLD}")
    return countries


def generate_benchmark(countries):
    """Generate L3/L4 benchmark questions for full-scale experiment (~20 questions)."""
    import random
    rng = random.Random(42)

    questions = []
    idx = 1

    # L3: Top-K ranking across ALL countries (10 questions)
    for yr in YEARS[:5]:  # 5 years
        for k in [5, 10]:  # 2 K values
            ranked = sorted(
                [(c, countries[c]["years"].get(yr, 0))
                 for c in countries if yr in countries[c]["years"]],
                key=lambda x: x[1], reverse=True,
            )
            top_k = ranked[:k]
            top_names = [countries[c]["name"] for c, _ in top_k]
            top_codes = [c for c, _ in top_k]

            questions.append({
                "id": f"ggcr_who_full_L3_{idx:02d}",
                "dataset": "who_full",
                "level": "L3",
                "category": "cross_entity_ranking",
                "question": (
                    f"Among all {len(ranked)} countries, which {k} had the highest "
                    f"life expectancy at birth in {yr}? List them in order."
                ),
                "reference_answer": ", ".join(top_names),
                "reference_set": top_codes,
                "reference_numeric": None,
                "evaluation_type": "set_overlap",
                "entities_required": len(ranked),
                "language": "en",
                "metadata": {
                    "year": yr, "k": k, "total_countries": len(ranked),
                    "ranking": [{"code": c, "name": countries[c]["name"], "value": v}
                                for c, v in top_k],
                },
            })
            idx += 1

    # L4: Aggregation over ALL countries (10 questions)
    idx = 1

    # Count above threshold (4 questions)
    for yr in rng.sample(YEARS, 4):
        values = [countries[c]["years"][yr] for c in countries if yr in countries[c]["years"]]
        threshold = round(sorted(values)[len(values) // 2], 0)  # ~median
        count = sum(1 for v in values if v > threshold)

        questions.append({
            "id": f"ggcr_who_full_L4_{idx:02d}",
            "dataset": "who_full",
            "level": "L4",
            "category": "cross_entity_count",
            "question": (
                f"Among all countries with data, how many had life expectancy "
                f"above {threshold} years in {yr}?"
            ),
            "reference_answer": str(count),
            "reference_set": None,
            "reference_numeric": float(count),
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(values),
            "language": "en",
            "metadata": {"year": yr, "threshold": threshold, "total": len(values)},
        })
        idx += 1

    # Average across all countries (3 questions)
    for yr in rng.sample(YEARS, 3):
        values = [countries[c]["years"][yr] for c in countries if yr in countries[c]["years"]]
        avg = round(sum(values) / len(values), 2)

        questions.append({
            "id": f"ggcr_who_full_L4_{idx:02d}",
            "dataset": "who_full",
            "level": "L4",
            "category": "cross_entity_average",
            "question": (
                f"What was the average life expectancy at birth across all "
                f"{len(values)} countries in {yr}?"
            ),
            "reference_answer": str(avg),
            "reference_set": None,
            "reference_numeric": avg,
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(values),
            "language": "en",
            "metadata": {"year": yr, "count": len(values)},
        })
        idx += 1

    # Largest increase (3 questions)
    year_pairs = [("2000", "2021"), ("2005", "2019"), ("2000", "2015")]
    for y1, y2 in year_pairs:
        changes = []
        for c in countries:
            if y1 in countries[c]["years"] and y2 in countries[c]["years"]:
                delta = countries[c]["years"][y2] - countries[c]["years"][y1]
                changes.append((c, delta))
        changes.sort(key=lambda x: x[1], reverse=True)
        best_code, best_delta = changes[0]

        questions.append({
            "id": f"ggcr_who_full_L4_{idx:02d}",
            "dataset": "who_full",
            "level": "L4",
            "category": "cross_entity_max_change",
            "question": (
                f"Which country had the largest increase in life expectancy "
                f"from {y1} to {y2}?"
            ),
            "reference_answer": countries[best_code]["name"],
            "reference_set": [best_code],
            "reference_numeric": round(best_delta, 2),
            "evaluation_type": "exact_match",
            "entities_required": len(changes),
            "language": "en",
            "metadata": {
                "year_from": y1, "year_to": y2,
                "best_code": best_code, "best_delta": round(best_delta, 2),
                "top3": [{"code": c, "name": countries[c]["name"], "delta": round(d, 2)}
                         for c, d in changes[:3]],
            },
        })
        idx += 1

    with open(OUTPUT_BENCHMARK, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    l3_count = sum(1 for q in questions if q["level"] == "L3")
    l4_count = sum(1 for q in questions if q["level"] == "L4")
    print(f"Generated {len(questions)} full-scale benchmark questions (L3: {l3_count}, L4: {l4_count}) → {OUTPUT_BENCHMARK}")


if __name__ == "__main__":
    countries = read_who_csv()
    print(f"Read {len(countries)} countries from WHO CSV")
    generate_gold(countries)
    generate_benchmark(countries)
