#!/usr/bin/env python3
"""
Gold Standard generation for non-government datasets.

Fortune 500 Revenue (Type-II): 25 companies × 5 years = 125 facts
THE University Ranking (Type-III): 25 universities × 6 years = 150 facts

Source: github.com/cmusam/fortune500, github.com/arnaudbenard/university-ranking
"""
import csv
import json
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "non_gov"
EVAL_DIR = Path(__file__).resolve().parent
GOLD_DIR = EVAL_DIR / "gold"


def _write(triples: list[dict], output_path: Path, label: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  [{label}] wrote {len(triples)} facts → {output_path.name}")


def gen_fortune500(n_entities: int = 25):
    """Generate Gold Standard for Fortune 500 Revenue dataset.

    Structure: Company_Name, 2015, 2016, 2017, 2018, 2019
    Gold: 25 companies × 5 years = 125 facts
    """
    csv_path = DATASET_DIR / "fortune500_revenue.csv"
    output_path = GOLD_DIR / "gold_fortune500_revenue.jsonl"
    years = ["2015", "2016", "2017", "2018", "2019"]

    triples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_entities:
                break
            company = row["Company_Name"].strip()
            for yr in years:
                val = row[yr].strip()
                if not val:
                    continue
                # Remove trailing .0 for cleaner matching
                if val.endswith(".0"):
                    val = val[:-2]
                triples.append({
                    "source_file": "fortune500_revenue.csv",
                    "row_index": i,
                    "triple": {
                        "subject": company,
                        "subject_type": "Company",
                        "relation": "REVENUE",
                        "object": val,
                        "object_type": "StatValue",
                        "attributes": {
                            "year": yr,
                            "unit": "USD millions",
                            "domain": "finance",
                        },
                    },
                    "annotator": "gold_non_gov",
                    "confidence": "high",
                })

    _write(triples, output_path, "Fortune 500")
    return len(triples)


def gen_the_ranking(n_entities: int = 25):
    """Generate Gold Standard for THE University Ranking dataset.

    Structure: University_Name, Country, 2011, 2012, 2013, 2014, 2015, 2016
    Gold: 25 universities × 6 years = up to 150 facts
    """
    csv_path = DATASET_DIR / "the_university_ranking.csv"
    output_path = GOLD_DIR / "gold_the_university_ranking.jsonl"
    years = ["2011", "2012", "2013", "2014", "2015", "2016"]

    triples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_entities:
                break
            university = row["University_Name"].strip()
            for yr in years:
                val = row.get(yr, "").strip()
                if not val:
                    continue
                triples.append({
                    "source_file": "the_university_ranking.csv",
                    "row_index": i,
                    "triple": {
                        "subject": university,
                        "subject_type": "University",
                        "relation": "RANKING_SCORE",
                        "object": val,
                        "object_type": "StatValue",
                        "attributes": {
                            "year": yr,
                            "unit": "score (0-100)",
                            "domain": "academic",
                            "ranking_system": "Times Higher Education",
                        },
                    },
                    "annotator": "gold_non_gov",
                    "confidence": "high",
                })

    _write(triples, output_path, "THE Ranking")
    return len(triples)


if __name__ == "__main__":
    n1 = gen_fortune500(n_entities=25)
    n2 = gen_the_ranking(n_entities=25)
    print(f"\nTotal: {n1 + n2} facts ({n1} Fortune 500 + {n2} THE Ranking)")
