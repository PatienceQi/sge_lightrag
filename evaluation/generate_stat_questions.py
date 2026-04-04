"""
Generate 200+ statistical analysis questions from gold standard JSONL files.

Questions test whether knowledge graph construction preserves factual fidelity
across six question categories (L1-L4 difficulty levels).

Usage:
    python evaluation/generate_stat_questions.py
"""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

random.seed(42)

GOLD_DIR = Path(__file__).parent / "gold"
OUTPUT_FILE = GOLD_DIR / "stat_analysis_questions_200.jsonl"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_lookup(records: list[dict]) -> dict:
    """Return {(subject, year): value} from a flat JSONL record list."""
    lookup = {}
    for rec in records:
        triple = rec["triple"]
        subj = triple.get("attributes", {}).get("country_name") or triple["subject"]
        year = triple["attributes"]["year"]
        lookup[(subj, year)] = triple["object"]
    return lookup


def build_year_index(records: list[dict]) -> dict[str, dict[str, str]]:
    """Return {year: {subject: value}}."""
    index: dict[str, dict[str, str]] = defaultdict(dict)
    for rec in records:
        triple = rec["triple"]
        subj = triple.get("attributes", {}).get("country_name") or triple["subject"]
        year = triple["attributes"]["year"]
        index[year][subj] = triple["object"]
    return dict(index)


# ---------------------------------------------------------------------------
# Question generators — each returns a list of question dicts
# ---------------------------------------------------------------------------

def make_point_lookup(
    dataset_name: str,
    metric_label: str,
    unit_hint: str,
    year_index: dict,
    lookup: dict,
    n: int,
    id_counter: list,
) -> list[dict]:
    """L1: direct single-value retrieval."""
    questions = []
    pairs = list(lookup.keys())
    random.shuffle(pairs)
    for subj, year in pairs[:n]:
        value = lookup[(subj, year)]
        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "point_lookup",
            "level": "L1",
            "question": f"What was {subj}'s {metric_label} in {year}?",
            "answer": str(value),
            "answer_entities": [subj],
            "answer_years": [year],
            "requires_computation": False,
        })
    return questions


def make_ranking(
    dataset_name: str,
    metric_label: str,
    year_index: dict,
    n_questions: int,
    id_counter: list,
) -> list[dict]:
    """L2: rank countries by metric value for a given year."""
    questions = []
    years = sorted(year_index.keys())
    configs = [
        ("top", 5, "highest"),
        ("top", 3, "highest"),
        ("bottom", 3, "lowest"),
        ("bottom", 5, "lowest"),
    ]
    random.shuffle(years)

    config_cycle = list(configs) * (n_questions // len(configs) + 1)
    random.shuffle(config_cycle)

    used = 0
    for year in years * 2:
        if used >= n_questions:
            break
        direction, k, direction_word = config_cycle[used % len(config_cycle)]
        year_data = year_index[year]
        if len(year_data) < k:
            continue

        sorted_countries = sorted(
            year_data.items(),
            key=lambda x: float(x[1]),
            reverse=(direction == "top"),
        )
        top_k = [c for c, _ in sorted_countries[:k]]
        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "ranking",
            "level": "L2",
            "question": (
                f"Which {k} countries had the {direction_word} {metric_label} in {year}?"
            ),
            "answer": json.dumps(top_k),
            "answer_entities": top_k,
            "answer_years": [year],
            "requires_computation": True,
        })
        used += 1
    return questions


def make_filtering(
    dataset_name: str,
    metric_label: str,
    year_index: dict,
    n_questions: int,
    id_counter: list,
) -> list[dict]:
    """L3: filter countries that exceed / fall below a threshold in a given year."""
    questions = []
    years = sorted(year_index.keys())

    for year in years:
        if len(questions) >= n_questions:
            break
        values = [(c, float(v)) for c, v in year_index[year].items()]
        sorted_vals = sorted(values, key=lambda x: x[1])
        n = len(sorted_vals)
        if n < 6:
            continue

        # choose thresholds that yield 3-8 results
        # above threshold: use ~30th percentile value
        idx_low = max(1, int(n * 0.30))
        threshold_above = sorted_vals[idx_low][1]
        above = [c for c, v in values if v > threshold_above]
        if 3 <= len(above) <= 8:
            qid = id_counter[0]
            id_counter[0] += 1
            questions.append({
                "id": qid,
                "dataset": dataset_name,
                "category": "filtering",
                "level": "L3",
                "question": (
                    f"Which countries had a {metric_label} greater than "
                    f"{threshold_above} in {year}?"
                ),
                "answer": json.dumps(sorted(above)),
                "answer_entities": sorted(above),
                "answer_years": [year],
                "requires_computation": True,
            })

        if len(questions) >= n_questions:
            break

        # below threshold: use ~70th percentile value
        idx_high = min(n - 2, int(n * 0.70))
        threshold_below = sorted_vals[idx_high][1]
        below = [c for c, v in values if v < threshold_below]
        if 3 <= len(below) <= 8:
            qid = id_counter[0]
            id_counter[0] += 1
            questions.append({
                "id": qid,
                "dataset": dataset_name,
                "category": "filtering",
                "level": "L3",
                "question": (
                    f"Which countries had a {metric_label} less than "
                    f"{threshold_below} in {year}?"
                ),
                "answer": json.dumps(sorted(below)),
                "answer_entities": sorted(below),
                "answer_years": [year],
                "requires_computation": True,
            })

    return questions[:n_questions]


def make_aggregation(
    dataset_name: str,
    metric_label: str,
    year_index: dict,
    n_questions: int,
    id_counter: list,
) -> list[dict]:
    """L3: compute mean/median/min/max across countries for a given year."""
    questions = []
    years = sorted(year_index.keys())
    agg_configs = [
        ("average", "mean"),
        ("median", "median"),
        ("minimum", "min"),
        ("maximum", "max"),
    ]

    config_cycle = list(agg_configs) * (n_questions // len(agg_configs) + 1)
    idx = 0
    for year in years * 3:
        if len(questions) >= n_questions:
            break
        agg_word, agg_fn = config_cycle[idx % len(config_cycle)]
        idx += 1
        values = [float(v) for v in year_index[year].values()]
        if not values:
            continue

        if agg_fn == "mean":
            result = round(statistics.mean(values), 4)
        elif agg_fn == "median":
            result = round(statistics.median(values), 4)
        elif agg_fn == "min":
            result = round(min(values), 4)
        else:
            result = round(max(values), 4)

        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "aggregation",
            "level": "L3",
            "question": (
                f"What was the {agg_word} {metric_label} across all countries in {year}?"
            ),
            "answer": str(result),
            "answer_entities": [],
            "answer_years": [year],
            "requires_computation": True,
        })
    return questions[:n_questions]


def make_trend(
    dataset_name: str,
    metric_label: str,
    year_index: dict,
    lookup: dict,
    n_questions: int,
    id_counter: list,
) -> list[dict]:
    """L4: trend analysis — largest change between two years, or direction for one country."""
    questions = []
    years = sorted(year_index.keys())
    year_pairs = [(years[i], years[j]) for i in range(len(years)) for j in range(i + 1, len(years))]
    random.shuffle(year_pairs)

    used = 0
    for year1, year2 in year_pairs:
        if used >= n_questions:
            break
        # countries present in both years
        common = {
            c for c in year_index[year1] if c in year_index[year2]
        }
        if len(common) < 5:
            continue

        # largest-increase question
        deltas = {
            c: float(year_index[year2][c]) - float(year_index[year1][c])
            for c in common
        }
        top3_increase = sorted(deltas, key=lambda c: deltas[c], reverse=True)[:3]
        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "trend",
            "level": "L4",
            "question": (
                f"Which 3 countries showed the largest increase in "
                f"{metric_label} from {year1} to {year2}?"
            ),
            "answer": json.dumps(top3_increase),
            "answer_entities": top3_increase,
            "answer_years": [year1, year2],
            "requires_computation": True,
        })
        used += 1

        if used >= n_questions:
            break

        # direction question for a random country
        country = random.choice(list(common))
        v1 = float(year_index[year1][country])
        v2 = float(year_index[year2][country])
        delta = round(v2 - v1, 4)
        direction = "increase" if delta > 0 else ("decrease" if delta < 0 else "no change")

        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "trend",
            "level": "L4",
            "question": (
                f"Did {country}'s {metric_label} increase or decrease "
                f"from {year1} to {year2}?"
            ),
            "answer": json.dumps({"direction": direction, "delta": delta}),
            "answer_entities": [country],
            "answer_years": [year1, year2],
            "requires_computation": True,
        })
        used += 1

    return questions[:n_questions]


def make_comparison(
    dataset_name: str,
    metric_label: str,
    year_index: dict,
    n_questions: int,
    id_counter: list,
) -> list[dict]:
    """L4: pairwise comparison between two countries for a given year."""
    questions = []
    years = sorted(year_index.keys())
    random.shuffle(years)

    used = 0
    for year in years * 5:
        if used >= n_questions:
            break
        countries = list(year_index[year].keys())
        if len(countries) < 2:
            continue
        a, b = random.sample(countries, 2)
        va = float(year_index[year][a])
        vb = float(year_index[year][b])
        higher = a if va > vb else b
        answer = f"Yes" if va > vb else "No"

        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": dataset_name,
            "category": "comparison",
            "level": "L4",
            "question": (
                f"Was {a}'s {metric_label} higher than {b}'s in {year}?"
            ),
            "answer": json.dumps({
                "answer": answer,
                "higher_entity": higher,
                f"{a}_value": va,
                f"{b}_value": vb,
            }),
            "answer_entities": [a, b],
            "answer_years": [year],
            "requires_computation": False,
        })
        used += 1
    return questions[:n_questions]


# ---------------------------------------------------------------------------
# Inpatient dataset (cross-entity, no time dimension)
# ---------------------------------------------------------------------------

def generate_inpatient_questions(
    records: list[dict],
    id_counter: list,
) -> list[dict]:
    """Generate questions for the inpatient 2023 dataset (no year dimension)."""
    questions = []

    # Build per-relation lookups
    rel_data: dict[str, dict[str, str]] = defaultdict(dict)
    for rec in records:
        triple = rec["triple"]
        subj = triple["subject"]
        rel = triple["relation"]
        val = triple["object"]
        rel_data[rel][subj] = val

    relation_labels = {
        "INPATIENT_TOTAL": "total inpatient discharges and deaths",
        "INPATIENT_HA_HOSPITAL": "inpatient discharges at HA hospitals",
        "REGISTERED_DEATHS": "registered deaths",
    }

    for rel, label in relation_labels.items():
        entities = rel_data.get(rel, {})
        if not entities:
            continue
        values = {e: float(v) for e, v in entities.items()}
        sorted_ents = sorted(values, key=lambda e: values[e], reverse=True)

        # Point lookup — one per entity
        for entity, value in entities.items():
            qid = id_counter[0]
            id_counter[0] += 1
            questions.append({
                "id": qid,
                "dataset": "Inpatient",
                "category": "point_lookup",
                "level": "L1",
                "question": f"What was the {label} count for {entity} in Hong Kong in 2023?",
                "answer": str(value),
                "answer_entities": [entity],
                "answer_years": ["2023"],
                "requires_computation": False,
            })

        # Ranking — top 3
        top3 = sorted_ents[:3]
        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": "Inpatient",
            "category": "ranking",
            "level": "L2",
            "question": (
                f"Which 3 diseases had the highest {label} in Hong Kong in 2023?"
            ),
            "answer": json.dumps(top3),
            "answer_entities": top3,
            "answer_years": ["2023"],
            "requires_computation": True,
        })

        # Aggregation — total
        total = round(sum(values.values()), 2)
        qid = id_counter[0]
        id_counter[0] += 1
        questions.append({
            "id": qid,
            "dataset": "Inpatient",
            "category": "aggregation",
            "level": "L3",
            "question": (
                f"What was the total {label} across all listed diseases "
                f"in Hong Kong in 2023?"
            ),
            "answer": str(total),
            "answer_entities": [],
            "answer_years": ["2023"],
            "requires_computation": True,
        })

        # Pairwise comparisons
        ent_list = list(entities.keys())
        for _ in range(3):
            if len(ent_list) < 2:
                break
            a, b = random.sample(ent_list, 2)
            va, vb = values[a], values[b]
            higher = a if va > vb else b
            qid = id_counter[0]
            id_counter[0] += 1
            questions.append({
                "id": qid,
                "dataset": "Inpatient",
                "category": "comparison",
                "level": "L4",
                "question": (
                    f"Did {a} have more {label} cases than {b} "
                    f"in Hong Kong in 2023?"
                ),
                "answer": json.dumps({
                    "answer": "Yes" if va > vb else "No",
                    "higher_entity": higher,
                    f"{a}_value": va,
                    f"{b}_value": vb,
                }),
                "answer_entities": [a, b],
                "answer_years": ["2023"],
                "requires_computation": False,
            })

    return questions


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "WHO",
        "file": "gold_who_life_expectancy_v2.jsonl",
        "metric": "life expectancy",
        "unit": "years",
        "use_country_name": True,
    },
    {
        "name": "WB_ChildMortality",
        "file": "gold_wb_child_mortality_v2.jsonl",
        "metric": "under-5 mortality rate",
        "unit": "per 1,000 live births",
        "use_country_name": True,
    },
    {
        "name": "WB_Population",
        "file": "gold_wb_population_v2.jsonl",
        "metric": "total population",
        "unit": "persons",
        "use_country_name": False,
    },
    {
        "name": "WB_Maternal",
        "file": "gold_wb_maternal_v2.jsonl",
        "metric": "maternal mortality rate",
        "unit": "per 100,000 live births",
        "use_country_name": False,
    },
]

# Questions per category per dataset (international datasets)
# 4 datasets × 10 per category × 6 categories = 240 base questions
# Plus ~50 inpatient = ~290 total (well above 200)
PER_CATEGORY = 10


def generate_all_questions() -> list[dict]:
    id_counter = [1]
    all_questions: list[dict] = []

    for ds in DATASETS:
        path = GOLD_DIR / ds["file"]
        records = load_jsonl(path)
        lookup = build_lookup(records)
        year_index = build_year_index(records)

        metric = ds["metric"]
        name = ds["name"]

        all_questions.extend(
            make_point_lookup(name, metric, ds["unit"], year_index, lookup, PER_CATEGORY, id_counter)
        )
        all_questions.extend(
            make_ranking(name, metric, year_index, PER_CATEGORY, id_counter)
        )
        all_questions.extend(
            make_filtering(name, metric, year_index, PER_CATEGORY, id_counter)
        )
        all_questions.extend(
            make_aggregation(name, metric, year_index, PER_CATEGORY, id_counter)
        )
        all_questions.extend(
            make_trend(name, metric, year_index, lookup, PER_CATEGORY, id_counter)
        )
        all_questions.extend(
            make_comparison(name, metric, year_index, PER_CATEGORY, id_counter)
        )

    # Inpatient dataset
    inpatient_records = load_jsonl(GOLD_DIR / "gold_inpatient_2023.jsonl")
    all_questions.extend(generate_inpatient_questions(inpatient_records, id_counter))

    # Re-assign sequential IDs after shuffling categories
    for i, q in enumerate(all_questions, start=1):
        q["id"] = i

    return all_questions


# ---------------------------------------------------------------------------
# Output and summary
# ---------------------------------------------------------------------------

def write_output(questions: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def print_summary(questions: list[dict]) -> None:
    print(f"\nTotal questions generated: {len(questions)}")
    print()

    by_dataset: dict[str, list] = defaultdict(list)
    by_category: dict[str, list] = defaultdict(list)
    by_level: dict[str, list] = defaultdict(list)

    for q in questions:
        by_dataset[q["dataset"]].append(q)
        by_category[q["category"]].append(q)
        by_level[q["level"]].append(q)

    print("--- By dataset ---")
    for ds, qs in sorted(by_dataset.items()):
        print(f"  {ds}: {len(qs)}")

    print()
    print("--- By category ---")
    for cat, qs in sorted(by_category.items()):
        print(f"  {cat}: {len(qs)}")

    print()
    print("--- By level ---")
    for lvl, qs in sorted(by_level.items()):
        print(f"  {lvl}: {len(qs)}")

    computation_required = sum(1 for q in questions if q["requires_computation"])
    print(f"\n  Requires computation: {computation_required}")
    print(f"  Direct lookup:        {len(questions) - computation_required}")
    print(f"\nOutput written to: {OUTPUT_FILE}")


def main() -> None:
    print("Loading gold standard files and generating questions...")
    questions = generate_all_questions()
    write_output(questions, OUTPUT_FILE)
    print_summary(questions)


if __name__ == "__main__":
    main()
