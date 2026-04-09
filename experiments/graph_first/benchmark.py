#!/usr/bin/env python3
"""
benchmark.py — Generate 150+ natural statistical questions across all 7 main datasets.

Datasets:
  WHO Life Expectancy, WB Child Mortality, WB Population, WB Maternal Mortality,
  HK Inpatient, Fortune 500 Revenue, THE University Ranking

Question types: lookup, comparison, ranking, trend, aggregation
Questions are natural language (as a real user would ask), not metadata-driven.
Reference answers are computed deterministically from gold standards.

Output: experiments/graph_first/benchmark_statistical.jsonl

Usage:
    python3 experiments/graph_first/benchmark.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "graph_first" / "benchmark_statistical.jsonl"
SEED = 42


# ---------------------------------------------------------------------------
# Gold data loaders
# ---------------------------------------------------------------------------

def load_type_ii_gold(gold_path: Path) -> tuple[dict, dict]:
    """Load Type-II gold into ({entity: {year: value}}, {entity: display_name})."""
    data: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["triple"]
            subject = t["subject"]
            year = t["attributes"]["year"]
            value = float(t["object"])
            display = t["attributes"].get("country_name", subject)
            if subject not in data:
                data[subject] = {}
                names[subject] = display
            data[subject][year] = value
    return data, names


def load_type_iii_gold(gold_path: Path) -> tuple[dict, dict]:
    """Load Type-III gold into ({entity: {relation: value}}, {entity: icd_code})."""
    data: dict[str, dict[str, float]] = {}
    icd_codes: dict[str, str] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["triple"]
            entity = t["subject"]
            relation = t["relation"]
            value = float(t["object"])
            icd = t["attributes"].get("icd_code", "")
            if entity not in data:
                data[entity] = {}
                icd_codes[entity] = icd
            data[entity][relation] = value
    return data, icd_codes


# ---------------------------------------------------------------------------
# Type-II question generators (country/company/university × year)
# ---------------------------------------------------------------------------

def _gen_lookup(data, names, dataset, metric_phrase, unit_phrase, rng, n=6):
    """Point lookup: 'What was X's [metric] in [year]?'"""
    all_pairs = [
        (code, year, val)
        for code, years in data.items()
        for year, val in years.items()
    ]
    selected = rng.sample(all_pairs, min(n, len(all_pairs)))
    questions = []
    for i, (code, year, val) in enumerate(selected, 1):
        name = names.get(code, code)
        questions.append({
            "id": f"gf_{dataset}_lookup_{i:02d}",
            "dataset": dataset,
            "query_type": "lookup",
            "question": f"What was {name}'s {metric_phrase} in {year}?",
            "reference_answer": str(val),
            "reference_numeric": val,
            "reference_set": None,
            "evaluation_type": "numeric_tolerance",
            "language": "en",
            "metadata": {"entity": code, "year": year},
        })
    return questions


def _gen_comparison(data, names, dataset, metric_phrase, unit_phrase, rng, n=4):
    """Comparison: 'Which had higher [metric] in [year], A or B?'"""
    codes = list(data.keys())
    all_years = sorted({y for yrs in data.values() for y in yrs})
    questions = []
    for i in range(min(n, len(codes) // 2)):
        c1, c2 = rng.sample(codes, 2)
        year = rng.choice(all_years)
        if year not in data[c1] or year not in data[c2]:
            continue
        v1, v2 = data[c1][year], data[c2][year]
        winner = names.get(c1, c1) if v1 >= v2 else names.get(c2, c2)
        questions.append({
            "id": f"gf_{dataset}_comparison_{i + 1:02d}",
            "dataset": dataset,
            "query_type": "comparison",
            "question": (
                f"Which had a higher {metric_phrase} in {year}: "
                f"{names.get(c1, c1)} or {names.get(c2, c2)}?"
            ),
            "reference_answer": winner,
            "reference_numeric": None,
            "reference_set": [c1 if v1 >= v2 else c2],
            "evaluation_type": "exact_match",
            "language": "en",
            "metadata": {
                "entity_a": c1, "entity_b": c2,
                "year": year, "val_a": v1, "val_b": v2,
            },
        })
    return questions


def _gen_ranking(data, names, dataset, metric_phrase, unit_phrase, rng, n=5):
    """Ranking: 'Which [k] entities had the highest/lowest [metric] in [year]?'"""
    all_years = sorted({y for yrs in data.values() for y in yrs})
    k_values = [3, 5]
    questions = []
    combos = [(y, k, direction)
              for y in all_years
              for k in k_values
              for direction in ["highest", "lowest"]]
    selected = rng.sample(combos, min(n, len(combos)))

    for i, (year, k, direction) in enumerate(selected, 1):
        valid = [(code, data[code][year]) for code in data if year in data[code]]
        ranked = sorted(valid, key=lambda x: x[1], reverse=(direction == "highest"))
        top_k = ranked[:k]
        top_names = [names.get(c, c) for c, _ in top_k]
        top_codes = [c for c, _ in top_k]
        questions.append({
            "id": f"gf_{dataset}_ranking_{i:02d}",
            "dataset": dataset,
            "query_type": "ranking",
            "question": (
                f"Which {k} {'countries' if dataset not in ('fortune500', 'the') else 'entities'} "
                f"had the {direction} {metric_phrase} in {year}? List them in order."
            ),
            "reference_answer": ", ".join(top_names),
            "reference_numeric": None,
            "reference_set": top_codes,
            "evaluation_type": "set_overlap",
            "language": "en",
            "metadata": {
                "year": year, "k": k, "direction": direction,
                "ranking": [{"code": c, "name": names.get(c, c), "value": v}
                            for c, v in top_k],
            },
        })
    return questions


def _gen_trend(data, names, dataset, metric_phrase, unit_phrase, rng, n=4):
    """Trend: 'Did X's [metric] consistently increase/decrease from Y1 to Y2?'"""
    codes = list(data.keys())
    selected = rng.sample(codes, min(n, len(codes)))
    questions = []
    for i, code in enumerate(selected, 1):
        years_sorted = sorted(data[code].keys())
        if len(years_sorted) < 2:
            continue
        vals = [data[code][y] for y in years_sorted]
        y1, y2 = years_sorted[0], years_sorted[-1]
        is_inc = all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))
        is_dec = all(vals[j] >= vals[j + 1] for j in range(len(vals) - 1))
        if is_inc:
            direction, answer = "increase", "Yes"
        elif is_dec:
            direction, answer = "decrease", "Yes"
        else:
            direction = rng.choice(["increase", "decrease"])
            answer = "No"
        name = names.get(code, code)
        questions.append({
            "id": f"gf_{dataset}_trend_{i:02d}",
            "dataset": dataset,
            "query_type": "trend",
            "question": (
                f"Did {name}'s {metric_phrase} consistently {direction} "
                f"from {y1} to {y2}?"
            ),
            "reference_answer": answer,
            "reference_numeric": None,
            "reference_set": None,
            "evaluation_type": "direction",
            "language": "en",
            "metadata": {
                "entity": code, "direction": direction,
                "y1": y1, "y2": y2,
                "values": {y: v for y, v in zip(years_sorted, vals)},
            },
        })
    return questions


def _gen_aggregation(data, names, dataset, metric_phrase, unit_phrase, rng, n=5):
    """Aggregation: average, count above threshold, max change, range."""
    all_years = sorted({y for yrs in data.values() for y in yrs})
    questions = []
    idx = 1

    # Average across all entities
    for year in rng.sample(all_years, min(2, len(all_years))):
        vals = [data[c][year] for c in data if year in data[c]]
        avg = round(sum(vals) / len(vals), 2)
        questions.append({
            "id": f"gf_{dataset}_aggregation_{idx:02d}",
            "dataset": dataset,
            "query_type": "aggregation",
            "question": f"What was the average {metric_phrase} across all {len(vals)} entities in {year}?",
            "reference_answer": str(avg),
            "reference_numeric": avg,
            "reference_set": None,
            "evaluation_type": "numeric_tolerance",
            "language": "en",
            "metadata": {"year": year, "count": len(vals), "subtype": "average"},
        })
        idx += 1

    # Count above threshold
    year = rng.choice(all_years)
    vals = [data[c][year] for c in data if year in data[c]]
    median = sorted(vals)[len(vals) // 2]
    threshold = round(median, 0)
    count = sum(1 for v in vals if v > threshold)
    questions.append({
        "id": f"gf_{dataset}_aggregation_{idx:02d}",
        "dataset": dataset,
        "query_type": "aggregation",
        "question": (
            f"How many entities had {metric_phrase} above {threshold} "
            f"{unit_phrase} in {year}?"
        ),
        "reference_answer": str(count),
        "reference_numeric": float(count),
        "reference_set": None,
        "evaluation_type": "numeric_tolerance",
        "language": "en",
        "metadata": {"year": year, "threshold": threshold, "subtype": "count_above"},
    })
    idx += 1

    # Largest increase
    if len(all_years) >= 2:
        y1, y2 = all_years[0], all_years[-1]
        changes = [
            (code, data[code][y2] - data[code][y1])
            for code in data
            if y1 in data[code] and y2 in data[code]
        ]
        if changes:
            changes.sort(key=lambda x: x[1], reverse=True)
            best_code, _ = changes[0]
            best_name = names.get(best_code, best_code)
            questions.append({
                "id": f"gf_{dataset}_aggregation_{idx:02d}",
                "dataset": dataset,
                "query_type": "aggregation",
                "question": (
                    f"Which entity had the largest increase in {metric_phrase} "
                    f"from {y1} to {y2}?"
                ),
                "reference_answer": best_name,
                "reference_numeric": None,
                "reference_set": [best_code],
                "evaluation_type": "exact_match",
                "language": "en",
                "metadata": {
                    "y1": y1, "y2": y2, "best_code": best_code,
                    "subtype": "max_increase",
                },
            })
            idx += 1

    return questions[:n]


# ---------------------------------------------------------------------------
# Type-III question generators (Inpatient HK diseases)
# ---------------------------------------------------------------------------

_REL_LABELS = {
    "INPATIENT_TOTAL": "住院病人出院及死亡总人次",
    "REGISTERED_DEATHS": "在医院登记死亡人数",
    "INPATIENT_HA_HOSPITAL": "医院管理局辖下医院住院人次",
}


def _gen_inpatient_lookup(data, icd_codes, rng, n=6):
    """Lookup: single disease × single relation."""
    pairs = [
        (entity, rel, val)
        for entity, rels in data.items()
        for rel, val in rels.items()
    ]
    selected = rng.sample(pairs, min(n, len(pairs)))
    questions = []
    for i, (entity, rel, val) in enumerate(selected, 1):
        label = _REL_LABELS.get(rel, rel)
        questions.append({
            "id": f"gf_inpatient_lookup_{i:02d}",
            "dataset": "inpatient",
            "query_type": "lookup",
            "question": f"{entity}的{label}是多少？",
            "reference_answer": str(int(val)),
            "reference_numeric": val,
            "reference_set": None,
            "evaluation_type": "numeric_tolerance",
            "language": "zh",
            "metadata": {"entity": entity, "relation": rel, "icd": icd_codes.get(entity, "")},
        })
    return questions


def _gen_inpatient_comparison(data, icd_codes, rng, n=4):
    """Comparison: A vs B on total inpatients."""
    entities = list(data.keys())
    rel = "INPATIENT_TOTAL"
    label = _REL_LABELS[rel]
    questions = []
    for i in range(min(n, len(entities) // 2)):
        e1, e2 = rng.sample(entities, 2)
        if rel not in data[e1] or rel not in data[e2]:
            continue
        v1, v2 = data[e1][rel], data[e2][rel]
        winner = e1 if v1 >= v2 else e2
        questions.append({
            "id": f"gf_inpatient_comparison_{i + 1:02d}",
            "dataset": "inpatient",
            "query_type": "comparison",
            "question": f"{e1}和{e2}哪种疾病的{label}更多？",
            "reference_answer": winner,
            "reference_numeric": None,
            "reference_set": [winner],
            "evaluation_type": "exact_match",
            "language": "zh",
            "metadata": {"entity_a": e1, "entity_b": e2, "val_a": v1, "val_b": v2},
        })
    return questions


def _gen_inpatient_ranking(data, icd_codes, rng, n=4):
    """Ranking: top K diseases by relation."""
    questions = []
    idx = 1
    for rel, label in [("INPATIENT_TOTAL", _REL_LABELS["INPATIENT_TOTAL"]),
                        ("REGISTERED_DEATHS", _REL_LABELS["REGISTERED_DEATHS"])]:
        valid = [(e, data[e][rel]) for e in data if rel in data[e]]
        for k in [3, 5]:
            if len(valid) < k:
                continue
            ranked_desc = sorted(valid, key=lambda x: x[1], reverse=True)[:k]
            top_names = [e for e, _ in ranked_desc]
            questions.append({
                "id": f"gf_inpatient_ranking_{idx:02d}",
                "dataset": "inpatient",
                "query_type": "ranking",
                "question": f"{label}最多的前{k}种疾病是哪些？请按顺序列出。",
                "reference_answer": "、".join(top_names),
                "reference_numeric": None,
                "reference_set": top_names,
                "evaluation_type": "set_overlap",
                "language": "zh",
                "metadata": {
                    "relation": rel, "k": k,
                    "ranking": [{"entity": e, "value": int(v)} for e, v in ranked_desc],
                },
            })
            idx += 1
    return questions[:n]


def _gen_inpatient_aggregation(data, icd_codes, rng, n=5):
    """Aggregation: sum, average, mortality rate, count above threshold."""
    questions = []
    idx = 1

    # Total sum
    for rel, label in [("INPATIENT_TOTAL", _REL_LABELS["INPATIENT_TOTAL"]),
                        ("REGISTERED_DEATHS", _REL_LABELS["REGISTERED_DEATHS"])]:
        vals = [data[e][rel] for e in data if rel in data[e]]
        total = sum(vals)
        questions.append({
            "id": f"gf_inpatient_aggregation_{idx:02d}",
            "dataset": "inpatient",
            "query_type": "aggregation",
            "question": f"所有{len(vals)}种疾病的{label}合计是多少？",
            "reference_answer": str(int(total)),
            "reference_numeric": float(total),
            "reference_set": None,
            "evaluation_type": "numeric_tolerance",
            "language": "zh",
            "metadata": {"relation": rel, "subtype": "sum"},
        })
        idx += 1

    # Highest mortality rate
    mortality = []
    for e in data:
        t = data[e].get("INPATIENT_TOTAL")
        d = data[e].get("REGISTERED_DEATHS")
        if t and d and t > 0:
            rate = round(d / t * 100, 2)
            mortality.append((e, rate))

    if mortality:
        mortality.sort(key=lambda x: x[1], reverse=True)
        best_e = mortality[0][0]
        questions.append({
            "id": f"gf_inpatient_aggregation_{idx:02d}",
            "dataset": "inpatient",
            "query_type": "aggregation",
            "question": "哪种疾病的死亡率（登记死亡÷住院总人次）最高？",
            "reference_answer": best_e,
            "reference_numeric": mortality[0][1],
            "reference_set": [best_e],
            "evaluation_type": "exact_match",
            "language": "zh",
            "metadata": {
                "subtype": "max_mortality_rate",
                "ranking": [{"entity": e, "rate": r} for e, r in mortality],
            },
        })
        idx += 1

    return questions[:n]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

_DATASET_CONFIGS = {
    "who": {
        "gold": GOLD_DIR / "gold_who_life_expectancy_v2.jsonl",
        "type": "II",
        "metric": "life expectancy at birth",
        "unit": "years",
        "entity_label": "countries",
    },
    "wb_cm": {
        "gold": GOLD_DIR / "gold_wb_child_mortality_v2.jsonl",
        "type": "II",
        "metric": "under-5 child mortality rate",
        "unit": "per 1,000 live births",
        "entity_label": "countries",
    },
    "wb_pop": {
        "gold": GOLD_DIR / "gold_wb_population_v2.jsonl",
        "type": "II",
        "metric": "total population",
        "unit": "persons",
        "entity_label": "countries",
    },
    "wb_mat": {
        "gold": GOLD_DIR / "gold_wb_maternal_v2.jsonl",
        "type": "II",
        "metric": "maternal mortality rate",
        "unit": "per 100,000 live births",
        "entity_label": "countries",
    },
    "fortune500": {
        "gold": GOLD_DIR / "gold_fortune500_revenue.jsonl",
        "type": "II",
        "metric": "annual revenue",
        "unit": "USD millions",
        "entity_label": "companies",
    },
    "the": {
        "gold": GOLD_DIR / "gold_the_university_ranking.jsonl",
        "type": "II",
        "metric": "THE ranking score",
        "unit": "score (0-100)",
        "entity_label": "universities",
    },
}


def generate_benchmark() -> list[dict]:
    """Generate the full benchmark across all 7 datasets."""
    rng = random.Random(SEED)
    all_questions: list[dict] = []

    # Type-II datasets (6 datasets)
    for dataset, cfg in _DATASET_CONFIGS.items():
        data, names = load_type_ii_gold(cfg["gold"])
        metric = cfg["metric"]
        unit = cfg["unit"]

        all_questions.extend(_gen_lookup(data, names, dataset, metric, unit, rng, n=6))
        all_questions.extend(_gen_comparison(data, names, dataset, metric, unit, rng, n=4))
        all_questions.extend(_gen_ranking(data, names, dataset, metric, unit, rng, n=5))
        all_questions.extend(_gen_trend(data, names, dataset, metric, unit, rng, n=4))
        all_questions.extend(_gen_aggregation(data, names, dataset, metric, unit, rng, n=5))

    # Type-III: HK Inpatient (Chinese questions)
    inp_data, inp_icd = load_type_iii_gold(GOLD_DIR / "gold_inpatient_2023.jsonl")
    all_questions.extend(_gen_inpatient_lookup(inp_data, inp_icd, rng, n=6))
    all_questions.extend(_gen_inpatient_comparison(inp_data, inp_icd, rng, n=4))
    all_questions.extend(_gen_inpatient_ranking(inp_data, inp_icd, rng, n=4))
    all_questions.extend(_gen_inpatient_aggregation(inp_data, inp_icd, rng, n=5))

    return all_questions


def main():
    questions = generate_benchmark()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    by_dataset: dict = defaultdict(lambda: defaultdict(int))
    for q in questions:
        by_dataset[q["dataset"]][q["query_type"]] += 1

    print(f"Generated {len(questions)} questions → {OUTPUT_PATH}")
    print()
    for ds in sorted(by_dataset):
        types = by_dataset[ds]
        total = sum(types.values())
        parts = " | ".join(f"{qt}: {types[qt]}" for qt in sorted(types))
        print(f"  {ds:15s} — {parts} | Total: {total}")


if __name__ == "__main__":
    main()
