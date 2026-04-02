#!/usr/bin/env python3
"""
benchmark_generator.py — Generate L1-L4 complexity-graded benchmark for GGCR evaluation.

Deterministically generates questions from Gold Standard JSONL files (seed=42).
No LLM involved — all reference answers are computed directly from gold data.

Levels:
  L1: Single entity, single value lookup
  L2: Single entity, trend/ratio analysis
  L3: Cross-entity ranking (top-K)
  L4: Cross-entity conditional aggregation

Usage:
    python3 experiments/ggcr/benchmark_generator.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"
OUTPUT_PATH = GOLD_DIR / "ggcr_benchmark.jsonl"

SEED = 42


# ---------------------------------------------------------------------------
# Gold data loaders
# ---------------------------------------------------------------------------

def load_type_ii_gold(gold_path: str) -> dict[str, dict[str, float]]:
    """Load Type-II gold into {entity_code: {year: value}}."""
    data: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["triple"]
            code = t["subject"]
            year = t["attributes"]["year"]
            value = float(t["object"])
            name = t["attributes"].get("country_name", code)
            if code not in data:
                data[code] = {}
                names[code] = name
            data[code][year] = value
    return data, names


def load_type_iii_gold(gold_path: str) -> dict[str, dict[str, float]]:
    """Load Type-III gold into {entity: {relation: value}}."""
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
# Type-II question generators (WHO, WB_CM)
# ---------------------------------------------------------------------------

def gen_type_ii_l1(data, names, dataset, relation, unit, rng, n=10):
    """L1: Single entity, single value lookup."""
    all_pairs = [(code, year, val)
                 for code, years in data.items()
                 for year, val in years.items()]
    selected = rng.sample(all_pairs, min(n, len(all_pairs)))
    questions = []
    for i, (code, year, val) in enumerate(selected, 1):
        name = names[code]
        questions.append({
            "id": f"ggcr_{dataset}_L1_{i:02d}",
            "dataset": dataset,
            "level": "L1",
            "category": "single_entity_lookup",
            "question": f"What was {name}'s {relation} in {year}?",
            "reference_answer": str(val),
            "reference_set": None,
            "reference_numeric": val,
            "evaluation_type": "exact_match",
            "entities_required": 1,
            "language": "en",
            "metadata": {"entity": code, "year": year},
        })
    return questions


def gen_type_ii_l2(data, names, dataset, relation, unit, rng, n=8):
    """L2: Single entity, multi-year trend analysis."""
    codes = list(data.keys())
    selected = rng.sample(codes, min(n, len(codes)))
    questions = []
    for i, code in enumerate(selected, 1):
        years_sorted = sorted(data[code].keys())
        values = [data[code][y] for y in years_sorted]
        first_year, last_year = years_sorted[0], years_sorted[-1]
        first_val, last_val = values[0], values[-1]

        # Determine trend
        is_monotonic_inc = all(values[j] <= values[j + 1] for j in range(len(values) - 1))
        is_monotonic_dec = all(values[j] >= values[j + 1] for j in range(len(values) - 1))
        if is_monotonic_inc:
            direction = "consistently increased"
        elif is_monotonic_dec:
            direction = "consistently decreased"
        else:
            direction = "fluctuated"

        # Ask about the actual direction so answer is always "Yes"/"No" correctly
        name = names[code]
        if is_monotonic_inc:
            q_direction = "increase"
            answer = "Yes"
        elif is_monotonic_dec:
            q_direction = "decrease"
            answer = "Yes"
        else:
            # For fluctuating, randomly ask increase or decrease → answer is No
            q_direction = rng.choice(["increase", "decrease"])
            answer = "No"

        questions.append({
            "id": f"ggcr_{dataset}_L2_{i:02d}",
            "dataset": dataset,
            "level": "L2",
            "category": "single_entity_trend",
            "question": (
                f"Did {name}'s {relation} consistently {q_direction} from "
                f"{first_year} to {last_year}?"
            ),
            "reference_answer": answer,
            "reference_set": None,
            "reference_numeric": None,
            "evaluation_type": "direction",
            "entities_required": 1,
            "language": "en",
            "metadata": {
                "entity": code,
                "direction": direction,
                "first_value": first_val,
                "last_value": last_val,
                "all_values": {y: v for y, v in zip(years_sorted, values)},
            },
        })
    return questions


def gen_type_ii_l3(data, names, dataset, relation, unit, rng, n=8):
    """L3: Cross-entity ranking (top-K countries in a given year)."""
    all_years = sorted({y for yrs in data.values() for y in yrs})
    k_values = [3, 5]
    combos = [(y, k) for y in all_years for k in k_values]
    selected = rng.sample(combos, min(n, len(combos)))
    questions = []
    for i, (year, k) in enumerate(selected, 1):
        # Rank all entities by value in this year
        ranked = sorted(
            [(code, data[code][year]) for code in data if year in data[code]],
            key=lambda x: x[1],
            reverse=True,
        )
        top_k = ranked[:k]
        top_codes = [code for code, _ in top_k]
        top_names = [names[code] for code in top_codes]

        questions.append({
            "id": f"ggcr_{dataset}_L3_{i:02d}",
            "dataset": dataset,
            "level": "L3",
            "category": "cross_entity_ranking",
            "question": (
                f"Which {k} countries had the highest {relation} in {year}? "
                f"List them in order."
            ),
            "reference_answer": ", ".join(top_names),
            "reference_set": top_codes,
            "reference_numeric": None,
            "evaluation_type": "set_overlap",
            "entities_required": len(data),
            "language": "en",
            "metadata": {
                "year": year,
                "k": k,
                "ranking": [{"code": c, "name": names[c], "value": v}
                            for c, v in top_k],
            },
        })
    return questions


def gen_type_ii_l4(data, names, dataset, relation, unit, rng, n=8):
    """L4: Cross-entity conditional aggregation."""
    all_years = sorted({y for yrs in data.values() for y in yrs})
    questions = []
    idx = 1

    # Sub-type A: Count entities above threshold (4 questions)
    for year in rng.sample(all_years, min(2, len(all_years))):
        values_in_year = [data[c][year] for c in data if year in data[c]]
        median_val = sorted(values_in_year)[len(values_in_year) // 2]
        # Use round threshold near median
        threshold = round(median_val, 0)

        count = sum(1 for v in values_in_year if v > threshold)
        matching = [names[c] for c in data
                    if year in data[c] and data[c][year] > threshold]

        questions.append({
            "id": f"ggcr_{dataset}_L4_{idx:02d}",
            "dataset": dataset,
            "level": "L4",
            "category": "cross_entity_count",
            "question": (
                f"How many countries had {relation} above {threshold} {unit} "
                f"in {year}?"
            ),
            "reference_answer": str(count),
            "reference_set": None,
            "reference_numeric": float(count),
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "en",
            "metadata": {"year": year, "threshold": threshold,
                         "matching_countries": matching},
        })
        idx += 1

    # Sub-type B: Average across all entities (2 questions)
    for year in rng.sample(all_years, min(2, len(all_years))):
        values_in_year = [data[c][year] for c in data if year in data[c]]
        avg_val = round(sum(values_in_year) / len(values_in_year), 2)

        questions.append({
            "id": f"ggcr_{dataset}_L4_{idx:02d}",
            "dataset": dataset,
            "level": "L4",
            "category": "cross_entity_average",
            "question": (
                f"What was the average {relation} across all {len(values_in_year)} "
                f"countries in {year}?"
            ),
            "reference_answer": str(avg_val),
            "reference_set": None,
            "reference_numeric": avg_val,
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "en",
            "metadata": {"year": year, "count": len(values_in_year)},
        })
        idx += 1

    # Sub-type C: Largest change between two years (2 questions)
    if len(all_years) >= 2:
        year_pairs = [(all_years[0], all_years[-1]),
                      (all_years[1], all_years[-2])]
        for y1, y2 in year_pairs[:2]:
            changes = []
            for code in data:
                if y1 in data[code] and y2 in data[code]:
                    delta = data[code][y2] - data[code][y1]
                    changes.append((code, delta))
            if not changes:
                continue
            changes.sort(key=lambda x: x[1], reverse=True)
            best_code, best_delta = changes[0]
            best_name = names[best_code]

            questions.append({
                "id": f"ggcr_{dataset}_L4_{idx:02d}",
                "dataset": dataset,
                "level": "L4",
                "category": "cross_entity_max_change",
                "question": (
                    f"Which country had the largest increase in {relation} "
                    f"from {y1} to {y2}?"
                ),
                "reference_answer": best_name,
                "reference_set": [best_code],
                "reference_numeric": round(best_delta, 2),
                "evaluation_type": "exact_match",
                "entities_required": len(data),
                "language": "en",
                "metadata": {
                    "year_from": y1, "year_to": y2,
                    "best_code": best_code, "best_delta": best_delta,
                    "top3": [{"code": c, "name": names[c], "delta": round(d, 2)}
                             for c, d in changes[:3]],
                },
            })
            idx += 1

    # Sub-type D: Range (max - min) in a year (2 questions)
    for year in rng.sample(all_years, min(2, len(all_years))):
        values_in_year = [data[c][year] for c in data if year in data[c]]
        range_val = round(max(values_in_year) - min(values_in_year), 2)

        questions.append({
            "id": f"ggcr_{dataset}_L4_{idx:02d}",
            "dataset": dataset,
            "level": "L4",
            "category": "cross_entity_range",
            "question": (
                f"What was the range (highest minus lowest) of {relation} "
                f"across all countries in {year}?"
            ),
            "reference_answer": str(range_val),
            "reference_set": None,
            "reference_numeric": range_val,
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "en",
            "metadata": {"year": year, "max": max(values_in_year),
                         "min": min(values_in_year)},
        })
        idx += 1

    return questions[:n]


# ---------------------------------------------------------------------------
# Type-III question generators (Inpatient)
# ---------------------------------------------------------------------------

def gen_type_iii_l1(data, icd_codes, rng, n=10):
    """L1: Single entity, single dimension lookup."""
    all_pairs = [(entity, rel, val)
                 for entity, rels in data.items()
                 for rel, val in rels.items()]
    selected = rng.sample(all_pairs, min(n, len(all_pairs)))

    rel_labels = {
        "INPATIENT_TOTAL": "住院病人出院及死亡总人次",
        "REGISTERED_DEATHS": "在医院登记死亡人数",
        "INPATIENT_HA_HOSPITAL": "医院管理局辖下医院住院人次",
    }

    questions = []
    for i, (entity, rel, val) in enumerate(selected, 1):
        label = rel_labels.get(rel, rel)
        questions.append({
            "id": f"ggcr_inpatient_L1_{i:02d}",
            "dataset": "inpatient",
            "level": "L1",
            "category": "single_entity_lookup",
            "question": f"{entity}的{label}是多少？",
            "reference_answer": str(int(val)),
            "reference_set": None,
            "reference_numeric": val,
            "evaluation_type": "exact_match",
            "entities_required": 1,
            "language": "zh",
            "metadata": {"entity": entity, "relation": rel,
                         "icd_code": icd_codes.get(entity, "")},
        })
    return questions


def gen_type_iii_l2(data, icd_codes, rng, n=8):
    """L2: Single entity, cross-dimension comparison."""
    entities_with_both = [e for e, rels in data.items()
                         if "INPATIENT_TOTAL" in rels and "REGISTERED_DEATHS" in rels]
    selected = rng.sample(entities_with_both, min(n, len(entities_with_both)))

    questions = []
    for i, entity in enumerate(selected, 1):
        total = data[entity]["INPATIENT_TOTAL"]
        deaths = data[entity]["REGISTERED_DEATHS"]
        ratio = round(deaths / total * 100, 1)
        is_above_10 = ratio > 10.0

        questions.append({
            "id": f"ggcr_inpatient_L2_{i:02d}",
            "dataset": "inpatient",
            "level": "L2",
            "category": "single_entity_ratio",
            "question": (
                f"{entity}的登记死亡人数是否超过住院总人次的10%？"
            ),
            "reference_answer": "是" if is_above_10 else "否",
            "reference_set": None,
            "reference_numeric": ratio,
            "evaluation_type": "direction",
            "entities_required": 1,
            "language": "zh",
            "metadata": {
                "entity": entity, "deaths": deaths,
                "total": total, "ratio_pct": ratio,
            },
        })
    return questions


def gen_type_iii_l3(data, icd_codes, rng, n=8):
    """L3: Cross-entity ranking by different dimensions."""
    questions = []
    idx = 1

    rel_labels = {
        "INPATIENT_TOTAL": "住院病人出院及死亡总人次",
        "REGISTERED_DEATHS": "在医院登记死亡人数",
    }

    for rel, label in rel_labels.items():
        entities_with_rel = [(e, data[e][rel]) for e in data if rel in data[e]]
        if len(entities_with_rel) < 3:
            continue

        for k in [3, 5]:
            if k > len(entities_with_rel):
                k = len(entities_with_rel)
            ranked = sorted(entities_with_rel, key=lambda x: x[1], reverse=True)
            top_k = ranked[:k]
            top_names = [e for e, _ in top_k]

            questions.append({
                "id": f"ggcr_inpatient_L3_{idx:02d}",
                "dataset": "inpatient",
                "level": "L3",
                "category": "cross_entity_ranking",
                "question": f"{label}最多的前{k}种疾病是哪些？请按顺序列出。",
                "reference_answer": "、".join(top_names),
                "reference_set": top_names,
                "reference_numeric": None,
                "evaluation_type": "set_overlap",
                "entities_required": len(data),
                "language": "zh",
                "metadata": {
                    "relation": rel, "k": k,
                    "ranking": [{"entity": e, "value": int(v)}
                                for e, v in top_k],
                },
            })
            idx += 1

    # Also: lowest ranking
    for rel, label in rel_labels.items():
        entities_with_rel = [(e, data[e][rel]) for e in data if rel in data[e]]
        if len(entities_with_rel) < 3:
            continue
        ranked = sorted(entities_with_rel, key=lambda x: x[1])
        top_3 = ranked[:3]
        top_names = [e for e, _ in top_3]

        questions.append({
            "id": f"ggcr_inpatient_L3_{idx:02d}",
            "dataset": "inpatient",
            "level": "L3",
            "category": "cross_entity_ranking",
            "question": f"{label}最少的前3种疾病是哪些？",
            "reference_answer": "、".join(top_names),
            "reference_set": top_names,
            "reference_numeric": None,
            "evaluation_type": "set_overlap",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {
                "relation": rel, "k": 3, "direction": "ascending",
                "ranking": [{"entity": e, "value": int(v)}
                            for e, v in top_3],
            },
        })
        idx += 1

    return questions[:n]


def gen_type_iii_l4(data, icd_codes, rng, n=8):
    """L4: Cross-entity aggregation over all diseases."""
    questions = []
    idx = 1

    # Sub-type A: Total sum across all entities
    for rel, label in [("INPATIENT_TOTAL", "住院总人次"),
                       ("REGISTERED_DEATHS", "登记死亡人数")]:
        values = [data[e][rel] for e in data if rel in data[e]]
        total = sum(values)

        questions.append({
            "id": f"ggcr_inpatient_L4_{idx:02d}",
            "dataset": "inpatient",
            "level": "L4",
            "category": "cross_entity_sum",
            "question": f"所有{len(values)}种疾病的{label}合计是多少？",
            "reference_answer": str(int(total)),
            "reference_set": None,
            "reference_numeric": float(total),
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {"relation": rel, "count": len(values)},
        })
        idx += 1

    # Sub-type B: Average
    for rel, label in [("INPATIENT_TOTAL", "住院总人次")]:
        values = [data[e][rel] for e in data if rel in data[e]]
        avg = round(sum(values) / len(values), 1)

        questions.append({
            "id": f"ggcr_inpatient_L4_{idx:02d}",
            "dataset": "inpatient",
            "level": "L4",
            "category": "cross_entity_average",
            "question": f"这{len(values)}种疾病的平均{label}是多少？",
            "reference_answer": str(avg),
            "reference_set": None,
            "reference_numeric": avg,
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {"relation": rel},
        })
        idx += 1

    # Sub-type C: Highest mortality rate (deaths / total)
    mortality = []
    for e in data:
        if "INPATIENT_TOTAL" in data[e] and "REGISTERED_DEATHS" in data[e]:
            rate = data[e]["REGISTERED_DEATHS"] / data[e]["INPATIENT_TOTAL"] * 100
            mortality.append((e, round(rate, 2)))
    if mortality:
        mortality.sort(key=lambda x: x[1], reverse=True)
        best_entity, best_rate = mortality[0]

        questions.append({
            "id": f"ggcr_inpatient_L4_{idx:02d}",
            "dataset": "inpatient",
            "level": "L4",
            "category": "cross_entity_derived_metric",
            "question": "哪种疾病的死亡率（登记死亡人数÷住院总人次）最高？",
            "reference_answer": best_entity,
            "reference_set": [best_entity],
            "reference_numeric": best_rate,
            "evaluation_type": "exact_match",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {
                "ranking": [{"entity": e, "rate": r} for e, r in mortality],
            },
        })
        idx += 1

        # Lowest mortality rate
        worst_entity, worst_rate = mortality[-1]
        questions.append({
            "id": f"ggcr_inpatient_L4_{idx:02d}",
            "dataset": "inpatient",
            "level": "L4",
            "category": "cross_entity_derived_metric",
            "question": "哪种疾病的死亡率（登记死亡人数÷住院总人次）最低？",
            "reference_answer": worst_entity,
            "reference_set": [worst_entity],
            "reference_numeric": worst_rate,
            "evaluation_type": "exact_match",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {
                "ranking": [{"entity": e, "rate": r} for e, r in mortality],
            },
        })
        idx += 1

    # Sub-type D: Count above threshold
    for rel, label, threshold in [("INPATIENT_TOTAL", "住院总人次", 50000),
                                  ("REGISTERED_DEATHS", "登记死亡人数", 2000)]:
        values = [(e, data[e][rel]) for e in data if rel in data[e]]
        count = sum(1 for _, v in values if v > threshold)
        matching = [e for e, v in values if v > threshold]

        questions.append({
            "id": f"ggcr_inpatient_L4_{idx:02d}",
            "dataset": "inpatient",
            "level": "L4",
            "category": "cross_entity_count",
            "question": f"有多少种疾病的{label}超过{threshold}？",
            "reference_answer": str(count),
            "reference_set": None,
            "reference_numeric": float(count),
            "evaluation_type": "numeric_tolerance",
            "entities_required": len(data),
            "language": "zh",
            "metadata": {"threshold": threshold, "matching": matching},
        })
        idx += 1

    return questions[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_benchmark():
    """Generate complete GGCR benchmark."""
    rng = random.Random(SEED)
    all_questions = []

    # WHO Life Expectancy (Type-II)
    who_data, who_names = load_type_ii_gold(
        GOLD_DIR / "gold_who_life_expectancy_v2.jsonl"
    )
    all_questions.extend(gen_type_ii_l1(
        who_data, who_names, "who", "life expectancy at birth", "(years)", rng, n=10))
    all_questions.extend(gen_type_ii_l2(
        who_data, who_names, "who", "life expectancy at birth", "(years)", rng, n=8))
    all_questions.extend(gen_type_ii_l3(
        who_data, who_names, "who", "life expectancy at birth", "(years)", rng, n=8))
    all_questions.extend(gen_type_ii_l4(
        who_data, who_names, "who", "life expectancy at birth", "(years)", rng, n=8))

    # WB Child Mortality (Type-II)
    wb_data, wb_names = load_type_ii_gold(
        GOLD_DIR / "gold_wb_child_mortality_v2.jsonl"
    )
    all_questions.extend(gen_type_ii_l1(
        wb_data, wb_names, "wb_cm", "under-5 mortality rate",
        "(per 1,000 live births)", rng, n=10))
    all_questions.extend(gen_type_ii_l2(
        wb_data, wb_names, "wb_cm", "under-5 mortality rate",
        "(per 1,000 live births)", rng, n=8))
    all_questions.extend(gen_type_ii_l3(
        wb_data, wb_names, "wb_cm", "under-5 mortality rate",
        "(per 1,000 live births)", rng, n=8))
    all_questions.extend(gen_type_ii_l4(
        wb_data, wb_names, "wb_cm", "under-5 mortality rate",
        "(per 1,000 live births)", rng, n=8))

    # Inpatient 2023 (Type-III)
    inp_data, inp_icd = load_type_iii_gold(
        GOLD_DIR / "gold_inpatient_2023.jsonl"
    )
    all_questions.extend(gen_type_iii_l1(inp_data, inp_icd, rng, n=10))
    all_questions.extend(gen_type_iii_l2(inp_data, inp_icd, rng, n=8))
    all_questions.extend(gen_type_iii_l3(inp_data, inp_icd, rng, n=8))
    all_questions.extend(gen_type_iii_l4(inp_data, inp_icd, rng, n=8))

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    by_dataset = defaultdict(lambda: defaultdict(int))
    for q in all_questions:
        by_dataset[q["dataset"]][q["level"]] += 1

    print(f"Generated {len(all_questions)} GGCR benchmark questions → {OUTPUT_PATH}")
    print()
    for ds in sorted(by_dataset):
        levels = by_dataset[ds]
        total = sum(levels.values())
        parts = " | ".join(f"{lv}: {levels[lv]}" for lv in sorted(levels))
        print(f"  {ds:15s} — {parts} | Total: {total}")


if __name__ == "__main__":
    generate_benchmark()
