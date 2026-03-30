#!/usr/bin/env python3
"""
graph_native_probe_inpatient.py — Graph-Native Downstream Probe: Inpatient 2023

Adapts the WHO/WB probe architecture for Hong Kong inpatient discharge
statistics (Type-III hierarchical data). Validates that SGE's higher graph
construction fidelity (FC=0.938 vs Baseline FC=0.438) translates to superior
performance on queries requiring cross-entity graph traversal.

Design: 15 queries across 4 categories:
  - Cross-entity ranking (5): top/bottom diseases by inpatient count or deaths
  - Cross-entity filtering (3): threshold-based disease selection
  - Dimension comparison (4): within-disease multi-dimension comparisons
  - Cross-entity aggregation (3): sum / range / count across all diseases

Data characteristics:
  - Single year 2023, no time dimension
  - ~308 disease categories × 8 statistical dimensions
  - 8 dimensions: HA hospital, CI hospital, private hospital, total inpatient,
    male death, female death, unknown sex death, total death
  - Values: integers 0–200,000

Method: Pure graph traversal + deterministic rule-based answering.
  No LLM used — values extracted from graph node names and edge descriptions.

SGE graph structure:
  Disease → (HAS_SUB_ITEM) → ICD_code_node → (HAS_VALUE) → float_value_node
  Edge keywords encode the dimension: e.g. "HAS_VALUE,住院病人出院及死亡人次,合计"

Baseline graph structure:
  Hospital_org ↔ Disease  (edge description contains integer value)
  全港登记死亡人数 → Disease  (edge description contains death total)

Usage:
    python3 experiments/graph_native_probe_inpatient.py
"""

import csv
import json
import re
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SGE_GRAPH = (
    BASE_DIR
    / "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml"
)
BASELINE_GRAPH = (
    BASE_DIR
    / "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml"
)
CSV_PATH = (
    BASE_DIR
    / "dataset/住院病人统计"
    / "Inpatient Discharges and Deaths in Hospitals and Registered"
    " Deaths in Hong Kong by Disease 2023 (SC).csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "graph_native_probe_inpatient_results.json"
)

# ---------------------------------------------------------------------------
# Dimension keys — short names used throughout
# ---------------------------------------------------------------------------
DIM_HA = "ha_inpatient"           # 住院病人出院及死亡人次*-医院管理局辖下医院
DIM_CI = "ci_inpatient"           # 住院病人出院及死亡人次*-惩教署辖下医院
DIM_PRIVATE = "private_inpatient" # 住院病人出院及死亡人次*-私家医院
DIM_TOTAL_INP = "total_inpatient" # 住院病人出院及死亡人次*-合计
DIM_MALE_D = "male_death"         # 全港登记死亡人数-男性
DIM_FEMALE_D = "female_death"     # 全港登记死亡人数-女性
DIM_UNK_D = "unk_death"           # 全港登记死亡人数-性别不详
DIM_TOTAL_D = "total_death"       # 全港登记死亡人数-合计

# Chinese keywords that identify dimensions in SGE graph edge keywords
SGE_DIM_KEYWORDS = {
    DIM_HA: ("住院病人出院及死亡人次", "医院管理局"),
    DIM_CI: ("住院病人出院及死亡人次", "惩教署"),
    DIM_PRIVATE: ("住院病人出院及死亡人次", "私家"),
    DIM_TOTAL_INP: ("住院病人出院及死亡人次", "合计"),
    DIM_MALE_D: ("全港登记死亡人数", "男性"),
    DIM_FEMALE_D: ("全港登记死亡人数", "女性"),
    DIM_UNK_D: ("全港登记死亡人数", "性别不详"),
    DIM_TOTAL_D: ("全港登记死亡人数", "合计"),
}

# Baseline edge description patterns for dimension extraction
BASELINE_EDGE_PATTERNS = {
    DIM_HA: re.compile(
        r"医院管理局.*?(?:出院及死亡.*?共|人次为|人次[^\d]*)([\d,]+\.?\d*)(?:人次|人|\.0)?",
        re.DOTALL,
    ),
    DIM_CI: re.compile(
        r"惩教署.*?(?:出院及死亡.*?共|人次为|人次[^\d]*)([\d,]+\.?\d*)(?:人次|人|\.0)?",
        re.DOTALL,
    ),
    DIM_PRIVATE: re.compile(
        r"私家医院.*?(?:出院及死亡.*?共|人次为|人次[^\d]*)([\d,]+\.?\d*)(?:人次|人|\.0)?",
        re.DOTALL,
    ),
    DIM_TOTAL_INP: re.compile(
        r"(?:住院病人出院及死亡.*?合计|出院及死亡.*?共)[\D]*([\d,]+\.?\d*)(?:人次|人|\.0)",
        re.DOTALL,
    ),
    DIM_TOTAL_D: re.compile(
        r"(?:全港登记死亡总数|死亡总数|死亡总计|死亡.*?合计)[^\d]*([\d,]+\.?\d*)人",
        re.DOTALL,
    ),
    DIM_MALE_D: re.compile(r"男性([\d,]+\.?\d*)人"),
    DIM_FEMALE_D: re.compile(r"女性([\d,]+\.?\d*)人"),
}

# Generic number extractor for fallback
_INT_IN_STR = re.compile(r"(?:^|[\s为共])(\d[\d,]*)(?:\.0)?(?:人次|人|\b)")


# ---------------------------------------------------------------------------
# CSV Gold Data Loader
# ---------------------------------------------------------------------------

def load_gold_data(csv_path: str) -> dict:
    """
    Parse the inpatient CSV and return:
        {disease_name: {DIM_*: int_value, 'icd': str}}

    CSV layout (from row 3 onward, 0-indexed):
        col0: ICD code
        col1: disease name (Chinese)
        col2: HA inpatient
        col3: CI inpatient
        col4: private inpatient
        col5: total inpatient
        col6: male death
        col7: female death
        col8: unknown sex death
        col9: total death
    """
    data = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Data starts at row index 3 (after 2-row title + 1 blank + 1 header)
    for raw_line in lines[3:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Stop at footer notes
        if stripped.startswith("注：") or stripped.startswith("由二零"):
            break
        row = next(csv.reader([stripped]))
        if len(row) < 10:
            continue
        icd = row[0].strip().strip('"')
        name = row[1].strip()
        if not name or name in ("合计", "诊断不详"):
            continue

        def _parse_int(s: str) -> int:
            s = s.strip()
            if not s or s == "-":
                return 0
            try:
                return int(float(s.replace(",", "")))
            except ValueError:
                return 0

        data[name] = {
            "icd": icd,
            DIM_HA: _parse_int(row[2]),
            DIM_CI: _parse_int(row[3]),
            DIM_PRIVATE: _parse_int(row[4]),
            DIM_TOTAL_INP: _parse_int(row[5]),
            DIM_MALE_D: _parse_int(row[6]),
            DIM_FEMALE_D: _parse_int(row[7]),
            DIM_UNK_D: _parse_int(row[8]),
            DIM_TOTAL_D: _parse_int(row[9]),
        }

    return data


# ---------------------------------------------------------------------------
# Query Builder
# ---------------------------------------------------------------------------

def build_queries(gold: dict) -> list:
    """
    Build 15 graph-native queries with deterministic reference answers
    computed from the gold CSV data.

    Categories:
      5 × cross_entity_ranking
      3 × cross_entity_filtering
      4 × dimension_comparison
      3 × cross_entity_aggregation
    """
    queries = []
    diseases = sorted(gold.keys())

    def v(disease: str, dim: str):
        return gold.get(disease, {}).get(dim)

    # =========================================================================
    # Category 1: Cross-entity ranking (5 queries)
    # =========================================================================

    # Q1: Top 5 diseases by total inpatient
    ranked_inp = sorted(
        [(d, v(d, DIM_TOTAL_INP)) for d in diseases if v(d, DIM_TOTAL_INP) is not None],
        key=lambda x: x[1], reverse=True,
    )
    top5_inp = ranked_inp[:5]
    queries.append({
        "id": "rank_01",
        "category": "cross_entity_ranking",
        "query": "住院合计人次最多的前5种疾病是哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in top5_inp],
        "reference_codes": [d for d, _ in top5_inp],
        "evaluation_type": "top_k_match",
        "k": 5,
        "dim": DIM_TOTAL_INP,
    })

    # Q2: Top 3 diseases by total death
    ranked_death = sorted(
        [(d, v(d, DIM_TOTAL_D)) for d in diseases if v(d, DIM_TOTAL_D) is not None],
        key=lambda x: x[1], reverse=True,
    )
    top3_death = ranked_death[:3]
    queries.append({
        "id": "rank_02",
        "category": "cross_entity_ranking",
        "query": "全港登记死亡合计人数最多的前3种疾病是哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in top3_death],
        "reference_codes": [d for d, _ in top3_death],
        "evaluation_type": "top_k_match",
        "k": 3,
        "dim": DIM_TOTAL_D,
    })

    # Q3: Top 3 diseases by HA hospital inpatient
    ranked_ha = sorted(
        [(d, v(d, DIM_HA)) for d in diseases if v(d, DIM_HA) is not None],
        key=lambda x: x[1], reverse=True,
    )
    top3_ha = ranked_ha[:3]
    queries.append({
        "id": "rank_03",
        "category": "cross_entity_ranking",
        "query": "医院管理局辖下医院住院人次最多的前3种疾病是哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in top3_ha],
        "reference_codes": [d for d, _ in top3_ha],
        "evaluation_type": "top_k_match",
        "k": 3,
        "dim": DIM_HA,
    })

    # Q4: Top 3 diseases by private hospital inpatient
    ranked_priv = sorted(
        [(d, v(d, DIM_PRIVATE)) for d in diseases if v(d, DIM_PRIVATE) is not None],
        key=lambda x: x[1], reverse=True,
    )
    top3_priv = ranked_priv[:3]
    queries.append({
        "id": "rank_04",
        "category": "cross_entity_ranking",
        "query": "私家医院住院人次最多的前3种疾病是哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in top3_priv],
        "reference_codes": [d for d, _ in top3_priv],
        "evaluation_type": "top_k_match",
        "k": 3,
        "dim": DIM_PRIVATE,
    })

    # Q5: Top 5 diseases by total inpatient (bottom — diseases with most zero counts)
    ranked_inp_asc = sorted(
        [(d, v(d, DIM_TOTAL_INP)) for d in diseases if v(d, DIM_TOTAL_INP) is not None],
        key=lambda x: x[1],
    )
    # Find diseases with zero inpatient (for set_match)
    zero_inp = [d for d, val in ranked_inp_asc if val == 0]
    queries.append({
        "id": "rank_05",
        "category": "cross_entity_ranking",
        "query": "住院合计人次为零的疾病有哪些？",
        "reference_answer": [{"disease": d, "value": 0} for d in sorted(zero_inp)],
        "reference_codes": sorted(zero_inp),
        "evaluation_type": "set_match",
        "dim": DIM_TOTAL_INP,
    })

    # =========================================================================
    # Category 2: Cross-entity filtering (3 queries)
    # =========================================================================

    # Q6: Diseases with total inpatient > 50000
    above50k = sorted(
        [(d, v(d, DIM_TOTAL_INP)) for d in diseases
         if v(d, DIM_TOTAL_INP) is not None and v(d, DIM_TOTAL_INP) > 50000],
        key=lambda x: x[1], reverse=True,
    )
    queries.append({
        "id": "filter_01",
        "category": "cross_entity_filtering",
        "query": "住院合计人次超过50000的疾病有哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in above50k],
        "reference_codes": [d for d, _ in above50k],
        "evaluation_type": "set_match",
        "threshold": 50000,
        "direction": "above",
        "dim": DIM_TOTAL_INP,
    })

    # Q7: Diseases with total death > 1000
    above1k_death = sorted(
        [(d, v(d, DIM_TOTAL_D)) for d in diseases
         if v(d, DIM_TOTAL_D) is not None and v(d, DIM_TOTAL_D) > 1000],
        key=lambda x: x[1], reverse=True,
    )
    queries.append({
        "id": "filter_02",
        "category": "cross_entity_filtering",
        "query": "全港登记死亡合计人数超过1000的疾病有哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in above1k_death],
        "reference_codes": [d for d, _ in above1k_death],
        "evaluation_type": "set_match",
        "threshold": 1000,
        "direction": "above",
        "dim": DIM_TOTAL_D,
    })

    # Q8: Diseases with HA inpatient > 10000 and private inpatient < 1000
    combined_filter = sorted(
        [
            (d, v(d, DIM_HA))
            for d in diseases
            if (
                v(d, DIM_HA) is not None
                and v(d, DIM_PRIVATE) is not None
                and v(d, DIM_HA) > 10000
                and v(d, DIM_PRIVATE) < 1000
            )
        ],
        key=lambda x: x[1], reverse=True,
    )
    queries.append({
        "id": "filter_03",
        "category": "cross_entity_filtering",
        "query": "医院管理局住院人次超过10000且私家医院住院人次低于1000的疾病有哪些？",
        "reference_answer": [{"disease": d, "value": val} for d, val in combined_filter],
        "reference_codes": [d for d, _ in combined_filter],
        "evaluation_type": "set_match",
        "dim": DIM_HA,
    })

    # =========================================================================
    # Category 3: Dimension comparison (4 queries)
    # =========================================================================

    # Q9: How many diseases have male death > female death?
    male_gt_female = sorted(
        [
            d for d in diseases
            if v(d, DIM_MALE_D) is not None
            and v(d, DIM_FEMALE_D) is not None
            and v(d, DIM_MALE_D) > v(d, DIM_FEMALE_D)
        ]
    )
    queries.append({
        "id": "dim_01",
        "category": "dimension_comparison",
        "query": "男性死亡人数超过女性死亡人数的疾病共有多少种？",
        "reference_answer": {
            "count": len(male_gt_female),
            "diseases": [{"disease": d} for d in male_gt_female],
        },
        "reference_value": len(male_gt_female),
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 0.0,
        "dim_a": DIM_MALE_D,
        "dim_b": DIM_FEMALE_D,
    })

    # Q10: Top 3 diseases by HA ratio (HA / total_inpatient), among total > 1000
    ha_ratio_items = [
        (d, v(d, DIM_HA) / v(d, DIM_TOTAL_INP))
        for d in diseases
        if v(d, DIM_HA) is not None
        and v(d, DIM_TOTAL_INP) is not None
        and v(d, DIM_TOTAL_INP) > 1000
    ]
    ha_ratio_items.sort(key=lambda x: x[1], reverse=True)
    top3_ha_ratio = ha_ratio_items[:3]
    queries.append({
        "id": "dim_02",
        "category": "dimension_comparison",
        "query": "住院合计超过1000人次中，医院管理局占比最高的前3种疾病是哪些？",
        "reference_answer": [
            {"disease": d, "ha_ratio": round(r, 4)} for d, r in top3_ha_ratio
        ],
        "reference_codes": [d for d, _ in top3_ha_ratio],
        "evaluation_type": "top_k_match",
        "k": 3,
        "dim": "ha_ratio",
    })

    # Q11: Disease with the highest total death / total inpatient ratio (fatality rate)
    #       among diseases with total_inpatient > 1000
    fatality_items = [
        (d, v(d, DIM_TOTAL_D) / v(d, DIM_TOTAL_INP))
        for d in diseases
        if v(d, DIM_TOTAL_INP) is not None
        and v(d, DIM_TOTAL_D) is not None
        and v(d, DIM_TOTAL_INP) > 1000
    ]
    fatality_items.sort(key=lambda x: x[1], reverse=True)
    top1_fatality = fatality_items[:1]
    queries.append({
        "id": "dim_03",
        "category": "dimension_comparison",
        "query": "住院超过1000人次中，死亡人数占住院人次比例最高的疾病是哪种？",
        "reference_answer": [
            {"disease": d, "fatality_ratio": round(r, 4)} for d, r in top1_fatality
        ],
        "reference_codes": [d for d, _ in top1_fatality],
        "evaluation_type": "top_k_match",
        "k": 1,
        "dim": "fatality_ratio",
    })

    # Q12: Diseases where total_death > total_inpatient (more deaths registered
    #       than inpatient admissions — possible for community deaths)
    death_gt_inp = sorted(
        [
            d for d in diseases
            if v(d, DIM_TOTAL_D) is not None
            and v(d, DIM_TOTAL_INP) is not None
            and v(d, DIM_TOTAL_D) > v(d, DIM_TOTAL_INP)
        ]
    )
    queries.append({
        "id": "dim_04",
        "category": "dimension_comparison",
        "query": "全港登记死亡合计人数超过住院合计人次的疾病有哪些？",
        "reference_answer": [{"disease": d} for d in death_gt_inp],
        "reference_codes": death_gt_inp,
        "evaluation_type": "set_match",
        "dim_a": DIM_TOTAL_D,
        "dim_b": DIM_TOTAL_INP,
    })

    # =========================================================================
    # Category 4: Cross-entity aggregation (3 queries)
    # =========================================================================

    # Q13: Total sum of all inpatient cases
    total_all_inp = sum(v(d, DIM_TOTAL_INP) for d in diseases if v(d, DIM_TOTAL_INP) is not None)
    queries.append({
        "id": "agg_01",
        "category": "cross_entity_aggregation",
        "query": "所有疾病的住院合计人次总和是多少？",
        "reference_answer": {"total": total_all_inp, "disease_count": len(diseases)},
        "reference_value": total_all_inp,
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 2.0,
        "dim": DIM_TOTAL_INP,
    })

    # Q14: Range (max - min) of total death across all diseases
    all_death_vals = [v(d, DIM_TOTAL_D) for d in diseases if v(d, DIM_TOTAL_D) is not None]
    death_max = max(all_death_vals)
    death_min = min(all_death_vals)
    death_range = death_max - death_min
    queries.append({
        "id": "agg_02",
        "category": "cross_entity_aggregation",
        "query": "所有疾病中，全港登记死亡合计人数的极差（最大值减最小值）是多少？",
        "reference_answer": {
            "range": death_range,
            "max": death_max,
            "min": death_min,
        },
        "reference_value": death_range,
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 2.0,
        "dim": DIM_TOTAL_D,
    })

    # Q15: How many diseases had total inpatient > 10000?
    above10k_count = len([d for d in diseases if v(d, DIM_TOTAL_INP) is not None and v(d, DIM_TOTAL_INP) > 10000])
    queries.append({
        "id": "agg_03",
        "category": "cross_entity_aggregation",
        "query": "住院合计人次超过10000的疾病共有多少种？",
        "reference_answer": {"count": above10k_count},
        "reference_value": above10k_count,
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 0.0,
        "threshold": 10000,
        "dim": DIM_TOTAL_INP,
    })

    return queries


# ---------------------------------------------------------------------------
# SGE Graph Value Extractor
# ---------------------------------------------------------------------------

def _classify_dim_from_text(text: str) -> str | None:
    """
    Classify text containing dimension keywords to a DIM_* constant.
    Works on node names, edge keyword strings, and edge descriptions.

    Handles variants: standard keyword pair, abbreviated node names:
      '住院病人出院及死亡人次-合计: 3.0' → DIM_TOTAL_INP
      '住院病人出院及死亡人次*-合计: 0.0' → DIM_TOTAL_INP  (star variant)
      '全港登记死亡人数-男性: 90' → DIM_MALE_D
      '4.0(医院管理局辖下医院住院病人出院及死亡人次)' → DIM_HA
      '0.0(男性死亡)' → DIM_MALE_D  (short form)
      '0.0(合计-死亡)' → DIM_TOTAL_D  (short form)
    """
    for dim, (primary, secondary) in SGE_DIM_KEYWORDS.items():
        if primary in text and secondary in text:
            return dim
    # Short-form node names in parentheses: 'VAL(医院管理局辖下医院住院病人出院及死亡人次)'
    # Extra abbreviated forms used by some graph nodes
    short_forms = [
        (DIM_HA, ("医院管理局辖下医院住院病人出院及死亡人次",)),
        (DIM_CI, ("惩教署辖下医院住院病人出院及死亡人次",)),
        (DIM_PRIVATE, ("私家医院住院病人出院及死亡人次",)),
        (DIM_TOTAL_INP, ("住院病人出院及死亡人次合计",)),
        (DIM_MALE_D, ("男性死亡", "全港登记死亡人数-男性",)),
        (DIM_FEMALE_D, ("女性死亡", "全港登记死亡人数-女性",)),
        (DIM_UNK_D, ("性别不详死亡", "全港登记死亡人数-性别不详",)),
        (DIM_TOTAL_D, ("合计-死亡", "全港登记死亡人数-合计", "死亡人数合计",)),
    ]
    for dim, keywords in short_forms:
        if any(kw in text for kw in keywords):
            return dim
    return None


def _classify_dim_from_description(desc: str) -> str | None:
    """
    Classify dimension from Chinese edge description text.
    Handles patterns like:
      'M20-M21在医院管理局辖下医院的住院病人出院及死亡人次为1399'
      'C33-C34全港登记死亡人数合计为3880.0'
      'I63的住院病人出院及死亡人次(合计)为9803.0人次'
    """
    # First try exact keyword/short-form match
    dim = _classify_dim_from_text(desc)
    if dim is not None:
        return dim
    # Pattern matching for abbreviated description text
    dim_desc_patterns = [
        (DIM_HA, re.compile(r"住院病人出院及死亡人次.*?医院管理局")),
        (DIM_CI, re.compile(r"住院病人出院及死亡人次.*?惩教署")),
        (DIM_PRIVATE, re.compile(r"住院病人出院及死亡人次.*?私家")),
        (DIM_TOTAL_INP, re.compile(
            r"住院病人出院及死亡人次.*?合计|"
            r"住院病人出院及死亡人次[^男女性惩私医]+为\d"
        )),
        (DIM_MALE_D, re.compile(r"死亡人数.*?男性|死亡.*?男性.*?为\d")),
        (DIM_FEMALE_D, re.compile(r"死亡人数.*?女性|死亡.*?女性.*?为\d")),
        (DIM_UNK_D, re.compile(r"死亡人数.*?性别不详|性别不详.*?死亡")),
        (DIM_TOTAL_D, re.compile(
            r"(?:全港登记)?死亡人数.*?合计|死亡.*?合计.*?为\d"
        )),
    ]
    for dim, pattern in dim_desc_patterns:
        if pattern.search(desc):
            return dim
    return None


def _parse_trailing_float(name: str) -> float | None:
    """
    Extract a float from the end of a string.
    Works for: '60614.0', '住院病人出院及死亡人次-合计: 3.0', '3.0',
               '住院病人出院及死亡人次-合计：1260' (Chinese colon),
               '4.0(医院管理局辖下医院...)' (value at start with suffix in parens).
    """
    name = name.strip()
    # Normalize Chinese colon
    name_norm = name.replace("：", ": ")
    try:
        return float(name_norm)
    except ValueError:
        pass
    # Standard: ends with digits (possibly after ': ')
    m = re.search(r"[：:]\s*([\d,]+\.?\d*)\s*$", name_norm)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # Trailing digits at end
    m2 = re.search(r"([\d]+\.?\d*)\s*$", name_norm)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    # Value at START followed by parenthesized label: '4.0(医院管理局...)'
    m3 = re.match(r"^([\d]+\.?\d*)\s*\(", name_norm)
    if m3:
        try:
            return float(m3.group(1))
        except ValueError:
            pass
    return None


def _get_neighbors(G: nx.Graph, nid: str) -> list:
    """Return list of (neighbor_id, edge_data) for all edges incident to nid."""
    if G.is_directed():
        out = [(tgt, edata) for _, tgt, edata in G.out_edges(nid, data=True)]
        inc = [(src, edata) for src, _, edata in G.in_edges(nid, data=True)]
        return out + inc
    return [(nb, G.edges[nid, nb]) for nb in G.neighbors(nid)]


def _is_icd_code(name: str) -> bool:
    """Return True if name looks like an ICD-10 code (e.g. A00, J12-J18)."""
    return bool(re.match(r"^[A-Z]\d", name))


def _extract_sge_values(G: nx.Graph, id_to_name: dict) -> dict:
    """
    Extract {disease_name: {DIM_*: int}} from the SGE graph.

    Two UNKNOWN node naming patterns observed:
      Pattern A: node_id='住院病人出院及死亡人次-合计: 3.0'
                 edge_kw='HAS_VALUE'
                 → dim from node name, value from node name
      Pattern B: node_id='60614.0' (plain float)
                 edge_kw='HAS_VALUE,住院病人出院及死亡人次,合计'
                 → dim from edge keywords, value from node name

    For both patterns, source ICD node is the disease anchor.
    ICD nodes are linked to disease names via HAS_SUB_ITEM or HAS_METADATA.

    Note: disease name suffixes like '呼吸道结核病_A15-A16' are stripped to
    '呼吸道结核病' using the CSV disease name list for matching.
    """
    # Non-disease node names to exclude (aggregate / summary rows)
    _EXCLUDED_NAMES = {"合计", "诊断不详", "总计", "小计"}

    # Step 1: Build ICD → disease_name mapping
    icd_to_disease: dict[str, str] = {}
    for nid, ndata in G.nodes(data=True):
        if ndata.get("entity_type", "") != "disease":
            continue
        name = id_to_name[nid]
        if _is_icd_code(name) or name in _EXCLUDED_NAMES:
            continue
        for nb_nid, edata in _get_neighbors(G, nid):
            kw = str(edata.get("keywords", ""))
            if "HAS_SUB_ITEM" not in kw and "HAS_METADATA" not in kw:
                continue
            nb_name = id_to_name.get(nb_nid, nb_nid)
            if _is_icd_code(nb_name):
                # Strip ICD suffix from disease name (e.g. '呼吸道结核病_A15-A16' → '呼吸道结核病')
                clean_name = re.sub(r"_[A-Z]\d.*$", "", name).strip()
                if clean_name not in _EXCLUDED_NAMES:
                    icd_to_disease[nb_name] = clean_name

    # Step 1b: Identify ICD nodes that are also linked (HAS_SUB_ITEM) to
    # aggregate/summary nodes. These ICD nodes carry mixed-row HAS_VALUE
    # edges (both real disease values AND nation-wide summary values), so
    # we must skip their value collection entirely to avoid pollution.
    _icd_linked_to_excluded: set[str] = set()
    for nid, ndata in G.nodes(data=True):
        if ndata.get("entity_type", "") != "disease":
            continue
        node_name = id_to_name[nid]
        if not _is_icd_code(node_name):
            continue
        for nb_nid, edata in _get_neighbors(G, nid):
            kw = str(edata.get("keywords", ""))
            if "HAS_SUB_ITEM" not in kw and "HAS_METADATA" not in kw:
                continue
            nb_name = id_to_name.get(nb_nid, nb_nid)
            if nb_name in _EXCLUDED_NAMES:
                _icd_linked_to_excluded.add(node_name)
                break

    # Step 2: For each ICD node, collect dims from UNKNOWN neighbors
    # Pattern A: dim from node name (contains DIM keywords + value)
    # Pattern B: dim from edge keywords (plain float node name)
    icd_to_dims: dict[str, dict] = {}

    for nid, ndata in G.nodes(data=True):
        if ndata.get("entity_type", "") != "disease":
            continue
        node_name = id_to_name[nid]
        if not _is_icd_code(node_name):
            continue  # only process ICD nodes
        for nb_nid, edata in _get_neighbors(G, nid):
            kw = str(edata.get("keywords", ""))
            if "HAS_VALUE" not in kw:
                continue
            nb_etype = G.nodes[nb_nid].get("entity_type", "")
            if nb_etype not in ("UNKNOWN", ""):
                continue
            nb_name = id_to_name.get(nb_nid, nb_nid)
            if _is_icd_code(nb_name) or nb_etype == "disease":
                continue

            desc = str(edata.get("description", ""))
            # For ICD nodes linked to aggregate rows, skip HAS_VALUE neighbors
            # whose description uses "对应的" — this marks national aggregate
            # values (merged from multiple disease rows), not disease-specific.
            if node_name in _icd_linked_to_excluded and "对应的" in desc:
                continue
            # Pattern B: dim in edge keywords
            dim = _classify_dim_from_text(kw)
            # Pattern A: dim in node name (e.g. '住院病人出院及死亡人次-合计: 3.0')
            if dim is None:
                dim = _classify_dim_from_text(nb_name)
            # Pattern C: dim in edge description (e.g. 'M20-M21住院病人出院及死亡人次合计为1750')
            if dim is None:
                dim = _classify_dim_from_description(desc)
            if dim is None:
                continue

            fval = _parse_trailing_float(nb_name)
            if fval is None:
                # Extract value from edge description as fallback
                m_val = re.search(r"为\s*([\d,]+\.?\d*)\s*(?:人次|人|\.0)?", desc)
                if m_val:
                    try:
                        fval = float(m_val.group(1).replace(",", ""))
                    except ValueError:
                        pass
            if fval is None:
                continue

            if node_name not in icd_to_dims:
                icd_to_dims[node_name] = {}
            if dim not in icd_to_dims[node_name]:
                icd_to_dims[node_name][dim] = int(round(fval))

    # Step 2b: Some non-ICD disease nodes have direct HAS_VALUE edges
    # (no ICD intermediary). Collect their dims directly.
    # Node names may carry ICD suffixes (e.g. '呼吸道结核病_A15-A16'); strip them.
    direct_dims: dict[str, dict] = {}  # disease_name → {DIM_*: val}

    for nid, ndata in G.nodes(data=True):
        if ndata.get("entity_type", "") != "disease":
            continue
        node_name = id_to_name[nid]
        if _is_icd_code(node_name) or node_name in _EXCLUDED_NAMES:
            continue
        clean_name = re.sub(r"_[A-Z]\d.*$", "", node_name).strip()
        if clean_name in _EXCLUDED_NAMES:
            continue

        for nb_nid, edata in _get_neighbors(G, nid):
            kw = str(edata.get("keywords", ""))
            if "HAS_VALUE" not in kw:
                continue
            nb_etype = G.nodes[nb_nid].get("entity_type", "")
            if nb_etype not in ("UNKNOWN", ""):
                continue
            nb_name = id_to_name.get(nb_nid, nb_nid)
            if _is_icd_code(nb_name):
                continue

            desc = str(edata.get("description", ""))
            dim = _classify_dim_from_text(kw)
            if dim is None:
                dim = _classify_dim_from_text(nb_name)
            if dim is None:
                dim = _classify_dim_from_description(desc)
            if dim is None:
                continue

            fval = _parse_trailing_float(nb_name)
            if fval is None:
                m_val = re.search(r"为\s*([\d,]+\.?\d*)\s*(?:人次|人|\.0)?", desc)
                if m_val:
                    try:
                        fval = float(m_val.group(1).replace(",", ""))
                    except ValueError:
                        pass
            if fval is None:
                continue

            if clean_name not in direct_dims:
                direct_dims[clean_name] = {}
            if dim not in direct_dims[clean_name]:
                direct_dims[clean_name][dim] = int(round(fval))

    # Step 3: Map ICD dims → disease names
    result: dict[str, dict] = {}
    for icd_name, dims in icd_to_dims.items():
        disease_name = icd_to_disease.get(icd_name)
        if disease_name is None:
            continue
        if disease_name not in result:
            result[disease_name] = {}
        for dim, val in dims.items():
            if dim not in result[disease_name]:
                result[disease_name][dim] = val

    # Step 4: Merge direct dims (non-ICD disease nodes with direct HAS_VALUE)
    # These supplement or fill gaps left by ICD-based mapping.
    for disease_name, dims in direct_dims.items():
        if disease_name not in result:
            result[disease_name] = {}
        for dim, val in dims.items():
            if dim not in result[disease_name]:
                result[disease_name][dim] = val

    return result


# ---------------------------------------------------------------------------
# Baseline Graph Value Extractor
# ---------------------------------------------------------------------------

# Mapping of Baseline graph hospital/org node names to dimension keys
_BASELINE_ORG_TO_DIM = {
    "医院管理局辖下医院": DIM_HA,
    "惩教署辖下医院": DIM_CI,
    "私家医院": DIM_PRIVATE,
    "住院病人出院及死亡人次": DIM_TOTAL_INP,
}

_BASELINE_DEATH_NODE = "全港登记死亡人数"


def _extract_baseline_death_from_desc(desc: str, disease_name: str) -> dict:
    """
    Parse death dimensions from the baseline description string.
    The edge description format: "肺炎导致全港登记死亡总数为11334人，其中男性6166人，女性5168人"
    Returns {DIM_TOTAL_D: v, DIM_MALE_D: v, DIM_FEMALE_D: v} where available.
    """
    out = {}
    # Total death: "死亡总数为NNNN人" or "死亡总计NNNN" or "死亡NNNN人"
    m_total = re.search(
        r"(?:死亡总数为|死亡总计|死亡合计|合计[\D]{0,10})(\d[\d,]*\.?\d*)人",
        desc,
    )
    if m_total:
        try:
            out[DIM_TOTAL_D] = int(float(m_total.group(1).replace(",", "")))
        except ValueError:
            pass

    # Male: "男性NNNN人" or "其中男性NNNN"
    m_male = re.search(r"男性(\d[\d,]*\.?\d*)人", desc)
    if m_male:
        try:
            out[DIM_MALE_D] = int(float(m_male.group(1).replace(",", "")))
        except ValueError:
            pass

    # Female: "女性NNNN人"
    m_female = re.search(r"女性(\d[\d,]*\.?\d*)人", desc)
    if m_female:
        try:
            out[DIM_FEMALE_D] = int(float(m_female.group(1).replace(",", "")))
        except ValueError:
            pass

    return out


def _extract_baseline_inpatient_from_desc(desc: str, dim: str) -> int | None:
    """
    Extract an inpatient integer from a Baseline edge description.
    Typical patterns:
      "医院管理局辖下医院处理肺炎患者的住院、出院及死亡，共55487人次。"
      "医院管理局辖下医院收治患者，2051人出院或死亡。"
      "共有196617例患者出院及死亡"
    """
    # Try pattern: 共/计/总共 + digits + 人次
    for pattern in [
        re.compile(r"(?:共|计|总共|为)[\D]{0,5}(\d[\d,]*\.?\d*)(?:人次|\.0)"),
        re.compile(r"(?:出院及死亡|出院或死亡|出院和死亡)[^\d]*(\d[\d,]*\.?\d*)(?:人次|人|\.0)"),
        re.compile(r"(\d[\d,]*\.?\d*)(?:人次|\.0)"),
    ]:
        m = pattern.search(desc)
        if m:
            try:
                return int(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
    return None


def _extract_baseline_values(G: nx.Graph, id_to_name: dict) -> dict:
    """
    Extract {disease_name: {DIM_*: int}} from the Baseline graph.

    Baseline topology (undirected / weakly structured):
      Disease ↔ Hospital_org  (edge desc contains inpatient count for that org)
      全港登记死亡人数 ↔ Disease  (edge desc contains death breakdown)
      Hospital_org.description  (aggregate values across many diseases)

    Primary strategy: look at direct edges to/from disease nodes.
    """
    result = {}
    is_directed = G.is_directed()

    def iter_edges(nid):
        """Yield (neighbor_id, edge_data) for all incident edges of nid."""
        if is_directed:
            for _, tgt, edata in G.out_edges(nid, data=True):
                yield tgt, edata
            for src, _, edata in G.in_edges(nid, data=True):
                yield src, edata
        else:
            for nb in G.neighbors(nid):
                edata = G.edges[nid, nb]
                yield nb, edata

    for nid, ndata in G.nodes(data=True):
        etype = ndata.get("entity_type", "")
        disease_name = id_to_name[nid]

        # Only process disease/concept nodes (not org nodes)
        if etype not in ("concept", "disease"):
            continue

        dim_values = {}

        for nb_nid, edata in iter_edges(nid):
            nb_name = id_to_name.get(nb_nid, nb_nid)
            desc = str(edata.get("description", ""))

            # Hospital org edges → inpatient dimensions
            if nb_name in _BASELINE_ORG_TO_DIM:
                dim = _BASELINE_ORG_TO_DIM[nb_name]
                if dim not in dim_values:
                    val = _extract_baseline_inpatient_from_desc(desc, dim)
                    if val is not None:
                        dim_values[dim] = val

            # Death node edges → death dimensions
            elif nb_name == _BASELINE_DEATH_NODE:
                death_vals = _extract_baseline_death_from_desc(desc, disease_name)
                for ddim, dval in death_vals.items():
                    if ddim not in dim_values:
                        dim_values[ddim] = dval

        # Compute total inpatient if we have HA + private (CI often 0)
        if DIM_TOTAL_INP not in dim_values:
            ha = dim_values.get(DIM_HA)
            ci = dim_values.get(DIM_CI, 0)
            priv = dim_values.get(DIM_PRIVATE)
            if ha is not None and priv is not None:
                dim_values[DIM_TOTAL_INP] = ha + ci + priv

        if dim_values:
            result[disease_name] = dim_values

    return result


# ---------------------------------------------------------------------------
# Graph loader (unified entry point)
# ---------------------------------------------------------------------------

def extract_values_from_graph(graph_path: str) -> dict:
    """
    Load a graphml file and extract {disease_name: {DIM_*: int}}.

    Detects SGE vs Baseline by inspecting entity_type distribution.
    """
    G = nx.read_graphml(graph_path)
    id_to_name = {
        nid: str(data.get("entity_name", data.get("name", nid))).strip()
        for nid, data in G.nodes(data=True)
    }

    # Detect graph type: SGE has many 'UNKNOWN' entity_type nodes (value nodes)
    unk_count = sum(
        1 for _, d in G.nodes(data=True) if d.get("entity_type", "") == "UNKNOWN"
    )
    is_sge = unk_count > 50

    if is_sge:
        return _extract_sge_values(G, id_to_name)
    else:
        return _extract_baseline_values(G, id_to_name)


# ---------------------------------------------------------------------------
# Query Answering
# ---------------------------------------------------------------------------

def answer_ranking_query(graph_values: dict, query: dict) -> dict:
    """Answer cross_entity_ranking queries."""
    dim = query.get("dim", DIM_TOTAL_INP)
    k = query.get("k")
    eval_type = query.get("evaluation_type", "top_k_match")

    items = [
        (disease, vals[dim])
        for disease, vals in graph_values.items()
        if dim in vals
    ]

    if eval_type == "set_match":
        # Filter to items with value == 0
        zero_items = [d for d, v in items if v == 0]
        return {
            "retrieved_codes": sorted(zero_items),
            "total_diseases_found": len(items),
        }

    # Determine sort direction from query text
    query_lower = query["query"].lower()
    is_bottom = any(kw in query_lower for kw in ("最少", "最低", "最小", "为零", "zero"))
    items.sort(key=lambda x: x[1], reverse=(not is_bottom))

    top_k = items[:k] if k else items
    return {
        "retrieved_codes": [d for d, _ in top_k],
        "retrieved_values": {d: v for d, v in top_k},
        "total_diseases_found": len(items),
    }


def answer_filter_query(graph_values: dict, query: dict) -> dict:
    """Answer cross_entity_filtering queries."""
    dim = query.get("dim", DIM_TOTAL_INP)
    threshold = query.get("threshold")
    direction = query.get("direction", "above")

    items = [
        (disease, vals.get(dim))
        for disease, vals in graph_values.items()
        if vals.get(dim) is not None
    ]

    query_lower = query["query"].lower()

    if threshold is not None:
        if direction == "above" or "超过" in query_lower or "大于" in query_lower:
            filtered = [(d, v) for d, v in items if v > threshold]
        elif direction == "below" or "低于" in query_lower or "小于" in query_lower:
            filtered = [(d, v) for d, v in items if v < threshold]
        else:
            filtered = items
    elif "超过" in query_lower:
        # Extract threshold from query
        nums = re.findall(r"(\d[\d,]*)", query["query"])
        thr = int(nums[0].replace(",", "")) if nums else 0
        filtered = [(d, v) for d, v in items if v > thr]
    else:
        filtered = items

    # Handle combined filter (Q8: HA > X and private < Y)
    if query.get("id") == "filter_03":
        combined = []
        for disease, vals in graph_values.items():
            ha_val = vals.get(DIM_HA)
            priv_val = vals.get(DIM_PRIVATE)
            if ha_val is not None and priv_val is not None:
                if ha_val > 10000 and priv_val < 1000:
                    combined.append((disease, ha_val))
        combined.sort(key=lambda x: x[1], reverse=True)
        return {
            "retrieved_codes": [d for d, _ in combined],
            "retrieved_values": {d: v for d, v in combined},
            "total_diseases_found": len(list(graph_values.keys())),
        }

    filtered.sort(key=lambda x: x[1], reverse=True)
    return {
        "retrieved_codes": [d for d, _ in filtered],
        "retrieved_values": {d: v for d, v in filtered},
        "total_diseases_found": len(items),
    }


def answer_dim_comparison_query(graph_values: dict, query: dict) -> dict:
    """Answer dimension_comparison queries."""
    qid = query.get("id")

    if qid == "dim_01":
        # Count diseases where male_death > female_death
        matching = sorted([
            d for d, vals in graph_values.items()
            if vals.get(DIM_MALE_D) is not None
            and vals.get(DIM_FEMALE_D) is not None
            and vals.get(DIM_MALE_D, 0) > vals.get(DIM_FEMALE_D, 0)
        ])
        return {
            "computed_value": len(matching),
            "matching_codes": matching,
            "total_diseases_found": len(graph_values),
        }

    elif qid == "dim_02":
        # Top 3 by HA ratio
        items = [
            (d, vals[DIM_HA] / vals[DIM_TOTAL_INP])
            for d, vals in graph_values.items()
            if vals.get(DIM_HA) is not None
            and vals.get(DIM_TOTAL_INP) is not None
            and vals.get(DIM_TOTAL_INP, 0) > 1000
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        top3 = items[:3]
        return {
            "retrieved_codes": [d for d, _ in top3],
            "retrieved_values": {d: round(r, 4) for d, r in top3},
            "total_diseases_found": len(items),
        }

    elif qid == "dim_03":
        # Top 1 by fatality ratio
        items = [
            (d, vals[DIM_TOTAL_D] / vals[DIM_TOTAL_INP])
            for d, vals in graph_values.items()
            if vals.get(DIM_TOTAL_D) is not None
            and vals.get(DIM_TOTAL_INP) is not None
            and vals.get(DIM_TOTAL_INP, 0) > 1000
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        top1 = items[:1]
        return {
            "retrieved_codes": [d for d, _ in top1],
            "retrieved_values": {d: round(r, 4) for d, r in top1},
            "total_diseases_found": len(items),
        }

    elif qid == "dim_04":
        # Diseases where total_death > total_inpatient.
        # Treat missing total_inpatient as 0: diseases with deaths but no
        # inpatient admissions (e.g. community deaths) satisfy the condition.
        matching = sorted([
            d for d, vals in graph_values.items()
            if vals.get(DIM_TOTAL_D) is not None
            and vals.get(DIM_TOTAL_D, 0) > vals.get(DIM_TOTAL_INP, 0)
        ])
        return {
            "retrieved_codes": matching,
            "total_diseases_found": len(graph_values),
        }

    return {"error": f"unknown dim_comparison query: {qid}"}


def answer_aggregation_query(graph_values: dict, query: dict) -> dict:
    """Answer cross_entity_aggregation queries."""
    qid = query.get("id")

    if qid == "agg_01":
        # Sum of all total inpatient
        vals = [v.get(DIM_TOTAL_INP, 0) for v in graph_values.values()
                if DIM_TOTAL_INP in v]
        total = sum(vals)
        return {
            "computed_value": total,
            "disease_count": len(vals),
            "total_diseases_found": len(graph_values),
        }

    elif qid == "agg_02":
        # Range of total death
        vals = [v[DIM_TOTAL_D] for v in graph_values.values() if DIM_TOTAL_D in v]
        if not vals:
            return {"computed_value": None, "error": "no death values found"}
        d_max, d_min = max(vals), min(vals)
        return {
            "computed_value": d_max - d_min,
            "max": d_max,
            "min": d_min,
            "total_diseases_found": len(graph_values),
        }

    elif qid == "agg_03":
        # Count of diseases with total inpatient > 10000
        count = sum(
            1 for v in graph_values.values()
            if v.get(DIM_TOTAL_INP) is not None and v[DIM_TOTAL_INP] > 10000
        )
        return {
            "computed_value": count,
            "total_diseases_found": len(graph_values),
        }

    return {"computed_value": None, "error": f"unknown aggregation: {qid}"}


def answer_query(graph_values: dict, query: dict) -> dict:
    """Route a query to the correct answering function."""
    cat = query["category"]
    if cat == "cross_entity_ranking":
        return answer_ranking_query(graph_values, query)
    elif cat == "cross_entity_filtering":
        return answer_filter_query(graph_values, query)
    elif cat == "dimension_comparison":
        return answer_dim_comparison_query(graph_values, query)
    elif cat == "cross_entity_aggregation":
        return answer_aggregation_query(graph_values, query)
    return {"error": f"unknown category: {cat}"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(query: dict, answer: dict) -> dict:
    """
    Evaluate a graph-derived answer against the reference.
    Returns {correct: bool, details: str, ...}
    """
    eval_type = query["evaluation_type"]

    if eval_type == "top_k_match":
        ref_codes = set(query["reference_codes"])
        ans_codes = set(answer.get("retrieved_codes", []))
        overlap = ref_codes & ans_codes
        correct = len(overlap) == len(ref_codes) and len(overlap) == len(ans_codes)
        precision = round(len(overlap) / len(ans_codes), 4) if ans_codes else 0.0
        recall = round(len(overlap) / len(ref_codes), 4) if ref_codes else 0.0
        return {
            "correct": correct,
            "overlap": len(overlap),
            "expected": len(ref_codes),
            "retrieved": len(ans_codes),
            "precision": precision,
            "recall": recall,
            "details": (
                f"matched {len(overlap)}/{len(ref_codes)} "
                f"(retrieved: {sorted(ans_codes)[:5]}, "
                f"expected: {sorted(ref_codes)[:5]})"
            ),
        }

    elif eval_type == "set_match":
        ref_codes = set(query["reference_codes"])
        ans_codes = set(answer.get("retrieved_codes", []))
        overlap = ref_codes & ans_codes
        correct = ref_codes == ans_codes
        precision = (
            round(len(overlap) / len(ans_codes), 4)
            if ans_codes
            else (1.0 if not ref_codes else 0.0)
        )
        recall = (
            round(len(overlap) / len(ref_codes), 4)
            if ref_codes
            else (1.0 if not ans_codes else 0.0)
        )
        return {
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "overlap": len(overlap),
            "expected_size": len(ref_codes),
            "retrieved_size": len(ans_codes),
            "details": (
                f"P={precision} R={recall} "
                f"(expected={len(ref_codes)}, retrieved={len(ans_codes)}, "
                f"overlap={len(overlap)})"
            ),
        }

    elif eval_type == "numeric_tolerance":
        ref_val = query["reference_value"]
        comp_val = answer.get("computed_value")
        tol_pct = query.get("tolerance_pct", 2.0)

        if comp_val is None:
            return {"correct": False, "details": "no value computed from graph"}

        if tol_pct == 0.0:
            correct = comp_val == ref_val
            pct_diff = 0.0 if correct else 100.0
        elif ref_val != 0:
            pct_diff = abs(comp_val - ref_val) / abs(ref_val) * 100
            correct = pct_diff <= tol_pct
        else:
            correct = comp_val == ref_val
            pct_diff = 0.0 if correct else 100.0

        return {
            "correct": correct,
            "computed": comp_val,
            "reference": ref_val,
            "pct_difference": round(pct_diff, 4),
            "tolerance_pct": tol_pct,
            "details": (
                f"computed={comp_val}, reference={ref_val}, "
                f"pct_diff={round(pct_diff, 4)}%, tol={tol_pct}%"
            ),
        }

    return {"correct": False, "details": f"unknown eval type: {eval_type}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path, label in [
        (SGE_GRAPH, "SGE graph"),
        (BASELINE_GRAPH, "Baseline graph"),
        (CSV_PATH, "CSV data"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 70)
    print("GRAPH-NATIVE DOWNSTREAM PROBE — INPATIENT 2023 (Type-III)")
    print("=" * 70)

    print("\n[1] Loading gold data from CSV...")
    gold = load_gold_data(str(CSV_PATH))
    total_facts = len(gold) * 8  # 8 dimensions per disease
    print(f"    {len(gold)} diseases, {total_facts} dimension-value facts")

    queries = build_queries(gold)
    print(f"    {len(queries)} queries generated")

    print("\n[2] Extracting values from SGE graph...")
    sge_values = extract_values_from_graph(str(SGE_GRAPH))
    sge_diseases = len(sge_values)
    sge_total = sum(len(v) for v in sge_values.values())
    print(f"    Found {sge_diseases} diseases, {sge_total} dimension-values")

    print("\n[3] Extracting values from Baseline graph...")
    base_values = extract_values_from_graph(str(BASELINE_GRAPH))
    base_diseases = len(base_values)
    base_total = sum(len(v) for v in base_values.values())
    print(f"    Found {base_diseases} diseases, {base_total} dimension-values")

    print("\n[4] Answering queries...")
    results = []
    sge_correct = 0
    base_correct = 0

    for q in queries:
        sge_answer = answer_query(sge_values, q)
        base_answer = answer_query(base_values, q)

        sge_eval = evaluate_answer(q, sge_answer)
        base_eval = evaluate_answer(q, base_answer)

        if sge_eval["correct"]:
            sge_correct += 1
        if base_eval["correct"]:
            base_correct += 1

        result = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "reference_answer": q["reference_answer"],
            "sge": {
                "answer": sge_answer,
                "evaluation": sge_eval,
                "correct": sge_eval["correct"],
            },
            "baseline": {
                "answer": base_answer,
                "evaluation": base_eval,
                "correct": base_eval["correct"],
            },
        }
        results.append(result)

        sge_mark = "PASS" if sge_eval["correct"] else "FAIL"
        base_mark = "PASS" if base_eval["correct"] else "FAIL"
        print(
            f"    [{q['id']}] SGE={sge_mark}  Base={base_mark}"
            f"  | {q['query'][:60]}"
        )

    # Summary
    n = len(queries)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Total queries:     {n}")
    print(f"  SGE accuracy:      {sge_correct}/{n} ({sge_correct/n*100:.1f}%)")
    print(f"  Baseline accuracy: {base_correct}/{n} ({base_correct/n*100:.1f}%)")
    print(f"  Delta:             +{sge_correct - base_correct} queries")

    # Per-category breakdown
    categories: dict = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"sge": 0, "base": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["sge"]["correct"]:
            categories[cat]["sge"] += 1
        if r["baseline"]["correct"]:
            categories[cat]["base"] += 1

    print("\n  Per-category breakdown:")
    for cat, counts in sorted(categories.items()):
        t = counts["total"]
        print(
            f"    {cat:35s}  SGE {counts['sge']}/{t}  "
            f"Base {counts['base']}/{t}"
        )

    print(f"\n  Graph extraction stats:")
    print(f"    SGE:      {sge_diseases} diseases, {sge_total} dim-values")
    print(f"    Baseline: {base_diseases} diseases, {base_total} dim-values")
    print("=" * 70)

    # Build output JSON
    output = {
        "dataset": "inpatient_2023",
        "type": "Type-III",
        "description": (
            "Pure graph-traversal queries requiring multi-entity aggregation "
            "over Hong Kong Inpatient 2023 data (~308 disease categories × 8 "
            "statistical dimensions). No LLM used — deterministic rule-based "
            "answering from extracted graph values. Tests whether higher graph "
            "construction fidelity (SGE FC=0.938 vs Baseline FC=0.438) "
            "translates to superior downstream task performance."
        ),
        "method": "graph_traversal + deterministic_computation",
        "graph_extraction_stats": {
            "sge": {"diseases_found": sge_diseases, "total_dim_values": sge_total},
            "baseline": {"diseases_found": base_diseases, "total_dim_values": base_total},
        },
        "summary": {
            "total_queries": n,
            "sge_correct": sge_correct,
            "sge_accuracy": round(sge_correct / n, 4),
            "baseline_correct": base_correct,
            "baseline_accuracy": round(base_correct / n, 4),
            "delta": sge_correct - base_correct,
        },
        "per_category": {
            cat: {
                "total": counts["total"],
                "sge_correct": counts["sge"],
                "sge_accuracy": round(counts["sge"] / counts["total"], 4),
                "baseline_correct": counts["base"],
                "baseline_accuracy": round(counts["base"] / counts["total"], 4),
            }
            for cat, counts in sorted(categories.items())
        },
        "queries": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
