#!/usr/bin/env python3
"""
graph_native_probe_wb_pop.py — Graph-Native Downstream Probe: WB Population

Adapts the WHO life expectancy probe for World Bank Population data.
Validates that SGE's higher graph construction fidelity translates to
superior performance on queries requiring graph structure (multi-entity
traversal) over population data.

Design: 15 queries across 4 categories:
  - Cross-entity ranking (5)
  - Cross-entity filtering (3)
  - Cross-entity trend comparison (4)
  - Cross-entity aggregation (3)

Method: Pure graph traversal + deterministic rule-based answering.
  No LLM is used — values are extracted from graph edges/descriptions
  via regex, then computed directly.

Key differences from WHO probe:
  - Values are integers (population counts, e.g. 37213984)
  - Subject names are full country names, not 3-letter codes
  - Years: 2000, 2005, 2010, 2015, 2020, 2023
  - Population value range: 1,000,000 to 2,000,000,000

Usage:
    python3 experiments/graph_native_probe_wb_pop.py
"""

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
    / "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml"
)
BASELINE_GRAPH = (
    BASE_DIR
    / "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml"
)
GOLD_JSONL = BASE_DIR / "evaluation/gold/gold_wb_population_v2.jsonl"
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "graph_native_probe_wb_pop_results.json"
)

# ---------------------------------------------------------------------------
# Known country names in the WB Population gold standard (25 countries)
# Full names as they appear in the data, plus known alternate forms/codes
# ---------------------------------------------------------------------------
GOLD_COUNTRY_NAMES = {
    "Argentina",
    "Australia",
    "Bangladesh",
    "Brazil",
    "Canada",
    "China",
    "Egypt Arab Rep.",
    "France",
    "Germany",
    "India",
    "Indonesia",
    "Italy",
    "Japan",
    "Korea Rep.",
    "Mexico",
    "Nigeria",
    "Pakistan",
    "Russian Federation",
    "Saudi Arabia",
    "South Africa",
    "Spain",
    "Thailand",
    "Turkiye",
    "United Kingdom",
    "United States",
}

# Mapping of alternate names / abbreviations to canonical gold names
COUNTRY_ALIASES = {
    # 3-letter codes → canonical name
    "ARG": "Argentina",
    "AUS": "Australia",
    "BGD": "Bangladesh",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHN": "China",
    "EGY": "Egypt Arab Rep.",
    "FRA": "France",
    "DEU": "Germany",
    "IND": "India",
    "IDN": "Indonesia",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "Korea Rep.",
    "MEX": "Mexico",
    "NGA": "Nigeria",
    "PAK": "Pakistan",
    "RUS": "Russian Federation",
    "SAU": "Saudi Arabia",
    "ZAF": "South Africa",
    "ESP": "Spain",
    "THA": "Thailand",
    "TUR": "Turkiye",
    "GBR": "United Kingdom",
    "USA": "United States",
    # Common English alternates
    "EGYPT": "Egypt Arab Rep.",
    "EGYPT ARAB REP": "Egypt Arab Rep.",
    "KOREA REP": "Korea Rep.",
    "KOREA": "Korea Rep.",
    "SOUTH KOREA": "Korea Rep.",
    "RUSSIA": "Russian Federation",
    "TURKEY": "Turkiye",
    "TÜRKIYE": "Turkiye",
    "UNITED STATES OF AMERICA": "United States",
    "US": "United States",
    "UK": "United Kingdom",
}


# ---------------------------------------------------------------------------
# Gold Standard loader
# ---------------------------------------------------------------------------

def load_gold_data(jsonl_path: str) -> dict:
    """
    Load gold standard into {country_name: {year: int_population}}.

    Gold format:
      {"triple": {"subject": "Argentina", "relation": "POPULATION",
                  "object": "37213984", "attributes": {"year": "2000"}}}
    """
    data = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triple = record["triple"]
            country = triple["subject"]
            # Normalize: strip commas to match graph node naming
            # "Korea, Rep." → "Korea Rep.", "Egypt, Arab Rep." → "Egypt Arab Rep."
            country_norm = country.replace(", ", " ")
            year = triple["attributes"]["year"]
            population = int(triple["object"])
            if country_norm not in data:
                data[country_norm] = {"name": country_norm, "years": {}}
            data[country_norm]["years"][year] = population
    return data


# ---------------------------------------------------------------------------
# Country name resolution
# ---------------------------------------------------------------------------

def _resolve_country_name(node_name: str) -> str | None:
    """
    Try to resolve a graph node name to a canonical gold country name.

    Returns the canonical name if matched, else None.
    """
    stripped = node_name.strip()

    # Exact match (case-sensitive)
    if stripped in GOLD_COUNTRY_NAMES:
        return stripped

    # Case-insensitive exact match
    stripped_upper = stripped.upper()
    for gold_name in GOLD_COUNTRY_NAMES:
        if gold_name.upper() == stripped_upper:
            return gold_name

    # Alias/code lookup
    alias_key = stripped_upper.replace(".", "").replace(",", "").strip()
    if alias_key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[alias_key]
    if stripped_upper in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[stripped_upper]

    # Substring: only for longer names (>10 chars) and within 3 chars length
    # Short names like Niger/Nigeria are too ambiguous for substring matching
    for gold_name in GOLD_COUNTRY_NAMES:
        gold_upper = gold_name.upper()
        shorter = min(len(gold_upper), len(stripped_upper))
        if shorter >= 10 and abs(len(gold_upper) - len(stripped_upper)) <= 3:
            if gold_upper in stripped_upper or stripped_upper in gold_upper:
                return gold_name

    return None


# ---------------------------------------------------------------------------
# Graph value extractor
# ---------------------------------------------------------------------------

# Regex patterns for extracting large integers (population counts, 6+ digits)
# Handles: "37213984", "37,213,984", scientific-adjacent, Chinese descriptions
POPULATION_PATTERN = re.compile(r"\b(\d[\d,]{5,})\b")
YEAR_EXACT = re.compile(r"\b(20\d{2})\b")


def _clean_int(s: str) -> int | None:
    """Strip commas and parse as integer. Return None on failure."""
    try:
        return int(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _is_population_value(n: int) -> bool:
    """Return True if n is plausibly a population count (1M–2B)."""
    return 1_000_000 <= n <= 2_000_000_000


def _extract_year_population_pairs(
    keywords: str, description: str, target_name: str
) -> list:
    """
    Extract (year_str, int_population) pairs from edge context.

    Handles multiple encoding patterns:
      0. SGE node naming: target="population_year2000=126382494_persons"
      1. SGE keywords: "2000年,POPULATION" or "2015,POPULATION" + target value
      2. Chinese description: "Nigeria在2000年的人口为126382494人"
      3. Inline year+value: "2020: 1411778724" or "2020年 1411778724"
    """
    pairs = []

    # Strategy 0: SGE structured node name — population_yearYYYY=VALUE_persons
    node_pattern = re.search(
        r"population_year(\d{4})[=_](\d{6,})", target_name, re.IGNORECASE
    )
    if node_pattern:
        year_str = node_pattern.group(1)
        pop = int(node_pattern.group(2))
        if _is_population_value(pop):
            pairs.append((year_str, pop))
            return pairs

    # Strategy 1: Year in keywords (YYYY年 or YYYY or year:YYYY) + value in target or description
    year_in_kw = re.search(r"(\d{4})年?", keywords)
    if year_in_kw:
        year_str = year_in_kw.group(1)
        if 1990 <= int(year_str) <= 2030:
            # Try target node as value
            target_pop = _clean_int(target_name.strip())
            if target_pop is not None and _is_population_value(target_pop):
                pairs.append((year_str, target_pop))
                return pairs
            # Try extracting value from target name (may have prefix/suffix)
            val_in_target = re.search(r"(\d{6,})", target_name)
            if val_in_target:
                pop = int(val_in_target.group(1))
                if _is_population_value(pop):
                    pairs.append((year_str, pop))
                    return pairs

    # Strategy 2: Chinese description — "在YYYY年.*为NNNNNN(人|万人)?"
    cn_pattern = re.compile(r"(\d{4})年.*?为\s*([\d,]+)")
    for match in cn_pattern.finditer(description):
        year_str = match.group(1)
        pop = _clean_int(match.group(2))
        if pop is not None and _is_population_value(pop):
            pairs.append((year_str, pop))

    if pairs:
        return pairs

    # Strategy 3: English description — "population ... YYYY ... VALUE" or "YYYY ... population ... VALUE"
    combined = f"{keywords} {description} {target_name}"
    yv_pattern = re.compile(r"(20\d{2})\D{0,15}?([\d,]{7,})")
    for match in yv_pattern.finditer(combined):
        year_str = match.group(1)
        pop = _clean_int(match.group(2))
        if pop is not None and _is_population_value(pop):
            pairs.append((year_str, pop))

    return pairs


def _build_edge_context(edata: dict, target_name: str) -> tuple:
    """Return (keywords, weight, description, target_name) from edge data."""
    kw = str(edata.get("keywords", ""))
    weight = str(edata.get("weight", ""))
    desc = str(edata.get("description", ""))
    return (kw, weight, desc, target_name)


def extract_values_from_graph(graph_path: str) -> dict:
    """
    Extract per-country, per-year population values from a graph.

    Returns: {country_name: {year_str: int_population}}

    Strategy:
      1. For each node, try to resolve to a gold country name.
      2. Traverse all edges (1-hop and 2-hop).
      3. Extract (year, population) pairs from edge context.
    """
    G = nx.read_graphml(graph_path)

    # Build node_id -> entity_name mapping
    id_to_name = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        id_to_name[node_id] = str(name).strip()

    country_values = {}  # canonical_name -> {year -> population}

    for node_id, data in G.nodes(data=True):
        node_name = id_to_name[node_id]
        canonical = _resolve_country_name(node_name)
        if canonical is None:
            continue

        if canonical not in country_values:
            country_values[canonical] = {}

        # Phase 1: Collect and parse 1-hop edges (highest priority)
        onehop_texts = []
        if G.is_directed():
            for _, target, edata in G.out_edges(node_id, data=True):
                target_name = id_to_name.get(target, target)
                onehop_texts.append(_build_edge_context(edata, target_name))
            for source, _, edata in G.in_edges(node_id, data=True):
                source_name = id_to_name.get(source, source)
                onehop_texts.append(_build_edge_context(edata, source_name))
        else:
            for neighbor_id in G.neighbors(node_id):
                edata = G.edges[node_id, neighbor_id]
                neighbor_name = id_to_name.get(neighbor_id, neighbor_id)
                onehop_texts.append(_build_edge_context(edata, neighbor_name))

        for kw, weight, desc, tgt_name in onehop_texts:
            pairs = _extract_year_population_pairs(kw, desc, tgt_name)
            for year_str, pop in pairs:
                if year_str not in country_values[canonical]:
                    country_values[canonical][year_str] = pop

        # Phase 2: 2-hop edges only fill gaps (years not found in 1-hop)
        neighbor_ids = list(G.neighbors(node_id)) if not G.is_directed() else list(
            set(t for _, t in G.out_edges(node_id))
            | set(s for s, _ in G.in_edges(node_id))
        )
        for nb_id in neighbor_ids:
            nb_name = id_to_name.get(nb_id, nb_id)
            nb_desc = G.nodes[nb_id].get("description", "")
            twohop_texts = []
            if nb_desc:
                twohop_texts.append(("", "", nb_desc, nb_name))
            if G.is_directed():
                for _, t2, ed2 in G.out_edges(nb_id, data=True):
                    t2_name = id_to_name.get(t2, t2)
                    twohop_texts.append(_build_edge_context(ed2, t2_name))
            else:
                for nb2_id in G.neighbors(nb_id):
                    ed2 = G.edges[nb_id, nb2_id]
                    nb2_name = id_to_name.get(nb2_id, nb2_id)
                    twohop_texts.append(_build_edge_context(ed2, nb2_name))

            for kw, weight, desc, tgt_name in twohop_texts:
                pairs = _extract_year_population_pairs(kw, desc, tgt_name)
                for year_str, pop in pairs:
                    # Only fill gaps — never overwrite 1-hop values
                    if year_str not in country_values[canonical]:
                        country_values[canonical][year_str] = pop

    return country_values


# ---------------------------------------------------------------------------
# Query definitions with reference answers
# ---------------------------------------------------------------------------

def build_queries(gold_data: dict) -> list:
    """
    Build 15 graph-native queries with deterministic reference answers
    computed from gold standard population data.
    """
    queries = []

    def val(country, year):
        return gold_data.get(country, {}).get("years", {}).get(year)

    def name(country):
        return gold_data.get(country, {}).get("name", country)

    countries = sorted(gold_data.keys())

    # =========================================================================
    # Category 1: Cross-entity ranking (5 queries)
    # =========================================================================

    # Q1: Top 5 most populous in 2023
    ranked_2023 = sorted(
        [(c, val(c, "2023")) for c in countries if val(c, "2023") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top5_2023 = ranked_2023[:5]
    queries.append({
        "id": "rank_01",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the largest population in 2023?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in top5_2023
        ],
        "reference_codes": [c for c, _ in top5_2023],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2023",
    })

    # Q2: Bottom 5 (least populous) in 2000
    ranked_2000_asc = sorted(
        [(c, val(c, "2000")) for c in countries if val(c, "2000") is not None],
        key=lambda x: x[1],
    )
    bottom5_2000 = ranked_2000_asc[:5]
    queries.append({
        "id": "rank_02",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the smallest population in 2000?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in bottom5_2000
        ],
        "reference_codes": [c for c, _ in bottom5_2000],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2000",
    })

    # Q3: Top 3 most populous in 2015
    ranked_2015 = sorted(
        [(c, val(c, "2015")) for c in countries if val(c, "2015") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top3_2015 = ranked_2015[:3]
    queries.append({
        "id": "rank_03",
        "category": "cross_entity_ranking",
        "query": "Which 3 countries had the largest population in 2015?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in top3_2015
        ],
        "reference_codes": [c for c, _ in top3_2015],
        "evaluation_type": "top_k_match",
        "k": 3,
        "year": "2015",
    })

    # Q4: Most populous country in 2020
    ranked_2020 = sorted(
        [(c, val(c, "2020")) for c in countries if val(c, "2020") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    queries.append({
        "id": "rank_04",
        "category": "cross_entity_ranking",
        "query": "Which country had the largest population in 2020?",
        "reference_answer": [
            {"country": ranked_2020[0][0], "value": ranked_2020[0][1]}
        ],
        "reference_codes": [ranked_2020[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
        "year": "2020",
    })

    # Q5: Top 5 most populous in 2010
    ranked_2010 = sorted(
        [(c, val(c, "2010")) for c in countries if val(c, "2010") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top5_2010 = ranked_2010[:5]
    queries.append({
        "id": "rank_05",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the largest population in 2010?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in top5_2010
        ],
        "reference_codes": [c for c, _ in top5_2010],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2010",
    })

    # =========================================================================
    # Category 2: Cross-entity filtering (3 queries)
    # =========================================================================

    # Q6: Countries with population > 200M in 2023
    above200m_2023 = sorted(
        [(c, val(c, "2023")) for c in countries
         if val(c, "2023") is not None and val(c, "2023") > 200_000_000],
        key=lambda x: x[1],
        reverse=True,
    )
    queries.append({
        "id": "filter_01",
        "category": "cross_entity_filtering",
        "query": "Which countries had a population above 200 million in 2023?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in above200m_2023
        ],
        "reference_codes": [c for c, _ in above200m_2023],
        "evaluation_type": "set_match",
        "year": "2023",
        "threshold": 200_000_000,
    })

    # Q7: Countries with population < 50M in 2000
    below50m_2000 = sorted(
        [(c, val(c, "2000")) for c in countries
         if val(c, "2000") is not None and val(c, "2000") < 50_000_000],
        key=lambda x: x[1],
    )
    queries.append({
        "id": "filter_02",
        "category": "cross_entity_filtering",
        "query": "Which countries had a population below 50 million in 2000?",
        "reference_answer": [
            {"country": c, "value": v} for c, v in below50m_2000
        ],
        "reference_codes": [c for c, _ in below50m_2000],
        "evaluation_type": "set_match",
        "year": "2000",
        "threshold": 50_000_000,
    })

    # Q8: Countries with population between 100M and 500M in 2020
    band_2020 = sorted(
        [(c, val(c, "2020")) for c in countries
         if val(c, "2020") is not None
         and 100_000_000 <= val(c, "2020") <= 500_000_000],
        key=lambda x: x[1],
    )
    queries.append({
        "id": "filter_03",
        "category": "cross_entity_filtering",
        "query": (
            "Which countries had a population between 100 million and "
            "500 million in 2020?"
        ),
        "reference_answer": [
            {"country": c, "value": v} for c, v in band_2020
        ],
        "reference_codes": [c for c, _ in band_2020],
        "evaluation_type": "set_match",
        "year": "2020",
    })

    # =========================================================================
    # Category 3: Cross-entity trend comparison (4 queries)
    # =========================================================================

    # Q9: Country with largest absolute population growth 2000→2023
    abs_changes = []
    for c in countries:
        v2000 = val(c, "2000")
        v2023 = val(c, "2023")
        if v2000 is not None and v2023 is not None:
            abs_changes.append((c, v2023 - v2000))
    abs_changes.sort(key=lambda x: x[1], reverse=True)
    queries.append({
        "id": "trend_01",
        "category": "cross_entity_trend",
        "query": (
            "Which country had the largest absolute population growth "
            "from 2000 to 2023?"
        ),
        "reference_answer": [
            {"country": abs_changes[0][0], "growth": abs_changes[0][1]}
        ],
        "reference_codes": [abs_changes[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
    })

    # Q10: Country with largest percentage growth 2000→2023
    pct_changes = []
    for c in countries:
        v2000 = val(c, "2000")
        v2023 = val(c, "2023")
        if v2000 is not None and v2023 is not None and v2000 > 0:
            pct_changes.append((c, round((v2023 - v2000) / v2000 * 100, 2)))
    pct_changes.sort(key=lambda x: x[1], reverse=True)
    queries.append({
        "id": "trend_02",
        "category": "cross_entity_trend",
        "query": (
            "Which country had the largest percentage population growth "
            "from 2000 to 2023?"
        ),
        "reference_answer": [
            {"country": pct_changes[0][0], "pct_growth": pct_changes[0][1]}
        ],
        "reference_codes": [pct_changes[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
    })

    # Q11: Top 3 countries by absolute population growth 2000→2020
    abs_changes_2020 = []
    for c in countries:
        v2000 = val(c, "2000")
        v2020 = val(c, "2020")
        if v2000 is not None and v2020 is not None:
            abs_changes_2020.append((c, v2020 - v2000))
    abs_changes_2020.sort(key=lambda x: x[1], reverse=True)
    top3_abs_2020 = abs_changes_2020[:3]
    queries.append({
        "id": "trend_03",
        "category": "cross_entity_trend",
        "query": (
            "Which 3 countries had the largest absolute population growth "
            "from 2000 to 2020?"
        ),
        "reference_answer": [
            {"country": c, "growth": g} for c, g in top3_abs_2020
        ],
        "reference_codes": [c for c, _ in top3_abs_2020],
        "evaluation_type": "top_k_match",
        "k": 3,
    })

    # Q12: Countries where population decreased 2000→2023 (if none, smallest growth)
    decreased_2023 = [(c, ch) for c, ch in abs_changes if ch < 0]
    decreased_2023.sort(key=lambda x: x[1])
    if decreased_2023:
        q12_ref_codes = [c for c, _ in decreased_2023]
        q12_ref_answer = [
            {"country": c, "change": ch} for c, ch in decreased_2023
        ]
        q12_eval = "set_match"
        q12_text = "Which countries had a smaller population in 2023 than in 2000?"
    else:
        # Fall back to countries with smallest growth
        smallest_growth = abs_changes[-3:]  # already sorted descending, take last 3
        smallest_growth.sort(key=lambda x: x[1])
        q12_ref_codes = [c for c, _ in smallest_growth]
        q12_ref_answer = [
            {"country": c, "growth": g} for c, g in smallest_growth
        ]
        q12_eval = "top_k_match"
        q12_text = (
            "Which 3 countries had the smallest population growth "
            "from 2000 to 2023?"
        )
    queries.append({
        "id": "trend_04",
        "category": "cross_entity_trend",
        "query": q12_text,
        "reference_answer": q12_ref_answer,
        "reference_codes": q12_ref_codes,
        "evaluation_type": q12_eval,
        "k": len(q12_ref_codes),
    })

    # =========================================================================
    # Category 4: Cross-entity aggregation (3 queries)
    # =========================================================================

    # Q13: Average population across all 25 countries in 2015
    vals_2015 = [val(c, "2015") for c in countries if val(c, "2015") is not None]
    avg_2015 = round(sum(vals_2015) / len(vals_2015)) if vals_2015 else 0
    queries.append({
        "id": "agg_01",
        "category": "cross_entity_aggregation",
        "query": (
            "What is the average population across all 25 countries in 2015?"
        ),
        "reference_answer": {"average": avg_2015, "count": len(vals_2015)},
        "reference_value": avg_2015,
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 2.0,
        "year": "2015",
    })

    # Q14: Range (max - min) of population in 2023
    vals_2023 = [val(c, "2023") for c in countries if val(c, "2023") is not None]
    range_2023 = max(vals_2023) - min(vals_2023) if vals_2023 else 0
    queries.append({
        "id": "agg_02",
        "category": "cross_entity_aggregation",
        "query": (
            "What is the range (max minus min) of population "
            "across all 25 countries in 2023?"
        ),
        "reference_answer": {
            "range": range_2023,
            "max": max(vals_2023),
            "min": min(vals_2023),
        },
        "reference_value": range_2023,
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 2.0,
        "year": "2023",
    })

    # Q15: How many countries had population > 100M in both 2000 and 2023?
    both_above100m = []
    for c in countries:
        v2000 = val(c, "2000")
        v2023 = val(c, "2023")
        if (
            v2000 is not None
            and v2023 is not None
            and v2000 > 100_000_000
            and v2023 > 100_000_000
        ):
            both_above100m.append(c)
    queries.append({
        "id": "agg_03",
        "category": "cross_entity_aggregation",
        "query": (
            "How many countries had a population above 100 million "
            "in both 2000 and 2023?"
        ),
        "reference_answer": {
            "count": len(both_above100m),
            "countries": [{"country": c} for c in sorted(both_above100m)],
        },
        "reference_value": len(both_above100m),
        "evaluation_type": "numeric_tolerance",
        "tolerance_pct": 0.0,
    })

    return queries


# ---------------------------------------------------------------------------
# Graph-based query answering (deterministic, no LLM)
# ---------------------------------------------------------------------------

def answer_ranking_query(graph_values: dict, query: dict) -> dict:
    """Answer a cross-entity ranking query from graph-extracted values."""
    year = query.get("year")
    k = query.get("k", 5)
    is_bottom = (
        "smallest" in query["query"].lower()
        or "bottom" in query["query"].lower()
        or "least" in query["query"].lower()
    )

    if year:
        items = [
            (country, vals.get(year))
            for country, vals in graph_values.items()
            if vals.get(year) is not None
        ]
    else:
        return answer_trend_query(graph_values, query)

    items.sort(key=lambda x: x[1], reverse=(not is_bottom))
    top_k = items[:k]
    return {
        "retrieved_codes": [c for c, _ in top_k],
        "retrieved_values": {c: v for c, v in top_k},
        "total_countries_found": len(items),
    }


def answer_filter_query(graph_values: dict, query: dict) -> dict:
    """Answer a cross-entity filtering query from graph-extracted values."""
    year = query.get("year")
    threshold = query.get("threshold")

    items = [
        (country, vals.get(year))
        for country, vals in graph_values.items()
        if vals.get(year) is not None
    ]

    query_lower = query["query"].lower()

    if "above" in query_lower or "exceed" in query_lower or "larger" in query_lower:
        filtered = [(c, v) for c, v in items if v > threshold]
    elif "below" in query_lower or "under" in query_lower or "smaller" in query_lower:
        filtered = [(c, v) for c, v in items if v < threshold]
    elif "between" in query_lower:
        # Extract numeric bounds from query text (values may be in millions)
        between_match = re.search(
            r"between\s+([\d,]+)\s+(?:million\s+)?and\s+([\d,]+)\s*(?:million)?",
            query_lower,
        )
        if between_match:
            lo_raw = int(between_match.group(1).replace(",", ""))
            hi_raw = int(between_match.group(2).replace(",", ""))
            # Scale by million if raw value < 1000 (i.e., expressed as "100 million")
            lo = lo_raw * 1_000_000 if lo_raw < 100_000 else lo_raw
            hi = hi_raw * 1_000_000 if hi_raw < 100_000 else hi_raw
            filtered = [(c, v) for c, v in items if lo <= v <= hi]
        else:
            filtered = []
    else:
        filtered = items

    return {
        "retrieved_codes": sorted([c for c, _ in filtered]),
        "retrieved_values": {c: v for c, v in filtered},
        "total_countries_found": len(items),
    }


def answer_trend_query(graph_values: dict, query: dict) -> dict:
    """Answer a cross-entity trend comparison query."""
    k = query.get("k", 1)
    query_lower = query["query"].lower()

    years_in_query = sorted(set(re.findall(r"(20\d{2})", query["query"])))
    if len(years_in_query) >= 2:
        y_start, y_end = years_in_query[0], years_in_query[-1]
    else:
        y_start, y_end = "2000", "2023"

    is_pct = "percentage" in query_lower or "percent" in query_lower or "%" in query_lower

    if is_pct:
        changes = []
        for country, vals in graph_values.items():
            v_start = vals.get(y_start)
            v_end = vals.get(y_end)
            if v_start is not None and v_end is not None and v_start > 0:
                pct = round((v_end - v_start) / v_start * 100, 2)
                changes.append((country, pct))
    else:
        changes = []
        for country, vals in graph_values.items():
            v_start = vals.get(y_start)
            v_end = vals.get(y_end)
            if v_start is not None and v_end is not None:
                changes.append((country, v_end - v_start))

    is_set = query.get("evaluation_type") == "set_match"

    if is_set:
        if "smaller" in query_lower or "decrease" in query_lower or "lower" in query_lower:
            filtered = [(c, ch) for c, ch in changes if ch < 0]
        else:
            filtered = [(c, ch) for c, ch in changes if ch > 0]
        filtered.sort(key=lambda x: x[1])
        return {
            "retrieved_codes": [c for c, _ in filtered],
            "retrieved_values": {c: ch for c, ch in filtered},
            "total_countries_found": len(changes),
        }

    changes.sort(key=lambda x: x[1], reverse=True)
    top_k = changes[:k]
    return {
        "retrieved_codes": [c for c, _ in top_k],
        "retrieved_values": {c: v for c, v in top_k},
        "total_countries_found": len(changes),
    }


def answer_aggregation_query(graph_values: dict, query: dict) -> dict:
    """Answer a cross-entity aggregation query."""
    year = query.get("year")
    query_lower = query["query"].lower()

    if "average" in query_lower or "mean" in query_lower:
        vals = [
            v.get(year) for v in graph_values.values() if v.get(year) is not None
        ]
        computed = round(sum(vals) / len(vals)) if vals else None
        return {
            "computed_value": computed,
            "count_used": len(vals),
            "total_countries_found": len(graph_values),
        }

    elif "range" in query_lower:
        vals = [
            v.get(year) for v in graph_values.values() if v.get(year) is not None
        ]
        computed = max(vals) - min(vals) if vals else None
        return {
            "computed_value": computed,
            "max": max(vals) if vals else None,
            "min": min(vals) if vals else None,
            "total_countries_found": len(graph_values),
        }

    elif "how many" in query_lower:
        years_in_q = re.findall(r"(20\d{2})", query["query"])
        # Extract threshold: "above 100 million" → 100000000
        threshold_match = re.search(
            r"above\s+(\d+)\s*(?:million)?", query_lower
        )
        if threshold_match:
            raw = int(threshold_match.group(1))
            threshold = raw * 1_000_000 if raw < 100_000 else raw
        else:
            threshold = 100_000_000

        count = 0
        matching_codes = []
        for country, vals in graph_values.items():
            if all(
                vals.get(y) is not None and vals.get(y) > threshold
                for y in years_in_q
            ):
                count += 1
                matching_codes.append(country)
        return {
            "computed_value": count,
            "matching_codes": sorted(matching_codes),
            "total_countries_found": len(graph_values),
        }

    return {"computed_value": None, "error": "unrecognized aggregation type"}


def answer_query(graph_values: dict, query: dict) -> dict:
    """Route a query to the appropriate answering function."""
    cat = query["category"]
    if cat == "cross_entity_ranking":
        return answer_ranking_query(graph_values, query)
    elif cat == "cross_entity_filtering":
        return answer_filter_query(graph_values, query)
    elif cat == "cross_entity_trend":
        return answer_trend_query(graph_values, query)
    elif cat == "cross_entity_aggregation":
        return answer_aggregation_query(graph_values, query)
    return {"error": f"unknown category: {cat}"}


# ---------------------------------------------------------------------------
# Evaluation: compare graph answer to reference
# ---------------------------------------------------------------------------

def evaluate_answer(query: dict, answer: dict) -> dict:
    """
    Evaluate a graph-derived answer against the reference.

    Returns: {correct: bool, details: str, ...}

    For aggregation queries with population values, uses percentage tolerance
    (tolerance_pct) instead of absolute tolerance, since population values
    are very large integers.
    """
    eval_type = query["evaluation_type"]

    if eval_type == "top_k_match":
        ref_codes = set(query["reference_codes"])
        ans_codes = set(answer.get("retrieved_codes", []))
        overlap = ref_codes & ans_codes
        correct = len(overlap) == len(ref_codes)
        return {
            "correct": correct,
            "overlap": len(overlap),
            "expected": len(ref_codes),
            "precision": round(len(overlap) / len(ans_codes), 4) if ans_codes else 0.0,
            "recall": round(len(overlap) / len(ref_codes), 4) if ref_codes else 0.0,
            "details": (
                f"matched {len(overlap)}/{len(ref_codes)} "
                f"(retrieved: {sorted(ans_codes)}, "
                f"expected: {sorted(ref_codes)})"
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
                f"(retrieved: {sorted(ans_codes)}, "
                f"expected: {sorted(ref_codes)})"
            ),
        }

    elif eval_type == "numeric_tolerance":
        ref_val = query["reference_value"]
        comp_val = answer.get("computed_value")
        tol_pct = query.get("tolerance_pct", 2.0)

        if comp_val is None:
            return {
                "correct": False,
                "details": "no value computed from graph",
            }

        # Use percentage tolerance for large population values
        if ref_val != 0:
            pct_diff = abs(comp_val - ref_val) / abs(ref_val) * 100
            correct = pct_diff <= tol_pct
        else:
            # ref_val == 0: exact integer match required
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
    # Validate paths
    for path, label in [
        (SGE_GRAPH, "SGE graph"),
        (BASELINE_GRAPH, "Baseline graph"),
        (GOLD_JSONL, "Gold standard"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 65)
    print("GRAPH-NATIVE DOWNSTREAM PROBE — WB POPULATION")
    print("=" * 65)

    # Load gold standard and build queries
    print("\n[1] Loading gold standard...")
    gold_data = load_gold_data(str(GOLD_JSONL))
    print(
        f"    {len(gold_data)} countries, "
        f"{sum(len(d['years']) for d in gold_data.values())} facts"
    )

    queries = build_queries(gold_data)
    print(f"    {len(queries)} queries generated")

    # Extract values from both graphs
    print("\n[2] Extracting values from SGE graph...")
    sge_values = extract_values_from_graph(str(SGE_GRAPH))
    sge_countries = len(sge_values)
    sge_total_vals = sum(len(v) for v in sge_values.values())
    print(f"    Found {sge_countries} countries, {sge_total_vals} year-values")

    print("\n[3] Extracting values from Baseline graph...")
    base_values = extract_values_from_graph(str(BASELINE_GRAPH))
    base_countries = len(base_values)
    base_total_vals = sum(len(v) for v in base_values.values())
    print(f"    Found {base_countries} countries, {base_total_vals} year-values")

    # Answer queries and evaluate
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
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
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
            f"    {cat:30s}  SGE {counts['sge']}/{t}  "
            f"Base {counts['base']}/{t}"
        )

    print(f"\n  Graph extraction stats:")
    print(f"    SGE:      {sge_countries} countries, {sge_total_vals} values")
    print(f"    Baseline: {base_countries} countries, {base_total_vals} values")
    print("=" * 65)

    # Save results
    output = {
        "experiment": "graph_native_downstream_probe_wb_population",
        "description": (
            "Pure graph-traversal queries requiring multi-entity aggregation "
            "over WB Population data (25 countries x 6 years). "
            "No LLM used — deterministic rule-based answering from extracted "
            "graph values. Tests whether higher graph construction fidelity "
            "(SGE FC=1.000 vs Baseline FC=0.187) translates to superior "
            "downstream task performance on structure-dependent queries."
        ),
        "method": "graph_traversal + deterministic_computation",
        "dataset": "WB Population (25 countries x 6 years: 2000,2005,2010,2015,2020,2023)",
        "graph_extraction_stats": {
            "sge": {
                "countries_found": sge_countries,
                "total_year_values": sge_total_vals,
            },
            "baseline": {
                "countries_found": base_countries,
                "total_year_values": base_total_vals,
            },
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
                "baseline_accuracy": round(
                    counts["base"] / counts["total"], 4
                ),
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
