#!/usr/bin/env python3
"""
graph_native_probe.py — Graph-Native Downstream Probe Experiment

Validates that SGE's higher graph construction fidelity translates to
superior performance on queries that *require* graph structure (multi-entity
traversal) rather than single-chunk text matching.

Design: 15 queries across 4 categories:
  - Cross-entity ranking (5)
  - Cross-entity filtering (3)
  - Cross-entity trend comparison (4)
  - Cross-entity aggregation (3)

Method: Pure graph traversal + deterministic rule-based answering.
  No LLM is used — values are extracted from graph edges/descriptions
  via regex, then computed directly. This eliminates LLM reasoning as
  a confounding variable.

Usage:
    python3 experiments/graph_native_probe.py
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
    / "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml"
)
BASELINE_GRAPH = (
    BASE_DIR
    / "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml"
)
GOLD_JSONL = BASE_DIR / "evaluation/gold/gold_who_life_expectancy_v2.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "graph_native_probe_results.json"

# ---------------------------------------------------------------------------
# Gold Standard loader
# ---------------------------------------------------------------------------

def load_gold_data(jsonl_path: str) -> dict:
    """Load gold standard into {country_code: {year: value}}."""
    data = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triple = record["triple"]
            code = triple["subject"]
            year = triple["attributes"]["year"]
            value = float(triple["object"])
            country_name = triple["attributes"].get("country_name", code)
            if code not in data:
                data[code] = {"name": country_name, "years": {}}
            data[code]["years"][year] = value
    return data


# ---------------------------------------------------------------------------
# Graph value extractor
# ---------------------------------------------------------------------------

# Regex patterns for extracting numeric values with year context
# Handles patterns like: "72.44" in keyword "HAS_VALUE,year:2005"
# and description "CHN在2005年的出生时预期寿命为72.44年"
VALUE_PATTERN = re.compile(r"(\d+\.\d+)")
YEAR_PATTERN = re.compile(r"(?:year[:\s]*|(\d{4})年)")
YEAR_EXACT = re.compile(r"\b(20\d{2})\b")


def extract_values_from_graph(
    graph_path: str,
) -> dict:
    """
    Extract per-country, per-year life expectancy values from a graph.

    Returns: {country_code_upper: {year_str: float_value}}

    Strategy:
      1. For each node, check if it looks like a country code (2-3 uppercase letters).
      2. Traverse all edges from that node.
      3. From edge keywords + description, extract (year, value) pairs.
      4. Also check edge target node names for numeric values.
    """
    G = nx.read_graphml(graph_path)

    # Build node_id -> entity_name mapping
    id_to_name = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        id_to_name[node_id] = str(name).strip()

    # Identify country-code nodes (uppercase 2-3 letter codes)
    # Also collect all node names for flexible matching
    country_code_pattern = re.compile(r"^[A-Z]{2,3}$")
    all_names_lower = {}  # lower_name -> original_name
    for nid, name in id_to_name.items():
        all_names_lower[name.lower()] = name

    # Known gold country codes for matching
    gold_codes = {
        "ARG", "AUS", "BGD", "BRA", "CAN", "CHN", "DEU", "EGY", "ESP",
        "FRA", "GBR", "IDN", "IND", "ITA", "JPN", "KOR", "MEX", "NGA",
        "PAK", "RUS", "SAU", "THA", "TUR", "USA", "ZAF",
    }

    country_values = {}  # code -> {year -> value}

    for node_id, data in G.nodes(data=True):
        name = id_to_name[node_id]
        name_upper = name.upper().strip()

        # Check if this is a gold country code node
        if name_upper not in gold_codes:
            continue

        code = name_upper
        if code not in country_values:
            country_values[code] = {}

        # Collect text from ALL edges connected to this node
        # (both outgoing in directed graphs and via neighbors in undirected)
        edge_texts = []
        if G.is_directed():
            for _, target, edata in G.out_edges(node_id, data=True):
                target_name = id_to_name.get(target, target)
                edge_texts.append(_build_edge_context(edata, target_name))
            for source, _, edata in G.in_edges(node_id, data=True):
                source_name = id_to_name.get(source, source)
                edge_texts.append(_build_edge_context(edata, source_name))
        else:
            for neighbor_id in G.neighbors(node_id):
                edata = G.edges[node_id, neighbor_id]
                neighbor_name = id_to_name.get(neighbor_id, neighbor_id)
                edge_texts.append(_build_edge_context(edata, neighbor_name))

        # Also do 2-hop: for each neighbor, collect THEIR edges too
        neighbor_ids = list(G.neighbors(node_id)) if not G.is_directed() else []
        if G.is_directed():
            neighbor_ids = list(
                set(t for _, t in G.out_edges(node_id))
                | set(s for s, _ in G.in_edges(node_id))
            )
        for nb_id in neighbor_ids:
            nb_name = id_to_name.get(nb_id, nb_id)
            nb_desc = G.nodes[nb_id].get("description", "")
            if nb_desc:
                edge_texts.append(("", "", nb_desc, nb_name))
            # Edges of neighbor
            if G.is_directed():
                for _, t2, ed2 in G.out_edges(nb_id, data=True):
                    t2_name = id_to_name.get(t2, t2)
                    edge_texts.append(_build_edge_context(ed2, t2_name))
            else:
                for nb2_id in G.neighbors(nb_id):
                    ed2 = G.edges[nb_id, nb2_id]
                    nb2_name = id_to_name.get(nb2_id, nb2_id)
                    edge_texts.append(_build_edge_context(ed2, nb2_name))

        # Parse (year, value) from collected edge texts
        for kw, weight, desc, target_name in edge_texts:
            pairs = _extract_year_value_pairs(kw, desc, target_name)
            for year_str, val in pairs:
                # Keep the first value found for each year (avoid duplicates)
                if year_str not in country_values[code]:
                    country_values[code][year_str] = val

    return country_values


def _build_edge_context(edata: dict, target_name: str) -> tuple:
    """Return (keywords, weight, description, target_name) from edge data."""
    kw = str(edata.get("keywords", ""))
    weight = str(edata.get("weight", ""))
    desc = str(edata.get("description", ""))
    return (kw, weight, desc, target_name)


def _extract_year_value_pairs(
    keywords: str, description: str, target_name: str
) -> list:
    """
    Extract (year_str, float_value) pairs from edge context.

    Handles multiple graph encoding patterns:
      1. SGE pattern: keywords="HAS_VALUE,year:2005", target="72.44"
      2. Baseline pattern: description contains "2005年...72.44年" or similar
      3. Value in description with year in keywords
    """
    pairs = []
    combined = f"{keywords} {description} {target_name}"

    # Strategy 1: SGE-style — year in keywords, value as target node name
    year_in_kw = re.search(r"year[:\s]*(\d{4})", keywords)
    target_val = _try_parse_float(target_name)
    if year_in_kw and target_val is not None:
        # Life expectancy values are typically 30-90
        if 20.0 <= target_val <= 100.0:
            pairs.append((year_in_kw.group(1), target_val))
            return pairs

    # Strategy 2: Chinese description pattern — "在YYYY年的...为XX.XX年"
    cn_pattern = re.compile(r"(\d{4})年.*?为\s*(\d+\.?\d*)\s*年")
    for match in cn_pattern.finditer(description):
        year_str = match.group(1)
        val = float(match.group(2))
        if 20.0 <= val <= 100.0:
            pairs.append((year_str, val))

    if pairs:
        return pairs

    # Strategy 3: Description with explicit year-value associations
    # e.g., "2000年: 70.83, 2005年: 72.44"
    yv_pattern = re.compile(r"(20\d{2})\D{0,5}?(\d{2,3}\.\d+)")
    for match in yv_pattern.finditer(combined):
        year_str = match.group(1)
        val = float(match.group(2))
        if 20.0 <= val <= 100.0:
            pairs.append((year_str, val))

    return pairs


def _try_parse_float(s: str) -> float | None:
    """Try to parse a string as float, return None on failure."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Query definitions with reference answers
# ---------------------------------------------------------------------------

def build_queries(gold_data: dict) -> list:
    """
    Build 15 graph-native queries with deterministic reference answers
    computed from gold standard data.
    """
    queries = []

    # Helper: get value for (code, year), return None if missing
    def val(code, year):
        return gold_data.get(code, {}).get("years", {}).get(year)

    def name(code):
        return gold_data.get(code, {}).get("name", code)

    # All 25 country codes
    codes = sorted(gold_data.keys())

    # =====================================================================
    # Category 1: Cross-entity ranking (5 queries)
    # =====================================================================

    # Q1: Top 5 countries by life expectancy in 2015
    ranked_2015 = sorted(
        [(c, val(c, "2015")) for c in codes if val(c, "2015") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top5_2015 = ranked_2015[:5]
    queries.append({
        "id": "rank_01",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the highest life expectancy in 2015?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in top5_2015
        ],
        "reference_codes": [c for c, _ in top5_2015],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2015",
    })

    # Q2: Bottom 5 countries by life expectancy in 2021
    ranked_2021 = sorted(
        [(c, val(c, "2021")) for c in codes if val(c, "2021") is not None],
        key=lambda x: x[1],
    )
    bottom5_2021 = ranked_2021[:5]
    queries.append({
        "id": "rank_02",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the lowest life expectancy in 2021?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in bottom5_2021
        ],
        "reference_codes": [c for c, _ in bottom5_2021],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2021",
    })

    # Q3: Top 3 countries by life expectancy in 2000
    ranked_2000 = sorted(
        [(c, val(c, "2000")) for c in codes if val(c, "2000") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top3_2000 = ranked_2000[:3]
    queries.append({
        "id": "rank_03",
        "category": "cross_entity_ranking",
        "query": "Which 3 countries had the highest life expectancy in 2000?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in top3_2000
        ],
        "reference_codes": [c for c, _ in top3_2000],
        "evaluation_type": "top_k_match",
        "k": 3,
        "year": "2000",
    })

    # Q4: Rank all 25 countries by 2019 life expectancy — which is #1?
    ranked_2019 = sorted(
        [(c, val(c, "2019")) for c in codes if val(c, "2019") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    queries.append({
        "id": "rank_04",
        "category": "cross_entity_ranking",
        "query": "Which country had the highest life expectancy in 2019?",
        "reference_answer": [
            {"code": ranked_2019[0][0], "name": name(ranked_2019[0][0]),
             "value": ranked_2019[0][1]}
        ],
        "reference_codes": [ranked_2019[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
        "year": "2019",
    })

    # Q5: Top 5 countries by life expectancy in 2010
    ranked_2010 = sorted(
        [(c, val(c, "2010")) for c in codes if val(c, "2010") is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    top5_2010 = ranked_2010[:5]
    queries.append({
        "id": "rank_05",
        "category": "cross_entity_ranking",
        "query": "Which 5 countries had the highest life expectancy in 2010?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in top5_2010
        ],
        "reference_codes": [c for c, _ in top5_2010],
        "evaluation_type": "top_k_match",
        "k": 5,
        "year": "2010",
    })

    # =====================================================================
    # Category 2: Cross-entity filtering (3 queries)
    # =====================================================================

    # Q6: Countries with 2021 life expectancy > 80
    above80_2021 = sorted(
        [(c, val(c, "2021")) for c in codes
         if val(c, "2021") is not None and val(c, "2021") > 80.0],
        key=lambda x: x[1],
        reverse=True,
    )
    queries.append({
        "id": "filter_01",
        "category": "cross_entity_filtering",
        "query": "Which countries had life expectancy above 80 years in 2021?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in above80_2021
        ],
        "reference_codes": [c for c, _ in above80_2021],
        "evaluation_type": "set_match",
        "year": "2021",
        "threshold": 80.0,
    })

    # Q7: Countries with 2000 life expectancy < 60
    below60_2000 = sorted(
        [(c, val(c, "2000")) for c in codes
         if val(c, "2000") is not None and val(c, "2000") < 60.0],
        key=lambda x: x[1],
    )
    queries.append({
        "id": "filter_02",
        "category": "cross_entity_filtering",
        "query": "Which countries had life expectancy below 60 years in 2000?",
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in below60_2000
        ],
        "reference_codes": [c for c, _ in below60_2000],
        "evaluation_type": "set_match",
        "year": "2000",
        "threshold": 60.0,
    })

    # Q8: Countries with 2019 life expectancy between 75 and 80
    band_2019 = sorted(
        [(c, val(c, "2019")) for c in codes
         if val(c, "2019") is not None and 75.0 <= val(c, "2019") <= 80.0],
        key=lambda x: x[1],
    )
    queries.append({
        "id": "filter_03",
        "category": "cross_entity_filtering",
        "query": (
            "Which countries had life expectancy between 75 and 80 years in 2019?"
        ),
        "reference_answer": [
            {"code": c, "name": name(c), "value": v} for c, v in band_2019
        ],
        "reference_codes": [c for c, _ in band_2019],
        "evaluation_type": "set_match",
        "year": "2019",
    })

    # =====================================================================
    # Category 3: Cross-entity trend comparison (4 queries)
    # =====================================================================

    # Q9: Country with largest increase 2000→2021
    increases = []
    for c in codes:
        v2000 = val(c, "2000")
        v2021 = val(c, "2021")
        if v2000 is not None and v2021 is not None:
            increases.append((c, round(v2021 - v2000, 2)))
    increases.sort(key=lambda x: x[1], reverse=True)
    queries.append({
        "id": "trend_01",
        "category": "cross_entity_trend",
        "query": (
            "Which country had the largest increase in life expectancy "
            "from 2000 to 2021?"
        ),
        "reference_answer": [
            {"code": increases[0][0], "name": name(increases[0][0]),
             "increase": increases[0][1]}
        ],
        "reference_codes": [increases[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
    })

    # Q10: Country with largest decrease 2019→2021 (COVID impact)
    decreases = []
    for c in codes:
        v2019 = val(c, "2019")
        v2021 = val(c, "2021")
        if v2019 is not None and v2021 is not None:
            decreases.append((c, round(v2019 - v2021, 2)))
    decreases.sort(key=lambda x: x[1], reverse=True)
    queries.append({
        "id": "trend_02",
        "category": "cross_entity_trend",
        "query": (
            "Which country experienced the largest drop in life expectancy "
            "from 2019 to 2021?"
        ),
        "reference_answer": [
            {"code": decreases[0][0], "name": name(decreases[0][0]),
             "decrease": decreases[0][1]}
        ],
        "reference_codes": [decreases[0][0]],
        "evaluation_type": "top_k_match",
        "k": 1,
    })

    # Q11: Top 3 countries by life expectancy gain 2000→2019
    gains_2000_2019 = []
    for c in codes:
        v2000 = val(c, "2000")
        v2019 = val(c, "2019")
        if v2000 is not None and v2019 is not None:
            gains_2000_2019.append((c, round(v2019 - v2000, 2)))
    gains_2000_2019.sort(key=lambda x: x[1], reverse=True)
    top3_gain = gains_2000_2019[:3]
    queries.append({
        "id": "trend_03",
        "category": "cross_entity_trend",
        "query": (
            "Which 3 countries had the largest life expectancy gain "
            "from 2000 to 2019?"
        ),
        "reference_answer": [
            {"code": c, "name": name(c), "gain": g} for c, g in top3_gain
        ],
        "reference_codes": [c for c, _ in top3_gain],
        "evaluation_type": "top_k_match",
        "k": 3,
    })

    # Q12: Countries where life expectancy decreased 2000→2021
    decreased = [
        (c, inc) for c, inc in increases if inc < 0
    ]
    decreased.sort(key=lambda x: x[1])
    queries.append({
        "id": "trend_04",
        "category": "cross_entity_trend",
        "query": (
            "Which countries had a lower life expectancy in 2021 "
            "compared to 2000?"
        ),
        "reference_answer": [
            {"code": c, "name": name(c), "change": ch} for c, ch in decreased
        ],
        "reference_codes": [c for c, _ in decreased],
        "evaluation_type": "set_match",
    })

    # =====================================================================
    # Category 4: Cross-entity aggregation (3 queries)
    # =====================================================================

    # Q13: Average life expectancy across all 25 countries in 2015
    vals_2015 = [val(c, "2015") for c in codes if val(c, "2015") is not None]
    avg_2015 = round(sum(vals_2015) / len(vals_2015), 2) if vals_2015 else 0
    queries.append({
        "id": "agg_01",
        "category": "cross_entity_aggregation",
        "query": (
            "What is the average life expectancy across all 25 countries in 2015?"
        ),
        "reference_answer": {"average": avg_2015, "count": len(vals_2015)},
        "reference_value": avg_2015,
        "evaluation_type": "numeric_tolerance",
        "tolerance": 1.0,
        "year": "2015",
    })

    # Q14: Range (max - min) of life expectancy in 2021
    vals_2021 = [val(c, "2021") for c in codes if val(c, "2021") is not None]
    range_2021 = round(max(vals_2021) - min(vals_2021), 2) if vals_2021 else 0
    queries.append({
        "id": "agg_02",
        "category": "cross_entity_aggregation",
        "query": (
            "What is the range (max minus min) of life expectancy "
            "across all 25 countries in 2021?"
        ),
        "reference_answer": {
            "range": range_2021,
            "max": max(vals_2021),
            "min": min(vals_2021),
        },
        "reference_value": range_2021,
        "evaluation_type": "numeric_tolerance",
        "tolerance": 1.0,
        "year": "2021",
    })

    # Q15: How many countries had life expectancy > 75 in both 2000 and 2021?
    both_above75 = []
    for c in codes:
        v2000 = val(c, "2000")
        v2021 = val(c, "2021")
        if v2000 is not None and v2021 is not None:
            if v2000 > 75.0 and v2021 > 75.0:
                both_above75.append(c)
    queries.append({
        "id": "agg_03",
        "category": "cross_entity_aggregation",
        "query": (
            "How many countries had life expectancy above 75 in both 2000 and 2021?"
        ),
        "reference_answer": {
            "count": len(both_above75),
            "countries": [
                {"code": c, "name": name(c)} for c in sorted(both_above75)
            ],
        },
        "reference_value": len(both_above75),
        "evaluation_type": "numeric_tolerance",
        "tolerance": 0,
    })

    return queries


# ---------------------------------------------------------------------------
# Graph-based query answering (deterministic, no LLM)
# ---------------------------------------------------------------------------

def answer_ranking_query(
    graph_values: dict, query: dict
) -> dict:
    """Answer a cross-entity ranking query from graph-extracted values."""
    year = query.get("year")
    k = query.get("k", 5)
    is_bottom = "lowest" in query["query"].lower() or "bottom" in query["query"].lower()

    if year:
        # Single-year ranking
        items = [
            (code, vals.get(year))
            for code, vals in graph_values.items()
            if vals.get(year) is not None
        ]
    else:
        # Trend-based: need to compute from two years
        return answer_trend_query(graph_values, query)

    items.sort(key=lambda x: x[1], reverse=(not is_bottom))
    top_k = items[:k]
    return {
        "retrieved_codes": [c for c, _ in top_k],
        "retrieved_values": {c: v for c, v in top_k},
        "total_countries_found": len(items),
    }


def answer_filter_query(
    graph_values: dict, query: dict
) -> dict:
    """Answer a cross-entity filtering query from graph-extracted values."""
    year = query.get("year")
    threshold = query.get("threshold")

    items = [
        (code, vals.get(year))
        for code, vals in graph_values.items()
        if vals.get(year) is not None
    ]

    query_lower = query["query"].lower()

    if "above" in query_lower or "exceed" in query_lower:
        filtered = [(c, v) for c, v in items if v > threshold]
    elif "below" in query_lower or "under" in query_lower:
        filtered = [(c, v) for c, v in items if v < threshold]
    elif "between" in query_lower:
        # Parse "between X and Y" — extract the two bounds before "years"
        between_match = re.search(
            r"between\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)", query_lower
        )
        if between_match:
            lo, hi = float(between_match.group(1)), float(between_match.group(2))
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


def answer_trend_query(
    graph_values: dict, query: dict
) -> dict:
    """Answer a cross-entity trend comparison query."""
    k = query.get("k", 1)
    query_lower = query["query"].lower()

    # Determine year range from query (always chronological order)
    years_in_query = sorted(set(re.findall(r"(20\d{2})", query["query"])))
    if len(years_in_query) >= 2:
        y_start, y_end = years_in_query[0], years_in_query[-1]
    else:
        y_start, y_end = "2000", "2021"

    changes = []
    for code, vals in graph_values.items():
        v_start = vals.get(y_start)
        v_end = vals.get(y_end)
        if v_start is not None and v_end is not None:
            changes.append((code, round(v_end - v_start, 2)))

    is_decrease = "drop" in query_lower or "decrease" in query_lower
    is_set = query.get("evaluation_type") == "set_match"

    if is_set:
        # Return all countries matching the criterion
        if "lower" in query_lower or "decrease" in query_lower:
            filtered = [(c, ch) for c, ch in changes if ch < 0]
        else:
            filtered = [(c, ch) for c, ch in changes if ch > 0]
        filtered.sort(key=lambda x: x[1])
        return {
            "retrieved_codes": [c for c, _ in filtered],
            "retrieved_values": {c: ch for c, ch in filtered},
            "total_countries_found": len(changes),
        }

    if is_decrease:
        # For "largest drop": sort by decrease (largest positive difference
        # when computing start - end)
        changes_abs = [(c, -ch) for c, ch in changes]
        changes_abs.sort(key=lambda x: x[1], reverse=True)
        top_k = changes_abs[:k]
        return {
            "retrieved_codes": [c for c, _ in top_k],
            "retrieved_values": {c: v for c, v in top_k},
            "total_countries_found": len(changes),
        }
    else:
        changes.sort(key=lambda x: x[1], reverse=True)
        top_k = changes[:k]
        return {
            "retrieved_codes": [c for c, _ in top_k],
            "retrieved_values": {c: v for c, v in top_k},
            "total_countries_found": len(changes),
        }


def answer_aggregation_query(
    graph_values: dict, query: dict
) -> dict:
    """Answer a cross-entity aggregation query."""
    year = query.get("year")
    query_lower = query["query"].lower()

    if "average" in query_lower or "mean" in query_lower:
        vals = [
            v.get(year) for v in graph_values.values() if v.get(year) is not None
        ]
        computed = round(sum(vals) / len(vals), 2) if vals else None
        return {
            "computed_value": computed,
            "count_used": len(vals),
            "total_countries_found": len(graph_values),
        }

    elif "range" in query_lower:
        vals = [
            v.get(year) for v in graph_values.values() if v.get(year) is not None
        ]
        computed = round(max(vals) - min(vals), 2) if vals else None
        return {
            "computed_value": computed,
            "max": max(vals) if vals else None,
            "min": min(vals) if vals else None,
            "total_countries_found": len(graph_values),
        }

    elif "how many" in query_lower:
        # Count countries meeting multi-year criteria
        years_in_q = re.findall(r"(20\d{2})", query["query"])
        threshold_match = re.search(r"above\s+(\d+\.?\d*)", query_lower)
        threshold = float(threshold_match.group(1)) if threshold_match else 75.0

        count = 0
        matching_codes = []
        for code, vals in graph_values.items():
            if all(
                vals.get(y) is not None and vals.get(y) > threshold
                for y in years_in_q
            ):
                count += 1
                matching_codes.append(code)
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

    Returns: {correct: bool, details: str}
    """
    eval_type = query["evaluation_type"]

    if eval_type == "top_k_match":
        ref_codes = set(query["reference_codes"])
        ans_codes = set(answer.get("retrieved_codes", []))
        # For top-k: check if the retrieved set matches reference set
        # (order-insensitive, since ranking ties may shift order)
        overlap = ref_codes & ans_codes
        k = query.get("k", len(ref_codes))
        correct = len(overlap) == len(ref_codes)
        return {
            "correct": correct,
            "overlap": len(overlap),
            "expected": len(ref_codes),
            "precision": round(len(overlap) / len(ans_codes), 4)
            if ans_codes
            else 0.0,
            "recall": round(len(overlap) / len(ref_codes), 4)
            if ref_codes
            else 0.0,
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
            round(len(overlap) / len(ans_codes), 4) if ans_codes else (1.0 if not ref_codes else 0.0)
        )
        recall = (
            round(len(overlap) / len(ref_codes), 4) if ref_codes else (1.0 if not ans_codes else 0.0)
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
        tolerance = query.get("tolerance", 1.0)
        if comp_val is None:
            return {
                "correct": False,
                "details": "no value computed from graph",
            }
        diff = abs(comp_val - ref_val)
        correct = diff <= tolerance
        return {
            "correct": correct,
            "computed": comp_val,
            "reference": ref_val,
            "difference": round(diff, 4),
            "tolerance": tolerance,
            "details": (
                f"computed={comp_val}, reference={ref_val}, "
                f"diff={round(diff, 4)}, tol={tolerance}"
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
    print("GRAPH-NATIVE DOWNSTREAM PROBE EXPERIMENT")
    print("=" * 65)

    # Load gold standard and build queries
    print("\n[1] Loading gold standard...")
    gold_data = load_gold_data(str(GOLD_JSONL))
    print(f"    {len(gold_data)} countries, "
          f"{sum(len(d['years']) for d in gold_data.values())} facts")

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
            "reference_answer": q["reference_answer"]
            if not isinstance(q.get("reference_answer"), dict)
            else q["reference_answer"],
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

        # Print per-query summary
        sge_mark = "PASS" if sge_eval["correct"] else "FAIL"
        base_mark = "PASS" if base_eval["correct"] else "FAIL"
        print(f"    [{q['id']}] SGE={sge_mark}  Base={base_mark}  | {q['query'][:60]}")

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
    categories = {}
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

    # Graph extraction stats
    print(f"\n  Graph extraction stats:")
    print(f"    SGE:      {sge_countries} countries, {sge_total_vals} values")
    print(f"    Baseline: {base_countries} countries, {base_total_vals} values")
    print("=" * 65)

    # Save results
    output = {
        "experiment": "graph_native_downstream_probe",
        "description": (
            "Pure graph-traversal queries requiring multi-entity aggregation. "
            "No LLM used — deterministic rule-based answering from extracted "
            "graph values. Tests whether higher graph construction fidelity "
            "(SGE FC=1.000 vs Baseline FC=0.167) translates to superior "
            "downstream task performance on structure-dependent queries."
        ),
        "method": "graph_traversal + deterministic_computation",
        "dataset": "WHO Life Expectancy (25 countries x 6 years)",
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
