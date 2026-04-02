#!/usr/bin/env python3
"""
run_stratified_precision.py — Stratified Precision Sampling for SGE-LightRAG

Samples 250 edges (50 per dataset) from 5 SGE graphs, stratified by
dataset × edge-type × node-degree, and validates each edge against the source CSV.

Stratification axes:
  - edge_type:   "entity-value" (target is pure number) vs "entity-entity"
  - degree_class: "high" (source degree > median) vs "low"

Outputs to evaluation/results/stratified_precision_results.json.

Usage:
    python3 evaluation/run_stratified_precision.py

Run from sge_lightrag/ directory.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import math
import statistics

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from statsmodels.stats.proportion import proportion_confint
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent   # sge_lightrag/
EVAL_DIR = Path(__file__).resolve().parent           # sge_lightrag/evaluation/
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"

GRAPHS = {
    "who":       OUTPUT_DIR / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    "wb_cm":     OUTPUT_DIR / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    "wb_pop":    OUTPUT_DIR / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    "wb_mat":    OUTPUT_DIR / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    "inpatient": OUTPUT_DIR / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
}

CSVS = {
    "who":       DATASET_DIR / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv",
    "wb_cm":     DATASET_DIR / "世界银行数据" / "child_mortality" / "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
    "wb_pop":    DATASET_DIR / "世界银行数据" / "population" / "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
    "wb_mat":    DATASET_DIR / "世界银行数据" / "maternal_mortality" / "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
    "inpatient": DATASET_DIR / "住院病人统计" / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
}

SAMPLE_PER_DATASET = 50
RANDOM_SEED = 42

# Numeric pattern: a node name is a pure number (possibly decimal)
_PURE_NUMBER_RE = re.compile(r"^\d+[\.\d]*$")

# Value-label pattern: "entity_name: 123.4" used by WB/inpatient
_VALUE_LABEL_RE = re.compile(
    r"(?:mortality_rate_year|population_year|maternal_mortality_year).*?=([\d.]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Return (lower, upper) Wilson CI for k successes in n trials."""
    if _HAS_STATSMODELS:
        lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
        return round(lo, 4), round(hi, 4)

    # Manual Wilson score interval
    z = 1.959964  # z_{0.975}
    p_hat = k / n if n > 0 else 0.0
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


# ---------------------------------------------------------------------------
# Edge classification helpers
# ---------------------------------------------------------------------------

def classify_edge_type(target_name: str) -> str:
    """Classify an edge as 'entity-value' or 'entity-entity'.

    An edge is entity-value if the target node name matches a pure number pattern,
    or if the target name contains an embedded numeric value label (WB/inpatient style).
    """
    name = str(target_name).strip()

    # Pure number target (WHO style: "53.82332641")
    if _PURE_NUMBER_RE.match(name):
        return "entity-value"

    # Embedded value label (WB style: "mortality_rate_year2000=136.7_per_1000")
    if _VALUE_LABEL_RE.search(name):
        return "entity-value"

    # Inpatient style: "A00子类别在医院管理局辖下医院的住院病人出院及死亡人次为1.0"
    # Target node name itself may embed the value after "为" or ":"
    if re.search(r"[:：为=][\s]?[\d,]+\.?\d*$", name):
        return "entity-value"

    return "entity-entity"


def classify_degree(degree: int, median_degree: float) -> str:
    return "high" if degree > median_degree else "low"


# ---------------------------------------------------------------------------
# Graph loading + stratified sampling
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    """Load GraphML and return (G, node_name_map) where node_name_map maps node id → display name."""
    G = nx.read_graphml(str(path))
    node_names = {}
    for nid, data in G.nodes(data=True):
        name = str(data.get("entity_id") or data.get("entity_name") or data.get("name") or nid).strip()
        node_names[nid] = name
    return G, node_names


def stratified_sample(G, node_names: dict, n: int, seed: int) -> list[dict]:
    """Sample n edges from G stratified by edge_type × degree_class.

    Returns a list of edge record dicts (not yet validated).
    """
    all_edges = list(G.edges(data=True))
    if len(all_edges) == 0:
        return []

    # Compute median source degree
    degrees = [G.degree(u) for u, _v, _d in all_edges]
    median_deg = statistics.median(degrees)

    # Assign each edge to a stratum
    strata: dict[str, list[dict]] = {}
    for u, v, data in all_edges:
        src_name = node_names.get(u, u)
        tgt_name = node_names.get(v, v)
        edge_type = classify_edge_type(tgt_name)
        deg_class = classify_degree(G.degree(u), median_deg)
        stratum_key = f"{edge_type}_{deg_class}"

        record = {
            "source_node": src_name,
            "target_node": tgt_name,
            "keywords":    str(data.get("keywords", "")),
            "description": str(data.get("description", "")),
            "stratum":     stratum_key,
            "edge_type":   edge_type,
            "degree_class": deg_class,
            "source_degree": G.degree(u),
        }
        strata.setdefault(stratum_key, []).append(record)

    # Proportional allocation
    total_edges = len(all_edges)
    rng = random.Random(seed)
    sampled: list[dict] = []

    stratum_keys = sorted(strata.keys())
    allocations: dict[str, int] = {}
    leftover = n

    for i, key in enumerate(stratum_keys):
        if i == len(stratum_keys) - 1:
            # Last stratum gets the remainder
            alloc = leftover
        else:
            proportion = len(strata[key]) / total_edges
            alloc = max(0, round(proportion * n))
        alloc = min(alloc, len(strata[key]))
        allocations[key] = alloc
        leftover -= alloc

    for key in stratum_keys:
        alloc = allocations[key]
        if alloc <= 0:
            continue
        pool = strata[key]
        drawn = rng.sample(pool, min(alloc, len(pool)))
        sampled.extend(drawn)

    # If we under-sampled due to rounding, top up from all edges
    if len(sampled) < n:
        sampled_set = {(r["source_node"], r["target_node"]) for r in sampled}
        remaining = [
            r for r in [
                {
                    "source_node": node_names.get(u, u),
                    "target_node": node_names.get(v, v),
                    "keywords":    str(d.get("keywords", "")),
                    "description": str(d.get("description", "")),
                    "stratum":     f"{classify_edge_type(node_names.get(v, v))}_{classify_degree(G.degree(u), median_deg)}",
                    "edge_type":   classify_edge_type(node_names.get(v, v)),
                    "degree_class": classify_degree(G.degree(u), median_deg),
                    "source_degree": G.degree(u),
                }
                for u, v, d in all_edges
            ]
            if (r["source_node"], r["target_node"]) not in sampled_set
        ]
        rng.shuffle(remaining)
        sampled.extend(remaining[:n - len(sampled)])

    return sampled[:n]


# ---------------------------------------------------------------------------
# WHO CSV loading + validation
# ---------------------------------------------------------------------------

def load_who_csv(path: Path):
    """Load WHO life expectancy CSV. Returns (df indexed by Country Code, name→code map)."""
    df = pd.read_csv(str(path), header=0)
    name_to_code: dict[str, str] = {}
    for _, row in df.iterrows():
        name = row.get("Country Name")
        code = row.get("Country Code")
        if isinstance(name, str) and isinstance(code, str):
            name_to_code[name.lower().strip()] = code.strip()
    df = df.set_index("Country Code")
    return df, name_to_code


def _resolve_who_entity(entity: str, all_codes: set, name_to_code: dict):
    entity = entity.strip()
    if entity in all_codes:
        return entity
    if "/" in entity:
        code_part = entity.split("/")[0].strip()
        if code_part in all_codes:
            return code_part
    lower = entity.lower()
    if lower in name_to_code:
        return name_to_code[lower]
    for name, code in name_to_code.items():
        if lower in name or name in lower:
            return code
    return None


def _extract_year_from_keywords(keywords: str):
    for part in keywords.split(","):
        part = part.strip()
        if part.startswith("year:"):
            return part[5:].strip()
    return None


def _lookup_wb_value(df: pd.DataFrame, country_code: str, year: str):
    year_str = str(year).strip()
    if country_code not in df.index:
        return None
    if year_str not in df.columns:
        return None
    val = df.loc[country_code, year_str]
    if pd.isna(val):
        return None
    return str(val)


def check_who_edge(record: dict, df: pd.DataFrame, name_to_code: dict) -> tuple:
    """Validate a WHO SGE edge against the CSV. Returns (is_correct, reason)."""
    src = record["source_node"]
    tgt = record["target_node"]
    keywords = record["keywords"]
    desc = record["description"]
    all_codes = set(df.index.tolist())

    # Case 1: pure-number target — verify value
    if _PURE_NUMBER_RE.match(str(tgt).strip()):
        year = _extract_year_from_keywords(keywords)
        if not year:
            year_m = re.search(r'\b(20\d{2}|199\d)\b', desc)
            if year_m:
                year = year_m.group(1)

        resolved = _resolve_who_entity(src, all_codes, name_to_code)
        if year and resolved:
            csv_val = _lookup_wb_value(df, resolved, year)
            if csv_val is not None:
                try:
                    if abs(float(csv_val) - float(tgt.replace(",", ""))) < 0.001:
                        return True, f"CSV[{resolved}][{year}]={csv_val} matches edge target {tgt}."
                    else:
                        return False, f"CSV[{resolved}][{year}]={csv_val} ≠ edge target {tgt}."
                except ValueError:
                    pass
            return False, f"No CSV value for country={resolved!r} year={year!r}."
        return False, f"Cannot verify numeric target {tgt!r}: no year or country match for {src!r}."

    # Case 2: source resolves to a known country — structural edge
    src_resolved = _resolve_who_entity(src, all_codes, name_to_code)
    if src_resolved:
        return True, f"Source {src!r} → country {src_resolved!r}; structural edge supported by CSV."

    # Case 3: target resolves to a known country
    tgt_resolved = _resolve_who_entity(tgt, all_codes, name_to_code)
    if tgt_resolved:
        return True, f"Target {tgt!r} → country {tgt_resolved!r}; structural edge supported by CSV."

    # Case 4: description references a known country code
    code_hits = [c for c in all_codes if re.search(r'\b' + c + r'\b', desc)]
    if code_hits:
        return True, f"Description references country code(s) {code_hits}; supported by CSV."

    return False, f"Cannot verify: {src!r} → {tgt!r} not found in CSV."


# ---------------------------------------------------------------------------
# World Bank (Type-II) CSV loading + validation
# ---------------------------------------------------------------------------

def load_wb_csv(path: Path):
    """Load a World Bank CSV (4 header rows). Returns df indexed by Country Code + name map."""
    df = pd.read_csv(str(path), skiprows=4, header=0)
    name_to_code: dict[str, str] = {}
    for _, row in df.iterrows():
        name = row.get("Country Name")
        code = row.get("Country Code")
        if isinstance(name, str) and isinstance(code, str):
            name_to_code[name.lower().strip()] = code.strip()
    df = df.set_index("Country Code")
    return df, name_to_code


def _extract_value_from_wb_target(target: str):
    """Extract numeric value from WB-style target like 'mortality_rate_year2000=136.7_per_1000'."""
    m = re.search(r"=([\d.]+)", target)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_year_from_wb_target(target: str):
    """Extract year from WB-style target like 'mortality_rate_year2000=136.7'."""
    m = re.search(r"year(\d{4})", target, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fall back to 4-digit year anywhere
    m2 = re.search(r'\b(20\d{2}|199\d)\b', target)
    if m2:
        return m2.group(1)
    return None


def _resolve_wb_entity(entity: str, all_codes: set, name_to_code: dict):
    """Resolve entity to a country code for WB data (supports full names and codes)."""
    entity = entity.strip()
    if entity in all_codes:
        return entity
    lower = entity.lower()
    if lower in name_to_code:
        return name_to_code[lower]
    for name, code in name_to_code.items():
        if lower == name:
            return code
    # Partial match (e.g. "Africa Eastern and Southern" in name)
    for name, code in name_to_code.items():
        if lower in name or name in lower:
            return code
    return None


def check_wb_edge(record: dict, df: pd.DataFrame, name_to_code: dict) -> tuple:
    """Validate a World Bank SGE edge against the CSV. Returns (is_correct, reason)."""
    src = record["source_node"]
    tgt = record["target_node"]
    keywords = record["keywords"]
    desc = record["description"]
    all_codes = set(df.index.tolist())

    # Case 1: WB value-label target (e.g. "mortality_rate_year2000=136.7_per_1000")
    emb_val = _extract_value_from_wb_target(tgt)
    emb_year = _extract_year_from_wb_target(tgt)

    if emb_val is not None and emb_year is not None:
        src_resolved = _resolve_wb_entity(src, all_codes, name_to_code)
        if src_resolved:
            csv_val = _lookup_wb_value(df, src_resolved, emb_year)
            if csv_val is not None:
                try:
                    diff = abs(float(csv_val) - emb_val)
                    # Allow rounding tolerance: 0.1% of value or 0.5 absolute
                    tol = max(0.5, abs(float(csv_val)) * 0.001)
                    if diff <= tol:
                        return True, (
                            f"CSV[{src_resolved}][{emb_year}]={csv_val} matches "
                            f"edge embedded value {emb_val}."
                        )
                    else:
                        return False, (
                            f"CSV[{src_resolved}][{emb_year}]={csv_val} ≠ "
                            f"edge embedded value {emb_val} (diff={diff:.3f})."
                        )
                except ValueError:
                    pass
            return False, f"No CSV value for {src_resolved!r} year={emb_year!r}."

        # Source is a full name not in WB country code list (e.g. regional aggregate)
        # Try to match by full country name lookup
        return False, f"Source {src!r} not resolved to CSV country; cannot verify value {emb_val}."

    # Case 2: Entity-entity edge — check if source/target are known countries
    src_resolved = _resolve_wb_entity(src, all_codes, name_to_code)
    if src_resolved:
        return True, f"Source {src!r} → {src_resolved!r}; structural edge supported by CSV."

    tgt_resolved = _resolve_wb_entity(tgt, all_codes, name_to_code)
    if tgt_resolved:
        return True, f"Target {tgt!r} → {tgt_resolved!r}; structural edge supported by CSV."

    # Case 3: Description/keywords reference a country
    combined = desc + " " + keywords
    code_hits = [c for c in all_codes if re.search(r'\b' + c + r'\b', combined)]
    if code_hits:
        return True, f"Edge text references country code(s) {code_hits}; supported by CSV."

    year_in_desc = re.search(r'\b(20\d{2}|199\d)\b', combined)
    val_in_desc = re.search(r'\b\d+\.?\d+\b', combined)
    if year_in_desc and val_in_desc:
        # Has year + number but no matching country; flag as not verifiable
        return False, (
            f"Edge describes year {year_in_desc.group()} / value {val_in_desc.group()} "
            f"but neither {src!r} nor {tgt!r} resolved to a CSV country."
        )

    return False, f"Cannot verify: {src!r} → {tgt!r} not found in CSV."


# ---------------------------------------------------------------------------
# Inpatient CSV loading + validation
# ---------------------------------------------------------------------------

def load_inpatient_csv(path: Path) -> pd.DataFrame:
    """Load Inpatient 2023 CSV. Returns DataFrame with standardized column names."""
    df = pd.read_csv(str(path), encoding="utf-8-sig", skiprows=1, header=0)
    col_names = [
        "icd_code", "disease_name",
        "inpatient_ha", "inpatient_cs", "inpatient_private", "inpatient_total",
        "death_male", "death_female", "death_unknown", "death_total",
    ]
    df.columns = col_names
    # Drop the header-row-as-data (contains the actual column header text)
    df = df[df["icd_code"] != "《疾病和有关健康问题的国际统计分类》第十次修订本的详细序号"].copy()
    df = df.reset_index(drop=True)
    return df


def _get_all_icd_entities(df: pd.DataFrame) -> set:
    codes = set(df["icd_code"].dropna().astype(str).str.strip().tolist())
    names = set(df["disease_name"].dropna().astype(str).str.strip().tolist())
    return codes | names


_INPATIENT_COL_PATTERNS = [
    (r"医院管理局", "inpatient_ha"),
    (r"惩教署", "inpatient_cs"),
    (r"私家医院", "inpatient_private"),
    (r"合计|总计", "inpatient_total"),
    (r"死亡.*男|男.*死亡", "death_male"),
    (r"死亡.*女|女.*死亡", "death_female"),
    (r"死亡.*合计|总.*死亡", "death_total"),
]


def _lookup_inpatient_value(df: pd.DataFrame, key: str, col: str):
    mask = df["icd_code"].astype(str).str.strip() == key.strip()
    if mask.any():
        val = df[mask].iloc[0].get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val)
    mask2 = df["disease_name"].astype(str).str.strip() == key.strip()
    if mask2.any():
        val = df[mask2].iloc[0].get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val)
    return None


def _infer_inpatient_col(keywords: str, desc: str):
    combined = keywords + " " + desc
    for pattern, col in _INPATIENT_COL_PATTERNS:
        if re.search(pattern, combined):
            return col
    return None


def check_inpatient_edge(record: dict, df: pd.DataFrame) -> tuple:
    """Validate an Inpatient SGE edge against the CSV. Returns (is_correct, reason)."""
    src = record["source_node"]
    tgt = record["target_node"]
    keywords = record["keywords"]
    desc = record["description"]
    all_entities = _get_all_icd_entities(df)

    # Case 1a: Pure-number target (e.g. "1605.0") — try all numeric columns
    if _PURE_NUMBER_RE.match(str(tgt).strip()):
        numeric_val = float(str(tgt).strip())
        src_clean = src.split("_")[0].strip() if "_" in src else src
        if src_clean in all_entities or src in all_entities:
            lookup_key = src_clean if src_clean in all_entities else src
            for col_try in ["inpatient_total", "inpatient_ha", "inpatient_cs",
                            "inpatient_private", "death_total", "death_male", "death_female"]:
                csv_val = _lookup_inpatient_value(df, lookup_key, col_try)
                if csv_val is not None:
                    try:
                        if abs(float(str(csv_val).replace(",", "")) - numeric_val) < 0.5:
                            return True, (
                                f"CSV[{lookup_key!r}][{col_try}]={csv_val} matches "
                                f"edge value {numeric_val}."
                            )
                    except ValueError:
                        pass
            return False, (
                f"{src!r} in CSV but numeric value {numeric_val} not verified "
                f"in any column."
            )
        else:
            return False, (
                f"Source {src!r} not a known ICD entity; cannot verify "
                f"pure-numeric target {tgt!r}."
            )

    # Case 1b: Target has embedded numeric value "entity_label: N.N"
    val_m = re.search(r"[:：为=]\s*([\d,]+\.?\d*)$", str(tgt))
    if val_m:
        numeric_val = float(val_m.group(1).replace(",", ""))
        # Extract ICD entity from source
        src_clean = src.split("_")[0].strip() if "_" in src else src
        if src_clean in all_entities or src in all_entities:
            lookup_key = src_clean if src_clean in all_entities else src
            col = _infer_inpatient_col(keywords, desc)
            if col:
                csv_val = _lookup_inpatient_value(df, lookup_key, col)
                if csv_val is not None:
                    try:
                        diff = abs(float(str(csv_val).replace(",", "")) - numeric_val)
                        if diff < 0.5:
                            return True, (
                                f"CSV[{lookup_key!r}][{col}]={csv_val} matches "
                                f"edge value {numeric_val}."
                            )
                        else:
                            return False, (
                                f"CSV[{lookup_key!r}][{col}]={csv_val} ≠ "
                                f"edge value {numeric_val}."
                            )
                    except ValueError:
                        pass
            # Try all columns
            for col_try in ["inpatient_total", "inpatient_ha", "death_total", "death_male", "death_female"]:
                csv_val = _lookup_inpatient_value(df, lookup_key, col_try)
                if csv_val is not None:
                    try:
                        if abs(float(str(csv_val).replace(",", "")) - numeric_val) < 0.5:
                            return True, f"CSV[{lookup_key!r}][{col_try}]={csv_val} matches {numeric_val}."
                    except ValueError:
                        pass
            return False, f"Source {src!r} in CSV but numeric value {numeric_val} not verified."
        else:
            return False, f"Source {src!r} not a known ICD entity; cannot verify numeric claim."

    # Case 2: HAS_SUB_ITEM hierarchy
    if "HAS_SUB_ITEM" in keywords or "HAS_SUB_ITEM" in desc:
        src_in = src in all_entities
        tgt_in = tgt in all_entities
        if src_in and tgt_in:
            return True, f"Both {src!r} and {tgt!r} in CSV; HAS_SUB_ITEM supported."
        elif src_in or tgt_in:
            found = src if src_in else tgt
            missing = tgt if src_in else src
            return False, f"{found!r} in CSV but {missing!r} not found."
        else:
            return False, f"Neither {src!r} nor {tgt!r} in CSV."

    # Case 3: Hospital administrative edge
    hospital_nodes = {"医院管理局辖下医院", "惩教署辖下医院", "私家医院", "住院病人出院及死亡人次"}
    src_is_hosp = src in hospital_nodes or any(h in src for h in hospital_nodes)
    tgt_is_hosp = tgt in hospital_nodes or any(h in tgt for h in hospital_nodes)
    if src_is_hosp or tgt_is_hosp:
        other = tgt if src_is_hosp else src
        other_clean = other.split("(")[0].strip()
        if other_clean in all_entities or other in all_entities:
            return True, f"Hospital admin edge; {other!r} is a valid ICD entity."
        stat_concepts = {"住院病人出院及死亡人次", "全港登记死亡人数", "出院及死亡统计"}
        if any(c in other for c in stat_concepts):
            return True, f"Hospital edge to stat concept {other!r}; structurally valid."
        return False, f"Hospital edge: {other!r} not found as ICD entity."

    # Case 4: Both nodes are known entities
    src_in = src in all_entities
    tgt_in = tgt in all_entities
    if src_in and tgt_in:
        return True, f"Both {src!r} and {tgt!r} in CSV; structural edge supported."
    if src_in or tgt_in:
        found = src if src_in else tgt
        other = tgt if src_in else src
        admin_terms = ["医院", "死亡", "统计", "人次", "登记"]
        if any(t in other for t in admin_terms):
            return True, f"{found!r} in CSV; admin edge to {other!r} plausible."
        return False, f"{found!r} in CSV but {other!r} not verified."

    return False, f"Cannot verify: neither {src!r} nor {tgt!r} in CSV."


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def validate_edge(dataset: str, record: dict, csv_data: dict) -> tuple:
    """Route edge validation to the correct checker based on dataset type."""
    if dataset == "who":
        df, name_to_code = csv_data["who"]
        return check_who_edge(record, df, name_to_code)
    elif dataset in ("wb_cm", "wb_pop", "wb_mat"):
        df, name_to_code = csv_data[dataset]
        return check_wb_edge(record, df, name_to_code)
    elif dataset == "inpatient":
        df = csv_data["inpatient"]
        return check_inpatient_edge(record, df)
    else:
        return False, f"Unknown dataset: {dataset!r}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate paths ---
    for key, path in GRAPHS.items():
        if not path.exists():
            print(f"ERROR: Graph not found for {key!r}: {path}", file=sys.stderr)
            sys.exit(1)
    for key, path in CSVS.items():
        if not path.exists():
            print(f"ERROR: CSV not found for {key!r}: {path}", file=sys.stderr)
            sys.exit(1)

    # --- Load CSVs ---
    print("Loading CSVs...", flush=True)
    csv_data = {}

    df_who, who_name_to_code = load_who_csv(CSVS["who"])
    csv_data["who"] = (df_who, who_name_to_code)
    print(f"  WHO: {len(df_who)} countries")

    for ds_key in ("wb_cm", "wb_pop", "wb_mat"):
        df_wb, wb_name_to_code = load_wb_csv(CSVS[ds_key])
        csv_data[ds_key] = (df_wb, wb_name_to_code)
        print(f"  {ds_key}: {len(df_wb)} countries")

    df_inp = load_inpatient_csv(CSVS["inpatient"])
    csv_data["inpatient"] = df_inp
    print(f"  inpatient: {len(df_inp)} rows")

    # --- Sample + validate per dataset ---
    all_results: dict[str, dict] = {}
    stratum_results: dict[str, dict] = {}
    total_sampled = 0
    total_correct = 0

    for ds_key in ("who", "wb_cm", "wb_pop", "wb_mat", "inpatient"):
        print(f"\n[{ds_key}] Loading graph...", flush=True)
        G, node_names = load_graph(GRAPHS[ds_key])
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        print(f"  Stratified sampling {SAMPLE_PER_DATASET} edges (seed={RANDOM_SEED})...", flush=True)
        records = stratified_sample(G, node_names, SAMPLE_PER_DATASET, RANDOM_SEED)
        print(f"  Sampled: {len(records)}")

        # Show stratum breakdown
        strata_counts: dict[str, int] = {}
        for r in records:
            strata_counts[r["stratum"]] = strata_counts.get(r["stratum"], 0) + 1
        for sk, cnt in sorted(strata_counts.items()):
            print(f"    {sk}: {cnt}")

        # Validate
        ds_correct = 0
        annotated: list[dict] = []
        for i, record in enumerate(records):
            is_correct, reason = validate_edge(ds_key, record, csv_data)
            if is_correct:
                ds_correct += 1
            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"  [{i+1:2d}/{len(records)}] {status}: "
                  f"{record['source_node']!r} -> {record['target_node']!r}")
            print(f"       {reason}")

            ann = {**record, "correct": is_correct, "reason": reason}
            annotated.append(ann)

            # Accumulate stratum stats
            sk = record["stratum"]
            if sk not in stratum_results:
                stratum_results[sk] = {"sampled": 0, "correct": 0}
            stratum_results[sk]["sampled"] += 1
            if is_correct:
                stratum_results[sk]["correct"] += 1

        ds_precision = round(ds_correct / len(records), 4) if records else 0.0
        all_results[ds_key] = {
            "sampled": len(records),
            "correct": ds_correct,
            "precision": ds_precision,
            "samples": annotated,
        }
        total_sampled += len(records)
        total_correct += ds_correct
        print(f"  => {ds_key}: {ds_correct}/{len(records)} correct ({ds_precision:.1%})")

    # --- Compute global Wilson CI ---
    overall_precision = round(total_correct / total_sampled, 4) if total_sampled > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(total_correct, total_sampled)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("STRATIFIED PRECISION SAMPLING RESULTS")
    print("=" * 70)
    for ds_key, res in all_results.items():
        print(f"  {ds_key:12s}: {res['correct']}/{res['sampled']} ({res['precision']:.1%})")
    print(f"  {'TOTAL':12s}: {total_correct}/{total_sampled} ({overall_precision:.1%})")
    print(f"  Wilson 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print("=" * 70)

    # --- Build output structure ---
    per_dataset_out = {
        ds: {"sampled": v["sampled"], "correct": v["correct"], "precision": v["precision"]}
        for ds, v in all_results.items()
    }
    per_stratum_out = {
        sk: {"sampled": v["sampled"], "correct": v["correct"],
             "precision": round(v["correct"] / v["sampled"], 4) if v["sampled"] > 0 else 0.0}
        for sk, v in stratum_results.items()
    }

    output = {
        "total_sampled":  total_sampled,
        "total_correct":  total_correct,
        "precision":      overall_precision,
        "wilson_ci_95":   [ci_lo, ci_hi],
        "per_dataset":    per_dataset_out,
        "per_stratum":    per_stratum_out,
        "samples_detail": {ds: v["samples"] for ds, v in all_results.items()},
    }

    out_path = EVAL_DIR / "results" / "stratified_precision_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
