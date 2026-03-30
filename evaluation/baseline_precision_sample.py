#!/usr/bin/env python3
"""
baseline_precision_sample.py — Baseline Precision Sampling for SGE-LightRAG

Randomly samples 25 edges each from:
  - Baseline WHO Life Expectancy graph
  - Baseline Inpatient 2023 graph

Cross-checks each edge against the source CSV to determine if the claim
is factually supported. Saves annotated results to
evaluation/results/baseline_precision_results.json.

Usage:
    python3 evaluation/baseline_precision_sample.py

Run from sge_lightrag/ directory.
"""

import json
import random
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # sge_lightrag/
EVAL_DIR = Path(__file__).parent         # sge_lightrag/evaluation/
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = BASE_DIR / "dataset"

BASELINE_WHO_GRAPH = OUTPUT_DIR / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"
BASELINE_INPATIENT_GRAPH = OUTPUT_DIR / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"

WHO_CSV = DATASET_DIR / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"
INPATIENT_CSV = DATASET_DIR / "住院病人统计" / "Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv"

SAMPLE_SIZE = 25
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Graph loading (immutable — returns new objects)
# ---------------------------------------------------------------------------

def load_graph_edges(graphml_path: Path) -> tuple:
    """Load graph and build a name→description index.

    Returns:
        (G, node_desc): networkx.Graph, dict mapping node display name to
        its description string.
    """
    G = nx.read_graphml(str(graphml_path))
    node_desc: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        name = str(data.get("entity_name") or data.get("name") or nid).strip()
        node_desc[name] = str(data.get("description", ""))
    return G, node_desc


def sample_edges(G, n: int, seed: int) -> list[dict]:
    """Randomly sample n edges from graph G. Returns list of annotation records."""
    rng = random.Random(seed)
    all_edges = list(G.edges(data=True))
    sampled = rng.sample(all_edges, min(n, len(all_edges)))

    records = []
    for u, v, data in sampled:
        u_name = str(G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u).strip()
        v_name = str(G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v).strip()
        records.append({
            "source_node": u_name,
            "target_node": v_name,
            "keywords": str(data.get("keywords", "")),
            "description": str(data.get("description", "")),
            "correct": None,
            "reason": "",
        })
    return records


# ---------------------------------------------------------------------------
# WHO CSV lookup helpers
# ---------------------------------------------------------------------------

def load_who_csv(csv_path: Path) -> tuple:
    """Load WHO life expectancy CSV.

    Returns:
        (df_indexed, country_name_to_code): DataFrame indexed by Country Code,
        and a dict mapping lowercase country names to country codes.
    """
    df = pd.read_csv(str(csv_path), header=0)
    # Columns: Country Name, Country Code, Indicator Name, Indicator Code, 2000, 2001, ...
    name_to_code: dict[str, str] = {}
    for _, row in df.iterrows():
        name = row.get("Country Name")
        code = row.get("Country Code")
        if isinstance(name, str) and isinstance(code, str):
            name_to_code[name.lower().strip()] = code.strip()
    df = df.set_index("Country Code")
    return df, name_to_code


def lookup_who_value(df: pd.DataFrame, country_code: str, year: str) -> str | None:
    """Return the WHO CSV value for (country_code, year), or None if not found."""
    year_str = str(year).strip()
    if country_code not in df.index:
        return None
    if year_str not in df.columns:
        return None
    val = df.loc[country_code, year_str]
    if pd.isna(val):
        return None
    return str(val)


def extract_year_from_keywords(keywords: str) -> str | None:
    """Try to extract year from keywords string like 'HAS_VALUE,year:2010'."""
    for part in keywords.split(","):
        part = part.strip()
        if part.startswith("year:"):
            return part[5:].strip()
    return None


def _resolve_who_entity(entity: str, all_codes: set, name_to_code: dict) -> str | None:
    """Try to resolve an entity string to a CSV country code.

    Handles:
    - ISO 3-letter country codes (e.g. "KAZ")
    - Composite node names like "BDI / WHOSIS_000001" → "BDI"
    - Country full names (e.g. "Nicaragua", "Bosnia And Herzegovina") → code lookup
    - Returns None if the entity cannot be resolved to a known CSV entry.
    """
    entity = entity.strip()
    # Direct match
    if entity in all_codes:
        return entity
    # Composite: "CODE / INDICATOR" pattern
    if "/" in entity:
        code_part = entity.split("/")[0].strip()
        if code_part in all_codes:
            return code_part
    # Full country name lookup (case-insensitive)
    lower = entity.lower()
    if lower in name_to_code:
        return name_to_code[lower]
    # Partial match for common name variations (e.g. "Cote d'Ivoire" vs "Côte d'Ivoire")
    for name, code in name_to_code.items():
        if lower in name or name in lower:
            return code
    return None


def check_who_edge(
    record: dict,
    df: pd.DataFrame,
    name_to_code: dict,
    node_desc: dict | None = None,
) -> tuple[bool, str]:
    """Check if a WHO Baseline edge is supported by the source CSV.

    Strategy:
    1. If the description/target contains a numeric value and a year reference,
       try to match (source_node, year, value) against the CSV.
    2. If the edge relates a known country (code or name) to a concept or
       indicator, mark as correct (structural/summary edge).
    3. If the edge involves a composite node "CODE / WHOSIS_000001" that
       resolves to a known country, mark as correct.
    4. If the edge involves an external entity (e.g. "基线数据集" — baseline
       dataset node) that refers to a country provably in the CSV, mark correct.
    5. Otherwise mark as incorrect.
    """
    import re

    src = record["source_node"]
    tgt = record["target_node"]
    keywords = record["keywords"]
    desc = record["description"]

    all_codes = set(df.index.tolist())

    # --- Case 1: Edge encodes a numeric value (target is a number) ---
    try:
        float(tgt.replace(",", ""))
        target_is_numeric = True
    except (ValueError, AttributeError):
        target_is_numeric = False

    if target_is_numeric:
        year = extract_year_from_keywords(keywords)
        # Try to resolve source to country code
        resolved = _resolve_who_entity(src, all_codes, name_to_code)
        if year and resolved:
            csv_val = lookup_who_value(df, resolved, year)
            if csv_val is not None:
                try:
                    csv_float = float(csv_val)
                    tgt_float = float(tgt.replace(",", ""))
                    if abs(csv_float - tgt_float) < 0.001:
                        return True, (
                            f"CSV[{resolved}][{year}] = {csv_val}, "
                            f"edge target = {tgt}, exact match."
                        )
                    else:
                        return False, (
                            f"CSV[{resolved}][{year}] = {csv_val}, "
                            f"but edge claims {tgt} (mismatch)."
                        )
                except ValueError:
                    pass
            return False, (
                f"Country {resolved!r} year {year!r} not found or value mismatch."
            )
        elif not year:
            year_match = re.search(r'\b(20\d{2}|199\d)\b', desc)
            if year_match and resolved:
                year = year_match.group(1)
                csv_val = lookup_who_value(df, resolved, year)
                if csv_val is not None:
                    try:
                        csv_float = float(csv_val)
                        tgt_float = float(tgt.replace(",", ""))
                        if abs(csv_float - tgt_float) < 0.001:
                            return True, (
                                f"CSV[{resolved}][{year}] = {csv_val}, "
                                f"edge = {tgt}, match (year from desc)."
                            )
                        else:
                            return False, (
                                f"CSV[{resolved}][{year}] = {csv_val}, "
                                f"but edge claims {tgt} (mismatch)."
                            )
                    except ValueError:
                        pass
        return False, (
            f"Numeric target {tgt!r} cannot be verified: "
            f"no year or country match for {src!r}."
        )

    # --- Case 2: Source is WHOSIS_000001 or a composite indicator node ---
    if "WHOSIS_000001" in src or src == "WHOSIS_000001":
        # Check if target resolves to a known country or concept
        tgt_resolved = _resolve_who_entity(tgt, all_codes, name_to_code)
        if tgt_resolved:
            return True, (
                f"WHO indicator node {src!r} → country {tgt_resolved!r}; "
                f"structural edge supported by CSV."
            )
        # Target is a named time series summary or concept string
        if any(c in tgt for c in ["预期寿命", "出生时", "时间序列", "数据", "统计"]):
            return True, (
                f"WHO indicator {src!r} → concept summary {tgt!r}; "
                f"structural edge is valid in CSV context."
            )
        # Check target node description for country codes or English names
        if node_desc is not None:
            tgt_node_text = node_desc.get(tgt, "") + " " + node_desc.get(src, "")
            import re as _re2
            nd_code_hits = [c for c in all_codes if _re2.search(r'\b' + c + r'\b', tgt_node_text)]
            if nd_code_hits:
                return True, (
                    f"WHO indicator {src!r} → {tgt!r}: node descriptions "
                    f"reference CSV country code(s) {nd_code_hits!r}; supported."
                )
            nd_name_hits = [n for n in name_to_code if n in tgt_node_text.lower()]
            if nd_name_hits:
                code_hit = name_to_code[nd_name_hits[0]]
                return True, (
                    f"WHO indicator {src!r} → {tgt!r}: node description mentions "
                    f"country {nd_name_hits[0]!r} → {code_hit!r}; supported by CSV."
                )
        return False, (
            f"WHO indicator node {src!r} → {tgt!r}: target not verifiable in CSV."
        )

    # --- Case 3: Source resolves to a known country code ---
    src_resolved = _resolve_who_entity(src, all_codes, name_to_code)
    if src_resolved:
        return True, (
            f"Source {src!r} resolves to CSV country {src_resolved!r}; "
            f"structural edge is supported."
        )

    # --- Case 4: Target resolves to a known country code ---
    tgt_resolved = _resolve_who_entity(tgt, all_codes, name_to_code)
    if tgt_resolved:
        return True, (
            f"Target {tgt!r} resolves to CSV country {tgt_resolved!r}; "
            f"structural edge is supported."
        )

    # --- Case 5: "出生时预期寿命" concept node with descriptive target ---
    # The concept node links to named country time-series summaries.
    # These are valid if the description references a known country.
    if src == "出生时预期寿命":
        # Check if description or target references a known country
        countries_mentioned = [
            code for name, code in name_to_code.items()
            if name in desc.lower()
        ]
        codes_mentioned = [c for c in all_codes if c in desc]
        if countries_mentioned or codes_mentioned:
            found = (countries_mentioned + codes_mentioned)[0]
            return True, (
                f"Concept edge from '出生时预期寿命' to {tgt!r}; "
                f"description references known country {found!r}."
            )
        # Target is a named time series referencing a known country in its name
        tgt_resolved2 = _resolve_who_entity(
            re.sub(r'出生时预期寿命.*', '', tgt).strip(), all_codes, name_to_code
        )
        if tgt_resolved2:
            return True, (
                f"Concept edge from '出生时预期寿命' to time-series summary "
                f"{tgt!r}; country {tgt_resolved2!r} found in CSV."
            )

    # --- Case 6: Fallback — scan edge description + node descriptions ---
    # The graph may encode Chinese country names whose node descriptions contain
    # the ISO code or English name (e.g. "摩洛哥(Morocco) … MAR / WHOSIS_000001").
    import re as _re

    # Gather all text to search: edge desc + src/tgt node descriptions (if available)
    extra_texts = []
    if node_desc is not None:
        extra_texts.append(node_desc.get(src, ""))
        extra_texts.append(node_desc.get(tgt, ""))
    search_text = desc + " " + " ".join(extra_texts)

    # Check for 3-letter country codes
    code_hits = [c for c in all_codes if _re.search(r'\b' + c + r'\b', search_text)]
    if code_hits:
        return True, (
            f"Edge/node descriptions reference CSV country code(s) {code_hits!r}; "
            f"edge is supported by CSV data."
        )
    # Check for English country names
    name_hits = [name for name in name_to_code if name in search_text.lower()]
    if name_hits:
        code_hit = name_to_code[name_hits[0]]
        return True, (
            f"Edge/node descriptions mention country {name_hits[0]!r} → {code_hit!r}; "
            f"edge is supported by CSV data."
        )
    # For node names containing time-series pattern, check target name itself
    if "时间序列数据" in tgt or "出生时预期寿命" in tgt:
        name_in_tgt = [name for name in name_to_code if name in tgt.lower()]
        if name_in_tgt:
            code_hit = name_to_code[name_in_tgt[0]]
            return True, (
                f"Target {tgt!r} contains country name {name_in_tgt[0]!r} → "
                f"{code_hit!r}; time-series summary edge supported by CSV."
            )

    # --- Case 7: Cannot verify ---
    return False, (
        f"Cannot verify: source {src!r} and target {tgt!r} "
        f"not found in CSV as country codes, names, or indicator."
    )


# ---------------------------------------------------------------------------
# Inpatient CSV lookup helpers
# ---------------------------------------------------------------------------

def load_inpatient_csv(csv_path: Path) -> pd.DataFrame:
    """Load Inpatient 2023 CSV. Returns DataFrame with cleaned column names."""
    # Row 0 is title, row 1 is actual header
    df = pd.read_csv(str(csv_path), encoding="utf-8-sig", skiprows=1, header=0)
    # Rename columns based on the actual header row content
    col_names = [
        "icd_code", "disease_name",
        "inpatient_ha", "inpatient_cs", "inpatient_private", "inpatient_total",
        "death_male", "death_female", "death_unknown", "death_total",
    ]
    df.columns = col_names
    # Drop the header row that was read as data (first row is actual column headers)
    df = df[df["icd_code"] != "《疾病和有关健康问题的国际统计分类》第十次修订本的详细序号"].copy()
    df = df.reset_index(drop=True)
    return df


def lookup_inpatient_value(
    df: pd.DataFrame,
    key: str,
    col: str,
) -> str | None:
    """Return the inpatient CSV value for (icd_code or disease_name, col), or None."""
    # Try matching by ICD code first
    mask = df["icd_code"].astype(str).str.strip() == key.strip()
    if mask.any():
        row = df[mask].iloc[0]
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val)

    # Try matching by disease name
    mask2 = df["disease_name"].astype(str).str.strip() == key.strip()
    if mask2.any():
        row = df[mask2].iloc[0]
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val)

    return None


# Column keyword to DataFrame column mapping for inpatient
INPATIENT_COL_MAP = {
    "医院管理局辖下医院": "inpatient_ha",
    "惩教署辖下医院": "inpatient_cs",
    "私家医院": "inpatient_private",
    "合计": "inpatient_total",
    "总计": "inpatient_total",
    "全港登记死亡人数-男性": "death_male",
    "死亡.*男": "death_male",
    "全港登记死亡人数-女性": "death_female",
    "死亡.*女": "death_female",
    "全港登记死亡人数-合计": "death_total",
}


def infer_inpatient_col(keywords: str, description: str) -> str | None:
    """Infer which CSV column to look up from keywords/description."""
    import re
    combined = keywords + " " + description
    for pattern, col in INPATIENT_COL_MAP.items():
        if re.search(pattern, combined):
            return col
    return None


def get_all_icd_codes(df: pd.DataFrame) -> set:
    """Return all ICD codes and disease names from the inpatient CSV."""
    codes = set(df["icd_code"].dropna().astype(str).str.strip().tolist())
    names = set(df["disease_name"].dropna().astype(str).str.strip().tolist())
    return codes | names


def check_inpatient_edge(record: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """Check if an Inpatient Baseline edge is supported by the source CSV.

    Strategy:
    1. If source or target is a known ICD code / disease name and the edge
       encodes a numeric value, verify the numeric claim against the CSV.
    2. If the edge is a structural relation (HAS_SUB_ITEM, category hierarchy),
       verify both nodes exist in the CSV.
    3. If the edge is about hospital type (structural/administrative), verify
       that the ICD entity is in the CSV.
    4. Otherwise mark as incorrect.
    """
    src = record["source_node"]
    tgt = record["target_node"]
    keywords = record["keywords"]
    desc = record["description"]

    all_entities = get_all_icd_codes(df)

    # --- Case 1: Target is a numeric value ---
    try:
        float(str(tgt).replace(",", "").split("(")[0].split(":")[0].strip())
        target_is_numeric = True
        numeric_val = float(str(tgt).replace(",", "").split("(")[0].split(":")[0].strip())
    except (ValueError, AttributeError):
        target_is_numeric = False
        numeric_val = None

    if target_is_numeric:
        # Source should be an ICD code or disease name
        if src in all_entities or src.split("_")[0].strip() in all_entities:
            col = infer_inpatient_col(keywords, desc)
            lookup_key = src.split("_")[0].strip() if "_" in src else src

            # Try direct lookup
            if col:
                csv_val = lookup_inpatient_value(df, lookup_key, col)
                if csv_val is not None:
                    try:
                        csv_float = float(str(csv_val).replace(",", ""))
                        if abs(csv_float - numeric_val) < 0.5:
                            return True, f"CSV[{lookup_key!r}][{col}] = {csv_val}, edge = {tgt}, exact match."
                        else:
                            return False, f"CSV[{lookup_key!r}][{col}] = {csv_val}, but edge claims {tgt} (mismatch)."
                    except ValueError:
                        pass
            # No column identified — try all columns
            for col_try in ["inpatient_total", "inpatient_ha", "inpatient_private", "death_total", "death_male", "death_female"]:
                csv_val = lookup_inpatient_value(df, lookup_key, col_try)
                if csv_val is not None:
                    try:
                        csv_float = float(str(csv_val).replace(",", ""))
                        if abs(csv_float - numeric_val) < 0.5:
                            return True, f"CSV[{lookup_key!r}][{col_try}] = {csv_val}, edge = {tgt}, match found."
                    except ValueError:
                        pass
            return False, f"Source {src!r} in CSV but numeric value {tgt!r} not verified in any column."
        else:
            # Source not a known ICD entity
            return False, f"Source {src!r} not a known ICD code or disease name; cannot verify numeric claim."

    # --- Case 2: HAS_SUB_ITEM or category hierarchy ---
    if "HAS_SUB_ITEM" in keywords or "HAS_SUB_ITEM" in desc:
        src_in = src in all_entities
        tgt_in = tgt in all_entities
        if src_in and tgt_in:
            return True, f"Both {src!r} and {tgt!r} found in CSV; HAS_SUB_ITEM relation supported."
        elif src_in or tgt_in:
            found = src if src_in else tgt
            missing = tgt if src_in else src
            return False, f"{found!r} in CSV but {missing!r} not found; partial match only."
        else:
            return False, f"Neither {src!r} nor {tgt!r} found in CSV; HAS_SUB_ITEM not supported."

    # --- Case 3: Structural administrative edges ---
    # e.g. "霍乱 → 医院管理局辖下医院" or "医院管理局辖下医院 → 伤寒和副伤寒"
    hospital_nodes = {"医院管理局辖下医院", "惩教署辖下医院", "私家医院", "住院病人出院及死亡人次"}
    src_is_hospital = src in hospital_nodes or any(h in src for h in hospital_nodes)
    tgt_is_hospital = tgt in hospital_nodes or any(h in tgt for h in hospital_nodes)

    if src_is_hospital or tgt_is_hospital:
        # The other node should be an ICD entity
        other = tgt if src_is_hospital else src
        # Strip possible suffixes
        other_clean = other.split("(")[0].strip()
        if other_clean in all_entities or other in all_entities:
            return True, f"Hospital administrative edge: {other!r} is a valid ICD entity in CSV."
        else:
            # Check if it's a generic statistical concept
            statistical_concepts = {"住院病人出院及死亡人次", "全港登记死亡人数", "出院及死亡统计"}
            if any(c in other for c in statistical_concepts):
                return True, f"Hospital edge to statistical concept {other!r}; structurally valid in CSV context."
            return False, f"Hospital administrative edge: {other!r} not found as ICD entity in CSV."

    # --- Case 4: Both nodes in CSV entities (generic structural) ---
    src_in = src in all_entities
    tgt_in = tgt in all_entities
    if src_in and tgt_in:
        return True, f"Both {src!r} and {tgt!r} found in CSV; structural edge supported."
    if src_in or tgt_in:
        found = src if src_in else tgt
        other = tgt if src_in else src
        # Allow if the other is a general hospital/admin concept
        admin_terms = ["医院", "死亡", "统计", "人次", "登记"]
        if any(t in other for t in admin_terms):
            return True, f"{found!r} in CSV; administrative edge to {other!r} is structurally plausible."
        return False, f"{found!r} in CSV but {other!r} not verified; edge partially supported only."

    # --- Case 5: Cannot verify ---
    return False, f"Cannot verify: neither {src!r} nor {tgt!r} found in CSV entities."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading WHO Baseline graph...", flush=True)
    if not BASELINE_WHO_GRAPH.exists():
        print(f"ERROR: Baseline WHO graph not found: {BASELINE_WHO_GRAPH}", file=sys.stderr)
        sys.exit(1)
    G_who, who_node_desc = load_graph_edges(BASELINE_WHO_GRAPH)
    print(f"  Nodes: {G_who.number_of_nodes()}, Edges: {G_who.number_of_edges()}")

    print("Loading Inpatient Baseline graph...", flush=True)
    if not BASELINE_INPATIENT_GRAPH.exists():
        print(f"ERROR: Baseline Inpatient graph not found: {BASELINE_INPATIENT_GRAPH}", file=sys.stderr)
        sys.exit(1)
    G_inpatient, _inpatient_node_desc = load_graph_edges(BASELINE_INPATIENT_GRAPH)
    print(f"  Nodes: {G_inpatient.number_of_nodes()}, Edges: {G_inpatient.number_of_edges()}")

    print("\nLoading WHO CSV...", flush=True)
    if not WHO_CSV.exists():
        print(f"ERROR: WHO CSV not found: {WHO_CSV}", file=sys.stderr)
        sys.exit(1)
    df_who, who_name_to_code = load_who_csv(WHO_CSV)
    print(f"  Loaded {len(df_who)} country rows, {len(df_who.columns)} year columns")

    print("Loading Inpatient CSV...", flush=True)
    if not INPATIENT_CSV.exists():
        print(f"ERROR: Inpatient CSV not found: {INPATIENT_CSV}", file=sys.stderr)
        sys.exit(1)
    df_inpatient = load_inpatient_csv(INPATIENT_CSV)
    print(f"  Loaded {len(df_inpatient)} rows")

    # --- Sample WHO edges ---
    print(f"\nSampling {SAMPLE_SIZE} edges from WHO Baseline (seed={RANDOM_SEED})...", flush=True)
    who_samples = sample_edges(G_who, SAMPLE_SIZE, RANDOM_SEED)

    print("Cross-checking WHO samples against source CSV...", flush=True)
    who_annotated = []
    who_correct = 0
    for i, record in enumerate(who_samples):
        is_correct, reason = check_who_edge(record, df_who, who_name_to_code, who_node_desc)
        status = "CORRECT" if is_correct else "INCORRECT"
        annotated = {**record, "correct": is_correct, "reason": reason}
        who_annotated.append(annotated)
        if is_correct:
            who_correct += 1
        print(f"  [{i+1:2d}] {status}: {record['source_node']!r} -> {record['target_node']!r}")
        print(f"       {reason}")

    # --- Sample Inpatient edges ---
    print(f"\nSampling {SAMPLE_SIZE} edges from Inpatient Baseline (seed={RANDOM_SEED})...", flush=True)
    inpatient_samples = sample_edges(G_inpatient, SAMPLE_SIZE, RANDOM_SEED)

    print("Cross-checking Inpatient samples against source CSV...", flush=True)
    inpatient_annotated = []
    inpatient_correct = 0
    for i, record in enumerate(inpatient_samples):
        is_correct, reason = check_inpatient_edge(record, df_inpatient)
        status = "CORRECT" if is_correct else "INCORRECT"
        annotated = {**record, "correct": is_correct, "reason": reason}
        inpatient_annotated.append(annotated)
        if is_correct:
            inpatient_correct += 1
        print(f"  [{i+1:2d}] {status}: {record['source_node']!r} -> {record['target_node']!r}")
        print(f"       {reason}")

    # --- Summary ---
    total_correct = who_correct + inpatient_correct
    total_samples = SAMPLE_SIZE * 2

    print("\n" + "=" * 70)
    print("BASELINE PRECISION SAMPLING RESULTS")
    print("=" * 70)
    print(f"WHO Life Expectancy:  {who_correct}/{SAMPLE_SIZE} correct "
          f"({who_correct/SAMPLE_SIZE:.1%})")
    print(f"Inpatient 2023:       {inpatient_correct}/{SAMPLE_SIZE} correct "
          f"({inpatient_correct/SAMPLE_SIZE:.1%})")
    print(f"TOTAL:                {total_correct}/{total_samples} correct "
          f"({total_correct/total_samples:.1%})")
    print("=" * 70)

    # --- Save results ---
    results = {
        "metadata": {
            "seed": RANDOM_SEED,
            "sample_size_per_dataset": SAMPLE_SIZE,
            "total_samples": total_samples,
        },
        "summary": {
            "who_life_expectancy": {
                "correct": who_correct,
                "total": SAMPLE_SIZE,
                "precision": round(who_correct / SAMPLE_SIZE, 4),
            },
            "inpatient_2023": {
                "correct": inpatient_correct,
                "total": SAMPLE_SIZE,
                "precision": round(inpatient_correct / SAMPLE_SIZE, 4),
            },
            "combined": {
                "correct": total_correct,
                "total": total_samples,
                "precision": round(total_correct / total_samples, 4),
            },
        },
        "who_life_expectancy_samples": who_annotated,
        "inpatient_2023_samples": inpatient_annotated,
    }

    output_path = EVAL_DIR / "results" / "baseline_precision_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
