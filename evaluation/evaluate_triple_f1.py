#!/usr/bin/env python3
"""
evaluate_triple_f1.py — Canonical Triple F1 Evaluation for SGE-LightRAG

Computes standard IE/KG-construction metrics (Precision / Recall / F1) by
extracting canonical triples from both the gold standard and the system graph,
then matching them with flexible normalization rules.

Gold triple format  : (subject_normalized, year, value)
System triple format: extracted from graph edges using three complementary
                      strategies that cover both SGE and baseline output styles.

Extraction strategies:
  A) "year:XXXX" in edge keywords + numeric destination node name  (SGE style)
  B) Year and value both extracted from edge description text       (text style)
  C) Numeric value in edge keywords alongside a year pattern        (mixed style)

Matching rules:
  - subject : substring inclusion (case-insensitive)
  - year    : exact string match (skipped if gold year is empty)
  - value   : numeric equivalence after comma-stripping, trailing-zero
              normalization, and decimal-prefix tolerance

Usage:
    python3 evaluate_triple_f1.py --graph <graphml> --gold <jsonl>
    python3 evaluate_triple_f1.py --graph <graphml> --gold <jsonl> \
        --output results.json --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx not installed.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class GoldTriple(NamedTuple):
    subject: str   # normalized (lowercase)
    year: str      # may be empty string
    value: str     # raw string from gold ("70.83")


class SystemTriple(NamedTuple):
    subject: str   # lowercase node name
    year: str      # may be empty string
    value: str     # raw string extracted from graph


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches 4-digit years in range 1900-2099.
# Uses lookahead/lookbehind instead of \b to work with CJK text.
_YEAR_INLINE = re.compile(r"(?<!\d)(1[9][0-9]{2}|20[0-9]{2})(?!\d)")
# Matches "year:2000" or "year: 2000" style keyword (with optional space)
_YEAR_KW = re.compile(r"year:\s*(\d{4})")
# Matches numeric values (decimals and comma-separated large numbers preferred
# over bare integers to avoid false positives from year-like integers).
# Uses lookahead/lookbehind to work with CJK text.
_VALUE_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)(?!\d)"
)
# Fallback: bare integers (only used in strategy C where other patterns failed)
_VALUE_INT_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")
# A node name that is purely numeric (the SGE graph stores values as node names)
_NUMERIC_NODE = re.compile(r"^-?\d+(?:\.\d+)?$")


# ---------------------------------------------------------------------------
# Gold loading
# ---------------------------------------------------------------------------

def load_gold_triples(jsonl_path: str) -> list[GoldTriple]:
    """Load gold triples as (subject_normalized, year, value) tuples.

    Only StatValue / BudgetAmount / Literal object types are included;
    entity-only records (HAS_SUB_ITEM, etc.) are skipped.
    """
    triples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARN: skipping malformed JSON at line {line_no}: {exc}",
                    file=sys.stderr,
                )
                continue

            triple = record.get("triple", {})
            obj_type = triple.get("object_type", "")
            if obj_type not in ("StatValue", "BudgetAmount", "Literal"):
                continue

            subject = triple.get("subject", "").strip().lower()
            value = triple.get("object", "").strip()
            year = triple.get("attributes", {}).get("year", "").strip()

            if not subject or not value:
                continue

            triples.append(GoldTriple(subject=subject, year=year, value=value))

    return triples


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _node_name(G: nx.Graph, node_id: str) -> str:
    """Return the canonical entity name for a node."""
    data = G.nodes[node_id]
    name = data.get("entity_name") or data.get("name") or data.get("entity_id") or node_id
    return str(name).strip()


def _is_numeric_node(name: str) -> bool:
    """Return True if a node name is a plain numeric value."""
    return bool(_NUMERIC_NODE.match(name))


# ---------------------------------------------------------------------------
# System triple extraction  (three complementary strategies)
# ---------------------------------------------------------------------------

def _extract_values_from_text(text: str, year_set: set[str]) -> list[str]:
    """Extract numeric values from text, excluding year tokens.

    Returns decimal/comma-formatted numbers only (not bare integers that
    could be mistaken for years or row counts).
    """
    return [v for v in _VALUE_RE.findall(text) if v not in year_set]


def _extract_strategy_a(G: nx.Graph) -> list[SystemTriple]:
    """Strategy A: SGE-style — year:XXXX in keywords + numeric destination name.

    Covers the dominant SGE WHO/WB output format:
        AFG → 53.82332641  |  keywords="HAS_VALUE,year:2000"
    """
    triples = []
    for u, v, data in G.edges(data=True):
        kw = str(data.get("keywords", ""))
        m = _YEAR_KW.search(kw)
        if not m:
            continue
        year = m.group(1)

        u_name = _node_name(G, u)
        v_name = _node_name(G, v)

        # Determine which endpoint is the value node
        # Skip when the "numeric node" is actually the year itself (e.g. THE Ranking)
        for subj_name, val_name in ((u_name, v_name), (v_name, u_name)):
            if _is_numeric_node(val_name) and val_name != year:
                triples.append(
                    SystemTriple(
                        subject=subj_name.lower(),
                        year=year,
                        value=val_name,
                    )
                )
    return triples


def _extract_strategy_b(G: nx.Graph) -> list[SystemTriple]:
    """Strategy B: extract year from keywords/description + value from description
    or embedded in destination node name.

    Covers alternate SGE output formats such as WB child mortality:
        ARG → mortality_rate_year2000=19.4_per_1000
        keywords="2000,HAS_VALUE"
        desc="ARG在2000年的五岁以下儿童死亡率为19.4/千活产婴儿"

    Year extraction uses lookahead/lookbehind to work with CJK-adjacent text.
    Year tokens are excluded from value candidates to prevent false positives.
    """
    triples = []
    for u, v, data in G.edges(data=True):
        kw = str(data.get("keywords", ""))
        desc = str(data.get("description", ""))
        v_node_name = _node_name(G, v)

        # Skip edges already fully handled by strategy A
        # (but NOT when the numeric node equals the year — A skips those too)
        m_kw = _YEAR_KW.search(kw)
        if m_kw and _is_numeric_node(v_node_name) and v_node_name != m_kw.group(1):
            continue

        # Collect year candidates from keywords and description
        all_text = f"{kw} {desc}"
        years = _YEAR_INLINE.findall(all_text)
        if not years:
            continue

        year_set = set(years)

        # Collect value candidates:
        # 1. Decimal/comma-formatted numbers in description
        # 2. Decimal/comma-formatted numbers embedded in destination node name
        desc_vals = _extract_values_from_text(desc, year_set)
        node_vals = _extract_values_from_text(v_node_name, year_set)
        values = list(dict.fromkeys(desc_vals + node_vals))  # deduplicate order

        if not values:
            continue

        u_name = _node_name(G, u).lower()
        v_name_lower = v_node_name.lower()

        # Determine subject: prefer the non-numeric, non-year endpoint
        # For reversed edges like "2011 → Stanford University", u is the year
        u_is_year = u_name in year_set or _is_numeric_node(_node_name(G, u))
        subject = v_name_lower if u_is_year else u_name

        for yr in years:
            for val in values:
                triples.append(SystemTriple(subject=subject, year=yr, value=val))

    return triples


# Matches "yearXXXX=VALUE" (standard WB format)
_YEAR_EQ_VALUE = re.compile(r"year(\d{4})\s*=\s*([0-9]+(?:\.\d+)?)")
# Matches bare "_XXXX=VALUE" or "-XXXX=VALUE" (alternate format without "year" prefix)
# e.g. "mortality_rate_2005=4.6_per_1000"
_BARE_YEAR_EQ_VALUE = re.compile(r"[_\-](1[9][0-9]{2}|20[0-9]{2})\s*=\s*([0-9]+(?:\.\d+)?)")


def _year_eq_value_from_name(name: str) -> tuple[str, str] | None:
    """Extract (year, value) from a node name that encodes a year=value pattern.

    Handles two formats:
      - "yearXXXX=VALUE"  (e.g. mortality_rate_year2005=4.6_per_1000)
      - "_XXXX=VALUE"     (e.g. mortality_rate_2005=4.6_per_1000, alternate format)
    Returns None if no pattern is found.
    """
    m = _YEAR_EQ_VALUE.search(name)
    if m:
        return m.group(1), m.group(2)
    m = _BARE_YEAR_EQ_VALUE.search(name)
    if m:
        return m.group(1), m.group(2)
    return None


def _extract_strategy_d(G: nx.Graph) -> list[SystemTriple]:
    """Strategy D: one endpoint encodes yearXXXX=VALUE_unit; the other is the subject.

    Covers SGE output for WB-style datasets in both edge directions and formats:
        Forward (standard)  : Aruba → population_year2000=90588_persons
        Forward (bare year) : FRA → mortality_rate_2005=4.6_per_1000
        Reversed (standard) : mortality_rate_year2000=7_per100k → Germany
        Reversed (bare year): mortality_rate_2005=41.4_per_1000 → IDN
    """
    triples = []
    for u, v, _data in G.edges(data=True):
        u_name = _node_name(G, u)
        v_name = _node_name(G, v)

        # Forward direction: value encoded in destination node
        parsed_v = _year_eq_value_from_name(v_name)
        if parsed_v:
            year, value = parsed_v
            triples.append(SystemTriple(subject=u_name.lower(), year=year, value=value))

        # Reversed direction: value encoded in source node, subject is destination
        parsed_u = _year_eq_value_from_name(u_name)
        if parsed_u:
            year, value = parsed_u
            triples.append(SystemTriple(subject=v_name.lower(), year=year, value=value))

    return triples


def _extract_strategy_c(G: nx.Graph) -> list[SystemTriple]:
    """Strategy C: year and decimal value both in keywords field (no 'year:' prefix).

    Covers cases where a compact keyword like "2005,73.21,LIFE_EXPECTANCY" encodes
    the year and value together.  Only triggers when a decimal/comma-formatted
    value is present alongside a year token.
    """
    triples = []
    for u, v, data in G.edges(data=True):
        kw = str(data.get("keywords", ""))
        # Skip if already handled by strategy A
        if _YEAR_KW.search(kw):
            continue

        years = _YEAR_INLINE.findall(kw)
        if not years:
            continue

        year_set = set(years)
        values = _extract_values_from_text(kw, year_set)

        if not values:
            continue

        u_name = _node_name(G, u).lower()
        v_name = _node_name(G, v).lower()

        for subj in (u_name, v_name):
            for yr in years:
                for val in values:
                    triples.append(SystemTriple(subject=subj, year=yr, value=val))

    return triples


def extract_system_triples(G: nx.Graph) -> list[SystemTriple]:
    """Extract system triples using all three strategies combined."""
    triples: list[SystemTriple] = []
    triples.extend(_extract_strategy_a(G))
    triples.extend(_extract_strategy_b(G))
    triples.extend(_extract_strategy_c(G))
    triples.extend(_extract_strategy_d(G))
    return triples


# ---------------------------------------------------------------------------
# Value normalization for matching
# ---------------------------------------------------------------------------

def _normalize_value(v: str) -> str:
    """Normalize a value string for numeric comparison.

    Steps:
      1. Strip trailing percent sign
      2. Remove comma separators ("1,411,778,724" → "1411778724")
      3. Parse as float and reformat, stripping trailing zeros
         ("70.830" → "70.83", "100.00" → "100")
    Returns the stripped original string if parsing fails.
    """
    v = v.strip().rstrip("%")
    v = v.replace(",", "")
    try:
        f = float(v)
        # Avoid scientific notation for very large/small numbers
        formatted = f"{f:.10f}".rstrip("0").rstrip(".")
        if "e" in formatted.lower():
            return v
        return formatted
    except ValueError:
        return v


def _values_match(gold_val: str, sys_val: str) -> bool:
    """Return True if the two value strings represent the same number.

    Handles:
      - Trailing zeros: "70.830" == "70.83"
      - Comma separators: "1,411" == "1411"
      - Decimal-prefix tolerance: gold "70.83" matches sys "70.83886138"
        (gold is a rounded/truncated form of the full-precision system value)
    """
    gn = _normalize_value(gold_val)
    sn = _normalize_value(sys_val)

    if gn == sn:
        return True

    # Decimal-prefix tolerance: gold truncated to fewer decimals
    # "70.83" should match "70.83886138" but not "70.84"
    if "." in gn and "." in sn:
        if sn.startswith(gn) or gn.startswith(sn):
            return True

    # Integer-prefix tolerance: "70" matches "70.83" only when no decimal in gold
    if "." not in gn and sn.startswith(gn + "."):
        return True

    # Substring containment as final fallback
    return gn in sn or sn in gn


# ---------------------------------------------------------------------------
# Subject matching
# ---------------------------------------------------------------------------

def _subjects_match(gold_subj: str, sys_subj: str) -> bool:
    """Return True if the subjects refer to the same entity (case-insensitive)."""
    g = gold_subj.lower()
    s = sys_subj.lower()
    return g in s or s in g


# ---------------------------------------------------------------------------
# Triple matching
# ---------------------------------------------------------------------------

def _triple_matches(gold: GoldTriple, sys: SystemTriple) -> bool:
    """Return True if a system triple satisfies a gold triple."""
    if not _subjects_match(gold.subject, sys.subject):
        return False
    if gold.year and gold.year != sys.year:
        return False
    return _values_match(gold.value, sys.value)


def compute_matches(
    gold_triples: list[GoldTriple],
    system_triples: list[SystemTriple],
) -> tuple[int, int, int]:
    """Return (matched_gold_count, total_gold, matched_system_unique_count).

    matched_gold_count        : number of gold triples with at least one
                                matching system triple (drives Recall)
    matched_system_unique_count : number of unique system triples that match
                                  at least one gold triple (drives Precision)
    """
    matched_gold_count = 0
    matched_sys_indices: set[int] = set()

    for gold in gold_triples:
        for i, sys in enumerate(system_triples):
            if _triple_matches(gold, sys):
                matched_gold_count += 1
                matched_sys_indices.add(i)
                break  # one match per gold triple is sufficient for recall

    return matched_gold_count, len(gold_triples), len(matched_sys_indices)


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def compute_prf(
    matched_gold: int,
    total_gold: int,
    matched_sys: int,
    total_sys: int,
) -> dict:
    """Compute Precision, Recall, and F1 from match counts."""
    recall = matched_gold / total_gold if total_gold > 0 else 0.0
    precision = matched_sys / total_sys if total_sys > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_gold": matched_gold,
        "total_gold": total_gold,
        "matched_system": matched_sys,
        "total_system": total_sys,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    print()
    print("=" * 55)
    print("CANONICAL TRIPLE F1 EVALUATION")
    print("=" * 55)
    print(
        f"\n  Precision : {results['precision']:.4f}  "
        f"({results['matched_system']} / {results['total_system']})"
    )
    print(
        f"  Recall    : {results['recall']:.4f}  "
        f"({results['matched_gold']} / {results['total_gold']})"
    )
    print(f"  F1        : {results['f1']:.4f}")
    print("=" * 55)


def _print_unmatched(
    gold_triples: list[GoldTriple],
    system_triples: list[SystemTriple],
) -> None:
    print("\n[Unmatched Gold Triples]")
    for gold in gold_triples:
        matched = any(_triple_matches(gold, sys) for sys in system_triples)
        if not matched:
            print(
                f"  subj={gold.subject!r}  year={gold.year!r}  "
                f"value={gold.value!r}"
            )


def _maybe_write(results: dict, output_path: str | None) -> None:
    if output_path is None:
        print("\n[JSON]")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical triple F1 evaluation for SGE-LightRAG graphs."
    )
    parser.add_argument("--graph", required=True, help="Path to graphml file")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument(
        "--output", default=None, help="Write JSON results to this file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print unmatched gold triples"
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    graph_path = Path(args.graph)
    gold_path = Path(args.gold)

    if not graph_path.exists():
        print(f"ERROR: graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)
    if not gold_path.exists():
        print(f"ERROR: gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    # Load
    gold_triples = load_gold_triples(str(gold_path))
    if not gold_triples:
        print(
            "WARN: no gold triples loaded — check object_type filter",
            file=sys.stderr,
        )

    G = nx.read_graphml(str(graph_path))
    system_triples = extract_system_triples(G)

    print(
        f"Gold  : {len(gold_triples)} triples\n"
        f"Graph : {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(system_triples)} extracted system triples"
    )

    if not gold_triples or not system_triples:
        results = compute_prf(0, len(gold_triples), 0, len(system_triples))
        _print_results(results)
        _maybe_write(results, args.output)
        return

    # Match and score
    matched_gold, total_gold, matched_sys = compute_matches(
        gold_triples, system_triples
    )
    results = compute_prf(matched_gold, total_gold, matched_sys, len(system_triples))

    _print_results(results)

    if args.verbose:
        _print_unmatched(gold_triples, system_triples)

    _maybe_write(results, args.output)


if __name__ == "__main__":
    main()
