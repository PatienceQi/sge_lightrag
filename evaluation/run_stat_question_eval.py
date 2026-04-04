#!/usr/bin/env python3
"""
run_stat_question_eval.py — Evaluates answerability of stat analysis questions
on SGE vs Baseline knowledge graphs.

Input:  evaluation/gold/stat_analysis_questions_200.jsonl (231 questions)
Output: evaluation/results/stat_question_eval_results.json

Answerability logic:
  L1 (point_lookup): answer value reachable from answer entity within 2 hops
  L2 (ranking):      all answer entities have their year-facts bound correctly
  L3 (aggregation):  all year-facts for the given year exist in the graph
  L4 (trend/comparison): answer entities have facts for all required years
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow running from project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.graph_loaders import load_graph_auto  # noqa: E402

# ---------------------------------------------------------------------------
# Graph path configuration
# ---------------------------------------------------------------------------

_OUTPUT = _ROOT / "output"

GRAPH_PATHS: dict[str, dict[str, str]] = {
    "sge": {
        "WHO": str(_OUTPUT / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_ChildMortality": str(_OUTPUT / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_Population": str(_OUTPUT / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_Maternal": str(_OUTPUT / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "Inpatient": str(_OUTPUT / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
    },
    "baseline": {
        "WHO": str(_OUTPUT / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_ChildMortality": str(_OUTPUT / "baseline_wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_Population": str(_OUTPUT / "baseline_wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "WB_Maternal": str(_OUTPUT / "baseline_wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
        "Inpatient": str(_OUTPUT / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml"),
    },
}

QUESTIONS_FILE = Path(__file__).parent / "gold" / "stat_analysis_questions_200.jsonl"
RESULTS_FILE = Path(__file__).parent / "results" / "stat_question_eval_results.json"


# ---------------------------------------------------------------------------
# Entity matching helpers
# ---------------------------------------------------------------------------

def _build_search_index(nodes: dict) -> dict[str, str]:
    """
    Build a lowercase → original name index for fuzzy entity lookup.
    Returns {lowercase_key: original_node_name}.
    """
    index: dict[str, str] = {}
    for name in nodes:
        index[name.lower()] = name
    return index


# ISO 3166-1 alpha-3 codes used in World Bank datasets
_COUNTRY_ALIASES: dict[str, list[str]] = {
    "afg": ["afghanistan"], "ago": ["angola"], "alb": ["albania"],
    "are": ["united arab emirates"], "arg": ["argentina"], "arm": ["armenia"],
    "aus": ["australia"], "aut": ["austria"], "aze": ["azerbaijan"],
    "bdi": ["burundi"], "bel": ["belgium"], "ben": ["benin"],
    "bfa": ["burkina faso"], "bgd": ["bangladesh"], "bgr": ["bulgaria"],
    "bhr": ["bahrain"], "bhs": ["bahamas"], "bih": ["bosnia and herzegovina"],
    "blr": ["belarus"], "blz": ["belize"], "bol": ["bolivia"],
    "bra": ["brazil"], "brb": ["barbados"], "brn": ["brunei"],
    "btn": ["bhutan"], "bwa": ["botswana"], "caf": ["central african republic"],
    "can": ["canada"], "che": ["switzerland"], "chl": ["chile"],
    "chn": ["china"], "civ": ["cote d'ivoire", "ivory coast"],
    "cmr": ["cameroon"], "cod": ["congo, dem. rep.", "democratic republic of congo"],
    "cog": ["congo, rep.", "republic of congo"], "col": ["colombia"],
    "com": ["comoros"], "cpv": ["cabo verde", "cape verde"],
    "cri": ["costa rica"], "cub": ["cuba"], "cyp": ["cyprus"],
    "cze": ["czech republic", "czechia"], "deu": ["germany"],
    "dji": ["djibouti"], "dnk": ["denmark"], "dom": ["dominican republic"],
    "dza": ["algeria"], "ecu": ["ecuador"],
    "egy": ["egypt", "egypt, arab rep."], "eri": ["eritrea"],
    "esp": ["spain"], "est": ["estonia"], "eth": ["ethiopia"],
    "fin": ["finland"], "fji": ["fiji"], "fra": ["france"],
    "gab": ["gabon"], "gbr": ["united kingdom"],
    "geo": ["georgia"], "gha": ["ghana"], "gin": ["guinea"],
    "gmb": ["gambia"], "gnb": ["guinea-bissau"], "gnq": ["equatorial guinea"],
    "grc": ["greece"], "gtm": ["guatemala"], "guy": ["guyana"],
    "hkg": ["hong kong"], "hnd": ["honduras"], "hrv": ["croatia"],
    "hti": ["haiti"], "hun": ["hungary"], "idn": ["indonesia"],
    "ind": ["india"], "irl": ["ireland"], "irn": ["iran"],
    "irq": ["iraq"], "isl": ["iceland"], "isr": ["israel"],
    "ita": ["italy"], "jam": ["jamaica"], "jor": ["jordan"],
    "jpn": ["japan"], "kaz": ["kazakhstan"], "ken": ["kenya"],
    "kgz": ["kyrgyz republic", "kyrgyzstan"],
    "khm": ["cambodia"], "kor": ["korea, rep.", "south korea"],
    "kwt": ["kuwait"], "lao": ["lao pdr", "laos"],
    "lbn": ["lebanon"], "lbr": ["liberia"], "lby": ["libya"],
    "lca": ["st. lucia"], "lka": ["sri lanka"],
    "lso": ["lesotho"], "ltu": ["lithuania"], "lux": ["luxembourg"],
    "lva": ["latvia"], "mar": ["morocco"], "mda": ["moldova"],
    "mdg": ["madagascar"], "mdv": ["maldives"], "mex": ["mexico"],
    "mli": ["mali"], "mlt": ["malta"], "mmr": ["myanmar"],
    "mne": ["montenegro"], "mng": ["mongolia"], "moz": ["mozambique"],
    "mrt": ["mauritania"], "mus": ["mauritius"], "mwi": ["malawi"],
    "mys": ["malaysia"], "nam": ["namibia"], "ner": ["niger"],
    "nga": ["nigeria"], "nic": ["nicaragua"], "nld": ["netherlands"],
    "nor": ["norway"], "npl": ["nepal"], "nzl": ["new zealand"],
    "omn": ["oman"], "pak": ["pakistan"], "pan": ["panama"],
    "per": ["peru"], "phl": ["philippines"], "png": ["papua new guinea"],
    "pol": ["poland"], "prt": ["portugal"], "pry": ["paraguay"],
    "qat": ["qatar"], "rou": ["romania"], "rus": ["russian federation", "russia"],
    "rwa": ["rwanda"], "sau": ["saudi arabia"], "sdn": ["sudan"],
    "sen": ["senegal"], "sgp": ["singapore"], "slb": ["solomon islands"],
    "sle": ["sierra leone"], "slv": ["el salvador"], "som": ["somalia"],
    "srb": ["serbia"], "ssd": ["south sudan"], "stp": ["sao tome and principe"],
    "sur": ["suriname"], "svk": ["slovak republic", "slovakia"],
    "svn": ["slovenia"], "swe": ["sweden"], "swz": ["eswatini", "swaziland"],
    "syc": ["seychelles"], "syr": ["syrian arab republic", "syria"],
    "tcd": ["chad"], "tgo": ["togo"], "tha": ["thailand"],
    "tjk": ["tajikistan"], "tkm": ["turkmenistan"],
    "tls": ["timor-leste"], "ton": ["tonga"], "tto": ["trinidad and tobago"],
    "tun": ["tunisia"], "tur": ["turkiye", "turkey"],
    "tza": ["tanzania"], "uga": ["uganda"], "ukr": ["ukraine"],
    "ury": ["uruguay"], "usa": ["united states"],
    "uzb": ["uzbekistan"], "ven": ["venezuela"],
    "vnm": ["vietnam", "viet nam"], "vut": ["vanuatu"],
    "yem": ["yemen"], "zaf": ["south africa"],
    "zmb": ["zambia"], "zwe": ["zimbabwe"],
}

# Build reverse mapping: country_name_lower → [code1, code2, ...]
_NAME_TO_CODES: dict[str, list[str]] = {}
for _code, _names in _COUNTRY_ALIASES.items():
    for _name in _names:
        _NAME_TO_CODES.setdefault(_name, []).append(_code)


def find_entity_in_graph(
    entity: str,
    nodes: dict,
    search_index: dict[str, str],
) -> str | None:
    """
    Return matched node name for an entity string, or None.

    Matching order:
      1. Exact case-insensitive match on node name
      2. Substring: entity_lower in node_lower or node_lower in entity_lower
      3. Description substring: entity_lower in node description
    """
    entity_lower = entity.lower()

    # 1. Exact
    if entity_lower in search_index:
        return search_index[entity_lower]

    # 2a. Entity contained in node name (strong match, e.g., "china" in "china_population")
    candidates_strong: list[tuple[str, int]] = []
    for node_lower, orig_name in search_index.items():
        if entity_lower in node_lower:
            candidates_strong.append((orig_name, len(node_lower)))
    if candidates_strong:
        # Prefer shortest containing node (closest match)
        candidates_strong.sort(key=lambda x: x[1])
        return candidates_strong[0][0]

    # 2b. Node contained in entity (weak match, e.g., "afg" in "afghanistan")
    #     Require significant overlap or prefix match to avoid false positives
    #     like "eri" in "algeria" (3/7=0.43)
    candidates_weak: list[tuple[str, int, float]] = []
    for node_lower, orig_name in search_index.items():
        if node_lower in entity_lower:
            ratio = len(node_lower) / len(entity_lower)
            if ratio > 0.5 or entity_lower.startswith(node_lower):
                candidates_weak.append((orig_name, len(node_lower), ratio))
    if candidates_weak:
        candidates_weak.sort(key=lambda x: x[2], reverse=True)
        return candidates_weak[0][0]

    # 3. Country code alias lookup (e.g., "Australia" → "AUS")
    codes = _NAME_TO_CODES.get(entity_lower, [])
    for code in codes:
        if code in search_index:
            return search_index[code]

    # 4. Description substring
    for orig_name, data in nodes.items():
        desc = data.get("description", "").lower()
        if entity_lower in desc:
            return orig_name

    return None


def _entity_has_fact(
    node_name: str,
    year: str,
    value: str,
    nodes: dict,
    entity_text_2hop: dict,
    skip_year: bool = False,
) -> bool:
    """
    Return True if the entity's 2-hop neighborhood contains both value and year.
    Same logic as evaluate_coverage.py check_fact_coverage.
    If skip_year is True, only check value presence (for single-year datasets).
    """
    texts = entity_text_2hop.get(node_name, [])
    node_desc = nodes.get(node_name, {}).get("description", "")
    all_text = " ".join(texts) + " " + node_desc
    if skip_year:
        return value in all_text
    return (value in all_text) and (year in all_text)


def _entity_has_year(
    node_name: str,
    year: str,
    nodes: dict,
    entity_text_2hop: dict,
    skip_year: bool = False,
) -> bool:
    """Return True if the entity's 2-hop neighborhood mentions a year at all.
    If skip_year is True, always return True (for single-year datasets).
    """
    if skip_year:
        return True
    texts = entity_text_2hop.get(node_name, [])
    node_desc = nodes.get(node_name, {}).get("description", "")
    all_text = " ".join(texts) + " " + node_desc
    return year in all_text


# ---------------------------------------------------------------------------
# Per-level answerability checks
# ---------------------------------------------------------------------------

def check_l1_point_lookup(
    question: dict,
    nodes: dict,
    entity_text_2hop: dict,
    search_index: dict,
    skip_year: bool = False,
) -> bool:
    """
    L1: The answer value must be reachable from the answer entity within 2 hops.
    Requires exactly 1 answer entity and 1 answer year.
    """
    entities = question.get("answer_entities", [])
    years = question.get("answer_years", [])
    value = str(question.get("answer", "")).strip()

    if not entities or not years or not value:
        return False

    entity = entities[0]
    year = years[0]

    matched_node = find_entity_in_graph(entity, nodes, search_index)
    if matched_node is None:
        return False

    return _entity_has_fact(matched_node, year, value, nodes, entity_text_2hop, skip_year=skip_year)


def check_l2_ranking(
    question: dict,
    nodes: dict,
    entity_text_2hop: dict,
    search_index: dict,
    skip_year: bool = False,
) -> bool:
    """
    L2: All answer entities must have their year-facts present in the graph.
    Question is answerable only if ALL ranked entities have their values.
    """
    entities = question.get("answer_entities", [])
    years = question.get("answer_years", [])

    if not entities or not years:
        return False

    year = years[0]

    for entity in entities:
        matched_node = find_entity_in_graph(entity, nodes, search_index)
        if matched_node is None:
            return False
        if not _entity_has_year(matched_node, year, nodes, entity_text_2hop, skip_year=skip_year):
            return False

    return True


def check_l3_aggregation(
    question: dict,
    nodes: dict,
    entity_text_2hop: dict,
    search_index: dict,
    skip_year: bool = False,
) -> bool:
    """
    L3: Aggregation questions require the year to be broadly represented.
    We check by verifying the answer value appears somewhere in the graph
    (since aggregation values are computed, exact match is required),
    OR by checking that the relevant year is mentioned across multiple nodes.

    Strategy: if answer_entities is empty (global aggregation), check that the
    graph has at least one node with the given year in its 2-hop neighborhood
    (indicating year data exists). We use the answer value for presence check.
    """
    years = question.get("answer_years", [])
    value = str(question.get("answer", "")).strip()
    entities = question.get("answer_entities", [])

    if not years:
        return False

    year = years[0]

    # If there are specific entities, check them like L2
    if entities:
        for entity in entities:
            matched_node = find_entity_in_graph(entity, nodes, search_index)
            if matched_node is None:
                return False
            if not _entity_has_year(matched_node, year, nodes, entity_text_2hop, skip_year=skip_year):
                return False
        return True

    # Global aggregation (answer_entities=[]):
    # Check if the answer value appears anywhere in the graph
    if not value:
        return False

    # Search all nodes' 2-hop for the exact value
    for node_name in nodes:
        texts = entity_text_2hop.get(node_name, [])
        node_desc = nodes.get(node_name, {}).get("description", "")
        all_text = " ".join(texts) + " " + node_desc
        if skip_year:
            if value in all_text:
                return True
        else:
            if value in all_text and year in all_text:
                return True

    return False


def check_l4_trend_comparison(
    question: dict,
    nodes: dict,
    entity_text_2hop: dict,
    search_index: dict,
    skip_year: bool = False,
) -> bool:
    """
    L4 (trend + comparison): All answer entities must have facts for ALL
    required years. Both time points must be present in the graph.
    """
    entities = question.get("answer_entities", [])
    years = question.get("answer_years", [])

    if not entities or not years:
        return False

    for entity in entities:
        matched_node = find_entity_in_graph(entity, nodes, search_index)
        if matched_node is None:
            return False
        for year in years:
            if not _entity_has_year(matched_node, year, nodes, entity_text_2hop, skip_year=skip_year):
                return False

    return True


LEVEL_CHECKERS = {
    "L1": check_l1_point_lookup,
    "L2": check_l2_ranking,
    "L3": check_l3_aggregation,
    "L4": check_l4_trend_comparison,
}


def detect_single_year_datasets(questions: list[dict]) -> set[str]:
    """Detect datasets where ALL questions reference the same single year.
    For such datasets, year verification is skipped since the year may be
    implicit (not serialized into graph text).
    """
    from collections import defaultdict
    ds_years: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        ds = q["dataset"]
        for y in q.get("answer_years", []):
            ds_years[ds].add(y)
    return {ds for ds, years in ds_years.items() if len(years) == 1}


# ---------------------------------------------------------------------------
# Graph cache — avoid reloading the same graph multiple times
# ---------------------------------------------------------------------------

_graph_cache: dict[str, tuple] = {}


def load_graph_cached(path: str):
    if path not in _graph_cache:
        _graph_cache[path] = load_graph_auto(path)
    return _graph_cache[path]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_questions(
    questions: list[dict],
    system: str,
    graph_paths: dict[str, str],
    single_year_datasets: set[str] | None = None,
) -> list[dict]:
    """
    Evaluate all questions against graphs for one system (sge or baseline).
    Returns list of result dicts with question metadata + answerable flag.
    """
    if single_year_datasets is None:
        single_year_datasets = set()

    results = []
    # Track which single-year datasets actually lack year data in the graph
    _year_check_cache: dict[str, bool] = {}

    for q in questions:
        dataset = q["dataset"]
        level = q["level"]
        qid = q["id"]

        graph_path = graph_paths.get(dataset)
        if graph_path is None or not Path(graph_path).exists():
            results.append({
                "id": qid,
                "dataset": dataset,
                "level": level,
                "category": q["category"],
                "system": system,
                "answerable": False,
                "reason": "graph_not_found",
            })
            continue

        G, nodes, entity_text_2hop = load_graph_cached(graph_path)
        search_index = _build_search_index(nodes)

        # Determine if year check should be skipped for this dataset+graph
        skip_year = False
        if dataset in single_year_datasets:
            cache_key = f"{system}:{dataset}"
            if cache_key not in _year_check_cache:
                # Check if the year actually appears in graph text
                year_to_check = q.get("answer_years", [""])[0]
                year_found = False
                for nn in list(nodes.keys())[:50]:  # sample first 50 nodes
                    texts = entity_text_2hop.get(nn, [])
                    desc = nodes.get(nn, {}).get("description", "")
                    if year_to_check in (" ".join(texts) + " " + desc):
                        year_found = True
                        break
                _year_check_cache[cache_key] = not year_found
            skip_year = _year_check_cache[cache_key]

        checker = LEVEL_CHECKERS.get(level)
        if checker is None:
            results.append({
                "id": qid,
                "dataset": dataset,
                "level": level,
                "category": q["category"],
                "system": system,
                "answerable": False,
                "reason": "unknown_level",
            })
            continue

        answerable = checker(q, nodes, entity_text_2hop, search_index, skip_year=skip_year)
        results.append({
            "id": qid,
            "dataset": dataset,
            "level": level,
            "category": q["category"],
            "system": system,
            "answerable": answerable,
        })

    return results


def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["answerable"]) / len(results)


def group_accuracy(
    results: list[dict],
    key: str,
) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in results:
        k = r[key]
        groups.setdefault(k, []).append(r)
    return {
        k: {
            "answerable": sum(1 for r in v if r["answerable"]),
            "total": len(v),
            "accuracy": round(compute_accuracy(v), 4),
        }
        for k, v in sorted(groups.items())
    }


def print_summary_table(
    sge_results: list[dict],
    baseline_results: list[dict],
) -> None:
    """Print a formatted comparison table."""
    sge_by_id = {r["id"]: r for r in sge_results}
    base_by_id = {r["id"]: r for r in baseline_results}

    print()
    print("=" * 72)
    print("  STATISTICAL QUESTION ANSWERABILITY EVALUATION")
    print("=" * 72)

    total_sge = sum(1 for r in sge_results if r["answerable"])
    total_base = sum(1 for r in baseline_results if r["answerable"])
    n = len(sge_results)

    print(f"\n  Overall:  SGE {total_sge}/{n} ({total_sge/n:.1%})   "
          f"Baseline {total_base}/{n} ({total_base/n:.1%})")

    # By level
    print(f"\n  {'Level':<12} {'SGE':>12} {'Baseline':>12} {'Delta':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for level in ["L1", "L2", "L3", "L4"]:
        sge_lvl = [r for r in sge_results if r["level"] == level]
        base_lvl = [r for r in baseline_results if r["level"] == level]
        if not sge_lvl:
            continue
        s_acc = compute_accuracy(sge_lvl)
        b_acc = compute_accuracy(base_lvl)
        s_str = f"{sum(1 for r in sge_lvl if r['answerable'])}/{len(sge_lvl)} ({s_acc:.1%})"
        b_str = f"{sum(1 for r in base_lvl if r['answerable'])}/{len(base_lvl)} ({b_acc:.1%})"
        delta = f"+{s_acc - b_acc:.1%}" if s_acc >= b_acc else f"{s_acc - b_acc:.1%}"
        print(f"  {level:<12} {s_str:>12} {b_str:>12} {delta:>8}")

    # By dataset
    print(f"\n  {'Dataset':<22} {'SGE':>16} {'Baseline':>16} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*16} {'-'*16} {'-'*8}")
    for ds in ["WHO", "WB_ChildMortality", "WB_Population", "WB_Maternal", "Inpatient"]:
        sge_ds = [r for r in sge_results if r["dataset"] == ds]
        base_ds = [r for r in baseline_results if r["dataset"] == ds]
        if not sge_ds:
            continue
        s_acc = compute_accuracy(sge_ds)
        b_acc = compute_accuracy(base_ds)
        s_str = f"{sum(1 for r in sge_ds if r['answerable'])}/{len(sge_ds)} ({s_acc:.1%})"
        b_str = f"{sum(1 for r in base_ds if r['answerable'])}/{len(base_ds)} ({b_acc:.1%})"
        delta = f"+{s_acc - b_acc:.1%}" if s_acc >= b_acc else f"{s_acc - b_acc:.1%}"
        print(f"  {ds:<22} {s_str:>16} {b_str:>16} {delta:>8}")

    # By category
    print(f"\n  {'Category':<16} {'SGE':>14} {'Baseline':>14} {'Delta':>8}")
    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*8}")
    for cat in ["point_lookup", "ranking", "aggregation", "trend", "comparison"]:
        sge_cat = [r for r in sge_results if r["category"] == cat]
        base_cat = [r for r in baseline_results if r["category"] == cat]
        if not sge_cat:
            continue
        s_acc = compute_accuracy(sge_cat)
        b_acc = compute_accuracy(base_cat)
        s_str = f"{sum(1 for r in sge_cat if r['answerable'])}/{len(sge_cat)} ({s_acc:.1%})"
        b_str = f"{sum(1 for r in base_cat if r['answerable'])}/{len(base_cat)} ({b_acc:.1%})"
        delta = f"+{s_acc - b_acc:.1%}" if s_acc >= b_acc else f"{s_acc - b_acc:.1%}"
        print(f"  {cat:<16} {s_str:>14} {b_str:>14} {delta:>8}")

    print()
    print("=" * 72)


def build_output_dict(
    sge_results: list[dict],
    baseline_results: list[dict],
    questions: list[dict],
) -> dict[str, Any]:
    """Build the JSON output structure."""
    n = len(questions)

    def summary_for(results: list[dict]) -> dict:
        return {
            "overall": {
                "answerable": sum(1 for r in results if r["answerable"]),
                "total": n,
                "accuracy": round(compute_accuracy(results), 4),
            },
            "by_level": group_accuracy(results, "level"),
            "by_dataset": group_accuracy(results, "dataset"),
            "by_category": group_accuracy(results, "category"),
        }

    return {
        "sge": summary_for(sge_results),
        "baseline": summary_for(baseline_results),
        "per_question": {
            str(r["id"]): {
                "sge_answerable": sge_results[i]["answerable"],
                "baseline_answerable": baseline_results[i]["answerable"],
                "dataset": r["dataset"],
                "level": r["level"],
                "category": r["category"],
            }
            for i, r in enumerate(questions)
        },
    }


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    print(f"Loading questions from {QUESTIONS_FILE}")
    questions = load_questions(QUESTIONS_FILE)
    print(f"Loaded {len(questions)} questions.")

    single_year_ds = detect_single_year_datasets(questions)
    if single_year_ds:
        print(f"Single-year datasets detected: {single_year_ds}")

    print("\nEvaluating SGE graphs ...")
    sge_results = evaluate_questions(questions, "sge", GRAPH_PATHS["sge"], single_year_ds)

    # Clear cache between systems to avoid false sharing
    _graph_cache.clear()

    print("Evaluating Baseline graphs ...")
    baseline_results = evaluate_questions(questions, "baseline", GRAPH_PATHS["baseline"], single_year_ds)

    print_summary_table(sge_results, baseline_results)

    output = build_output_dict(sge_results, baseline_results, questions)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
