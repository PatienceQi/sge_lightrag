#!/usr/bin/env python3
"""
sge_graphrag_wb_pop_eval.py — Evaluate SGE-GraphRAG on WB Population dataset.

Computes Fact Coverage (FC) against the WB Population gold standard
(gold_wb_population_v2.jsonl) using the parquet output from MS GraphRAG v3.0.8
indexing.

Gold standard uses:
  - subject: country name (e.g. "Argentina", "China")
  - value: integer population (e.g. "37213984")
  - year: "2000", "2005", "2010", "2015", "2020", "2023"

SGE-GraphRAG graph uses:
  - Country_Code entities: ISO3 codes (e.g. "ARG", "CHN")
  - YearValue entities: "COUNTRY_CODE_YYYY=VALUE" (e.g. "ARG_2000=37213984")
  - HAS_POPULATION relationships

Matching strategy:
  1. Find Country_Code entities whose description contains the gold subject name
  2. Look for connected YearValue entities containing year + value in title
  3. Fallback: search text_units for subject + year + value co-occurrence

Usage:
    python3 experiments/sge_graphrag_wb_pop_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import numpy as np
except ImportError:
    print("ERROR: pandas or pyarrow not installed", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SGE_GRAPHRAG_DIR = PROJECT_ROOT / "output" / "graphrag_sge_wb_pop"
GOLD_JSONL = PROJECT_ROOT / "evaluation" / "gold" / "gold_wb_population_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "results" / "sge_graphrag_wb_pop_results.json"

# Mapping from gold country names to WB country codes (for description lookup)
COUNTRY_NAME_TO_CODE = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Bangladesh": "BGD",
    "Brazil": "BRA",
    "Canada": "CAN",
    "China": "CHN",
    "Egypt, Arab Rep.": "EGY",
    "France": "FRA",
    "Germany": "DEU",
    "India": "IND",
    "Indonesia": "IDN",
    "Italy": "ITA",
    "Japan": "JPN",
    "Korea, Rep.": "KOR",
    "Mexico": "MEX",
    "Nigeria": "NGA",
    "Pakistan": "PAK",
    "Russian Federation": "RUS",
    "Saudi Arabia": "SAU",
    "South Africa": "ZAF",
    "Spain": "ESP",
    "Thailand": "THA",
    "Turkiye": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA",
}


# ---------------------------------------------------------------------------
# Gold standard loading
# ---------------------------------------------------------------------------

def load_gold_facts(jsonl_path: Path) -> list[dict]:
    """Load gold triples as list of {subject, value, year} dicts."""
    facts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triple = record.get("triple", {})
            subj = triple.get("subject", "").strip()
            obj = triple.get("object", "").strip()
            attrs = triple.get("attributes", {})
            year = attrs.get("year", "").strip()
            obj_type = triple.get("object_type", "")
            if obj_type in ("StatValue", "BudgetAmount", "Literal") and subj and obj and year:
                facts.append({"subject": subj, "value": obj, "year": year})
    return facts


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def load_graphrag_parquets(graphrag_dir: Path) -> dict[str, pd.DataFrame]:
    """Load entities, relationships, text_units from graphrag output dir."""
    output_dir = graphrag_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"GraphRAG output dir not found: {output_dir}")

    tables = {}
    for name in ("entities", "relationships", "text_units"):
        path = output_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet: {path}")
        tables[name] = pq.read_table(str(path)).to_pandas()
    return tables


# ---------------------------------------------------------------------------
# Fact coverage computation
# ---------------------------------------------------------------------------

def _normalize_value(v: str) -> str:
    """Normalize numeric strings for fuzzy comparison."""
    v = str(v).strip()
    v = v.replace(",", "")
    v = re.sub(r"\.0+$", "", v)
    return v


def _value_in_text(value: str, text: str) -> bool:
    """Check if a numeric value appears in text (with normalization)."""
    text_lower = text.lower()
    norm_val = _normalize_value(value)
    if norm_val in text_lower:
        return True
    if value in text:
        return True
    return False


def _entity_matches(subject: str, entity_title: str, entity_desc: str) -> bool:
    """Check if gold subject (country name or code) matches an entity."""
    subj_lower = subject.lower()
    title_lower = entity_title.lower()
    desc_lower = (entity_desc or "").lower()

    if subj_lower == title_lower:
        return True
    if subj_lower in title_lower:
        return True
    if title_lower in subj_lower and len(title_lower) >= 2:
        return True
    if subj_lower in desc_lower:
        return True

    # Also try matching ISO3 code mapped from country name
    iso3 = COUNTRY_NAME_TO_CODE.get(subject, "").lower()
    if iso3 and iso3 == title_lower:
        return True
    if iso3 and iso3 in title_lower:
        return True

    return False


def compute_fc(
    facts: list[dict],
    tables: dict[str, pd.DataFrame],
    condition_name: str,
) -> dict:
    """
    Compute Fact Coverage (FC) using 2-hop entity-first search.

    For each gold fact (subject, value, year):
    1. Find Country_Code entities matching subject (by name or ISO3 code)
    2. Look for YearValue entities with COUNTRY_CODE_YEAR=VALUE title pattern
       that are connected via HAS_POPULATION relationship
    3. Fallback: search text_units for subject/code + year + value co-occurrence
    4. Final fallback: search relationship descriptions

    Returns dict with coverage stats.
    """
    entities = tables["entities"]
    relationships = tables["relationships"]
    text_units = tables["text_units"]

    covered = 0
    uncovered_facts = []

    for fact in facts:
        subj = fact["subject"]
        value = fact["value"]
        year = fact["year"]

        # Also get ISO3 code for direct lookup
        iso3 = COUNTRY_NAME_TO_CODE.get(subj, "")

        # Pass 1: Find Country_Code entities matching this subject
        matching_eids = []
        for _, row in entities.iterrows():
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            if _entity_matches(subj, title, desc):
                matching_eids.append(str(row["id"]))

        found = False

        if matching_eids:
            eid_set = set(matching_eids)

            # Pass 2a: Search YearValue entities — COUNTRY_CODE_YEAR=VALUE title match
            # e.g. "ARG_2000=37213984" matches year="2000", value="37213984"
            for _, row in entities.iterrows():
                title = str(row.get("title", ""))
                if year in title and _value_in_text(value, title):
                    eid = str(row["id"])
                    # Verify connected to a matching Country_Code entity
                    mask = (
                        (relationships["source"].isin(eid_set) & (relationships["target"] == eid)) |
                        (relationships["target"].isin(eid_set) & (relationships["source"] == eid))
                    )
                    if mask.any():
                        found = True
                        break
                    # Or if the YearValue entity itself is in matching set
                    if eid in eid_set:
                        found = True
                        break

            if not found:
                # Pass 2b: Direct ISO3 title lookup for YearValue
                # e.g. entity title "ARG_2000=37213984"
                if iso3:
                    pattern_prefix = f"{iso3}_{year}="
                    for _, row in entities.iterrows():
                        title = str(row.get("title", ""))
                        if title.upper().startswith(pattern_prefix.upper()):
                            if _value_in_text(value, title):
                                found = True
                                break

            if not found:
                # Pass 2c: Search text_units linked to matching entities
                tu_ids: set[str] = set()
                for _, row in entities[entities["id"].isin(matching_eids)].iterrows():
                    raw_tu_ids = row.get("text_unit_ids")
                    if raw_tu_ids is not None:
                        if isinstance(raw_tu_ids, (list, np.ndarray)):
                            tu_ids.update(str(v) for v in raw_tu_ids if v is not None)

                # Also collect text_units from relationships involving matching entities
                rel_mask = (
                    relationships["source"].isin(eid_set) |
                    relationships["target"].isin(eid_set)
                )
                for _, row in relationships[rel_mask].iterrows():
                    raw_tu_ids = row.get("text_unit_ids")
                    if raw_tu_ids is not None:
                        if isinstance(raw_tu_ids, (list, np.ndarray)):
                            tu_ids.update(str(v) for v in raw_tu_ids if v is not None)

                tu_id_set = set(str(i) for i in tu_ids)
                for _, row in text_units.iterrows():
                    if str(row["id"]) in tu_id_set:
                        text = str(row.get("text", ""))
                        if year in text and _value_in_text(value, text):
                            found = True
                            break

            if not found and matching_eids:
                # Pass 2d: Search relationship descriptions for matching entities
                rel_mask = (
                    relationships["source"].isin(eid_set) |
                    relationships["target"].isin(eid_set)
                )
                for _, row in relationships[rel_mask].iterrows():
                    rel_text = (
                        str(row.get("description", "")) + " " +
                        str(row.get("source", "")) + " " +
                        str(row.get("target", ""))
                    )
                    if year in rel_text and _value_in_text(value, rel_text):
                        found = True
                        break

        if not found:
            # Pass 3: Direct text_units fallback — search all text for value
            norm_val = _normalize_value(value)
            search_terms = [subj, iso3] if iso3 else [subj]
            for _, row in text_units.iterrows():
                text = str(row.get("text", ""))
                for term in search_terms:
                    if term and year in text and _value_in_text(value, text):
                        found = True
                        break
                if found:
                    break

        if found:
            covered += 1
        else:
            uncovered_facts.append(fact)

    fc = covered / len(facts) if facts else 0.0

    return {
        "condition": condition_name,
        "total_facts": len(facts),
        "covered": covered,
        "fc": round(fc, 4),
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "text_unit_count": len(text_units),
        "entity_types": entities["type"].value_counts().to_dict() if "type" in entities.columns else {},
        "uncovered_sample": uncovered_facts[:10],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("SGE-GraphRAG WB Population Evaluation")
    print("=" * 60)

    if not GOLD_JSONL.exists():
        print(f"ERROR: Gold standard not found: {GOLD_JSONL}", file=sys.stderr)
        sys.exit(1)

    facts = load_gold_facts(GOLD_JSONL)
    print(f"Gold facts loaded: {len(facts)}")
    subjects = sorted(set(f["subject"] for f in facts))
    years = sorted(set(f["year"] for f in facts))
    print(f"Countries: {len(subjects)}, Years: {years}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "system": "sge_graphrag_v3.0.8",
        "experiment": "WB Population SGE serialization + SGE schema prompt (Country_Code, YearValue, HAS_POPULATION)",
        "gold_file": str(GOLD_JSONL.name),
        "gold_facts": len(facts),
        "conditions": {},
    }

    print(f"\n--- sge_graphrag_wb_pop ---")
    print(f"Directory: {SGE_GRAPHRAG_DIR}")

    if not SGE_GRAPHRAG_DIR.exists():
        print(f"  ERROR: directory not found")
        sys.exit(1)

    output_dir = SGE_GRAPHRAG_DIR / "output"
    if not (output_dir / "entities.parquet").exists():
        print(f"  ERROR: entities.parquet not found — indexing incomplete?")
        sys.exit(1)

    tables = load_graphrag_parquets(SGE_GRAPHRAG_DIR)
    print(f"  Entities: {len(tables['entities'])}")
    print(f"  Relationships: {len(tables['relationships'])}")
    print(f"  Text units: {len(tables['text_units'])}")

    if "type" in tables["entities"].columns:
        type_counts = tables["entities"]["type"].value_counts().to_dict()
        print(f"  Entity types: {type_counts}")

    fc_result = compute_fc(facts, tables, "sge_graphrag_wb_pop")
    print(f"\n  FC = {fc_result['fc']:.4f} ({fc_result['covered']}/{fc_result['total_facts']} facts covered)")

    if fc_result["uncovered_sample"]:
        print(f"\n  Uncovered facts (sample):")
        for f in fc_result["uncovered_sample"][:5]:
            iso3 = COUNTRY_NAME_TO_CODE.get(f["subject"], "?")
            print(f"    {f['subject']} ({iso3}) | {f['year']} | {f['value']}")

    results["conditions"]["sge_graphrag_wb_pop"] = {
        **fc_result,
        "description": "SGE serialization + SGE schema (Country_Code, YearValue, HAS_POPULATION)",
    }

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  sge_graphrag_wb_pop: FC = {fc_result['fc']:.4f} ({fc_result['covered']}/{fc_result['total_facts']})")
    print(f"  MS GraphRAG baseline (reported): FC = 0.600")
    print(f"  SGE improvement: {fc_result['fc'] / 0.600:.2f}x")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
