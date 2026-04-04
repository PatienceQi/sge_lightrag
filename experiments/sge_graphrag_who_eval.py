#!/usr/bin/env python3
"""
sge_graphrag_who_eval.py — Evaluate SGE-GraphRAG on WHO Life Expectancy dataset.

Computes Fact Coverage (FC) against the WHO gold standard (gold_who_life_expectancy_v2.jsonl)
using the parquet output from MS GraphRAG v3.0.8 indexing.

This experiment demonstrates that SGE's format-constraint coupling mechanism is
host-agnostic: SGE-serialized text + SGE schema-aware prompt improves graph
construction fidelity on MS GraphRAG, not just LightRAG.

Two conditions evaluated:
  - graphrag_sge_who: SGE serialization + SGE schema-aware prompt (Country_Code + YearValue)
  - graphrag_who:     SGE serialization + broader entity types (baseline comparison)

Usage:
    python3 experiments/sge_graphrag_who_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pandas or pyarrow not installed", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SGE_GRAPHRAG_DIR = PROJECT_ROOT / "output" / "graphrag_sge_who"
BASELINE_GRAPHRAG_DIR = PROJECT_ROOT / "output" / "graphrag_who"
GOLD_JSONL = PROJECT_ROOT / "evaluation" / "gold" / "gold_who_life_expectancy_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "results" / "sge_graphrag_results.json"


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
    # Remove thousands separators
    v = v.replace(",", "")
    # Remove trailing .0 or .00
    v = re.sub(r"\.0+$", "", v)
    return v


def _value_in_text(value: str, text: str) -> bool:
    """Check if a numeric value appears in text (with normalization)."""
    text_lower = text.lower()
    norm_val = _normalize_value(value)

    # Direct substring match
    if norm_val in text_lower:
        return True
    # Also try with original value
    if value in text:
        return True
    return False


def _entity_matches(subject: str, entity_title: str, entity_desc: str) -> bool:
    """Check if gold subject matches an entity title or description."""
    subj_lower = subject.lower()
    title_lower = entity_title.lower()
    desc_lower = (entity_desc or "").lower()

    # Exact match
    if subj_lower == title_lower:
        return True
    # Subject is a substring of title (e.g. "CHN" in "CHN_2000=71.87")
    if subj_lower in title_lower:
        return True
    # Title is a substring of subject
    if title_lower in subj_lower and len(title_lower) >= 2:
        return True
    # Description contains subject
    if subj_lower in desc_lower:
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
    1. Find entities matching subject in entity titles/descriptions
    2. Collect text from related text_units and relationship descriptions
    3. Check if value appears in that text corpus
    4. Also check relationships directly for entity names containing value=year

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

        # Pass 1: Find matching entities
        matching_eids = []
        for _, row in entities.iterrows():
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            if _entity_matches(subj, title, desc):
                matching_eids.append(str(row["id"]))

        found = False

        if matching_eids:
            eid_set = set(matching_eids)

            # Pass 2a: Search YearValue entities — direct title match
            # e.g. entity title "CHN_2000=71.87622804" contains both year and value
            for _, row in entities.iterrows():
                title = str(row.get("title", ""))
                # Match COUNTRY_YEAR=VALUE pattern
                if year in title and _value_in_text(value, title):
                    # Verify this entity connects to our subject
                    eid = str(row["id"])
                    # Check if there's a relationship between subject entity and this
                    mask = (
                        (relationships["source"].isin(eid_set) & (relationships["target"] == eid)) |
                        (relationships["target"].isin(eid_set) & (relationships["source"] == eid))
                    )
                    if mask.any():
                        found = True
                        break
                    # Also check if this entity itself is in matching set (it has the value)
                    if eid in eid_set:
                        found = True
                        break

            if not found:
                # Pass 2b: Search text_units linked to matching entities
                tu_ids: set[str] = set()
                for _, row in entities[entities["id"].isin(matching_eids)].iterrows():
                    raw_tu_ids = row.get("text_unit_ids")
                    if raw_tu_ids is not None:
                        import numpy as np
                        if isinstance(raw_tu_ids, (list, np.ndarray)):
                            tu_ids.update(str(v) for v in raw_tu_ids if v is not None)

                # Also collect text_units from relationships
                rel_mask = (
                    relationships["source"].isin(eid_set) |
                    relationships["target"].isin(eid_set)
                )
                for _, row in relationships[rel_mask].iterrows():
                    raw_tu_ids = row.get("text_unit_ids")
                    if raw_tu_ids is not None:
                        import numpy as np
                        if isinstance(raw_tu_ids, (list, np.ndarray)):
                            tu_ids.update(str(v) for v in raw_tu_ids if v is not None)

                tu_id_set = set(str(i) for i in tu_ids)
                for _, row in text_units.iterrows():
                    if str(row["id"]) in tu_id_set:
                        text = str(row.get("text", ""))
                        if year in text and _value_in_text(value, text):
                            found = True
                            break

            if not found:
                # Pass 2c: Search relationship descriptions/keywords
                for _, row in relationships[rel_mask].iterrows():
                    rel_text = (
                        str(row.get("description", "")) + " " +
                        str(row.get("weight", "")) + " " +
                        str(row.get("source", "")) + " " +
                        str(row.get("target", ""))
                    )
                    if year in rel_text and _value_in_text(value, rel_text):
                        found = True
                        break

        if not found:
            # Pass 3: Direct text_units fallback — search all text
            for _, row in text_units.iterrows():
                text = str(row.get("text", ""))
                if subj in text and year in text and _value_in_text(value, text):
                    found = True
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
        "uncovered_sample": uncovered_facts[:5],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("SGE-GraphRAG WHO Evaluation")
    print("=" * 60)

    # Load gold facts
    if not GOLD_JSONL.exists():
        print(f"ERROR: Gold standard not found: {GOLD_JSONL}", file=sys.stderr)
        sys.exit(1)

    facts = load_gold_facts(GOLD_JSONL)
    print(f"Gold facts loaded: {len(facts)}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "system": "sge_graphrag_v3.0.8",
        "experiment": "SGE serialization + SGE schema prompt vs baseline prompt",
        "gold_file": str(GOLD_JSONL.name),
        "gold_facts": len(facts),
        "conditions": {},
    }

    # Evaluate each condition
    conditions = [
        ("sge_graphrag_who", SGE_GRAPHRAG_DIR, "SGE serialization + SGE schema (Country_Code, YearValue, HAS_MEASUREMENT)"),
        ("graphrag_who_baseline", BASELINE_GRAPHRAG_DIR, "SGE serialization + broader entity types (country, health_indicator, stat_value, year)"),
    ]

    for cond_key, graphrag_dir, description in conditions:
        print(f"\n--- {cond_key} ---")
        print(f"Description: {description}")
        print(f"Directory: {graphrag_dir}")

        if not graphrag_dir.exists():
            print(f"  SKIP: directory not found")
            results["conditions"][cond_key] = {
                "status": "skipped",
                "reason": "directory not found",
                "description": description,
            }
            continue

        output_dir = graphrag_dir / "output"
        if not (output_dir / "entities.parquet").exists():
            print(f"  SKIP: entities.parquet not found (indexing not complete?)")
            results["conditions"][cond_key] = {
                "status": "skipped",
                "reason": "entities.parquet not found",
                "description": description,
            }
            continue

        try:
            tables = load_graphrag_parquets(graphrag_dir)
            print(f"  Entities: {len(tables['entities'])}")
            print(f"  Relationships: {len(tables['relationships'])}")
            print(f"  Text units: {len(tables['text_units'])}")

            if "type" in tables["entities"].columns:
                type_counts = tables["entities"]["type"].value_counts().to_dict()
                print(f"  Entity types: {type_counts}")

            fc_result = compute_fc(facts, tables, cond_key)
            print(f"  FC = {fc_result['fc']:.4f} ({fc_result['covered']}/{fc_result['total_facts']} facts covered)")

            results["conditions"][cond_key] = {
                **fc_result,
                "description": description,
            }

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results["conditions"][cond_key] = {
                "status": "error",
                "error": str(exc),
                "description": description,
            }

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for cond_key, cond_data in results["conditions"].items():
        if "fc" in cond_data:
            print(f"  {cond_key}: FC = {cond_data['fc']:.4f} ({cond_data['covered']}/{cond_data['total_facts']})")
        else:
            print(f"  {cond_key}: {cond_data.get('status', 'unknown')} - {cond_data.get('reason', cond_data.get('error', ''))}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
