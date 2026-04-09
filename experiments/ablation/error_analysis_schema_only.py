"""
Error analysis of Schema-only ablation condition.

Schema-only = LightRAG default naive serialization + SGE schema prompt injection.
The schema prompt was designed for SGE's structured serialization, NOT for naive
text chunks. This mismatch causes the LLM to ignore schema constraints and
hallucinate entities with wrong types (Location/Data instead of Country_Code/StatValue).

Analyzes three conditions for WB Population dataset (FC known values):
  - Full SGE      (compact_who serves as SGE reference): FC=1.000
  - Schema-only   (ablation_schema_only_wb_pop):          FC=0.007
  - Baseline      (baseline_wb_population):               FC=0.187

Also analyzes WHO dataset across conditions.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DELIMITER = "<|#|>"

# Schema-defined entity types for WB population / WHO datasets
SCHEMA_ENTITY_TYPES = frozenset({"Country_Code", "StatValue"})

# Full SGE uses compact serialization — Country_Code is the dominant type
SGE_SCHEMA_TYPES = frozenset({"Country_Code", "StatValue"})


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict[str, Any]:
    """Load an LLM response cache JSON file. Returns empty dict if not found."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_extract_entries(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the extraction-type entries from a cache."""
    return [v for v in cache.values() if v.get("cache_type") == "extract"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a single LLM extraction response into entities and relations.

    Entity line format:  entity<|#|>name<|#|>type<|#|>description
    Relation line format: relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description
                          relationship<|#|>src<|#|>tgt<|#|>...

    Returns (entities, relations) as lists of dicts.
    """
    entities = []
    relations = []

    for raw_line in response_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Strip markdown backtick fences and completion markers
        if line.startswith("```") or line == "<|COMPLETE|>":
            continue

        parts = line.split(DELIMITER)
        record_type = parts[0].lower() if parts else ""

        if record_type == "entity" and len(parts) >= 3:
            entities.append({
                "name": parts[1].strip(),
                "type": parts[2].strip(),
                "description": parts[3].strip() if len(parts) > 3 else "",
            })
        elif record_type in ("relation", "relationship") and len(parts) >= 3:
            relations.append({
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "keywords": parts[3].strip() if len(parts) > 3 else "",
                "description": parts[4].strip() if len(parts) > 4 else "",
            })

    return entities, relations


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def classify_entity_type(entity_type: str, schema_types: frozenset) -> str:
    """Return 'schema' if type is in schema, else 'non_schema'."""
    return "schema" if entity_type in schema_types else "non_schema"


def compute_condition_metrics(
    entries: list[dict[str, Any]],
    schema_types: frozenset,
    condition_name: str,
) -> dict[str, Any]:
    """Compute per-condition extraction metrics from cache entries."""
    total_entities = 0
    total_relations = 0
    entity_type_counts: Counter = Counter()
    unique_entity_names: set[str] = set()
    unique_statvalue_names: set[str] = set()
    responses_with_relations = 0
    responses_with_refusal = 0
    refusal_examples: list[str] = []
    entity_per_response: list[int] = []
    relation_per_response: list[int] = []

    for entry in entries:
        response_text = entry.get("return", "")

        # Detect outright refusal / meta-commentary responses
        is_refusal = (
            "I appreciate" in response_text
            or "I'm Claude" in response_text
            or "I cannot follow" in response_text
            or "I need to clarify" in response_text
        ) and not response_text.strip().startswith("entity")

        if is_refusal:
            responses_with_refusal += 1
            if len(refusal_examples) < 3:
                refusal_examples.append(response_text[:200])
            entity_per_response.append(0)
            relation_per_response.append(0)
            continue

        entities, relations = parse_response(response_text)

        for ent in entities:
            entity_type_counts[ent["type"]] += 1
            unique_entity_names.add(ent["name"])
            if ent["type"] == "StatValue":
                unique_statvalue_names.add(ent["name"])

        n_ent = len(entities)
        n_rel = len(relations)
        total_entities += n_ent
        total_relations += n_rel
        entity_per_response.append(n_ent)
        relation_per_response.append(n_rel)

        if n_rel > 0:
            responses_with_relations += 1

    total_responses = len(entries)
    schema_entity_count = sum(
        cnt for etype, cnt in entity_type_counts.items()
        if etype in schema_types
    )
    schema_compliance_rate = (
        schema_entity_count / total_entities if total_entities > 0 else 0.0
    )
    relation_extraction_rate = (
        responses_with_relations / total_responses if total_responses > 0 else 0.0
    )
    mean_entities = (
        sum(entity_per_response) / len(entity_per_response)
        if entity_per_response else 0.0
    )
    mean_relations = (
        sum(relation_per_response) / len(relation_per_response)
        if relation_per_response else 0.0
    )
    refusal_rate = (
        responses_with_refusal / total_responses if total_responses > 0 else 0.0
    )

    # Top-10 entity types by count
    top_types = dict(entity_type_counts.most_common(10))

    # StatValue name explosion: schema-only creates one StatValue per (country, year) cell
    # instead of one Country_Code per country — this causes the FC to collapse because
    # gold facts are keyed on Country_Code entities that no longer exist in the graph.
    statvalue_explosion_factor = (
        len(unique_statvalue_names) / len(unique_entity_names)
        if unique_entity_names else 0.0
    )

    return {
        "condition": condition_name,
        "total_responses": total_responses,
        "total_entities": total_entities,
        "total_relations": total_relations,
        "unique_entity_names": len(unique_entity_names),
        "unique_statvalue_names": len(unique_statvalue_names),
        "statvalue_explosion_factor": round(statvalue_explosion_factor, 4),
        "schema_entity_count": schema_entity_count,
        "non_schema_entity_count": total_entities - schema_entity_count,
        "schema_compliance_rate": round(schema_compliance_rate, 4),
        "mean_entities_per_response": round(mean_entities, 2),
        "mean_relations_per_response": round(mean_relations, 2),
        "relation_extraction_rate": round(relation_extraction_rate, 4),
        "responses_with_refusal": responses_with_refusal,
        "refusal_rate": round(refusal_rate, 4),
        "entity_type_distribution": top_types,
        "refusal_examples": refusal_examples,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

BASE = str(_PROJECT_ROOT / "output")

CACHE_PATHS = {
    # WB Population dataset
    "wb_pop": {
        "Full_SGE": os.path.join(
            BASE,
            "ablation_c4_serial_only_wb_pop",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
        "Schema_only": os.path.join(
            BASE,
            "ablation_schema_only_wb_pop",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
        "Baseline": os.path.join(
            BASE,
            "baseline_wb_population",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
    },
    # WHO Life Expectancy dataset
    "who": {
        "Full_SGE": os.path.join(
            BASE,
            "compact_who",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
        "Schema_only": os.path.join(
            BASE,
            "ablation_schema_only_who",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
        "Baseline": os.path.join(
            BASE,
            "baseline_who_life",
            "lightrag_storage",
            "kv_store_llm_response_cache.json",
        ),
    },
}

# Schema types differ slightly per condition/dataset but we use canonical SGE types
SCHEMA_TYPES_BY_CONDITION = {
    "Full_SGE": frozenset({"Country_Code", "StatValue"}),
    "Schema_only": frozenset({"Country_Code", "StatValue"}),
    "Baseline": frozenset({"Country_Code", "StatValue", "Country"}),
}

# Known FC values for reference
KNOWN_FC = {
    "wb_pop": {"Full_SGE": 1.000, "Schema_only": 0.007, "Baseline": 0.187},
    "who": {"Full_SGE": 1.000, "Schema_only": None, "Baseline": 0.167},
}


def analyze_dataset(dataset_key: str) -> dict[str, Any]:
    """Run analysis for all conditions of a dataset."""
    paths = CACHE_PATHS[dataset_key]
    dataset_results: dict[str, Any] = {}

    for condition, cache_path in paths.items():
        cache = load_cache(cache_path)
        if not cache:
            print(f"  [SKIP] {condition}: cache not found at {cache_path}")
            dataset_results[condition] = {"condition": condition, "error": "cache not found"}
            continue

        entries = get_extract_entries(cache)
        schema_types = SCHEMA_TYPES_BY_CONDITION[condition]
        metrics = compute_condition_metrics(entries, schema_types, condition)

        known_fc = KNOWN_FC[dataset_key].get(condition)
        metrics["known_fc"] = known_fc
        dataset_results[condition] = metrics

    return dataset_results


def print_summary_table(results: dict[str, dict]) -> None:
    """Print a human-readable comparison table."""
    col_width = 22
    headers = ["Metric", "Full_SGE", "Schema_only", "Baseline"]
    header_line = "".join(h.ljust(col_width) for h in headers)
    print(header_line)
    print("-" * (col_width * len(headers)))

    def row(label: str, key: str, fmt: str = "{}") -> str:
        cells = [label.ljust(col_width)]
        for cond in ["Full_SGE", "Schema_only", "Baseline"]:
            data = results.get(cond, {})
            if "error" in data:
                cells.append("N/A".ljust(col_width))
            else:
                val = data.get(key, "N/A")
                if val is None:
                    cells.append("N/A".ljust(col_width))
                else:
                    cells.append(fmt.format(val).ljust(col_width))
        return "".join(cells)

    print(row("Known FC", "known_fc", "{:.3f}"))
    print(row("Total Responses", "total_responses"))
    print(row("Total Entities", "total_entities"))
    print(row("Unique Entity Names", "unique_entity_names"))
    print(row("Unique StatValue Names", "unique_statvalue_names"))
    print(row("StatVal Explosion Ratio", "statvalue_explosion_factor", "{:.3f}"))
    print(row("Total Relations", "total_relations"))
    print(row("Schema Entities", "schema_entity_count"))
    print(row("Non-Schema Entities", "non_schema_entity_count"))
    print(row("Schema Compliance %", "schema_compliance_rate", "{:.1%}"))
    print(row("Mean Entities/Resp", "mean_entities_per_response", "{:.1f}"))
    print(row("Mean Relations/Resp", "mean_relations_per_response", "{:.1f}"))
    print(row("Relation Rate", "relation_extraction_rate", "{:.1%}"))
    print(row("Refusal Count", "responses_with_refusal"))
    print(row("Refusal Rate", "refusal_rate", "{:.1%}"))


def print_type_distribution(results: dict[str, dict], top_n: int = 8) -> None:
    """Print entity type distribution for each condition."""
    for condition in ["Full_SGE", "Schema_only", "Baseline"]:
        data = results.get(condition, {})
        if "error" in data:
            continue
        dist = data.get("entity_type_distribution", {})
        print(f"\n  [{condition}] Top entity types:")
        for etype, cnt in sorted(dist.items(), key=lambda x: -x[1])[:top_n]:
            total = data["total_entities"]
            pct = cnt / total * 100 if total > 0 else 0
            marker = " <-- SCHEMA" if etype in SCHEMA_TYPES_BY_CONDITION[condition] else ""
            print(f"    {etype:<30} {cnt:>6} ({pct:5.1f}%){marker}")


def main() -> None:
    all_results: dict[str, Any] = {}

    for dataset_key in ("wb_pop", "who"):
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_key.upper()}")
        print("="*80)

        dataset_results = analyze_dataset(dataset_key)
        all_results[dataset_key] = dataset_results

        print("\n--- Summary Table ---")
        print_summary_table(dataset_results)

        print("\n--- Entity Type Distributions ---")
        print_type_distribution(dataset_results)

        # Print refusal examples for schema-only
        schema_data = dataset_results.get("Schema_only", {})
        refusals = schema_data.get("refusal_examples", [])
        if refusals:
            print(f"\n--- Schema-only Refusal Examples ({len(refusals)} shown) ---")
            for i, ex in enumerate(refusals):
                print(f"  Example {i+1}: {ex[:200]}")

    # Save to results file
    output_path = str(
        _PROJECT_ROOT / "experiments" / "results" / "error_analysis_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {output_path}")

    # Print key findings for paper
    print("\n" + "="*80)
    print("KEY FINDINGS FOR PAPER")
    print("="*80)
    for dataset_key in ("wb_pop", "who"):
        ds = all_results[dataset_key]
        schema = ds.get("Schema_only", {})
        sge = ds.get("Full_SGE", {})
        base = ds.get("Baseline", {})
        if any("error" in d for d in [schema, sge, base] if d):
            continue
        print(f"\n[{dataset_key.upper()}]")
        print(
            f"  Schema compliance: SGE={sge.get('schema_compliance_rate',0):.1%} | "
            f"Schema-only={schema.get('schema_compliance_rate',0):.1%} | "
            f"Baseline={base.get('schema_compliance_rate',0):.1%}"
        )
        print(
            f"  Unique entity names: SGE={sge.get('unique_entity_names',0)} | "
            f"Schema-only={schema.get('unique_entity_names',0)} | "
            f"Baseline={base.get('unique_entity_names',0)}"
        )
        print(
            f"  Unique StatValue names: SGE={sge.get('unique_statvalue_names',0)} | "
            f"Schema-only={schema.get('unique_statvalue_names',0)} | "
            f"Baseline={base.get('unique_statvalue_names',0)}"
        )
        print(
            f"  StatVal explosion: SGE={sge.get('statvalue_explosion_factor',0):.3f} | "
            f"Schema-only={schema.get('statvalue_explosion_factor',0):.3f} | "
            f"Baseline={base.get('statvalue_explosion_factor',0):.3f}"
        )
        print(
            f"  Relation rate:     SGE={sge.get('relation_extraction_rate',0):.1%} | "
            f"Schema-only={schema.get('relation_extraction_rate',0):.1%} | "
            f"Baseline={base.get('relation_extraction_rate',0):.1%}"
        )
        print(
            f"  Mean ents/resp:    SGE={sge.get('mean_entities_per_response',0):.1f} | "
            f"Schema-only={schema.get('mean_entities_per_response',0):.1f} | "
            f"Baseline={base.get('mean_entities_per_response',0):.1f}"
        )
        print(
            f"  Refusal rate:      SGE={sge.get('refusal_rate',0):.1%} | "
            f"Schema-only={schema.get('refusal_rate',0):.1%} | "
            f"Baseline={base.get('refusal_rate',0):.1%}"
        )


if __name__ == "__main__":
    main()
