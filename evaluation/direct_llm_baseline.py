#!/usr/bin/env python3
"""
direct_llm_baseline.py — Direct LLM extraction baseline for SGE-LightRAG comparison.

Feeds raw CSV chunks directly to an LLM (no pipeline, no graph) and evaluates
entity/fact coverage against the same gold standards used in the main evaluation.

This baseline answers: "Does the SGE pipeline add value over raw LLM extraction?"

Usage:
    python3 evaluation/direct_llm_baseline.py
    python3 evaluation/direct_llm_baseline.py --datasets budget who wb_child_mortality
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag.llm.openai import openai_complete_if_cache

API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

EXTRACT_PROMPT = """\
Extract all factual entity-value triples from the following data.

Output ONLY a JSON array, no explanation:
[{{"entity": "<entity name>", "attribute": "<attribute or relation>", "value": "<value>"}}]

Data:
{chunk}
"""

DATASETS = [
    {
        "label": "Annual Budget (ZH, Type-II)",
        "chunks_dir": "output/sge_budget/chunks",
        "gold": "evaluation/gold_budget.jsonl",
        "language": "zh",
    },
    {
        "label": "WHO Life Expectancy (EN, Type-II)",
        "chunks_dir": "output/who_life_expectancy/chunks",
        "gold": "evaluation/gold_who_life_expectancy.jsonl",
        "language": "en",
    },
    {
        "label": "WB Child Mortality (EN, Type-II)",
        "chunks_dir": "output/wb_child_mortality/chunks",
        "gold": "evaluation/gold_wb_child_mortality.jsonl",
        "language": "en",
    },
]


async def extract_triples(chunk_text: str) -> list[dict]:
    """Call LLM to extract triples from a raw chunk."""
    prompt = EXTRACT_PROMPT.format(chunk=chunk_text[:2000])
    try:
        response = await openai_complete_if_cache(
            MODEL, prompt,
            system_prompt="You are a precise information extraction assistant. Output only valid JSON.",
            api_key=API_KEY, base_url=BASE_URL,
            max_tokens=1024,
            timeout=120,
        )
        # Parse JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except Exception:
        pass
    return []


def _normalize(s: str) -> str:
    return s.lower().replace(",", "").replace("，", "").strip()


def check_entity_covered(entity_name: str, triples: list[dict]) -> bool:
    en = _normalize(entity_name)
    for t in triples:
        if en in _normalize(t.get("entity", "")) or _normalize(t.get("entity", "")) in en:
            return True
    return False


def check_fact_covered(subject: str, obj: str, triples: list[dict]) -> bool:
    subj_n = _normalize(subject)
    obj_n = _normalize(obj)
    for t in triples:
        entity_n = _normalize(t.get("entity", ""))
        value_n = _normalize(t.get("value", ""))
        if (subj_n in entity_n or entity_n in subj_n) and obj_n in value_n:
            return True
    return False


async def evaluate_dataset(ds: dict) -> dict:
    chunks_dir = PROJECT_ROOT / ds["chunks_dir"]
    gold_path = PROJECT_ROOT / ds["gold"]

    gold_items = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
    gold_entities = list({g["triple"]["subject"] for g in gold_items})
    gold_facts = [(g["triple"]["subject"], g["triple"]["object"]) for g in gold_items]

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    print(f"  Loading {len(chunk_files)} chunks...")

    # Extract triples from all chunks (concurrently, max 5)
    semaphore = asyncio.Semaphore(5)

    async def extract_one(cf: Path) -> list[dict]:
        async with semaphore:
            return await extract_triples(cf.read_text(encoding="utf-8", errors="replace"))

    all_triple_lists = await asyncio.gather(*[extract_one(cf) for cf in chunk_files])
    all_triples = [t for lst in all_triple_lists for t in lst]

    print(f"  Extracted {len(all_triples)} triples total")

    # Entity coverage
    covered_entities = sum(1 for e in gold_entities if check_entity_covered(e, all_triples))
    ec = covered_entities / len(gold_entities) if gold_entities else 0

    # Fact coverage
    covered_facts = sum(1 for subj, obj in gold_facts if check_fact_covered(subj, obj, all_triples))
    fc = covered_facts / len(gold_facts) if gold_facts else 0

    return {
        "label": ds["label"],
        "entity_coverage": {"coverage": ec, "covered": covered_entities, "total": len(gold_entities)},
        "fact_coverage": {"coverage": fc, "covered": covered_facts, "total": len(gold_facts)},
        "triples_extracted": len(all_triples),
    }


async def main():
    print("Direct LLM Extraction Baseline\n" + "=" * 50)
    print(f"{'Dataset':<45} | {'EC':>6} {'FC':>6} | Triples")
    print("-" * 70)

    results = []
    for ds in DATASETS:
        print(f"\nEvaluating: {ds['label']}")
        result = await evaluate_dataset(ds)
        results.append(result)
        ec = result["entity_coverage"]["coverage"]
        fc = result["fact_coverage"]["coverage"]
        triples = result["triples_extracted"]
        print(f"{'✓ ' + result['label']:<45} | {ec:>6.3f} {fc:>6.3f} | {triples}")

    out_path = PROJECT_ROOT / "evaluation" / "direct_llm_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")

    print("\n\nSummary for paper Table 4 (Direct LLM column):")
    for r in results:
        ec = r["entity_coverage"]["coverage"]
        fc = r["fact_coverage"]["coverage"]
        print(f"  {r['label']}: EC={ec:.3f}, FC={fc:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
