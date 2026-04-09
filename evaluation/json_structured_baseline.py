#!/usr/bin/env python3
"""
json_structured_baseline.py — JSON Structured Output Baseline.

Tests whether the format-constraint coupling effect is unique to SGE's
delimiter-based prompt template, or if JSON structured output can also
achieve high cell-fact binding fidelity.

Design:
  - Uses SGE's row-level serialization (same chunks as SGE)
  - LLM outputs structured JSON triples instead of delimiter-based format
  - Triples are assembled into a GraphML file for evaluation
  - No LightRAG entity extraction — direct LLM → graph construction

This answers: "Is the win from SGE's coupling mechanism, or from any
structured output mode?"

Usage:
    python3 evaluation/json_structured_baseline.py
    python3 evaluation/json_structured_baseline.py --dataset who
    python3 evaluation/json_structured_baseline.py --dataset who --fresh
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.preprocessor import preprocess_csv
from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

API_KEY  = os.environ.get("SGE_API_KEY", "")
BASE_URL = "https://wolfai.top/v1"
MODEL    = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# JSON extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a knowledge graph extractor for statistical CSV data.

Given a data row from a CSV table, extract structured facts as JSON.

Output a JSON array of objects. Each object has:
{
  "subject": "entity name (e.g., country code or category name)",
  "subject_type": "Country" or "Category",
  "relation": "HAS_VALUE_IN_YEAR" or "HAS_VALUE",
  "year": "the year (e.g., 2020) or empty string if not applicable",
  "value": "the numeric value as string",
  "value_description": "brief context (e.g., life expectancy in years)"
}

Rules:
- Extract ONE subject per row (the primary entity identifier)
- Extract ONE fact per non-empty numeric value column
- Preserve exact numeric values (do not round)
- Include the year from the column header
- Output ONLY valid JSON array, no other text
"""

USER_PROMPT_TEMPLATE = """\
Headers: {headers}
Data row: {row_values}

Extract all facts as a JSON array:"""

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "output_dir": "output/json_structured_who",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv_path": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "output_dir": "output/json_structured_wb_cm",
    },
    "wb_pop": {
        "label": "WB Population",
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "output_dir": "output/json_structured_wb_pop",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "csv_path": "dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "output_dir": "output/json_structured_wb_mat",
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "output_dir": "output/json_structured_inpatient",
    },
    "fortune500": {
        "label": "Fortune 500 Revenue",
        "csv_path": str(PROJECT_ROOT.parent / "dataset" / "non_gov" / "fortune500_revenue.csv"),
        "gold": "evaluation/gold/gold_fortune500_revenue.jsonl",
        "output_dir": "output/json_structured_fortune500",
    },
    "the": {
        "label": "THE University Ranking",
        "csv_path": str(PROJECT_ROOT.parent / "dataset" / "non_gov" / "the_university_ranking.csv"),
        "gold": "evaluation/gold/gold_the_university_ranking.jsonl",
        "output_dir": "output/json_structured_the",
    },
}


# ---------------------------------------------------------------------------
# LLM call (direct, no LightRAG)
# ---------------------------------------------------------------------------

_SEMAPHORE = asyncio.Semaphore(5)


async def call_llm(user_prompt: str, max_retries: int = 3) -> str:
    """Call Claude Haiku via OpenAI-compatible API with retry for 429."""
    import httpx

    async with _SEMAPHORE:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json={
                            "model": MODEL,
                            "temperature": 0,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": 4096,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    # Strip markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        # Try to find JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return []


# ---------------------------------------------------------------------------
# GraphML builder (same as det parser pattern)
# ---------------------------------------------------------------------------

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


class GraphMLBuilder:
    """Minimal GraphML builder for assembling extracted triples."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []

    def add_node(self, node_id: str, label: str, node_type: str = "",
                 description: str = "") -> None:
        nid = node_id.upper()
        if nid not in self._nodes:
            self._nodes[nid] = {
                "label": label,
                "entity_type": node_type,
                "description": description,
            }
        else:
            # Append description
            existing = self._nodes[nid]["description"]
            if description and description not in existing:
                self._nodes[nid]["description"] = f"{existing}\n{description}".strip()

    def add_edge(self, source: str, target: str, label: str = "",
                 keywords: str = "", description: str = "") -> None:
        self._edges.append({
            "source": source.upper(),
            "target": target.upper(),
            "label": label,
            "keywords": keywords,
            "description": description,
        })

    def to_graphml_string(self) -> str:
        root = ET.Element("graphml", xmlns=_GRAPHML_NS)

        # Key definitions
        for kid, attr, default in [
            ("d0", "label", ""), ("d1", "entity_type", ""),
            ("d2", "description", ""), ("d3", "keywords", ""),
        ]:
            key_el = ET.SubElement(root, "key", id=kid, **{
                "for": "node" if kid != "d3" else "edge",
                "attr.name": attr, "attr.type": "string",
            })
            ET.SubElement(key_el, "default").text = default

        graph = ET.SubElement(root, "graph", edgedefault="directed")

        # Nodes
        for nid, attrs in self._nodes.items():
            node = ET.SubElement(graph, "node", id=nid)
            ET.SubElement(node, "data", key="d0").text = attrs["label"]
            ET.SubElement(node, "data", key="d1").text = attrs["entity_type"]
            ET.SubElement(node, "data", key="d2").text = attrs["description"]

        # Edges
        for i, e in enumerate(self._edges):
            edge = ET.SubElement(graph, "edge", id=f"e{i}",
                                 source=e["source"], target=e["target"])
            ET.SubElement(edge, "data", key="d0").text = e["label"]
            ET.SubElement(edge, "data", key="d3").text = e["keywords"]
            ET.SubElement(edge, "data", key="d2").text = e["description"]

        ET.indent(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Core pipeline: CSV → LLM JSON → GraphML
# ---------------------------------------------------------------------------

async def extract_dataset(
    csv_path: str,
    output_dir: Path,
    fresh: bool = False,
) -> dict:
    """Extract triples from CSV using JSON structured output."""
    df, _ = preprocess_csv(str(PROJECT_ROOT / csv_path))

    headers_str = ", ".join(str(c) for c in df.columns)
    builder = GraphMLBuilder()
    total_facts = 0
    total_errors = 0

    tasks = []
    rows_data = []
    for _, row in df.iterrows():
        row_values = ", ".join(str(v) for v in row.values)
        prompt = USER_PROMPT_TEMPLATE.format(
            headers=headers_str,
            row_values=row_values,
        )
        tasks.append(call_llm(prompt))
        rows_data.append(row)

    print(f"  Sending {len(tasks)} LLM requests (concurrency=5)...")
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, (resp, row) in enumerate(zip(responses, rows_data)):
        if isinstance(resp, Exception):
            print(f"  [Row {i}] ERROR: {resp}")
            total_errors += 1
            continue

        facts = parse_json_response(resp)
        if not facts:
            total_errors += 1
            continue

        for fact in facts:
            subject = str(fact.get("subject", "")).strip()
            year = str(fact.get("year", "")).strip()
            value = str(fact.get("value", "")).strip()
            relation = str(fact.get("relation", "HAS_VALUE_IN_YEAR")).strip()
            subj_type = str(fact.get("subject_type", "Entity")).strip()
            desc = str(fact.get("value_description", "")).strip()

            if not subject or not value:
                continue

            # Build graph nodes and edges
            builder.add_node(subject, subject, subj_type)

            if year:
                val_node_id = f"{subject}_{year}_{value}"
                val_label = f"year:{year}, value:{value}"
                builder.add_node(val_node_id, val_label, "StatValue", desc)
                builder.add_edge(
                    subject, val_node_id,
                    label=relation,
                    keywords=f"{relation}, year:{year}",
                    description=f"{subject} has value {value} in year {year}. {desc}",
                )
            else:
                val_node_id = f"{subject}_{value}"
                val_label = f"value:{value}"
                builder.add_node(val_node_id, val_label, "StatValue", desc)
                builder.add_edge(
                    subject, val_node_id,
                    label=relation,
                    keywords=relation,
                    description=f"{subject} has value {value}. {desc}",
                )
            total_facts += 1

    # Write GraphML
    output_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = output_dir / "graph.graphml"
    graphml_path.write_text(builder.to_graphml_string(), encoding="utf-8")

    stats = {
        "rows_processed": len(rows_data),
        "facts_extracted": total_facts,
        "errors": total_errors,
        "nodes": len(builder._nodes),
        "edges": len(builder._edges),
    }
    print(f"  Extracted {total_facts} facts → {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Errors: {total_errors}")

    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    if not Path(graph_path).exists():
        print(f"  [warn] Graph not found: {graph_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  [{label}] EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
          f"FC={fc:.4f} ({len(covered)}/{len(facts)})")

    return {
        "label": label,
        "ec": round(ec, 4),
        "fc": round(fc, 4),
        "ec_matched": len(matched_entities),
        "ec_total": len(gold_entities),
        "fc_covered": len(covered),
        "fc_total": len(facts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_dataset(dataset_key: str, fresh: bool) -> dict:
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"JSON STRUCTURED BASELINE: {cfg['label']}")
    print(f"{'='*60}")

    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]

    # Extract via JSON structured output
    print(f"\n[Step 1] JSON structured extraction...")
    stats = await extract_dataset(cfg["csv_path"], output_dir, fresh=fresh)

    # Evaluate
    print(f"\n[Step 2] FC/EC Evaluation...")
    graph_path = str(output_dir / "graph.graphml")
    result = evaluate_graph_fc(graph_path, gold_path, "JSON-Structured")

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "baseline_type": "json_structured_output",
        "description": (
            "Per-row CSV → LLM JSON structured output → direct graph assembly. "
            "No LightRAG, no delimiter parsing, no schema induction. "
            "Tests whether JSON output mode alone achieves coupling."
        ),
        "stats": stats,
        "evaluation": result,
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(datasets: list[str], fresh: bool) -> None:
    output_path = PROJECT_ROOT / "evaluation" / "results" / "json_structured_baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)

    for ds in datasets:
        result = await run_dataset(ds, fresh)
        all_results[ds] = result

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'='*60}")
    print(f"{'Dataset':<25s} {'EC':>8s} {'FC':>8s} {'Facts':>8s} {'Errors':>8s}")
    print(f"{'-'*60}")
    for ds, r in all_results.items():
        ev = r.get("evaluation", {})
        st = r.get("stats", {})
        print(f"{r['label']:<25s} {ev.get('ec', 0):.4f}   {ev.get('fc', 0):.4f}   "
              f"{st.get('facts_extracted', 0):>6d}   {st.get('errors', 0):>6d}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON structured output baseline")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Run single dataset (default: all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Force fresh extraction")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    asyncio.run(main_async(datasets, args.fresh))


if __name__ == "__main__":
    main()
