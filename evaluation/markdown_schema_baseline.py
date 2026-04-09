#!/usr/bin/env python3
"""
markdown_schema_baseline.py — Markdown-native Schema Format-Constraint Coupling Validation.

Tests the format-constraint coupling hypothesis: Markdown serialization paired
with a Markdown-native extraction schema can achieve FC > 0, even though
Markdown + SGE Schema gives FC = 0.000.

Design:
  - Markdown serialization: CSV rows → Markdown table format (| col1 | col2 | ... |)
  - Markdown-native schema: extraction constraints expressed in Markdown table format
  - LLM outputs triples as Markdown table rows
  - Triples assembled into GraphML for evaluation

This validates: the coupling mechanism (format ↔ schema) matters more than
any specific format, making the effect a general principle.

Hypothesis:
  Markdown + SGE Schema = FC 0.000  (mismatch)
  SGE format + SGE Schema = FC 1.000  (match)
  Markdown + Markdown Schema = FC > 0  (matched, different format)

Usage:
    python3 evaluation/markdown_schema_baseline.py
    python3 evaluation/markdown_schema_baseline.py --fresh
"""

from __future__ import annotations

import asyncio
import argparse
import json
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
from evaluation.config import API_KEY, BASE_URL, MODEL

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DATASET = {
    "label": "WHO Life Expectancy",
    "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
    "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
    "output_dir": "output/markdown_schema_who",
}

CHUNK_SIZE = 5  # rows per chunk

# ---------------------------------------------------------------------------
# Prompt design: Markdown-native schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a knowledge graph extractor for statistical time-series data.

You will receive a Markdown table chunk from a life expectancy dataset.
Each row contains: country code, year columns with numeric values.

Your task is to extract subject-relation-object triples and output them
as a Markdown table.

Extraction rules:
1. Subject: the Country_Code from column 2 (e.g., AFG, CHN, USA)
2. For each year column with a non-empty numeric value, create one triple:
   - relation: HAS_LIFE_EXPECTANCY
   - object: format as "YEAR=VALUE_years" (e.g., 2000=53.82_years)
3. Skip rows where country code is empty or all-regional (no letters before numbers)
4. Preserve exact numeric values — do not round or truncate

Output format — a Markdown table with exactly these columns:
| Subject | Relation | Object |
|---------|----------|--------|
| AFG | HAS_LIFE_EXPECTANCY | 2000=53.82_years |
| AFG | HAS_LIFE_EXPECTANCY | 2005=56.79_years |

Output ONLY the Markdown table, no other text.
"""

USER_PROMPT_TEMPLATE = """\
Extract triples from this Markdown table chunk:

{markdown_table}

Output a Markdown table of all extracted triples:
| Subject | Relation | Object |
|---------|----------|--------|"""


# ---------------------------------------------------------------------------
# Markdown serialization
# ---------------------------------------------------------------------------

def build_markdown_table(headers: list[str], rows: list[list]) -> str:
    """Serialize headers + rows as a Markdown table string."""
    col_count = len(headers)

    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    data_rows = []
    for row in rows:
        cells = [str(v) if v is not None else "" for v in row]
        # Pad or trim to match header count
        while len(cells) < col_count:
            cells.append("")
        cells = cells[:col_count]
        data_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, separator] + data_rows)


def make_chunks(df, chunk_size: int) -> list[tuple[list[str], list[list]]]:
    """Split DataFrame into (headers, rows) chunks."""
    headers = list(df.columns)
    chunks = []
    rows_list = df.values.tolist()
    for i in range(0, len(rows_list), chunk_size):
        chunk_rows = rows_list[i : i + chunk_size]
        chunks.append((headers, chunk_rows))
    return chunks


# ---------------------------------------------------------------------------
# LLM call
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
    return ""


# ---------------------------------------------------------------------------
# Parse Markdown table output
# ---------------------------------------------------------------------------

def parse_markdown_triples(text: str) -> list[dict]:
    """Parse LLM Markdown table output into list of {subject, relation, object}."""
    triples = []
    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        # Skip header and separator rows
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        if "Subject" in line and "Relation" in line:
            continue

        # Split by pipe
        parts = [p.strip() for p in line.split("|")]
        # parts[0] is empty (before first |), parts[-1] is empty (after last |)
        parts = [p for p in parts if p]

        if len(parts) < 3:
            continue

        subject = parts[0].strip()
        relation = parts[1].strip()
        obj = parts[2].strip()

        if not subject or not relation or not obj:
            continue

        triples.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
        })

    return triples


def parse_object_year_value(obj_str: str) -> tuple[str, str]:
    """Parse 'YEAR=VALUE_years' format into (year, value)."""
    # Pattern: 2000=53.82_years or 2000=53.82
    match = re.match(r"^(\d{4})=([0-9.]+)(?:_years)?$", obj_str.strip())
    if match:
        return match.group(1), match.group(2)
    return "", ""


# ---------------------------------------------------------------------------
# GraphML builder
# ---------------------------------------------------------------------------

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


class GraphMLBuilder:
    """Minimal immutable-friendly GraphML builder."""

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
            existing_desc = self._nodes[nid]["description"]
            if description and description not in existing_desc:
                self._nodes[nid] = {
                    **self._nodes[nid],
                    "description": f"{existing_desc}\n{description}".strip(),
                }

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

        for kid, attr, for_el in [
            ("d0", "label", "node"),
            ("d1", "entity_type", "node"),
            ("d2", "description", "node"),
            ("d3", "keywords", "edge"),
        ]:
            key_el = ET.SubElement(root, "key", id=kid, **{
                "for": for_el,
                "attr.name": attr,
                "attr.type": "string",
            })
            ET.SubElement(key_el, "default").text = ""

        graph = ET.SubElement(root, "graph", edgedefault="directed")

        for nid, attrs in self._nodes.items():
            node = ET.SubElement(graph, "node", id=nid)
            ET.SubElement(node, "data", key="d0").text = attrs["label"]
            ET.SubElement(node, "data", key="d1").text = attrs["entity_type"]
            ET.SubElement(node, "data", key="d2").text = attrs["description"]

        for i, e in enumerate(self._edges):
            edge = ET.SubElement(graph, "edge", id=f"e{i}",
                                 source=e["source"], target=e["target"])
            ET.SubElement(edge, "data", key="d0").text = e["label"]
            ET.SubElement(edge, "data", key="d3").text = e["keywords"]
            ET.SubElement(edge, "data", key="d2").text = e["description"]

        ET.indent(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Core pipeline: CSV → Markdown chunks → LLM → GraphML
# ---------------------------------------------------------------------------

async def extract_dataset(csv_path: str, output_dir: Path) -> dict:
    """Extract triples via Markdown-native schema prompt."""
    df, _ = preprocess_csv(str(PROJECT_ROOT / csv_path))

    chunks = make_chunks(df, CHUNK_SIZE)
    builder = GraphMLBuilder()
    total_triples = 0
    total_errors = 0
    total_skipped = 0

    # Build prompts for all chunks
    prompts = []
    for headers, rows in chunks:
        md_table = build_markdown_table(headers, rows)
        prompt = USER_PROMPT_TEMPLATE.format(markdown_table=md_table)
        prompts.append(prompt)

    print(f"  Sending {len(prompts)} LLM requests (concurrency=5, chunk_size={CHUNK_SIZE})...")
    responses = await asyncio.gather(
        *[call_llm(p) for p in prompts],
        return_exceptions=True,
    )

    for chunk_idx, (resp, (headers, rows)) in enumerate(zip(responses, chunks)):
        if isinstance(resp, Exception):
            print(f"  [Chunk {chunk_idx}] ERROR: {resp}")
            total_errors += 1
            continue

        triples = parse_markdown_triples(resp)
        if not triples:
            total_skipped += 1
            continue

        for triple in triples:
            subject = triple["subject"].strip().upper()
            relation = triple["relation"].strip()
            obj_str = triple["object"].strip()

            if not subject or not obj_str:
                continue

            year, value = parse_object_year_value(obj_str)

            # Build graph: subject node + value node + edge
            builder.add_node(subject, subject, "Country_Code")

            if year and value:
                val_node_id = f"{subject}_{year}_{value}"
                val_label = f"year:{year}, value:{value}"
                description = (
                    f"{subject} life expectancy in {year}: {value} years"
                )
                builder.add_node(val_node_id, val_label, "StatValue", description)
                builder.add_edge(
                    subject, val_node_id,
                    label=relation,
                    keywords=f"{relation}, year:{year}, value:{value}",
                    description=description,
                )
                total_triples += 1
            else:
                # Object not in expected format — store as-is with raw description
                val_node_id = f"{subject}_RAW_{obj_str[:30]}"
                description = f"{subject} {relation} {obj_str}"
                builder.add_node(val_node_id, obj_str, "StatValue", description)
                builder.add_edge(
                    subject, val_node_id,
                    label=relation,
                    keywords=f"{relation}",
                    description=description,
                )
                total_triples += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = output_dir / "graph.graphml"
    graphml_path.write_text(builder.to_graphml_string(), encoding="utf-8")

    stats = {
        "chunks_processed": len(chunks),
        "triples_extracted": total_triples,
        "errors": total_errors,
        "skipped_empty": total_skipped,
        "nodes": len(builder._nodes),
        "edges": len(builder._edges),
    }
    print(
        f"  Extracted {total_triples} triples → "
        f"{stats['nodes']} nodes, {stats['edges']} edges"
    )
    print(f"  Errors: {total_errors}  Skipped (empty): {total_skipped}")

    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    """Compute EC and FC against gold standard."""
    if not Path(graph_path).exists():
        print(f"  [warn] Graph not found: {graph_path}", file=sys.stderr)
        return {"label": label, "ec": 0.0, "fc": 0.0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(
        f"  [{label}] EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
        f"FC={fc:.4f} ({len(covered)}/{len(facts)})"
    )

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

async def run_experiment(fresh: bool) -> dict:
    cfg = DATASET
    print(f"\n{'='*60}")
    print(f"MARKDOWN-NATIVE SCHEMA BASELINE: {cfg['label']}")
    print(f"{'='*60}")
    print(f"Hypothesis: Markdown + Markdown Schema → FC > 0")
    print(f"Compare:    Markdown + SGE Schema → FC = 0.000")
    print()

    gold_path = str(PROJECT_ROOT / cfg["gold"])
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    graph_path = str(output_dir / "graph.graphml")

    if not fresh and Path(graph_path).exists():
        print(f"[Step 1] Using cached graph: {graph_path}")
        stats = {"cached": True}
    else:
        print(f"[Step 1] Markdown-native extraction...")
        stats = await extract_dataset(cfg["csv_path"], output_dir)

    print(f"\n[Step 2] FC/EC Evaluation...")
    result = evaluate_graph_fc(graph_path, gold_path, "Markdown-Schema")

    return {
        "dataset": "who_life_expectancy",
        "label": cfg["label"],
        "baseline_type": "markdown_native_schema",
        "description": (
            "Markdown table serialization + Markdown-native extraction schema. "
            "Tests format-constraint coupling: when format and schema are matched "
            "(both Markdown-based), does FC recover from the 0.000 mismatch case?"
        ),
        "hypothesis": {
            "sge_format_sge_schema": 1.000,
            "markdown_sge_schema": 0.000,
            "markdown_markdown_schema": "this experiment",
        },
        "stats": stats,
        "evaluation": result,
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(fresh: bool) -> None:
    result = await run_experiment(fresh)

    output_path = (
        PROJECT_ROOT / "evaluation" / "results" / "markdown_schema_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    ev = result.get("evaluation", {})
    print(f"\n{'='*60}")
    print(f"RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:   {result['label']}")
    print(f"  EC:        {ev.get('ec', 0):.4f} ({ev.get('ec_matched', 0)}/{ev.get('ec_total', 0)})")
    print(f"  FC:        {ev.get('fc', 0):.4f} ({ev.get('fc_covered', 0)}/{ev.get('fc_total', 0)})")
    print(f"{'='*60}")
    print(f"\nComparison:")
    print(f"  SGE format + SGE Schema:      FC = 1.000 (full coupling)")
    print(f"  Markdown + SGE Schema:        FC = 0.000 (format mismatch)")
    print(f"  Markdown + Markdown Schema:   FC = {ev.get('fc', 0):.3f} (this experiment)")
    conclusion = "SUPPORTS" if ev.get("fc", 0) > 0 else "DOES NOT SUPPORT"
    print(f"\n  Conclusion: {conclusion} the coupling hypothesis")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Markdown-native schema format-constraint coupling baseline"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force fresh extraction (ignore cached graph)"
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.fresh))


if __name__ == "__main__":
    main()
