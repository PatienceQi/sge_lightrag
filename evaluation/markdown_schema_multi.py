#!/usr/bin/env python3
"""
markdown_schema_multi.py — Markdown-native Schema across multiple datasets.

Extends the WHO Markdown experiment to Inpatient (Type-III) and WB Pop (large Type-II)
to validate format-constraint coupling generalizability.

Results update Table 4 in the paper.

Usage:
    python3 evaluation/markdown_schema_multi.py
    python3 evaluation/markdown_schema_multi.py --dataset inpatient
    python3 evaluation/markdown_schema_multi.py --dataset wb_pop
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.preprocessor import preprocess_csv
from evaluation.evaluate_coverage import (
    load_gold, load_graph, check_entity_coverage, check_fact_coverage,
)
from evaluation.config import API_KEY, BASE_URL, MODEL

CHUNK_SIZE_WHO = 5
CHUNK_SIZE_INPATIENT = 3
CHUNK_SIZE_WB_POP = 5

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    "inpatient": {
        "label": "HK Inpatient 2023 (Type-III)",
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "output_dir": "output/markdown_schema_inpatient",
        "chunk_size": CHUNK_SIZE_INPATIENT,
    },
    "wb_pop": {
        "label": "WB Population (Large Type-II)",
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "output_dir": "output/markdown_schema_wb_pop",
        "chunk_size": CHUNK_SIZE_WB_POP,
        "skiprows": 4,
    },
}

# ---------------------------------------------------------------------------
# Markdown-native prompts per dataset
# ---------------------------------------------------------------------------

PROMPTS_INPATIENT = {
    "system": """\
You are a knowledge graph extractor for Hong Kong hospital statistics data.

You will receive a Markdown table chunk from a disease-level inpatient statistics dataset.
Each row contains: ICD code, disease category name, and numeric statistics columns
(inpatient counts by hospital type, registered deaths by gender).

Your task is to extract subject-relation-object triples and output them
as a Markdown table.

Extraction rules:
1. Subject: the disease category name from column "疾病类别" (e.g., 肺炎, 肾衰竭)
2. For each numeric value column with a non-empty value > 0, create one triple:
   - relation: the column header (e.g., 住院病人出院及死亡人次*-合计, 全港登记死亡人数-合计)
   - object: the numeric value as a string
3. Skip rows where disease name is empty or is a section header (roman numerals, etc.)
4. Skip cells with 0, empty, or non-numeric values
5. Preserve exact numeric values

Output format — a Markdown table with exactly these columns:
| Subject | Relation | Object |
|---------|----------|--------|
| 肺炎 | 住院病人出院及死亡人次*-合计 | 60614 |
| 肺炎 | 全港登记死亡人数-合计 | 11334 |

Output ONLY the Markdown table, no other text.
""",
    "user_template": """\
Extract triples from this Markdown table chunk of Hong Kong hospital statistics:

{markdown_table}

Output a Markdown table of all extracted triples:
| Subject | Relation | Object |
|---------|----------|--------|""",
}

PROMPTS_WB_POP = {
    "system": """\
You are a knowledge graph extractor for World Bank population statistics data.

You will receive a Markdown table chunk from a country-level population dataset.
Each row contains: country name, country code, indicator name/code, and year columns
with population values.

Your task is to extract subject-relation-object triples and output them
as a Markdown table.

Extraction rules:
1. Subject: the Country Name from column 1 (e.g., China, Japan, Argentina)
2. For each year column with a non-empty numeric value, create one triple:
   - relation: HAS_POPULATION
   - object: format as "YEAR=VALUE" (e.g., 2000=1262645000.0)
3. Skip rows where country name is empty
4. Preserve exact numeric values — do not round or truncate

Output format — a Markdown table with exactly these columns:
| Subject | Relation | Object |
|---------|----------|--------|
| China | HAS_POPULATION | 2000=1262645000.0 |
| China | HAS_POPULATION | 2010=1337705000.0 |

Output ONLY the Markdown table, no other text.
""",
    "user_template": """\
Extract triples from this Markdown table chunk of World Bank population data:

{markdown_table}

Output a Markdown table of all extracted triples:
| Subject | Relation | Object |
|---------|----------|--------|""",
}


# ---------------------------------------------------------------------------
# Markdown serialization
# ---------------------------------------------------------------------------

def build_markdown_table(headers: list[str], rows: list[list]) -> str:
    col_count = len(headers)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = []
    for row in rows:
        cells = [str(v) if v is not None and str(v) != "nan" else "" for v in row]
        while len(cells) < col_count:
            cells.append("")
        cells = cells[:col_count]
        data_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_row, separator] + data_rows)


def make_chunks(df, chunk_size: int) -> list[tuple[list[str], list[list]]]:
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


async def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    import httpx
    async with _SEMAPHORE:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    resp = await client.post(
                        f"{BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json={
                            "model": MODEL,
                            "temperature": 0,
                            "messages": [
                                {"role": "system", "content": system_prompt},
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
# Parse Markdown output → triples
# ---------------------------------------------------------------------------

def parse_markdown_triples(text: str) -> list[dict]:
    triples = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        if "Subject" in line and "Relation" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        subject, relation, obj = parts[0], parts[1], parts[2]
        if subject and relation and obj:
            triples.append({"subject": subject, "relation": relation, "object": obj})
    return triples


# ---------------------------------------------------------------------------
# GraphML builder
# ---------------------------------------------------------------------------

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


class GraphMLBuilder:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []

    def add_node(self, node_id: str, label: str, node_type: str = "",
                 description: str = "") -> None:
        nid = node_id.upper()
        if nid not in self._nodes:
            self._nodes[nid] = {"label": label, "entity_type": node_type, "description": description}
        else:
            existing = self._nodes[nid]["description"]
            if description and description not in existing:
                self._nodes[nid] = {**self._nodes[nid], "description": f"{existing}\n{description}".strip()}

    def add_edge(self, source: str, target: str, label: str = "",
                 keywords: str = "", description: str = "") -> None:
        self._edges.append({
            "source": source.upper(), "target": target.upper(),
            "label": label, "keywords": keywords, "description": description,
        })

    def to_graphml_string(self) -> str:
        root = ET.Element("graphml", xmlns=_GRAPHML_NS)
        for kid, attr, for_el in [
            ("d0", "label", "node"), ("d1", "entity_type", "node"),
            ("d2", "description", "node"), ("d3", "keywords", "edge"),
        ]:
            key_el = ET.SubElement(root, "key", id=kid, **{"for": for_el, "attr.name": attr, "attr.type": "string"})
            ET.SubElement(key_el, "default").text = ""
        graph = ET.SubElement(root, "graph", edgedefault="directed")
        for nid, attrs in self._nodes.items():
            node = ET.SubElement(graph, "node", id=nid)
            ET.SubElement(node, "data", key="d0").text = attrs["label"]
            ET.SubElement(node, "data", key="d1").text = attrs["entity_type"]
            ET.SubElement(node, "data", key="d2").text = attrs["description"]
        for i, e in enumerate(self._edges):
            edge = ET.SubElement(graph, "edge", id=f"e{i}", source=e["source"], target=e["target"])
            ET.SubElement(edge, "data", key="d0").text = e["label"]
            ET.SubElement(edge, "data", key="d3").text = e["keywords"]
            ET.SubElement(edge, "data", key="d2").text = e["description"]
        ET.indent(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Dataset-specific triple → graph logic
# ---------------------------------------------------------------------------

def build_graph_inpatient(all_triples: list[dict]) -> GraphMLBuilder:
    builder = GraphMLBuilder()
    for t in all_triples:
        subject = t["subject"].strip()
        relation = t["relation"].strip()
        value = t["object"].strip()
        if not subject or not value:
            continue
        builder.add_node(subject, subject, "Disease_Category")
        val_id = f"{subject}_{relation}_{value}"
        desc = f"{subject} {relation}: {value}"
        builder.add_node(val_id, f"{relation}: {value}", "StatValue", desc)
        builder.add_edge(subject, val_id, label=relation, keywords=f"{relation}, {value}", description=desc)
    return builder


def build_graph_wb_pop(all_triples: list[dict]) -> GraphMLBuilder:
    builder = GraphMLBuilder()
    for t in all_triples:
        subject = t["subject"].strip().upper()
        obj = t["object"].strip()
        if not subject or not obj:
            continue
        # Parse YEAR=VALUE format
        match = re.match(r"^(\d{4})[=:](.+)$", obj)
        if match:
            year, value = match.group(1), match.group(2).strip()
        else:
            year, value = "", obj
        builder.add_node(subject, subject, "Country_Code")
        val_id = f"{subject}_{year}_{value}" if year else f"{subject}_RAW_{value[:30]}"
        val_label = f"year:{year}, value:{value}" if year else value
        desc = f"{subject} population in {year}: {value}" if year else f"{subject}: {value}"
        builder.add_node(val_id, val_label, "StatValue", desc)
        kw = f"HAS_POPULATION, year:{year}, value:{value}" if year else f"HAS_POPULATION, {value}"
        builder.add_edge(subject, val_id, label="HAS_POPULATION", keywords=kw, description=desc)
    return builder


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def extract_dataset(dataset_key: str, cfg: dict) -> dict:
    csv_path = str(PROJECT_ROOT / cfg["csv_path"])
    skiprows = cfg.get("skiprows", 0)

    if skiprows > 0:
        import pandas as pd
        df = pd.read_csv(csv_path, skiprows=skiprows, encoding="utf-8-sig")
    else:
        df, _ = preprocess_csv(csv_path)

    prompts_cfg = PROMPTS_INPATIENT if dataset_key == "inpatient" else PROMPTS_WB_POP
    system_prompt = prompts_cfg["system"]
    user_template = prompts_cfg["user_template"]
    chunk_size = cfg["chunk_size"]

    # WB Pop: process all countries (full dataset)
    if dataset_key == "wb_pop":
        print(f"  Processing all {len(df)} rows (full dataset)")

    chunks = make_chunks(df, chunk_size)
    print(f"  {len(chunks)} chunks (chunk_size={chunk_size}, rows={len(df)})")

    # Build and send prompts
    user_prompts = []
    for headers, rows in chunks:
        md_table = build_markdown_table(headers, rows)
        user_prompts.append(user_template.format(markdown_table=md_table))

    print(f"  Sending {len(user_prompts)} LLM requests (concurrency=5)...")
    responses = await asyncio.gather(
        *[call_llm(system_prompt, p) for p in user_prompts],
        return_exceptions=True,
    )

    all_triples = []
    errors = 0
    for resp in responses:
        if isinstance(resp, Exception):
            errors += 1
            continue
        triples = parse_markdown_triples(resp)
        all_triples.extend(triples)

    print(f"  Extracted {len(all_triples)} triples (errors={errors})")

    # Build graph
    if dataset_key == "inpatient":
        builder = build_graph_inpatient(all_triples)
    else:
        builder = build_graph_wb_pop(all_triples)

    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.graphml"
    graph_path.write_text(builder.to_graphml_string(), encoding="utf-8")

    stats = {
        "chunks": len(chunks),
        "triples": len(all_triples),
        "errors": errors,
        "nodes": len(builder._nodes),
        "edges": len(builder._edges),
    }
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    return stats


def evaluate_fc(dataset_key: str, cfg: dict) -> dict:
    graph_path = str(PROJECT_ROOT / cfg["output_dir"] / "graph.graphml")
    gold_path = str(PROJECT_ROOT / cfg["gold"])

    if not Path(graph_path).exists():
        return {"ec": 0.0, "fc": 0.0, "error": "graph_not_found"}

    gold_entities, facts = load_gold(gold_path)
    G, graph_nodes, entity_text = load_graph(graph_path)

    matched_entities = check_entity_coverage(gold_entities, graph_nodes)
    ec = len(matched_entities) / len(gold_entities) if gold_entities else 0.0

    covered, _ = check_fact_coverage(facts, graph_nodes, entity_text)
    fc = len(covered) / len(facts) if facts else 0.0

    print(f"  EC={ec:.4f} ({len(matched_entities)}/{len(gold_entities)})  "
          f"FC={fc:.4f} ({len(covered)}/{len(facts)})")

    return {
        "ec": round(ec, 4), "fc": round(fc, 4),
        "ec_matched": len(matched_entities), "ec_total": len(gold_entities),
        "fc_covered": len(covered), "fc_total": len(facts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_dataset(dataset_key: str, fresh: bool) -> dict:
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"MARKDOWN SCHEMA: {cfg['label']}")
    print(f"{'='*60}")

    graph_path = PROJECT_ROOT / cfg["output_dir"] / "graph.graphml"
    if not fresh and graph_path.exists():
        print("  Using cached graph")
        stats = {"cached": True}
    else:
        stats = await extract_dataset(dataset_key, cfg)

    result = evaluate_fc(dataset_key, cfg)

    return {
        "dataset": dataset_key,
        "label": cfg["label"],
        "stats": stats,
        "evaluation": result,
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(args):
    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    all_results = {}
    for ds in datasets_to_run:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}. Available: {', '.join(DATASETS.keys())}")
            continue
        result = await run_dataset(ds, args.fresh)
        all_results[ds] = result

    # Summary
    print(f"\n{'='*60}")
    print("MARKDOWN SCHEMA MULTI-DATASET SUMMARY")
    print(f"{'='*60}")
    # Include known WHO result
    print(f"  {'Dataset':<30} {'EC':>6} {'FC':>6}")
    print(f"  {'-'*50}")
    print(f"  {'WHO (known result)':<30} {'1.000':>6} {'1.000':>6}")
    for ds, r in all_results.items():
        ev = r.get("evaluation", {})
        ec_s = f"{ev.get('ec', 0):.3f}"
        fc_s = f"{ev.get('fc', 0):.3f}"
        print(f"  {r['label']:<30} {ec_s:>6} {fc_s:>6}")
    print(f"{'='*60}")

    # Save
    output_path = PROJECT_ROOT / "evaluation" / "results" / "markdown_schema_multi_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"who_known": {"ec": 1.0, "fc": 1.0}, **all_results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Markdown Schema Multi-Dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run single dataset (inpatient, wb_pop)")
    parser.add_argument("--fresh", action="store_true", help="Force re-extraction")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
