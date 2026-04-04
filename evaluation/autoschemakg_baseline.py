#!/usr/bin/env python3
"""
autoschemakg_baseline.py — AutoSchemaKG (atlas-rag) baseline for multi-system comparison.

Runs AutoSchemaKG triple extraction on WHO CSV data, converts to GraphML,
then evaluates Fact Coverage (FC) against the gold standard.

Usage:
    python3 evaluation/autoschemakg_baseline.py
    python3 evaluation/autoschemakg_baseline.py --dataset who --output output/autoschemakg_who

Configuration:
    LLM: Claude Haiku via wolfai proxy (OpenAI-compatible)
    Dataset: WHO Life Expectancy (same chunks as LightRAG Baseline)

Requires:
    atlas-rag==0.0.5.post1
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_API_KEY = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
LLM_BASE_URL = "https://wolfai.top/v1"
LLM_MODEL = "claude-haiku-4-5-20251001"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CONFIG = {
    "who": {
        "csv_path": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold_path": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "chunks_dir": "output/who_life_expectancy/chunks",
        "filename_pattern": "who_chunks",
        "label": "WHO Life Expectancy",
    },
    "wb_cm": {
        "csv_path": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold_path": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "chunks_dir": "output/wb_child_mortality/chunks",
        "filename_pattern": "wb_cm_chunks",
        "label": "WB Child Mortality",
    },
    "wb_pop": {
        "csv_path": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold_path": "evaluation/gold/gold_wb_population_v2.jsonl",
        "chunks_dir": "output/wb_population/chunks",
        "filename_pattern": "wb_pop_chunks",
        "label": "WB Population",
    },
    "inpatient": {
        "csv_path": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold_path": "evaluation/gold/gold_inpatient_2023.jsonl",
        "chunks_dir": "output/inpatient_2023/chunks",
        "filename_pattern": "inpatient_chunks",
        "label": "HK Inpatient 2023",
    },
}


# ---------------------------------------------------------------------------
# Step 1: Prepare input JSONL from pre-computed chunks
# ---------------------------------------------------------------------------

def prepare_input_jsonl(dataset: str, jsonl_dir: str) -> str:
    """
    Write a JSONL file from the pre-computed text chunks used by LightRAG Baseline.

    AutoSchemaKG's load_dataset() expects .json/.jsonl files with records
    of shape: {"id": str, "text": str, "metadata": dict}.

    Returns the path to the written JSONL file.
    """
    cfg = DATASET_CONFIG[dataset]
    chunks_dir = PROJECT_ROOT / cfg["chunks_dir"]
    pattern = cfg["filename_pattern"]

    if not chunks_dir.exists():
        raise FileNotFoundError(
            f"Chunks directory not found: {chunks_dir}\n"
            "Run the LightRAG baseline first to generate serialized chunks, "
            "or check that output/who_life_expectancy/chunks/ exists."
        )

    chunk_files = sorted(chunks_dir.glob("*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"No .txt chunk files found in {chunks_dir}")

    print(f"Loading {len(chunk_files)} pre-computed chunks from {chunks_dir}")

    jsonl_path = os.path.join(jsonl_dir, f"{pattern}.jsonl")
    records = []
    for i, chunk_file in enumerate(chunk_files):
        text = chunk_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        records.append({
            "id": f"{pattern}_{i:04d}",
            "text": text,
            "metadata": {
                "source": str(chunk_file.name),
                "dataset": dataset,
            },
        })

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} records to {jsonl_path}")
    return jsonl_path


# ---------------------------------------------------------------------------
# Step 2: Run AutoSchemaKG extraction
# ---------------------------------------------------------------------------

def run_autoschemakg(dataset: str, output_dir: str) -> str:
    """
    Run AutoSchemaKG triple extraction.

    Returns the path to the output GraphML file.
    """
    from openai import OpenAI
    from atlas_rag.kg_construction.triple_extraction import (
        KnowledgeGraphExtractor,
        LLMGenerator,
        ProcessingConfig,
    )

    cfg = DATASET_CONFIG[dataset]
    pattern = cfg["filename_pattern"]
    output_path = PROJECT_ROOT / output_dir

    # Prepare input JSONL in a temp directory that AutoSchemaKG can scan
    input_dir = output_path / "input_texts"
    input_dir.mkdir(parents=True, exist_ok=True)
    prepare_input_jsonl(dataset, str(input_dir))

    print(f"\nInitializing AutoSchemaKG extractor...")
    print(f"  LLM: {LLM_MODEL} via {LLM_BASE_URL}")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_path}")

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    llm = LLMGenerator(client=client, model_name=LLM_MODEL, backend="openai")

    config = ProcessingConfig(
        model_path=LLM_MODEL,
        data_directory=str(input_dir),
        filename_pattern=pattern,
        output_directory=str(output_path),
        batch_size_triple=16,
        include_concept=False,
        chunk_size=8192,
    )

    extractor = KnowledgeGraphExtractor(model=llm, config=config)

    print("\nRunning triple extraction...")
    extractor.run_extraction()

    print("\nConverting to CSV format...")
    extractor.convert_json_to_csv()

    print("\nConverting to GraphML...")
    extractor.convert_to_graphml()

    graphml_path = output_path / "kg_graphml" / f"{pattern}_graph.graphml"
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"Expected GraphML not found at: {graphml_path}\n"
            "Check AutoSchemaKG output for errors."
        )

    print(f"\nGraphML written to: {graphml_path}")
    return str(graphml_path)


# ---------------------------------------------------------------------------
# Step 3: Load AutoSchemaKG GraphML with custom loader
# ---------------------------------------------------------------------------

def load_autoschemakg_graphml(graphml_path: str):
    """
    Load an AutoSchemaKG GraphML file and build a unified (G, nodes, entity_text_2hop) triple.

    AutoSchemaKG uses different attribute names than LightRAG:
      - Nodes: 'id' (entity name), 'type'  (no 'entity_name')
      - Edges: 'relation', 'type'           (no 'keywords' / 'description')
    """
    import networkx as nx

    G = nx.read_graphml(graphml_path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nodes: dict[str, dict] = {}
    node_id_to_name: dict[str, str] = {}

    for node_id, data in G.nodes(data=True):
        # AutoSchemaKG stores entity name in 'id' attribute
        name = (
            data.get("id")
            or data.get("entity_name")
            or data.get("name")
            or str(node_id)
        )
        name = str(name).strip()
        etype = str(data.get("type", "")).strip()
        nodes[name] = {"type": etype, "description": name}
        node_id_to_name[node_id] = name

    entity_text: dict[str, list[str]] = {}

    for u, v, data in G.edges(data=True):
        u_name = node_id_to_name.get(u, str(u))
        v_name = node_id_to_name.get(v, str(v))
        relation = str(data.get("relation", "")).strip()

        # Index the relation and neighbor name for fact search
        edge_text_uv = f"{relation} {v_name}"
        edge_text_vu = f"{relation} {u_name}"
        entity_text.setdefault(u_name, []).append(edge_text_uv)
        entity_text.setdefault(v_name, []).append(edge_text_vu)

    # Also add each entity name to its own text list (value-in-name matching)
    for name in nodes:
        entity_text.setdefault(name, []).append(name)

    # Build 2-hop index
    entity_text_2hop: dict[str, list[str]] = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, str(nb_id))
            texts.extend(entity_text.get(nb_name, []))
            # Include neighbor name itself for value matching
            texts.append(nb_name)
        entity_text_2hop[name] = texts

    return G, nodes, entity_text_2hop


# ---------------------------------------------------------------------------
# Step 4: Evaluate FC
# ---------------------------------------------------------------------------

def evaluate_fc(graphml_path: str, gold_path: str) -> dict:
    """
    Evaluate Fact Coverage against the gold standard.

    Returns a dict with keys: nodes, edges, fc, gold_facts, covered_facts.
    """
    import networkx as nx
    from evaluation.evaluate_coverage import load_gold, check_fact_coverage

    G, nodes, entity_text_2hop = load_autoschemakg_graphml(graphml_path)

    entities, facts = load_gold(gold_path)
    print(f"Gold: {len(entities)} entities, {len(facts)} value-facts")

    covered, not_covered = check_fact_coverage(facts, nodes, entity_text_2hop)
    fc = len(covered) / len(facts) if facts else 0.0

    reasons: dict[str, int] = {}
    for item in not_covered:
        r = item.get("reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    print(f"\nFact Coverage: {len(covered)}/{len(facts)} = {fc:.4f}")
    if reasons:
        print("Uncovered breakdown:")
        for r, c in sorted(reasons.items()):
            print(f"  {r}: {c}")

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "fc": round(fc, 4),
        "gold_facts": len(facts),
        "covered_facts": len(covered),
        "uncovered_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Step 5: Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, results_path: str) -> None:
    """Write evaluation results JSON."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoSchemaKG baseline: extract KG from CSV, evaluate FC."
    )
    parser.add_argument(
        "--dataset",
        default="who",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset to evaluate (default: who)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for AutoSchemaKG artifacts (default: output/autoschemakg_{dataset})",
    )
    parser.add_argument(
        "--results",
        default="evaluation/results/autoschemakg_results.json",
        help="Path to write FC evaluation results",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction; load existing GraphML and re-evaluate only",
    )
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output or f"output/autoschemakg_{dataset}"
    cfg = DATASET_CONFIG[dataset]
    gold_path = PROJECT_ROOT / cfg["gold_path"]
    pattern = cfg["filename_pattern"]

    print(f"AutoSchemaKG Baseline Evaluation")
    print(f"  Dataset : {cfg['label']}")
    print(f"  Output  : {output_dir}")
    print(f"  Gold    : {gold_path}")
    print()

    # Resolve GraphML path
    graphml_path = PROJECT_ROOT / output_dir / "kg_graphml" / f"{pattern}_graph.graphml"

    if args.skip_extraction:
        if not graphml_path.exists():
            print(f"ERROR: --skip-extraction set but no GraphML found at {graphml_path}")
            sys.exit(1)
        print(f"Skipping extraction; using existing GraphML: {graphml_path}")
    else:
        graphml_path_str = run_autoschemakg(dataset, output_dir)
        graphml_path = Path(graphml_path_str)

    # Evaluate
    print(f"\nEvaluating FC...")
    metrics = evaluate_fc(str(graphml_path), str(gold_path))

    # Build results payload — merge with existing if available
    results_path = PROJECT_ROOT / args.results
    existing: dict = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)

    results = {
        "system": "autoschemakg",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": LLM_MODEL,
            "atlas_rag_version": "0.0.5.post1",
        },
        "datasets": existing.get("datasets", {}),
    }
    results["datasets"][dataset] = {
        "label": cfg["label"],
        "nodes": metrics["nodes"],
        "edges": metrics["edges"],
        "fc": metrics["fc"],
        "gold_facts": metrics["gold_facts"],
        "covered_facts": metrics["covered_facts"],
        "uncovered_reasons": metrics["uncovered_reasons"],
    }

    print("\n" + "=" * 55)
    print("AUTOSCHEMAKG EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Dataset    : {cfg['label']}")
    print(f"  Graph nodes: {metrics['nodes']}")
    print(f"  Graph edges: {metrics['edges']}")
    print(f"  Gold facts : {metrics['gold_facts']}")
    print(f"  Covered    : {metrics['covered_facts']}")
    print(f"  FC         : {metrics['fc']:.4f}")
    print("=" * 55)

    results_path = PROJECT_ROOT / args.results
    save_results(results, str(results_path))


if __name__ == "__main__":
    main()
