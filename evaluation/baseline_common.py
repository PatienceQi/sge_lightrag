"""
baseline_common.py — Shared utilities for baseline evaluation scripts.

Provides LLM function, embedding function, and FC/EC evaluation helpers
used by all baseline scripts (row_local, fixed_stv, table_aware, json_structured, etc.).

Usage:
    from evaluation.baseline_common import (
        llm_model_func, EMBEDDING_FUNC, evaluate_graph_fc,
    )
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

from evaluation.config import API_KEY, BASE_URL, MODEL, EMBED_DIM

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM function (shared by all LightRAG-based baselines)
# ---------------------------------------------------------------------------

async def llm_model_func(
    prompt,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedding function (deterministic hash-based, no API cost)
# ---------------------------------------------------------------------------

def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(EMBED_DIM, len(h))):
        vec[i] = (h[i] - 128) / 128.0
    return vec


async def safe_embedding_func(texts: list[str]) -> np.ndarray:
    return np.array([_hash_embed(t) for t in texts], dtype=np.float32)


EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=512,
    func=safe_embedding_func,
)


# ---------------------------------------------------------------------------
# FC/EC evaluation helper
# ---------------------------------------------------------------------------

def evaluate_graph_fc(graph_path: str, gold_path: str, label: str) -> dict:
    """Run FC/EC evaluation on a graph against a gold standard."""
    from evaluation.evaluate_coverage import (
        load_gold, load_graph, check_entity_coverage, check_fact_coverage,
    )

    if not Path(graph_path).exists():
        import sys
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
# Dataset paths (canonical registry)
# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "who": {
        "label": "WHO Life Expectancy",
        "csv": "dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv",
        "gold": "evaluation/gold/gold_who_life_expectancy_v2.jsonl",
        "sge_graph": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "wb_cm": {
        "label": "WB Child Mortality",
        "csv": "dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "gold": "evaluation/gold/gold_wb_child_mortality_v2.jsonl",
        "sge_graph": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "wb_pop": {
        "label": "WB Population",
        "csv": "dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "gold": "evaluation/gold/gold_wb_population_v2.jsonl",
        "sge_graph": "output/wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_population/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "wb_mat": {
        "label": "WB Maternal Mortality",
        "csv": "dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "gold": "evaluation/gold/gold_wb_maternal_v2.jsonl",
        "sge_graph": "output/wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_wb_maternal/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "inpatient": {
        "label": "HK Inpatient 2023",
        "csv": "dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
        "gold": "evaluation/gold/gold_inpatient_2023.jsonl",
        "sge_graph": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
        "baseline_graph": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "fortune500": {
        "label": "Fortune 500 Revenue",
        "csv": "dataset/non_gov/fortune500_revenue.csv",
        "gold": "evaluation/gold/gold_fortune500_revenue.jsonl",
        "sge_graph": "output/fortune500_revenue/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
    "the": {
        "label": "THE University Ranking",
        "csv": "dataset/non_gov/the_university_ranking.csv",
        "gold": "evaluation/gold/gold_the_university_ranking.jsonl",
        "sge_graph": "output/the_university_ranking/lightrag_storage/graph_chunk_entity_relation.graphml",
        "language": "English",
    },
}
