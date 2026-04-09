#!/usr/bin/env python3
"""
graph_guided_retriever.py — Graph-guided entity enumeration + compact chunk retrieval.

Core insight: graph solves "WHICH entities", compact chunks solve "WHAT values".
  - L1/L2: vector retrieval on compact chunks (graph not needed)
  - L3/L4: graph BFS enumerates ALL entity nodes → retrieve their compact chunks
           → LLM has complete data to rank/filter/aggregate

Also implements Concat-All baseline (dump all compact chunks without any retrieval).
"""

from __future__ import annotations

import os
import re
import json
import networkx as nx
import numpy as np
import requests as _requests
from pathlib import Path

from .compact_chunks import CompactIndex, build_compact_index, _embed_text_sync, cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── LLM config ──────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT_EN = (
    "You are a precise QA assistant for statistical data. Answer based ONLY on "
    "the provided data context. When the question asks for a ranking, list entities "
    "in order. When it asks for a count or average, compute precisely from the data. "
    "Be concise. If the data is insufficient, say so."
)

SYSTEM_PROMPT_ZH = (
    "你是一个统计数据问答助手。只根据提供的数据上下文回答。"
    "如果问题要求排名，请按顺序列出。如果问题要求数量或均值，请精确计算。"
    "回答简洁明了。如果数据不足，请说明。"
)

# ── Graph paths ─────────────────────────────────────────────────────────────
GRAPH_PATHS = {
    "who": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
    "who_full": "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_cm": "output/wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    "inpatient": "output/inpatient_2023/lightrag_storage/graph_chunk_entity_relation.graphml",
}

BASELINE_GRAPH_PATHS = {
    "who": "output/baseline_who_life/lightrag_storage/graph_chunk_entity_relation.graphml",
    "wb_cm": "output/baseline_wb_child_mortality/lightrag_storage/graph_chunk_entity_relation.graphml",
    "inpatient": "output/baseline_inpatient23/lightrag_storage/graph_chunk_entity_relation.graphml",
}

# ── Entity type patterns for graph enumeration ──────────────────────────────
# WHO/WB_CM: country codes are 2-3 uppercase letters matching gold codes
WHO_GOLD_CODES = {
    "ARG", "AUS", "BGD", "BRA", "CAN", "CHN", "DEU", "EGY", "ESP",
    "FRA", "GBR", "IDN", "IND", "ITA", "JPN", "KOR", "MEX", "NGA",
    "PAK", "RUS", "SAU", "THA", "TUR", "USA", "ZAF",
}

# Inpatient: disease names in Chinese from gold standard
INPATIENT_GOLD_ENTITIES = {
    "肺炎", "肾衰竭", "乳房恶性肿瘤", "气管、支气管和肺恶性肿瘤",
    "其他缺血性心脏病", "呼吸道结核病", "胃炎和十二指肠炎", "白内障和晶状体的其他疾患",
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(context: str, question: str, language: str = "en", max_retries: int = 3) -> str:
    """Call Claude Haiku via wolfai proxy with retry."""
    import time
    system = SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN
    for attempt in range(max_retries):
        try:
            resp = _requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Data context:\n{context}\n\nQuestion: {question}"},
                    ],
                    "max_tokens": 512,
                    "temperature": 0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return f"[LLM Error: {e}]"


# ---------------------------------------------------------------------------
# Graph entity enumeration
# ---------------------------------------------------------------------------

def enumerate_entities_from_graph(
    dataset: str, graph_paths: dict | None = None
) -> list[str]:
    """
    BFS-enumerate all entity nodes from a graph.
    Returns list of entity identifiers (country codes for Type-II, disease names for Type-III).

    Args:
        dataset: Dataset key (e.g. "who", "wb_cm", "inpatient").
        graph_paths: Graph path mapping to use. Defaults to SGE GRAPH_PATHS.
    """
    if graph_paths is None:
        graph_paths = GRAPH_PATHS
    graph_rel_path = graph_paths.get(dataset)
    if not graph_rel_path:
        return []

    graph_path = PROJECT_ROOT / graph_rel_path
    if not graph_path.exists():
        print(f"  [WARN] Graph not found: {graph_path}")
        return []

    G = nx.read_graphml(str(graph_path))

    # Build node_id → entity_name mapping
    id_to_name = {}
    for node_id, ndata in G.nodes(data=True):
        name = ndata.get("entity_name") or ndata.get("name") or node_id
        id_to_name[node_id] = str(name).strip()

    if dataset == "who_full":
        # Full-scale: enumerate ALL 3-letter uppercase nodes from graph
        # that match any code in the full gold standard
        from .compact_chunks import DATASET_CONFIGS
        import re
        full_gold_path = DATASET_CONFIGS["who_full"]["gold_path"]
        full_codes = set()
        with open(full_gold_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    full_codes.add(json.loads(line)["triple"]["subject"])
        found = []
        for nid, name in id_to_name.items():
            name_upper = name.upper().strip()
            if name_upper in full_codes:
                found.append(name_upper)
        return sorted(set(found))

    if dataset in ("who", "wb_cm", "wb_pop", "wb_mat"):
        # Type-II: find country code nodes
        found = []
        for nid, name in id_to_name.items():
            name_upper = name.upper().strip()
            if name_upper in WHO_GOLD_CODES:
                found.append(name_upper)
        return sorted(set(found))

    elif dataset == "inpatient":
        # Type-III: find disease name nodes
        found = []
        for nid, name in id_to_name.items():
            for gold_entity in INPATIENT_GOLD_ENTITIES:
                if gold_entity in name or name in gold_entity:
                    found.append(gold_entity)
                    break
        return sorted(set(found))

    return []


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def retrieve_ggcr(
    question: dict, compact_index: CompactIndex, dataset: str
) -> str:
    """
    Graph-Guided Compact Retrieval.
    L1/L2: vector retrieval (top-5 compact chunks).
    L3/L4: graph enumeration → entity lookup → all matching chunks.
    """
    level = question["level"]
    query_text = question["question"]

    if level in ("L1", "L2"):
        chunks = compact_index.vector_retrieve(query_text, top_k=5)
    else:
        # L3/L4: enumerate ALL entities from graph, retrieve ALL compact chunks
        entity_keys = enumerate_entities_from_graph(dataset)
        chunks = compact_index.entity_retrieve(entity_keys)
        if not chunks:
            # Fallback to all chunks if entity enumeration fails
            chunks = compact_index.get_all_chunks()

    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


def retrieve_baseline_ggcr(
    question: dict, compact_index: CompactIndex, dataset: str
) -> str:
    """
    Baseline+GGCR: same retrieval logic as GGCR but entity enumeration from the
    Baseline graph instead of the SGE graph.

    L1/L2: identical vector retrieval (no graph used — results match sge_ggcr).
    L3/L4: entity enumeration via BASELINE_GRAPH_PATHS → compact chunk lookup.
    """
    level = question["level"]
    query_text = question["question"]

    if level in ("L1", "L2"):
        chunks = compact_index.vector_retrieve(query_text, top_k=5)
    else:
        # L3/L4: enumerate entities from the Baseline graph
        entity_keys = enumerate_entities_from_graph(dataset, graph_paths=BASELINE_GRAPH_PATHS)
        chunks = compact_index.entity_retrieve(entity_keys)
        if not chunks:
            # Fallback to all chunks if baseline entity enumeration fails
            chunks = compact_index.get_all_chunks()

    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


def retrieve_pure_compact(
    question: dict, compact_index: CompactIndex
) -> str:
    """
    Pure Compact baseline: vector retrieval only, no graph.
    Always retrieves top-5 by cosine similarity.
    """
    query_text = question["question"]
    chunks = compact_index.vector_retrieve(query_text, top_k=5)
    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


def retrieve_concat_all(
    question: dict, compact_index: CompactIndex
) -> str:
    """
    Concat-All baseline: dump ALL compact chunks as context.
    No retrieval — tests whether LLM can process complete data without graph guidance.
    """
    query_text = question["question"]
    chunks = compact_index.get_all_chunks()
    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


def retrieve_naive_rag(
    question: dict, naive_chunks: list[str], naive_embeddings: np.ndarray
) -> str:
    """
    Naive RAG baseline: vector retrieval on naive (row-by-row) serialization.
    """
    query_text = question["question"]
    query_emb = _embed_text_sync(query_text)
    scores = cosine_similarity(query_emb, naive_embeddings)
    top_indices = np.argsort(scores)[::-1][:5]
    chunks = [naive_chunks[i] for i in top_indices]
    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


# ── Oracle entity lists for oracle_ggcr ───────────────────────────────────
ORACLE_ENTITIES = {
    "who": sorted(WHO_GOLD_CODES),
    "wb_cm": sorted(WHO_GOLD_CODES),  # Same 25 countries
    "inpatient": sorted(INPATIENT_GOLD_ENTITIES),
}


def retrieve_oracle_ggcr(
    question: dict, compact_index: CompactIndex, dataset: str
) -> str:
    """
    Oracle+GGCR: perfect entity enumeration from Gold Standard lists.

    L1/L2: identical vector retrieval (same as sge_ggcr / baseline_ggcr).
    L3/L4: use oracle entity list → compact chunk lookup.

    This control isolates the entity coverage (EC) variable: if Oracle ≈ SGE,
    then the GGCR gap between SGE and Baseline is fully attributable to EC.
    """
    level = question["level"]
    query_text = question["question"]

    if level in ("L1", "L2"):
        chunks = compact_index.vector_retrieve(query_text, top_k=5)
    else:
        entity_keys = ORACLE_ENTITIES.get(dataset, [])
        chunks = compact_index.entity_retrieve(entity_keys)
        if not chunks:
            chunks = compact_index.get_all_chunks()

    context = "\n---\n".join(chunks)
    return _call_llm(context, query_text, question.get("language", "en"))


# ---------------------------------------------------------------------------
# Graph-Native (deterministic, no LLM)
# ---------------------------------------------------------------------------

def retrieve_graph_native(
    question: dict, dataset: str
) -> str:
    """
    Graph-native: pure graph traversal + deterministic computation.
    Only works for L3/L4 (cross-entity). Returns computed answer string.
    For L1/L2, falls back to graph value extraction.
    """
    graph_rel_path = GRAPH_PATHS.get(dataset)
    if not graph_rel_path:
        return "[No graph]"

    graph_path = PROJECT_ROOT / graph_rel_path
    if not graph_path.exists():
        return "[Graph not found]"

    # Import the extraction logic from the existing probe
    # We reuse the pattern but adapt for generic queries
    level = question["level"]
    metadata = question.get("metadata", {})

    if level == "L1":
        return _graph_native_l1(graph_path, metadata, dataset)
    elif level == "L2":
        return _graph_native_l2(graph_path, metadata, dataset)
    elif level == "L3":
        return _graph_native_l3(graph_path, metadata, dataset)
    elif level == "L4":
        return _graph_native_l4(graph_path, metadata, dataset, question)
    return "[Unsupported level]"


def _load_graph_values(graph_path: Path, dataset: str) -> dict:
    """Extract entity values from graph. Returns {entity_key: {dim: value}}."""
    G = nx.read_graphml(str(graph_path))
    id_to_name = {}
    for nid, ndata in G.nodes(data=True):
        name = ndata.get("entity_name") or ndata.get("name") or nid
        id_to_name[nid] = str(name).strip()

    if dataset in ("who", "wb_cm"):
        return _extract_type_ii_values(G, id_to_name, dataset)
    elif dataset == "inpatient":
        return _extract_type_iii_values(G, id_to_name)
    return {}


def _iter_edges(G, nid):
    """Iterate edges from a node, handling both directed and undirected graphs."""
    if G.is_directed():
        for _, target, edata in G.out_edges(nid, data=True):
            yield target, edata
        for source, _, edata in G.in_edges(nid, data=True):
            yield source, edata
    else:
        for neighbor in G.neighbors(nid):
            edata = G.edges[nid, neighbor]
            yield neighbor, edata


def _extract_type_ii_values(G, id_to_name, dataset):
    """Extract {country_code: {year: value}} from Type-II graph."""
    country_values = {}
    for nid, name in id_to_name.items():
        name_upper = name.upper().strip()
        if name_upper not in WHO_GOLD_CODES:
            continue
        code = name_upper
        if code not in country_values:
            country_values[code] = {}

        # Collect text from all edges
        for target, edata in _iter_edges(G, nid):
            kw = str(edata.get("keywords", ""))
            desc = str(edata.get("description", ""))
            target_name = id_to_name.get(target, target)

            # Pattern: year in keywords, value as target node name
            year_match = re.search(r"year[:\s]*(\d{4})", kw)
            if year_match:
                try:
                    val = float(target_name)
                    year = year_match.group(1)
                    if year not in country_values[code]:
                        country_values[code][year] = val
                except ValueError:
                    pass

            # Pattern: year-value pairs in description
            yv_pattern = re.compile(r"(20\d{2})\D{0,10}?(\d{2,3}\.?\d*)")
            for m in yv_pattern.finditer(f"{kw} {desc} {target_name}"):
                year, val_str = m.group(1), m.group(2)
                try:
                    val = float(val_str)
                    if year not in country_values[code]:
                        country_values[code][year] = val
                except ValueError:
                    pass

        # Also check node description for compact-style data
        node_desc = str(G.nodes[nid].get("description", ""))
        for m in re.finditer(r"(20\d{2})\s*[=:]\s*(\d+\.?\d*)", node_desc):
            year, val_str = m.group(1), m.group(2)
            try:
                val = float(val_str)
                if year not in country_values[code]:
                    country_values[code][year] = val
            except ValueError:
                pass

    return country_values


def _extract_type_iii_values(G, id_to_name):
    """Extract {disease: {relation: value}} from Type-III graph."""
    disease_values = {}
    for nid, name in id_to_name.items():
        matched_entity = None
        for gold in INPATIENT_GOLD_ENTITIES:
            if gold in name or name in gold:
                matched_entity = gold
                break
        if not matched_entity:
            continue
        if matched_entity not in disease_values:
            disease_values[matched_entity] = {}

        # Check edges for numeric values
        for target, edata in _iter_edges(G, nid):
            kw = str(edata.get("keywords", ""))
            desc = str(edata.get("description", ""))
            target_name = id_to_name.get(target, target)

            try:
                val = float(target_name.replace(",", ""))
                # Determine which relation based on keywords/description
                combined = f"{kw} {desc}".lower()
                if "死亡" in combined or "death" in combined:
                    disease_values[matched_entity]["REGISTERED_DEATHS"] = val
                elif "总" in combined or "total" in combined:
                    disease_values[matched_entity]["INPATIENT_TOTAL"] = val
                elif "管理局" in combined or "ha" in combined:
                    disease_values[matched_entity]["INPATIENT_HA_HOSPITAL"] = val
                else:
                    # Default to total if unclassified
                    if "INPATIENT_TOTAL" not in disease_values[matched_entity]:
                        disease_values[matched_entity]["INPATIENT_TOTAL"] = val
            except ValueError:
                pass

    return disease_values


def _graph_native_l1(graph_path, metadata, dataset):
    """L1: extract single value from graph."""
    values = _load_graph_values(graph_path, dataset)
    entity = metadata.get("entity", "")
    if dataset in ("who", "wb_cm"):
        year = metadata.get("year", "")
        entity_data = values.get(entity, {})
        val = entity_data.get(year)
        return str(val) if val is not None else "[Not found]"
    else:
        rel = metadata.get("relation", "")
        entity_data = values.get(entity, {})
        val = entity_data.get(rel)
        return str(int(val)) if val is not None else "[Not found]"


def _graph_native_l2(graph_path, metadata, dataset):
    """L2: extract trend from graph."""
    values = _load_graph_values(graph_path, dataset)
    entity = metadata.get("entity", "")
    entity_data = values.get(entity, {})

    if dataset in ("who", "wb_cm"):
        all_values = metadata.get("all_values", {})
        years = sorted(all_values.keys())
        graph_vals = [entity_data.get(y) for y in years]
        if any(v is None for v in graph_vals):
            return "No (insufficient data in graph)"
        is_inc = all(graph_vals[i] <= graph_vals[i + 1] for i in range(len(graph_vals) - 1))
        is_dec = all(graph_vals[i] >= graph_vals[i + 1] for i in range(len(graph_vals) - 1))
        direction = metadata.get("direction", "")
        if "increase" in direction and is_inc:
            return "Yes"
        if "decrease" in direction and is_dec:
            return "Yes"
        return "No"
    else:
        # Inpatient L2: ratio check
        total = entity_data.get("INPATIENT_TOTAL")
        deaths = entity_data.get("REGISTERED_DEATHS")
        if total and deaths:
            ratio = deaths / total * 100
            return "是" if ratio > 10 else "否"
        return "[Not found]"


def _graph_native_l3(graph_path, metadata, dataset):
    """L3: cross-entity ranking from graph."""
    values = _load_graph_values(graph_path, dataset)
    k = metadata.get("k", 5)

    if dataset in ("who", "wb_cm"):
        year = metadata.get("year", "")
        ranked = sorted(
            [(code, vals.get(year, 0)) for code, vals in values.items() if vals.get(year) is not None],
            key=lambda x: x[1], reverse=True,
        )
        return ", ".join(code for code, _ in ranked[:k])
    else:
        rel = metadata.get("relation", "INPATIENT_TOTAL")
        ranked = sorted(
            [(ent, vals.get(rel, 0)) for ent, vals in values.items() if vals.get(rel) is not None],
            key=lambda x: x[1],
            reverse=("ascending" not in str(metadata.get("direction", ""))),
        )
        return "、".join(ent for ent, _ in ranked[:k])


def _graph_native_l4(graph_path, metadata, dataset, question):
    """L4: cross-entity aggregation from graph."""
    values = _load_graph_values(graph_path, dataset)
    category = question.get("category", "")

    if dataset in ("who", "wb_cm"):
        year = metadata.get("year", "")
        vals_in_year = [v.get(year) for v in values.values() if v.get(year) is not None]

        if "count" in category:
            threshold = metadata.get("threshold", 0)
            return str(sum(1 for v in vals_in_year if v > threshold))
        elif "average" in category:
            return str(round(sum(vals_in_year) / len(vals_in_year), 2)) if vals_in_year else "[No data]"
        elif "max_change" in category:
            y1, y2 = metadata.get("year_from", ""), metadata.get("year_to", "")
            changes = []
            for code, yvals in values.items():
                if y1 in yvals and y2 in yvals:
                    changes.append((code, yvals[y2] - yvals[y1]))
            if changes:
                changes.sort(key=lambda x: x[1], reverse=True)
                return changes[0][0]
            return "[No data]"
        elif "range" in category:
            return str(round(max(vals_in_year) - min(vals_in_year), 2)) if vals_in_year else "[No data]"
    else:
        rel = metadata.get("relation", "INPATIENT_TOTAL")
        if "sum" in category:
            total = sum(v.get(rel, 0) for v in values.values() if v.get(rel) is not None)
            return str(int(total))
        elif "average" in category:
            vals = [v.get(rel) for v in values.values() if v.get(rel) is not None]
            return str(round(sum(vals) / len(vals), 1)) if vals else "[No data]"
        elif "derived" in category:
            # Mortality rate: deaths / total
            rates = []
            for ent, v in values.items():
                if "INPATIENT_TOTAL" in v and "REGISTERED_DEATHS" in v:
                    rate = v["REGISTERED_DEATHS"] / v["INPATIENT_TOTAL"] * 100
                    rates.append((ent, round(rate, 2)))
            if rates:
                rates.sort(key=lambda x: x[1], reverse=True)
                ref_answer = question.get("reference_answer", "")
                # Check if asking for highest or lowest
                if "最低" in question.get("question", ""):
                    return rates[-1][0]
                return rates[0][0]
            return "[No data]"
        elif "count" in category:
            threshold = metadata.get("threshold", 0)
            vals = [(v.get(rel, 0)) for v in values.values() if v.get(rel) is not None]
            return str(sum(1 for v in vals if v > threshold))

    return "[Unsupported query type]"
