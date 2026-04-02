"""
integrator.py — LightRAG integration layer for Stage 3.

Provides two public functions:

  patch_lightrag(schema) -> dict
      Prepares the PROMPTS override and returns a context dict that can be
      merged into LightRAG's context_base. Does NOT import or run LightRAG.

  prepare_chunks(csv_path, schema) -> list[str]
      Serializes a CSV file into text chunks using the extraction schema.

NOTE: LightRAG is not installed in this environment. This module prepares
everything LightRAG would need without actually importing it.
"""

from __future__ import annotations

import json
from typing import Optional

from .serializer import serialize_csv
from .prompt_injector import (
    generate_system_prompt,
    generate_user_prompt_template,
    TUPLE_DELIMITER,
    COMPLETION_DELIMITER,
)
from .compact_representation import should_use_compact, build_compact_system_prompt


def patch_lightrag(schema: dict, language: str = "Chinese") -> dict:
    """
    Prepare the LightRAG injection payload for a given extraction schema.

    Returns a dict containing:
      - "system_prompt"     : the overridden entity_extraction_system_prompt
      - "context_base_extra": extra fields to merge into context_base
      - "addon_params"      : dict to pass as LightRAG(addon_params=...)
      - "entity_types"      : list of entity type strings

    Usage (when LightRAG is available):
        from lightrag.prompt import PROMPTS
        payload = patch_lightrag(schema)
        PROMPTS["entity_extraction_system_prompt"] = payload["system_prompt"]
        rag = LightRAG(
            working_dir="./rag_storage",
            addon_params=payload["addon_params"],
        )

    Parameters
    ----------
    schema   : Stage 2 extraction schema dict
    language : output language for the LLM (default: "Chinese")

    Returns
    -------
    dict with keys: system_prompt, context_base_extra, addon_params, entity_types
    """
    # Adaptive mode: small/simple tables use baseline (no schema injection)
    if schema.get("use_baseline_mode", False):
        return {
            "system_prompt": None,          # None → Stage 3 keeps LightRAG default
            "context_base_extra": {},
            "addon_params": {"language": language},
            "entity_types": [],
            "use_baseline_mode": True,
            "adaptive_reason": schema.get("adaptive_reason", ""),
        }

    # Compact mode: large Type-II tables use a compact timeseries system prompt
    # that instructs LightRAG to create ONE entity node per country and embed
    # all year-value data in the description (prevents StatValue node explosion).
    n_rows = schema.get("_n_rows", 0)
    if should_use_compact(schema, n_rows):
        compact_prompt = build_compact_system_prompt(schema, language=language)
        entity_types_compact = [t for t in schema.get("entity_types", ["Country_Code"])
                                 if t != "StatValue"]
        if not entity_types_compact:
            entity_types_compact = ["Country_Code"]
        return {
            "system_prompt": compact_prompt,
            "context_base_extra": {},
            "addon_params": {
                "language": language,
                "entity_types": entity_types_compact,
            },
            "entity_types": entity_types_compact,
            "use_baseline_mode": False,
            "use_compact_mode": True,
        }

    entity_types = schema.get("entity_types", ["Entity"])
    schema_json_str = json.dumps(schema, ensure_ascii=False)

    system_prompt = generate_system_prompt(schema, language=language)

    # context_base_extra: fields to add to operate.py's context_base dict
    context_base_extra = {
        "schema_json": schema_json_str,
    }

    # addon_params: passed to LightRAG constructor
    addon_params = {
        "language": language,
        "entity_types": entity_types,
        "schema_json": schema_json_str,
    }

    return {
        "system_prompt": system_prompt,
        "context_base_extra": context_base_extra,
        "addon_params": addon_params,
        "entity_types": entity_types,
        "use_baseline_mode": False,
    }


def check_graph_degradation(
    graphml_path: str,
    edge_node_threshold: float = 0.90,
    isolated_threshold: float = 0.20,
) -> dict:
    """
    Post-extraction degradation check on the output graph.

    Returns a dict with:
      - "degraded" (bool): True if graph shows signs of SGE over-constraint
      - "edge_node_ratio": edges / nodes
      - "isolated_ratio": isolated nodes / total nodes
      - "reason": human-readable reason if degraded

    When degraded is True, the caller should re-run ingestion with
    baseline mode (no schema injection) to ensure worst-case >= baseline.
    """
    try:
        import networkx as nx
    except ImportError:
        return {"degraded": False, "reason": "networkx not available"}

    from pathlib import Path
    if not Path(graphml_path).exists():
        return {"degraded": False, "reason": "graph file not found"}

    G = nx.read_graphml(graphml_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "degraded": True,
            "edge_node_ratio": 0.0,
            "isolated_ratio": 1.0,
            "reason": "empty graph (0 nodes)",
        }

    edge_node_ratio = n_edges / n_nodes
    n_isolated = len(list(nx.isolates(G)))
    isolated_ratio = n_isolated / n_nodes

    degraded = (
        edge_node_ratio < edge_node_threshold
        or isolated_ratio > isolated_threshold
    )

    return {
        "degraded": degraded,
        "edge_node_ratio": round(edge_node_ratio, 4),
        "isolated_ratio": round(isolated_ratio, 4),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_isolated": n_isolated,
        "reason": (
            f"edge/node={edge_node_ratio:.3f}<{edge_node_threshold} "
            f"or isolated={isolated_ratio:.3f}>{isolated_threshold}"
            if degraded else "OK"
        ),
    }


def prepare_chunks(csv_path: str, schema: dict) -> list[str]:
    """
    Serialize a CSV file into text chunks ready for LightRAG ingestion.

    Parameters
    ----------
    csv_path : path to the CSV file
    schema   : Stage 2 extraction schema dict

    Returns
    -------
    list[str] — text chunks, one per entity row (or batch for Type I)
    """
    return serialize_csv(csv_path, schema)
