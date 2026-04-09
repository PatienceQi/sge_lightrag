#!/usr/bin/env python3
"""
graph_context.py — Build structured context from a GraphML knowledge graph.

Given a parsed query and a graph path, performs:
  1. Entity matching: case-insensitive substring match against all node names
  2. BFS 2-hop traversal from matched nodes
  3. Collects edge descriptions and neighbor data
  4. Formats as structured context text suitable for LLM QA

Returns empty string if no entity matches (signals fallback to vector retrieval).

Usage:
    from experiments.graph_first.graph_context import build_graph_context
    ctx = build_graph_context("output/who_life_expectancy/...", {"entities": ["China"]}, 3000)
"""

from __future__ import annotations

import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Token estimation (rough: 1 token ≈ 4 chars) ──────────────────────────────
def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _get_node_name(G: nx.Graph, node_id: str) -> str:
    """Extract display name from a graph node."""
    ndata = G.nodes[node_id]
    return str(ndata.get("entity_name") or ndata.get("name") or node_id).strip()


def _get_node_description(G: nx.Graph, node_id: str) -> str:
    """Extract node description, truncated to 200 chars."""
    ndata = G.nodes[node_id]
    desc = str(ndata.get("description") or "").strip()
    return desc[:200] if desc else ""


def _get_edge_description(edata: dict) -> str:
    """Format edge metadata as a short string."""
    parts = []
    kw = str(edata.get("keywords") or "").strip()
    desc = str(edata.get("description") or "").strip()
    weight = edata.get("weight")
    if kw:
        parts.append(f"[{kw}]")
    if desc:
        parts.append(desc[:150])
    if weight is not None:
        parts.append(f"(weight={weight})")
    return " ".join(parts) if parts else "(no description)"


def _match_entities(
    G: nx.Graph, entity_queries: list[str]
) -> list[str]:
    """
    Case-insensitive substring match of entity_queries against all node names.
    Returns list of matched node IDs.
    """
    id_to_name = {
        node_id: _get_node_name(G, node_id)
        for node_id in G.nodes()
    }

    matched = []
    for query in entity_queries:
        q_lower = query.lower().strip()
        if not q_lower:
            continue
        for node_id, name in id_to_name.items():
            if q_lower in name.lower() or name.lower() in q_lower:
                if node_id not in matched:
                    matched.append(node_id)

    return matched


def _bfs_2hop(G: nx.Graph, seed_nodes: list[str]) -> dict[str, int]:
    """
    BFS up to 2 hops from seed nodes.
    Returns {node_id: hop_distance} for all reachable nodes within 2 hops.
    """
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for seed in seed_nodes:
        if seed in G:
            queue.append((seed, 0))
            visited[seed] = 0

    while queue:
        node_id, depth = queue.popleft()
        if depth >= 2:
            continue
        neighbors = (
            list(G.successors(node_id)) + list(G.predecessors(node_id))
            if G.is_directed()
            else list(G.neighbors(node_id))
        )
        for neighbor in neighbors:
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return visited


def _iter_edges_between(G: nx.Graph, u: str, v: str):
    """Yield edge data between u and v, handling directed and undirected."""
    if G.is_directed():
        if G.has_edge(u, v):
            yield G.edges[u, v]
        if G.has_edge(v, u):
            yield G.edges[v, u]
    else:
        if G.has_edge(u, v):
            yield G.edges[u, v]


def _format_context(
    G: nx.Graph,
    seed_nodes: list[str],
    hop_map: dict[str, int],
    max_tokens: int,
) -> str:
    """
    Format graph neighborhood as structured context text.
    Prioritizes seed node data, then 1-hop edges, then 2-hop neighbors.
    """
    lines: list[str] = []
    total_tokens = 0

    def _add(text: str) -> bool:
        nonlocal total_tokens
        t = _estimate_tokens(text)
        if total_tokens + t > max_tokens:
            return False
        lines.append(text)
        total_tokens += t
        return True

    # Section 1: Seed node descriptions
    _add("=== Matched Entities ===")
    for node_id in seed_nodes:
        name = _get_node_name(G, node_id)
        desc = _get_node_description(G, node_id)
        entry = f"Entity: {name}"
        if desc:
            entry += f"\n  Description: {desc}"
        if not _add(entry):
            break

    # Section 2: Direct edges from seed nodes (1-hop)
    _add("\n=== Direct Relationships ===")
    seed_set = set(seed_nodes)
    for seed in seed_nodes:
        if G.is_directed():
            edge_iter = list(G.out_edges(seed, data=True)) + list(G.in_edges(seed, data=True))
        else:
            edge_iter = list(G.edges(seed, data=True))

        for u, v, edata in edge_iter:
            other = v if u == seed else u
            other_name = _get_node_name(G, other)
            seed_name = _get_node_name(G, seed)
            edge_desc = _get_edge_description(edata)
            entry = f"  {seed_name} → {other_name}: {edge_desc}"
            if not _add(entry):
                break

    # Section 3: 2-hop neighbors (non-seed nodes at distance 1 and 2)
    _add("\n=== Neighborhood Nodes ===")
    for node_id, depth in sorted(hop_map.items(), key=lambda x: x[1]):
        if node_id in seed_set:
            continue
        name = _get_node_name(G, node_id)
        desc = _get_node_description(G, node_id)
        entry = f"  [hop={depth}] {name}"
        if desc:
            entry += f": {desc[:100]}"
        if not _add(entry):
            break

    return "\n".join(lines)


def build_graph_context(
    graph_path: str,
    parsed_query: dict,
    max_tokens: int = 3000,
) -> str:
    """
    Build structured context text from a GraphML file for the given query.

    Args:
        graph_path: Absolute or relative path to the .graphml file.
        parsed_query: Output of parse_statistical_query() — dict with
            keys: entities, years, metric, query_type.
        max_tokens: Approximate token budget for context (1 token ≈ 4 chars).

    Returns:
        Structured context string for LLM consumption.
        Returns empty string if no entity match is found (fallback signal).
    """
    path = Path(graph_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return ""

    G = nx.read_graphml(str(path))
    if G.number_of_nodes() == 0:
        return ""

    entity_queries: list[str] = parsed_query.get("entities", [])
    if not entity_queries:
        return ""

    seed_nodes = _match_entities(G, entity_queries)
    if not seed_nodes:
        return ""

    hop_map = _bfs_2hop(G, seed_nodes)
    context = _format_context(G, seed_nodes, hop_map, max_tokens)
    return context


if __name__ == "__main__":
    # Quick smoke test
    graph = (
        PROJECT_ROOT
        / "output/who_life_expectancy/lightrag_storage/graph_chunk_entity_relation.graphml"
    )
    parsed = {"entities": ["CHN"], "years": [2020], "metric": "life expectancy", "query_type": "lookup"}
    ctx = build_graph_context(str(graph), parsed, max_tokens=2000)
    print(ctx[:1000] if ctx else "(no match)")
