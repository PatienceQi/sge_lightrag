"""
graph_loaders.py — Unified graph loading for evaluation.

Supports:
  - LightRAG GraphML format (standard path)
  - MS GraphRAG Parquet format (create_final_entities.parquet +
    create_final_relationships.parquet)
  - NetworkX DiGraph (passed directly)

Usage in evaluate_coverage.py:
    from evaluation.graph_loaders import load_graph_auto

    G, nodes, entity_text_2hop = load_graph_auto(path)

The returned (G, nodes, entity_text_2hop) triple has the same shape as
the original load_graph() in evaluate_coverage.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx")


# ---------------------------------------------------------------------------
# GraphML loader (existing LightRAG format)
# ---------------------------------------------------------------------------

def load_graphml(graphml_path: str):
    """Load a LightRAG GraphML file and build entity text index."""
    G = nx.read_graphml(graphml_path)

    nodes = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        name = str(name).strip()
        desc = str(data.get("description", ""))
        nodes[name] = {
            "type": data.get("entity_type", ""),
            "description": desc,
        }

    entity_text = {}
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("entity_name") or G.nodes[u].get("name") or u
        v_name = G.nodes[v].get("entity_name") or G.nodes[v].get("name") or v
        u_name = str(u_name).strip()
        v_name = str(v_name).strip()
        kw = str(data.get("keywords", ""))
        desc = str(data.get("description", ""))
        edge_text = f"{kw} {desc} {v_name}"
        entity_text.setdefault(u_name, []).append(edge_text)
        rev_text = f"{kw} {desc} {u_name}"
        entity_text.setdefault(v_name, []).append(rev_text)

    node_id_to_name = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or data.get("name") or node_id
        node_id_to_name[node_id] = str(name).strip()

    entity_text_2hop = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, nb_id)
            texts.extend(entity_text.get(nb_name, []))
            nb_desc = nodes.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
        entity_text_2hop[name] = texts

    return G, nodes, entity_text_2hop


# ---------------------------------------------------------------------------
# GraphRAG 3.x GraphML loader (output/graph.graphml)
# ---------------------------------------------------------------------------

def load_graphrag_graphml(graphml_path: str):
    """
    Load a GraphRAG 3.x graph.graphml file.

    GraphRAG 3.x uses node IDs as entity names (no entity_name attribute),
    so we build the name from the node ID itself.
    Edge data uses 'description' attributes.
    """
    G = nx.read_graphml(graphml_path)

    nodes = {}
    for node_id, data in G.nodes(data=True):
        # In GraphRAG 3.x, node_id IS the entity name (e.g., "93.8_MILLION_HKD")
        name = data.get("entity_name") or data.get("label") or str(node_id).strip()
        etype = data.get("entity_type") or data.get("type", "")
        desc = str(data.get("description", ""))
        nodes[name] = {"type": etype, "description": desc}

    entity_text: dict[str, list[str]] = {}
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("entity_name") or G.nodes[u].get("label") or str(u)
        v_name = G.nodes[v].get("entity_name") or G.nodes[v].get("label") or str(v)
        u_name = str(u_name).strip()
        v_name = str(v_name).strip()
        desc = str(data.get("description", ""))
        kw = str(data.get("keywords", ""))
        edge_text = f"{desc} {kw} {v_name}"
        entity_text.setdefault(u_name, []).append(edge_text)
        rev_text = f"{desc} {kw} {u_name}"
        entity_text.setdefault(v_name, []).append(rev_text)

    # Also add node name to entity_text (for value-in-name matching)
    for name in nodes:
        entity_text.setdefault(name, []).append(name)

    node_id_to_name = {
        node_id: (G.nodes[node_id].get("entity_name") or G.nodes[node_id].get("label") or str(node_id)).strip()
        for node_id in G.nodes()
    }

    entity_text_2hop: dict[str, list[str]] = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, str(nb_id))
            texts.extend(entity_text.get(nb_name, []))
            nb_desc = nodes.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
            # Also add neighbor name itself (value-in-name)
            texts.append(nb_name)
        entity_text_2hop[name] = texts

    return G, nodes, entity_text_2hop


# ---------------------------------------------------------------------------
# GraphRAG Parquet loader
# ---------------------------------------------------------------------------

def load_graphrag_parquet(root_dir: str):
    """
    Load MS GraphRAG Parquet output and convert to NetworkX graph.

    Expected Parquet files (under root_dir/output/artifacts/ or root_dir/output/):
      - create_final_entities.parquet
      - create_final_relationships.parquet

    Returns (G, nodes, entity_text_2hop) matching the LightRAG format.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet loading: pip install pandas pyarrow")

    root = Path(root_dir)

    # Find entities and relationships parquet files
    entities_path = _find_parquet(root, "create_final_entities")
    relations_path = _find_parquet(root, "create_final_relationships")

    if entities_path is None:
        raise FileNotFoundError(
            f"create_final_entities.parquet not found under {root}\n"
            f"Run 'graphrag index --root {root}' first."
        )

    ent_df = pd.read_parquet(entities_path)
    ent_df.columns = [c.strip() for c in ent_df.columns]

    G = nx.DiGraph()
    nodes = {}

    # Build entity nodes
    # Typical columns: id, name, type, description, graph_embedding, text_unit_ids
    for _, row in ent_df.iterrows():
        eid = str(row.get("id", "")).strip()
        name = str(row.get("name", eid)).strip()
        etype = str(row.get("type", "")).strip()
        desc = str(row.get("description", "")).strip()

        if not name or name.lower() in ("", "nan", "none"):
            continue

        G.add_node(eid, entity_name=name, entity_type=etype, description=desc)
        nodes[name] = {"type": etype, "description": desc, "_id": eid}

    # Build id→name lookup
    id_to_name: dict[str, str] = {
        str(row.get("id", "")): str(row.get("name", ""))
        for _, row in ent_df.iterrows()
    }

    entity_text: dict[str, list[str]] = {}

    if relations_path is not None:
        rel_df = pd.read_parquet(relations_path)
        rel_df.columns = [c.strip() for c in rel_df.columns]

        # Typical columns: id, source, target, description, text_unit_ids
        for _, row in rel_df.iterrows():
            src_id = str(row.get("source", "")).strip()
            tgt_id = str(row.get("target", "")).strip()
            rdesc = str(row.get("description", "")).strip()
            rweight = str(row.get("weight", "")).strip()

            src_name = id_to_name.get(src_id, src_id)
            tgt_name = id_to_name.get(tgt_id, tgt_id)

            G.add_edge(src_id, tgt_id, keywords=rdesc, description=rdesc)

            edge_text = f"{rdesc} {tgt_name} {rweight}"
            entity_text.setdefault(src_name, []).append(edge_text)
            rev_text = f"{rdesc} {src_name} {rweight}"
            entity_text.setdefault(tgt_name, []).append(rev_text)

    # Also index entity descriptions into entity_text for fact search
    for name, data in nodes.items():
        desc = data.get("description", "")
        if desc:
            entity_text.setdefault(name, []).append(desc)

    # Build 2-hop index
    node_id_to_name = {
        node_id: G.nodes[node_id].get("entity_name", node_id)
        for node_id in G.nodes()
    }

    entity_text_2hop: dict[str, list[str]] = {}
    for node_id in G.nodes():
        name = node_id_to_name[node_id]
        texts = list(entity_text.get(name, []))
        for nb_id in G.neighbors(node_id):
            nb_name = node_id_to_name.get(nb_id, nb_id)
            texts.extend(entity_text.get(nb_name, []))
            nb_desc = nodes.get(nb_name, {}).get("description", "")
            if nb_desc:
                texts.append(nb_desc)
        entity_text_2hop[name] = texts

    return G, nodes, entity_text_2hop


def _find_parquet(root: Path, stem: str) -> Path | None:
    """Find a parquet file by stem name anywhere under root."""
    for pattern in [
        f"output/artifacts/{stem}.parquet",
        f"output/{stem}.parquet",
        f"artifacts/{stem}.parquet",
        f"{stem}.parquet",
    ]:
        candidate = root / pattern
        if candidate.exists():
            return candidate
    # Recursive search as fallback
    matches = list(root.rglob(f"{stem}.parquet"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Auto-detect loader
# ---------------------------------------------------------------------------

def load_graph_auto(path: str):
    """
    Auto-detect graph format and load accordingly.

    - If path ends with .graphml → GraphML (LightRAG)
    - If path ends with .parquet → GraphRAG Parquet (root dir auto-detected)
    - If path is a directory containing a *.graphml file → GraphML
    - If path is a directory with output/artifacts/*.parquet → GraphRAG Parquet
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".graphml":
        return load_graphml(str(p))

    if p.is_file() and p.suffix == ".parquet":
        # Assume root is 2 levels up (output/artifacts/)
        return load_graphrag_parquet(str(p.parent.parent.parent))

    if p.is_dir():
        # Check for GraphRAG 3.x output/graph.graphml
        graphrag_graphml = p / "output" / "graph.graphml"
        if graphrag_graphml.exists():
            return load_graphrag_graphml(str(graphrag_graphml))

        # Check for GraphRAG Parquet artifacts (older versions)
        artifacts = list(p.rglob("create_final_entities.parquet"))
        if artifacts:
            return load_graphrag_parquet(str(p))

        # Check for GraphML
        graphmls = list(p.rglob("*.graphml"))
        if graphmls:
            # Prefer non-GraphRAG graphmls (LightRAG)
            lightrag = [g for g in graphmls if "graph_chunk_entity" in g.name]
            return load_graphml(str(lightrag[0] if lightrag else graphmls[0]))

    raise ValueError(
        f"Cannot detect graph format for: {path}\n"
        "Expected a .graphml file or a GraphRAG root directory with Parquet artifacts."
    )
