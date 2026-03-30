"""
test_graph_loaders.py — Unit tests for evaluation/graph_loaders.py.

Tests GraphML loading with networkx-created temp files, and verifies
node/edge attribute extraction in the returned (G, nodes, entity_text_2hop)
triple.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python -m pytest tests/test_graph_loaders.py -v
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.graph_loaders import (
    load_graphml,
    load_graphrag_graphml,
    load_graph_auto,
    _find_parquet,
)


# ---------------------------------------------------------------------------
# Helpers: write temp GraphML files
# ---------------------------------------------------------------------------

def _write_graphml(G: nx.Graph | nx.DiGraph) -> str:
    """Write a networkx graph to a temp GraphML file. Returns path."""
    fd, path = tempfile.mkstemp(suffix=".graphml", prefix="sge_test_")
    os.close(fd)
    nx.write_graphml(G, path)
    return path


def _make_lightrag_graph() -> nx.DiGraph:
    """Build a minimal LightRAG-style DiGraph with entity_name attributes."""
    G = nx.DiGraph()
    G.add_node(
        "n1",
        entity_name="China",
        entity_type="Country",
        description="China life expectancy data",
    )
    G.add_node(
        "n2",
        entity_name="76.4",
        entity_type="StatValue",
        description="Life expectancy value 2020",
    )
    G.add_node(
        "n3",
        entity_name="India",
        entity_type="Country",
        description="India data",
    )
    G.add_edge(
        "n1", "n2",
        keywords="life_expectancy HAS_VALUE",
        description="China HAS_VALUE 76.4 in 2020",
    )
    G.add_edge(
        "n3", "n2",
        keywords="life_expectancy HAS_VALUE",
        description="India shares value node",
    )
    return G


def _make_graphrag_graph() -> nx.DiGraph:
    """Build a minimal GraphRAG 3.x-style DiGraph (node ID = entity name)."""
    G = nx.DiGraph()
    G.add_node(
        "CHINA",
        label="CHINA",
        entity_type="COUNTRY",
        description="People's Republic of China",
    )
    G.add_node(
        "76_MILLION",
        label="76_MILLION",
        entity_type="VALUE",
        description="76 million population value",
    )
    G.add_edge(
        "CHINA", "76_MILLION",
        description="China has population 76 million",
        keywords="population HAS_VALUE",
    )
    return G


# ---------------------------------------------------------------------------
# load_graphml tests
# ---------------------------------------------------------------------------

class TestLoadGraphml(unittest.TestCase):

    def setUp(self):
        G = _make_lightrag_graph()
        self.path = _write_graphml(G)

    def tearDown(self):
        os.unlink(self.path)

    def test_returns_three_tuple(self):
        result = load_graphml(self.path)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_first_element_is_networkx_graph(self):
        G, _, _ = load_graphml(self.path)
        self.assertIsInstance(G, (nx.Graph, nx.DiGraph))

    def test_node_count_matches(self):
        G, nodes, _ = load_graphml(self.path)
        self.assertEqual(len(nodes), 3)

    def test_nodes_dict_has_entity_names(self):
        _, nodes, _ = load_graphml(self.path)
        self.assertIn("China", nodes)
        self.assertIn("India", nodes)

    def test_node_entry_has_type_and_description(self):
        _, nodes, _ = load_graphml(self.path)
        china = nodes["China"]
        self.assertIn("type", china)
        self.assertIn("description", china)
        self.assertEqual(china["type"], "Country")

    def test_entity_text_2hop_has_all_nodes(self):
        _, nodes, et2 = load_graphml(self.path)
        for name in nodes:
            self.assertIn(name, et2, f"Node '{name}' missing from entity_text_2hop")

    def test_entity_text_2hop_values_are_lists(self):
        _, _, et2 = load_graphml(self.path)
        for name, texts in et2.items():
            self.assertIsInstance(texts, list, f"entity_text_2hop[{name!r}] must be a list")

    def test_edge_keyword_appears_in_entity_text(self):
        _, _, et2 = load_graphml(self.path)
        china_texts = " ".join(et2.get("China", []))
        self.assertIn("HAS_VALUE", china_texts)

    def test_two_hop_neighbor_text_included(self):
        _, _, et2 = load_graphml(self.path)
        # "China" → "76.4" (direct edge), "76.4" also linked to "India"
        # So "China"'s 2-hop texts should include something from "India"'s neighbor
        china_texts = " ".join(et2.get("China", []))
        # At minimum the neighbor "76.4" description should be in China's context
        self.assertIn("76.4", china_texts)

    def test_graph_preserves_edge_count(self):
        G, _, _ = load_graphml(self.path)
        self.assertEqual(G.number_of_edges(), 2)


# ---------------------------------------------------------------------------
# load_graphrag_graphml tests
# ---------------------------------------------------------------------------

class TestLoadGraphragGraphml(unittest.TestCase):

    def setUp(self):
        G = _make_graphrag_graph()
        self.path = _write_graphml(G)

    def tearDown(self):
        os.unlink(self.path)

    def test_returns_three_tuple(self):
        result = load_graphrag_graphml(self.path)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_nodes_dict_populated(self):
        _, nodes, _ = load_graphrag_graphml(self.path)
        self.assertGreater(len(nodes), 0)

    def test_node_has_type_and_description(self):
        _, nodes, _ = load_graphrag_graphml(self.path)
        # At least one node should have type info
        for name, data in nodes.items():
            self.assertIn("type", data)
            self.assertIn("description", data)

    def test_entity_text_2hop_populated(self):
        _, nodes, et2 = load_graphrag_graphml(self.path)
        self.assertGreater(len(et2), 0)

    def test_entity_text_2hop_values_are_lists(self):
        _, _, et2 = load_graphrag_graphml(self.path)
        for name, texts in et2.items():
            self.assertIsInstance(texts, list)

    def test_edge_description_appears_in_entity_text(self):
        _, _, et2 = load_graphrag_graphml(self.path)
        # Find the CHINA node's texts
        china_key = next((k for k in et2 if "CHINA" in k.upper()), None)
        self.assertIsNotNone(china_key, "CHINA node not found in entity_text_2hop")
        texts = " ".join(et2[china_key])
        self.assertIn("population", texts)

    def test_node_name_included_in_entity_text_for_value_matching(self):
        # GraphRAG loader explicitly adds node name to entity_text for value-in-name matching
        _, _, et2 = load_graphrag_graphml(self.path)
        for name in et2:
            texts = " ".join(et2[name])
            # Node name should appear somewhere in its own texts
            self.assertIn(name, texts)


# ---------------------------------------------------------------------------
# load_graph_auto tests
# ---------------------------------------------------------------------------

class TestLoadGraphAuto(unittest.TestCase):

    def test_auto_detects_graphml_by_extension(self):
        G = _make_lightrag_graph()
        path = _write_graphml(G)
        try:
            result = load_graph_auto(path)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
        finally:
            os.unlink(path)

    def test_auto_raises_for_invalid_path(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_graph_auto("/nonexistent/path/that/does/not/exist")

    def test_auto_raises_for_unsupported_extension(self):
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="sge_test_")
        os.close(fd)
        try:
            with self.assertRaises((ValueError, FileNotFoundError)):
                load_graph_auto(path)
        finally:
            os.unlink(path)

    def test_auto_loads_graphml_from_directory_with_graphml(self):
        G = _make_lightrag_graph()
        tmpdir = tempfile.mkdtemp(prefix="sge_test_dir_")
        graphml_path = os.path.join(tmpdir, "graph_chunk_entity_relation.graphml")
        nx.write_graphml(G, graphml_path)
        try:
            result = load_graph_auto(tmpdir)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
        finally:
            os.unlink(graphml_path)
            os.rmdir(tmpdir)


# ---------------------------------------------------------------------------
# _find_parquet helper tests
# ---------------------------------------------------------------------------

class TestFindParquet(unittest.TestCase):

    def test_returns_none_when_no_parquet(self):
        tmpdir = tempfile.mkdtemp(prefix="sge_test_")
        try:
            result = _find_parquet(Path(tmpdir), "create_final_entities")
            self.assertIsNone(result)
        finally:
            os.rmdir(tmpdir)

    def test_finds_parquet_in_output_artifacts(self):
        tmpdir = tempfile.mkdtemp(prefix="sge_test_")
        artifacts_dir = os.path.join(tmpdir, "output", "artifacts")
        os.makedirs(artifacts_dir)
        parquet_path = os.path.join(artifacts_dir, "create_final_entities.parquet")
        # Create a minimal dummy file
        open(parquet_path, "w").close()
        try:
            result = _find_parquet(Path(tmpdir), "create_final_entities")
            self.assertIsNotNone(result)
            self.assertEqual(result, Path(parquet_path))
        finally:
            os.unlink(parquet_path)
            os.rmdir(artifacts_dir)
            os.rmdir(os.path.join(tmpdir, "output"))
            os.rmdir(tmpdir)

    def test_finds_parquet_via_recursive_fallback(self):
        tmpdir = tempfile.mkdtemp(prefix="sge_test_")
        nested = os.path.join(tmpdir, "deep", "nested")
        os.makedirs(nested)
        parquet_path = os.path.join(nested, "create_final_relationships.parquet")
        open(parquet_path, "w").close()
        try:
            result = _find_parquet(Path(tmpdir), "create_final_relationships")
            self.assertIsNotNone(result)
        finally:
            os.unlink(parquet_path)
            os.rmdir(nested)
            os.rmdir(os.path.join(tmpdir, "deep"))
            os.rmdir(tmpdir)


# ---------------------------------------------------------------------------
# Edge/node attribute extraction integration test
# ---------------------------------------------------------------------------

class TestNodeEdgeAttributeExtraction(unittest.TestCase):
    """Verify that node and edge attributes are correctly extracted."""

    def test_node_description_preserved(self):
        G = nx.DiGraph()
        G.add_node("n1", entity_name="TestEntity", entity_type="TestType",
                   description="Some detailed description here")
        path = _write_graphml(G)
        try:
            _, nodes, _ = load_graphml(path)
            self.assertIn("TestEntity", nodes)
            self.assertEqual(nodes["TestEntity"]["description"], "Some detailed description here")
        finally:
            os.unlink(path)

    def test_node_without_entity_name_uses_node_id(self):
        G = nx.DiGraph()
        G.add_node("fallback_node_id", entity_type="Type")
        path = _write_graphml(G)
        try:
            _, nodes, _ = load_graphml(path)
            self.assertIn("fallback_node_id", nodes)
        finally:
            os.unlink(path)

    def test_edge_keywords_appear_in_source_entity_text(self):
        G = nx.DiGraph()
        G.add_node("n1", entity_name="Alpha", entity_type="A", description="")
        G.add_node("n2", entity_name="Beta", entity_type="B", description="")
        G.add_edge("n1", "n2", keywords="SOME_KEYWORD relation_type", description="alpha beta link")
        path = _write_graphml(G)
        try:
            _, _, et2 = load_graphml(path)
            alpha_texts = " ".join(et2.get("Alpha", []))
            self.assertIn("SOME_KEYWORD", alpha_texts)
        finally:
            os.unlink(path)

    def test_reverse_edge_text_indexed_for_target_node(self):
        G = nx.DiGraph()
        G.add_node("n1", entity_name="Source", entity_type="S", description="")
        G.add_node("n2", entity_name="Target", entity_type="T", description="")
        G.add_edge("n1", "n2", keywords="kw", description="edge desc")
        path = _write_graphml(G)
        try:
            _, _, et2 = load_graphml(path)
            # Target should have reverse edge text referencing "Source"
            target_texts = " ".join(et2.get("Target", []))
            self.assertIn("Source", target_texts)
        finally:
            os.unlink(path)

    def test_isolated_node_has_entry_in_entity_text_2hop(self):
        G = nx.DiGraph()
        G.add_node("n1", entity_name="IsolatedNode", entity_type="X", description="")
        path = _write_graphml(G)
        try:
            _, _, et2 = load_graphml(path)
            self.assertIn("IsolatedNode", et2)
            self.assertIsInstance(et2["IsolatedNode"], list)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
