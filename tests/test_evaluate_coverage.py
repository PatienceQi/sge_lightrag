"""
test_evaluate_coverage.py — Unit tests for evaluation/evaluate_coverage.py

Tests all public functions:
  - load_gold(jsonl_path)
  - load_graph(graphml_path)
  - check_entity_coverage(gold_entities, graph_nodes)
  - check_fact_coverage(facts, graph_nodes, entity_text)

Uses in-memory fixtures only: tempfile for JSONL and networkx graphs saved to
tempfile GraphML.  No external files or API calls required.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python3 -m pytest tests/test_evaluate_coverage.py -v
"""

import json
import sys
import tempfile
import os
from pathlib import Path

import pytest

# Ensure the package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    pytest.skip("networkx not installed", allow_module_level=True)

from evaluation.evaluate_coverage import (
    load_gold,
    load_graph,
    check_entity_coverage,
    check_fact_coverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list, path: str) -> None:
    """Write a list of dicts as JSONL to path."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_graphml(G: nx.Graph) -> str:
    """Save an in-memory networkx graph to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".graphml")
    os.close(fd)
    nx.write_graphml(G, path)
    return path


def _build_node(G: nx.Graph, node_id: str, entity_name: str,
                entity_type: str = "ENTITY", description: str = "") -> None:
    """Add a node with standard SGE-LightRAG attributes."""
    G.add_node(node_id, entity_name=entity_name,
               entity_type=entity_type, description=description)


def _build_edge(G: nx.Graph, u: str, v: str,
                keywords: str = "", description: str = "") -> None:
    """Add an undirected edge with keyword/description attributes."""
    G.add_edge(u, v, keywords=keywords, description=description)


# ---------------------------------------------------------------------------
# load_gold tests
# ---------------------------------------------------------------------------

class TestLoadGold:
    def test_returns_entities_and_facts(self, tmp_path):
        records = [
            {
                "triple": {
                    "subject": "China",
                    "relation": "HAS_VALUE",
                    "object": "72.5",
                    "object_type": "StatValue",
                    "attributes": {"year": "2020"},
                }
            },
            {
                "triple": {
                    "subject": "India",
                    "relation": "HAS_VALUE",
                    "object": "69.4",
                    "object_type": "StatValue",
                    "attributes": {"year": "2020"},
                }
            },
        ]
        p = str(tmp_path / "gold.jsonl")
        _write_jsonl(records, p)

        entities, facts = load_gold(p)

        assert entities == {"China", "India"}
        assert len(facts) == 2

    def test_only_value_bearing_relations_counted_as_facts(self, tmp_path):
        """Non-value object_types (e.g., concept relations) should be excluded from facts."""
        records = [
            {
                "triple": {
                    "subject": "Country",
                    "relation": "PART_OF",
                    "object": "Region",
                    "object_type": "Entity",          # not a value type
                    "attributes": {},
                }
            },
            {
                "triple": {
                    "subject": "Country",
                    "relation": "HAS_BUDGET",
                    "object": "1500",
                    "object_type": "BudgetAmount",
                    "attributes": {"year": "2022"},
                }
            },
        ]
        p = str(tmp_path / "gold.jsonl")
        _write_jsonl(records, p)

        entities, facts = load_gold(p)

        assert "Country" in entities
        assert len(facts) == 1
        assert facts[0]["value"] == "1500"

    def test_all_three_value_object_types_are_included(self, tmp_path):
        """BudgetAmount, StatValue, and Literal all count as value-bearing facts."""
        records = [
            {"triple": {"subject": "A", "relation": "R1", "object": "100",
                        "object_type": "BudgetAmount", "attributes": {}}},
            {"triple": {"subject": "B", "relation": "R2", "object": "200",
                        "object_type": "StatValue",    "attributes": {}}},
            {"triple": {"subject": "C", "relation": "R3", "object": "hello",
                        "object_type": "Literal",      "attributes": {}}},
        ]
        p = str(tmp_path / "gold.jsonl")
        _write_jsonl(records, p)

        _, facts = load_gold(p)

        assert len(facts) == 3

    def test_year_attribute_propagated_to_facts(self, tmp_path):
        records = [
            {
                "triple": {
                    "subject": "Japan",
                    "relation": "HAS_VALUE",
                    "object": "84.3",
                    "object_type": "StatValue",
                    "attributes": {"year": "2019"},
                }
            },
        ]
        p = str(tmp_path / "gold.jsonl")
        _write_jsonl(records, p)

        _, facts = load_gold(p)

        assert facts[0]["year"] == "2019"
        assert facts[0]["subject"] == "Japan"
        assert facts[0]["value"] == "84.3"

    def test_empty_jsonl_returns_empty_sets(self, tmp_path):
        p = str(tmp_path / "empty.jsonl")
        _write_jsonl([], p)

        entities, facts = load_gold(p)

        assert entities == set()
        assert facts == []

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path):
        p = str(tmp_path / "blanks.jsonl")
        content = (
            json.dumps({"triple": {"subject": "X", "relation": "R",
                                   "object": "1", "object_type": "StatValue",
                                   "attributes": {}}})
            + "\n\n\n"
        )
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

        entities, facts = load_gold(p)

        assert len(entities) == 1
        assert len(facts) == 1


# ---------------------------------------------------------------------------
# load_graph tests
# ---------------------------------------------------------------------------

class TestLoadGraph:
    def _simple_graph(self):
        """Build a small graph with two nodes and one edge."""
        G = nx.Graph()
        _build_node(G, "n1", "China", entity_type="COUNTRY",
                    description="Large country in Asia")
        _build_node(G, "n2", "2020", entity_type="YEAR")
        _build_edge(G, "n1", "n2", keywords="72.5 life_expectancy",
                    description="China life expectancy in 2020 is 72.5")
        return G

    def test_returns_graph_nodes_and_entity_text(self, tmp_path):
        G = self._simple_graph()
        path = _make_graphml(G)

        try:
            result_G, graph_nodes, entity_text = load_graph(path)
        finally:
            os.unlink(path)

        assert "China" in graph_nodes
        assert "2020" in graph_nodes

    def test_entity_text_contains_edge_keywords(self, tmp_path):
        G = self._simple_graph()
        path = _make_graphml(G)

        try:
            _, _, entity_text = load_graph(path)
        finally:
            os.unlink(path)

        china_texts = " ".join(entity_text.get("China", []))
        assert "72.5" in china_texts

    def test_2hop_expansion_propagates_neighbor_descriptions(self):
        """A node's entity_text should include its neighbor's description."""
        G = nx.Graph()
        _build_node(G, "n1", "Disease A", description="Some disease")
        _build_node(G, "n2", "ICD_001",   description="code value 150")
        _build_edge(G, "n1", "n2", keywords="HAS_SUB_ITEM", description="")
        path = _make_graphml(G)

        try:
            _, _, entity_text = load_graph(path)
        finally:
            os.unlink(path)

        # Disease A should be able to see ICD_001's description via 2-hop
        disease_texts = " ".join(entity_text.get("Disease A", []))
        assert "code value 150" in disease_texts

    def test_empty_graph_returns_empty_structures(self):
        G = nx.Graph()
        path = _make_graphml(G)

        try:
            result_G, graph_nodes, entity_text = load_graph(path)
        finally:
            os.unlink(path)

        assert len(graph_nodes) == 0
        assert len(entity_text) == 0

    def test_node_without_entity_name_falls_back_to_node_id(self):
        """Nodes without entity_name attribute should use the raw node_id."""
        G = nx.Graph()
        G.add_node("raw_id_node", entity_type="MISC")   # no entity_name
        path = _make_graphml(G)

        try:
            _, graph_nodes, _ = load_graph(path)
        finally:
            os.unlink(path)

        assert "raw_id_node" in graph_nodes


# ---------------------------------------------------------------------------
# check_entity_coverage tests
# ---------------------------------------------------------------------------

class TestCheckEntityCoverage:
    def _nodes(self, names):
        """Build minimal graph_nodes dict from a list of names."""
        return {name: {"type": "ENTITY", "description": ""} for name in names}

    def test_exact_match(self):
        gold = {"China", "Japan"}
        nodes = self._nodes(["China", "Japan", "Korea"])

        matched = check_entity_coverage(gold, nodes)

        assert matched == {"China", "Japan"}

    def test_case_insensitive_exact_match(self):
        gold = {"china"}
        nodes = self._nodes(["China"])

        matched = check_entity_coverage(gold, nodes)

        assert "china" in matched

    def test_substring_match_gold_in_node(self):
        """Gold entity 'China' should match node 'China_Province'."""
        gold = {"China"}
        nodes = self._nodes(["China_Province"])

        matched = check_entity_coverage(gold, nodes)

        assert "China" in matched

    def test_substring_match_node_in_gold(self):
        """Node 'Life Exp' should match gold entity 'Life Expectancy At Birth'."""
        gold = {"Life Expectancy At Birth"}
        nodes = self._nodes(["Life Exp"])

        matched = check_entity_coverage(gold, nodes)

        assert "Life Expectancy At Birth" in matched

    def test_description_fallback_match(self):
        """Entity code in node description should count as a match."""
        gold = {"CHN"}
        nodes = {"China": {"type": "COUNTRY",
                            "description": "country code CHN, population large"}}

        matched = check_entity_coverage(gold, nodes)

        assert "CHN" in matched

    def test_no_match(self):
        gold = {"Germany"}
        nodes = self._nodes(["France", "Spain"])

        matched = check_entity_coverage(gold, nodes)

        assert matched == set()

    def test_empty_gold(self):
        nodes = self._nodes(["China"])

        matched = check_entity_coverage(set(), nodes)

        assert matched == set()

    def test_empty_graph(self):
        gold = {"China"}

        matched = check_entity_coverage(gold, {})

        assert matched == set()

    def test_partial_coverage(self):
        gold = {"China", "France", "Brazil"}
        nodes = self._nodes(["China", "Brazil"])

        matched = check_entity_coverage(gold, nodes)

        assert "China" in matched
        assert "Brazil" in matched
        assert "France" not in matched


# ---------------------------------------------------------------------------
# check_fact_coverage tests
# ---------------------------------------------------------------------------

class TestCheckFactCoverage:
    def _nodes(self, entries: dict):
        """entries: {name: description_str}"""
        return {
            name: {"type": "ENTITY", "description": desc}
            for name, desc in entries.items()
        }

    def _entity_text(self, entries: dict):
        """entries: {entity_name: [text1, text2, ...]}"""
        return dict(entries)

    def test_fact_covered_when_value_and_year_in_edge_text(self):
        facts = [{"subject": "China", "value": "72.5", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"China": ""})
        entity_text = self._entity_text({"China": ["72.5 2020 life expectancy"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 1
        assert len(not_covered) == 0

    def test_fact_not_covered_when_value_missing(self):
        facts = [{"subject": "China", "value": "72.5", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"China": ""})
        entity_text = self._entity_text({"China": ["2020 life expectancy"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 0
        assert not_covered[0]["reason"] == "value_not_found"

    def test_fact_not_covered_when_year_missing(self):
        facts = [{"subject": "China", "value": "72.5", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"China": ""})
        entity_text = self._entity_text({"China": ["72.5 life expectancy"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 0
        assert not_covered[0]["reason"] == "year_not_found"

    def test_fact_covered_when_no_year_specified(self):
        """Year-free facts should be covered as long as value is present."""
        facts = [{"subject": "Japan", "value": "84.3", "year": "",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"Japan": ""})
        entity_text = self._entity_text({"Japan": ["84.3 life expectancy"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 1

    def test_fact_not_covered_when_entity_not_in_graph(self):
        facts = [{"subject": "Germany", "value": "81.2", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"France": ""})
        entity_text = self._entity_text({"France": ["81.2 2020"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 0
        assert not_covered[0]["reason"] == "entity_not_found"

    def test_value_found_in_node_description_not_edge_text(self):
        """Value in node description (not edge text) should still cover the fact."""
        facts = [{"subject": "Brazil", "value": "75.9", "year": "2019",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"Brazil": "life expectancy 75.9 in 2019"})
        entity_text = self._entity_text({"Brazil": []})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 1

    def test_subject_matched_by_substring(self):
        """Subject 'China' should match graph node 'China_Entity'."""
        facts = [{"subject": "China", "value": "72.5", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = self._nodes({"China_Entity": ""})
        entity_text = self._entity_text({"China_Entity": ["72.5 2020"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 1

    def test_subject_matched_via_description_fallback(self):
        """Subject code in node description triggers description-fallback matching."""
        facts = [{"subject": "CHN", "value": "72.5", "year": "2020",
                  "relation": "HAS_VALUE"}]
        nodes = {"China": {"type": "COUNTRY",
                            "description": "country code CHN 72.5 2020"}}
        entity_text = self._entity_text({"China": []})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 1

    def test_empty_facts_returns_empty_lists(self):
        nodes = self._nodes({"China": ""})
        entity_text = self._entity_text({"China": ["72.5 2020"]})

        covered, not_covered = check_fact_coverage([], nodes, entity_text)

        assert covered == []
        assert not_covered == []

    def test_empty_graph_all_facts_uncovered(self):
        facts = [
            {"subject": "China", "value": "72.5", "year": "2020", "relation": "R"},
            {"subject": "Japan", "value": "84.3", "year": "2020", "relation": "R"},
        ]

        covered, not_covered = check_fact_coverage(facts, {}, {})

        assert covered == []
        assert len(not_covered) == 2
        assert all(nc["reason"] == "entity_not_found" for nc in not_covered)

    def test_multiple_facts_mixed_coverage(self):
        facts = [
            {"subject": "China", "value": "72.5", "year": "2020", "relation": "R"},
            {"subject": "India", "value": "69.4", "year": "2019", "relation": "R"},
            {"subject": "Brazil", "value": "75.9", "year": "2018", "relation": "R"},
        ]
        nodes = self._nodes({"China": "", "India": "", "Brazil": ""})
        entity_text = self._entity_text({
            "China": ["72.5 2020"],
            "India": ["2019"],          # value missing
            "Brazil": ["75.9 2018"],
        })

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 2
        assert len(not_covered) == 1
        assert not_covered[0]["subject"] == "India"
        assert not_covered[0]["reason"] == "value_not_found"

    def test_value_search_is_exact_substring(self):
        """Value '7' should NOT match against '72.5' — exact substring required."""
        facts = [{"subject": "China", "value": "777.0", "year": "",
                  "relation": "R"}]
        nodes = self._nodes({"China": ""})
        entity_text = self._entity_text({"China": ["72.5 life expectancy"]})

        covered, not_covered = check_fact_coverage(facts, nodes, entity_text)

        assert len(covered) == 0
        assert not_covered[0]["reason"] == "value_not_found"
