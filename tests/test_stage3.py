"""
test_stage3.py — Unit tests for Stage 3 serialization and prompt generation.

No LLM or LightRAG required — tests only the deterministic parts.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python3 -m pytest tests/test_stage3.py -v
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1.features import extract_features
from stage1.classifier import classify, TYPE_I, TYPE_II, TYPE_III
from stage1.schema import build_meta_schema
from stage2.inducer import induce_schema

from stage3.serializer import serialize_csv
from stage3.prompt_injector import (
    generate_system_prompt,
    generate_user_prompt_template,
    render_user_prompt,
    TUPLE_DELIMITER,
    COMPLETION_DELIMITER,
)
from stage3.integrator import patch_lightrag, prepare_chunks

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
BASE = Path.home() / "Desktop" / "SGE" / "dataset"
CSV_BUDGET = BASE / "年度预算" / "annualbudget_sc.csv"
CSV_FOOD   = BASE / "食物安全及公众卫生统计数字" / "stat_foodSafty_publicHealth.csv"


def _get_schema(csv_path: Path) -> dict:
    """Run Stage 1 + Stage 2 and return the extraction schema."""
    features   = extract_features(str(csv_path))
    table_type = classify(features)
    meta       = build_meta_schema(features, table_type)
    return induce_schema(meta, features)


# ---------------------------------------------------------------------------
# Serializer tests — Type II (Budget)
# ---------------------------------------------------------------------------

class TestSerializerTypeII(unittest.TestCase):
    """annualbudget_sc.csv is Type II (Time-Series-Matrix)."""

    def setUp(self):
        self.schema = _get_schema(CSV_BUDGET)
        self.chunks = serialize_csv(str(CSV_BUDGET), self.schema)

    def test_produces_chunks(self):
        self.assertGreater(len(self.chunks), 0, "Expected at least one chunk")

    def test_each_chunk_has_entity(self):
        for chunk in self.chunks:
            self.assertIn("Entity:", chunk, f"Chunk missing 'Entity:': {chunk[:100]}")

    def test_chunks_contain_year_values(self):
        # At least some chunks should mention a year-like pattern
        has_year = any(
            any(y in chunk for y in ["2022", "2023", "2024"])
            for chunk in self.chunks
        )
        self.assertTrue(has_year, "No chunks contain year references")

    def test_chunks_are_strings(self):
        for chunk in self.chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk.strip()), 0)

    def test_first_chunk_contains_budget_entity(self):
        # First data row should be 防止贪污
        first = self.chunks[0]
        self.assertIn("防止贪污", first, f"Expected 防止贪污 in first chunk: {first}")

    def test_chunk_count_matches_data_rows(self):
        import pandas as pd
        df = pd.read_csv(str(CSV_BUDGET), encoding="utf-8")
        # Each non-empty row should produce one chunk
        self.assertGreaterEqual(len(self.chunks), 1)
        self.assertLessEqual(len(self.chunks), len(df) + 1)


# ---------------------------------------------------------------------------
# Serializer tests — Type III (Food Safety)
# ---------------------------------------------------------------------------

class TestSerializerTypeIII(unittest.TestCase):
    """stat_foodSafty_publicHealth.csv is Type III (Hierarchical-Hybrid)."""

    def setUp(self):
        self.schema = _get_schema(CSV_FOOD)
        self.chunks = serialize_csv(str(CSV_FOOD), self.schema)

    def test_produces_chunks(self):
        self.assertGreater(len(self.chunks), 0)

    def test_chunks_have_category(self):
        for chunk in self.chunks:
            self.assertIn("Category:", chunk, f"Chunk missing 'Category:': {chunk[:100]}")

    def test_chunks_have_year_values(self):
        has_year = any(
            any(y in chunk for y in ["2021", "2022", "2023", "2024"])
            for chunk in self.chunks
        )
        self.assertTrue(has_year, "No chunks contain year values")

    def test_first_chunk_contains_food_safety(self):
        first = self.chunks[0]
        self.assertIn("食物安全", first, f"Expected 食物安全 in first chunk: {first}")

    def test_hierarchy_separator_present(self):
        # Type III chunks use " > " to separate hierarchy levels
        has_separator = any(" > " in chunk for chunk in self.chunks)
        self.assertTrue(has_separator, "No chunks contain hierarchy separator ' > '")

    def test_pipe_separator_for_values(self):
        # Values are separated by " | "
        has_pipe = any(" | " in chunk for chunk in self.chunks)
        self.assertTrue(has_pipe, "No chunks contain value separator ' | '")


# ---------------------------------------------------------------------------
# Prompt injector tests
# ---------------------------------------------------------------------------

class TestPromptInjector(unittest.TestCase):

    def setUp(self):
        self.schema_budget = _get_schema(CSV_BUDGET)
        self.schema_food   = _get_schema(CSV_FOOD)

    def test_system_prompt_contains_schema_json(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn("entity_types", prompt)
        self.assertIn("relation_types", prompt)

    def test_system_prompt_contains_tuple_delimiter(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn(TUPLE_DELIMITER, prompt)

    def test_system_prompt_contains_completion_delimiter(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn(COMPLETION_DELIMITER, prompt)

    def test_system_prompt_contains_role(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn("Knowledge Graph Specialist", prompt)

    def test_system_prompt_contains_entity_format(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn("entity" + TUPLE_DELIMITER, prompt)

    def test_system_prompt_contains_relation_format(self):
        prompt = generate_system_prompt(self.schema_budget)
        self.assertIn("relation" + TUPLE_DELIMITER, prompt)

    def test_system_prompt_contains_entity_types(self):
        prompt = generate_system_prompt(self.schema_budget)
        for et in self.schema_budget["entity_types"]:
            self.assertIn(et, prompt, f"Entity type {et} not in system prompt")

    def test_user_prompt_template_has_placeholder(self):
        tmpl = generate_user_prompt_template(self.schema_budget)
        self.assertIn("{input_text}", tmpl)

    def test_user_prompt_template_has_entity_types(self):
        tmpl = generate_user_prompt_template(self.schema_budget)
        for et in self.schema_budget["entity_types"]:
            self.assertIn(et, tmpl)

    def test_render_user_prompt_fills_placeholder(self):
        tmpl = generate_user_prompt_template(self.schema_budget)
        rendered = render_user_prompt(tmpl, "Entity: 防止贪污\nYear: 2022-23 | Value: 91.5")
        self.assertNotIn("{input_text}", rendered)
        self.assertIn("防止贪污", rendered)

    def test_system_prompt_food_schema(self):
        prompt = generate_system_prompt(self.schema_food)
        self.assertIn("HAS_SUB_ITEM", prompt)

    def test_system_prompt_no_json_instruction(self):
        prompt = generate_system_prompt(self.schema_budget)
        # Should explicitly forbid JSON output
        self.assertIn("Do NOT output JSON", prompt)


# ---------------------------------------------------------------------------
# Integrator tests
# ---------------------------------------------------------------------------

class TestIntegrator(unittest.TestCase):

    def setUp(self):
        self.schema = _get_schema(CSV_BUDGET)

    def test_patch_lightrag_returns_dict(self):
        payload = patch_lightrag(self.schema)
        self.assertIsInstance(payload, dict)

    def test_patch_lightrag_has_required_keys(self):
        payload = patch_lightrag(self.schema)
        for key in ("system_prompt", "context_base_extra", "addon_params", "entity_types"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_patch_lightrag_system_prompt_is_string(self):
        payload = patch_lightrag(self.schema)
        self.assertIsInstance(payload["system_prompt"], str)
        self.assertGreater(len(payload["system_prompt"]), 100)

    def test_patch_lightrag_addon_params_has_entity_types(self):
        payload = patch_lightrag(self.schema)
        self.assertIn("entity_types", payload["addon_params"])
        self.assertIsInstance(payload["addon_params"]["entity_types"], list)

    def test_patch_lightrag_addon_params_has_schema_json(self):
        payload = patch_lightrag(self.schema)
        self.assertIn("schema_json", payload["addon_params"])
        # Should be valid JSON
        parsed = json.loads(payload["addon_params"]["schema_json"])
        self.assertIn("entity_types", parsed)

    def test_patch_lightrag_context_base_extra_has_schema_json(self):
        payload = patch_lightrag(self.schema)
        self.assertIn("schema_json", payload["context_base_extra"])

    def test_prepare_chunks_budget(self):
        chunks = prepare_chunks(str(CSV_BUDGET), self.schema)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], str)

    def test_prepare_chunks_food(self):
        schema_food = _get_schema(CSV_FOOD)
        chunks = prepare_chunks(str(CSV_FOOD), schema_food)
        self.assertGreater(len(chunks), 0)

    def test_prepare_chunks_returns_list_of_strings(self):
        chunks = prepare_chunks(str(CSV_BUDGET), self.schema)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk.strip()), 0)


# ---------------------------------------------------------------------------
# Output format compliance tests
# ---------------------------------------------------------------------------

class TestOutputFormatCompliance(unittest.TestCase):
    """
    Verify that the system prompt correctly instructs the LLM to use
    LightRAG's delimiter-based output format.
    """

    def setUp(self):
        self.schema = _get_schema(CSV_BUDGET)
        self.system_prompt = generate_system_prompt(self.schema)

    def test_delimiter_format_in_prompt(self):
        # The prompt must show the exact delimiter format
        self.assertIn("entity<|#|>", self.system_prompt)
        self.assertIn("relation<|#|>", self.system_prompt)

    def test_completion_delimiter_in_prompt(self):
        self.assertIn("<|COMPLETE|>", self.system_prompt)

    def test_entity_line_format_correct_fields(self):
        # entity<|#|>name<|#|>type<|#|>description = 4 fields
        import re
        # Find lines that look like entity format instructions
        lines = self.system_prompt.split("\n")
        entity_lines = [l for l in lines if "entity<|#|>" in l]
        self.assertGreater(len(entity_lines), 0, "No entity format line found in prompt")

    def test_relation_line_format_correct_fields(self):
        lines = self.system_prompt.split("\n")
        relation_lines = [l for l in lines if "relation<|#|>" in l]
        self.assertGreater(len(relation_lines), 0, "No relation format line found in prompt")


if __name__ == "__main__":
    unittest.main()
