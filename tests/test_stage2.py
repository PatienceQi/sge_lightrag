"""
test_stage2.py — Unit tests for Stage 2 Rule-Based Schema Induction.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python3 -m unittest discover -s tests -p "test_stage2*" -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1.features import extract_features
from stage1.classifier import classify, TYPE_I, TYPE_II, TYPE_III
from stage1.schema import build_meta_schema
from stage2.inductor import induce_schema, induce_schema_from_meta

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
BASE = Path.home() / "Desktop" / "SGE" / "dataset"

CSV_BUDGET = BASE / "年度预算" / "annualbudget_sc.csv"
CSV_FOOD   = BASE / "食物安全及公众卫生统计数字" / "stat_foodSafty_publicHealth.csv"
CSV_HEALTH = BASE / "香港主要医疗卫生统计数字" / "healthstat_table1.csv"

ALL_CSVS = [CSV_BUDGET, CSV_FOOD, CSV_HEALTH]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_pipeline(csv_path):
    """Run Stage 1 + Stage 2 and return (schema, table_type)."""
    features = extract_features(str(csv_path))
    table_type = classify(features)
    meta_schema = build_meta_schema(features, table_type)
    schema = induce_schema_from_meta(features, table_type, meta_schema)
    return schema, table_type


# ---------------------------------------------------------------------------
# Schema structure tests
# ---------------------------------------------------------------------------

class TestSchemaStructure(unittest.TestCase):
    """Every schema must have the required top-level keys with correct types."""

    REQUIRED_KEYS = {"table_type", "entity_types", "relation_types",
                     "extraction_rules", "prompt_context"}
    EXTRACTION_RULE_KEYS = {"subject_extraction", "value_extraction",
                            "time_handling", "hierarchy", "remarks"}

    def _check(self, schema):
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, schema, f"Missing top-level key: {key}")

        self.assertIsInstance(schema["entity_types"], list)
        self.assertIsInstance(schema["relation_types"], list)
        self.assertIsInstance(schema["extraction_rules"], dict)
        self.assertIsInstance(schema["prompt_context"], str)

        # Baseline mode skips detailed extraction rules (adaptive small-table)
        if schema.get("use_baseline_mode"):
            return

        for sub in self.EXTRACTION_RULE_KEYS:
            self.assertIn(sub, schema["extraction_rules"],
                          f"Missing extraction_rules.{sub}")

    def test_budget_structure(self):
        schema, _ = _run_pipeline(CSV_BUDGET)
        self._check(schema)

    def test_food_structure(self):
        schema, _ = _run_pipeline(CSV_FOOD)
        self._check(schema)

    def test_health_structure(self):
        schema, _ = _run_pipeline(CSV_HEALTH)
        self._check(schema)


# ---------------------------------------------------------------------------
# Non-empty content tests
# ---------------------------------------------------------------------------

class TestSchemaContent(unittest.TestCase):
    """entity_types, relation_types, and prompt_context must be non-empty."""

    def _check_nonempty(self, schema, label):
        # Baseline mode has minimal entity_types and empty prompt_context by design
        if schema.get("use_baseline_mode"):
            return
        self.assertGreater(len(schema["entity_types"]), 0,
                           f"{label}: entity_types is empty")
        self.assertGreater(len(schema["relation_types"]), 0,
                           f"{label}: relation_types is empty")
        self.assertGreater(len(schema["prompt_context"].strip()), 0,
                           f"{label}: prompt_context is empty")

    def test_budget_nonempty(self):
        schema, _ = _run_pipeline(CSV_BUDGET)
        self._check_nonempty(schema, "BUDGET")

    def test_food_nonempty(self):
        schema, _ = _run_pipeline(CSV_FOOD)
        self._check_nonempty(schema, "FOOD")

    def test_health_nonempty(self):
        schema, _ = _run_pipeline(CSV_HEALTH)
        self._check_nonempty(schema, "HEALTH")


# ---------------------------------------------------------------------------
# Adaptive mode tests
# ---------------------------------------------------------------------------

class TestAdaptiveMode(unittest.TestCase):
    """Small tables (n_rows < 20) with Type-III should use baseline mode."""

    def test_food_adaptive_baseline(self):
        """Food safety has 13 rows → should trigger baseline mode."""
        schema, _ = _run_pipeline(CSV_FOOD)
        self.assertTrue(
            schema.get("use_baseline_mode", False),
            "Food safety (13 rows, Type-III) should use baseline mode"
        )
        self.assertEqual(schema.get("adaptive_reason"), "small_table_adaptive")

    def test_budget_no_adaptive(self):
        """Budget is Type-II (not Type-III) → should NOT use baseline mode."""
        schema, _ = _run_pipeline(CSV_BUDGET)
        self.assertFalse(
            schema.get("use_baseline_mode", False),
            "Budget (Type-II) should NOT use baseline mode"
        )


# ---------------------------------------------------------------------------
# Type-specific tests
# ---------------------------------------------------------------------------

class TestTypeII_Budget(unittest.TestCase):
    """annualbudget_sc.csv — Type II with compound headers."""

    def setUp(self):
        self.schema, self.table_type = _run_pipeline(CSV_BUDGET)

    def test_is_type_ii(self):
        self.assertEqual(self.table_type, TYPE_II)
        self.assertEqual(self.schema["table_type"], TYPE_II)

    def test_has_time_parsing_rules(self):
        self.assertIn("time_parsing_rules", self.schema)
        tpr = self.schema["time_parsing_rules"]
        self.assertIn("components", tpr)
        self.assertIn("year", tpr["components"])

    def test_relation_is_budget_related(self):
        # Budget CSV should produce HAS_BUDGET or HAS_VALUE relation
        rels = self.schema["relation_types"]
        self.assertTrue(
            any(r in ("HAS_BUDGET", "HAS_VALUE", "HAS_METRIC") for r in rels),
            f"Expected budget/value relation, got: {rels}"
        )

    def test_prompt_context_mentions_time_series(self):
        ctx = self.schema["prompt_context"]
        self.assertIn("Time-Series-Matrix", ctx)

    def test_prompt_context_mentions_compound_header(self):
        ctx = self.schema["prompt_context"]
        # Should mention compound header parsing
        self.assertTrue(
            "复合标题" in ctx or "compound header" in ctx or "fiscal year" in ctx,
            f"prompt_context should mention compound headers: {ctx[:200]}"
        )


class TestTypeIII_Food(unittest.TestCase):
    """stat_foodSafty_publicHealth.csv — Type III hierarchical (adaptive baseline mode)."""

    def setUp(self):
        self.schema, self.table_type = _run_pipeline(CSV_FOOD)

    def _skip_if_baseline(self):
        if self.schema.get("use_baseline_mode"):
            self.skipTest("Food safety uses adaptive baseline mode — detailed schema not generated")

    def test_is_type_iii(self):
        self.assertEqual(self.table_type, TYPE_III)
        self.assertEqual(self.schema["table_type"], TYPE_III)

    def test_has_hierarchy_relations(self):
        self._skip_if_baseline()
        self.assertIn("hierarchy_relations", self.schema)
        hr = self.schema["hierarchy_relations"]
        self.assertIsInstance(hr, list)
        self.assertGreater(len(hr), 0, "Expected at least one hierarchy relation")

    def test_hierarchy_relation_structure(self):
        self._skip_if_baseline()
        for hr in self.schema["hierarchy_relations"]:
            for key in ("parent_col", "child_col", "parent_entity", "child_entity", "relation"):
                self.assertIn(key, hr, f"Missing key in hierarchy_relation: {key}")

    def test_has_value_mapping(self):
        self._skip_if_baseline()
        self.assertIn("value_mapping", self.schema)
        vm = self.schema["value_mapping"]
        self.assertIsInstance(vm, dict)
        self.assertGreater(len(vm), 0, "Expected non-empty value_mapping")

    def test_has_sub_item_relation(self):
        self._skip_if_baseline()
        self.assertIn("HAS_SUB_ITEM", self.schema["relation_types"])

    def test_prompt_context_mentions_hierarchy(self):
        self._skip_if_baseline()
        ctx = self.schema["prompt_context"]
        self.assertIn("Hierarchical-Hybrid", ctx)

    def test_prompt_context_mentions_sparse_fill(self):
        self._skip_if_baseline()
        ctx = self.schema["prompt_context"]
        self.assertIn("sparse fill", ctx)


class TestTypeII_Health(unittest.TestCase):
    """healthstat_table1.csv — Type II transposed (UTF-16LE)."""

    def setUp(self):
        self.schema, self.table_type = _run_pipeline(CSV_HEALTH)

    def test_is_type_ii(self):
        self.assertEqual(self.table_type, TYPE_II)
        self.assertEqual(self.schema["table_type"], TYPE_II)

    def test_has_pivot_instruction(self):
        # Transposed table should have pivot_instruction
        self.assertIn("pivot_instruction", self.schema)
        self.assertIn("rows", self.schema["pivot_instruction"])

    def test_prompt_context_mentions_transposed(self):
        ctx = self.schema["prompt_context"]
        self.assertTrue(
            "transposed" in ctx or "转置" in ctx,
            f"prompt_context should mention transposed layout: {ctx[:200]}"
        )


# ---------------------------------------------------------------------------
# Prompt context language tests
# ---------------------------------------------------------------------------

class TestPromptContextLanguage(unittest.TestCase):
    """prompt_context should be a readable Chinese paragraph."""

    def _has_chinese(self, text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def test_budget_prompt_has_chinese(self):
        schema, _ = _run_pipeline(CSV_BUDGET)
        self.assertTrue(self._has_chinese(schema["prompt_context"]))

    def test_food_prompt_has_chinese(self):
        schema, _ = _run_pipeline(CSV_FOOD)
        if schema.get("use_baseline_mode"):
            return  # Baseline mode: no prompt generated
        self.assertTrue(self._has_chinese(schema["prompt_context"]))

    def test_health_prompt_has_chinese(self):
        schema, _ = _run_pipeline(CSV_HEALTH)
        self.assertTrue(self._has_chinese(schema["prompt_context"]))

    def test_all_prompts_min_length(self):
        """Each prompt_context should be at least 50 characters (skip baseline mode)."""
        for csv_path in ALL_CSVS:
            schema, _ = _run_pipeline(csv_path)
            if schema.get("use_baseline_mode"):
                continue  # Baseline mode intentionally has no prompt
            self.assertGreater(
                len(schema["prompt_context"]), 50,
                f"prompt_context too short for {csv_path.name}"
            )


# ---------------------------------------------------------------------------
# induce_schema() convenience function test
# ---------------------------------------------------------------------------

class TestInduceSchemaConvenience(unittest.TestCase):
    """Test the single-call induce_schema(csv_path) entry point."""

    def test_budget_via_induce_schema(self):
        schema = induce_schema(str(CSV_BUDGET))
        self.assertIn("table_type", schema)
        self.assertIn("prompt_context", schema)
        self.assertGreater(len(schema["entity_types"]), 0)

    def test_food_via_induce_schema(self):
        schema = induce_schema(str(CSV_FOOD))
        self.assertEqual(schema["table_type"], TYPE_III)

    def test_health_via_induce_schema(self):
        schema = induce_schema(str(CSV_HEALTH))
        self.assertEqual(schema["table_type"], TYPE_II)


if __name__ == "__main__":
    unittest.main()
