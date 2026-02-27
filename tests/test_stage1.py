"""
test_stage1.py — Unit tests for Stage 1 using the real CSV datasets.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python -m pytest tests/ -v
"""

import sys
import os
import unittest
from pathlib import Path

# Ensure the package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1.features import extract_features
from stage1.classifier import classify, TYPE_I, TYPE_II, TYPE_III
from stage1.schema import build_meta_schema

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
BASE = Path.home() / "Desktop" / "SGE" / "dataset"

CSV_TYPE_II_BUDGET    = BASE / "年度预算" / "annualbudget_sc.csv"
CSV_TYPE_III_FOOD     = BASE / "食物安全及公众卫生统计数字" / "stat_foodSafty_publicHealth.csv"
CSV_TYPE_II_HEALTH    = BASE / "香港主要医疗卫生统计数字" / "healthstat_table1.csv"


class TestFeatureExtraction(unittest.TestCase):
    """Verify that feature extraction produces sensible signals."""

    def test_budget_has_time_headers(self):
        """annualbudget_sc.csv should have year-pattern columns in headers."""
        f = extract_features(str(CSV_TYPE_II_BUDGET))
        self.assertGreater(
            len(f.time_cols_in_headers), 0,
            "Expected year-pattern columns in annualbudget headers"
        )

    def test_food_has_composite_key(self):
        """stat_foodSafty_publicHealth.csv should have ≥2 leading text columns."""
        f = extract_features(str(CSV_TYPE_III_FOOD))
        self.assertGreaterEqual(
            f.leading_text_col_count, 2,
            "Expected ≥2 leading text columns in food-safety CSV"
        )

    def test_food_has_remarks_column(self):
        """stat_foodSafty_publicHealth.csv should detect a remarks column."""
        f = extract_features(str(CSV_TYPE_III_FOOD))
        self.assertGreater(
            len(f.remarks_cols), 0,
            "Expected at least one remarks/notes column in food-safety CSV"
        )

    def test_health_transposed_time(self):
        """healthstat_table1.csv is transposed — years appear in data body rows."""
        f = extract_features(str(CSV_TYPE_II_HEALTH))
        # Either time in headers, first column, or data body (headerless transposed)
        has_time = (bool(f.time_cols_in_headers)
                    or f.time_cols_in_first_col
                    or f.time_in_data_body)
        self.assertTrue(has_time, "Expected time dimension in healthstat CSV")


class TestClassifier(unittest.TestCase):
    """Verify that the classifier returns the correct type for each dataset."""

    def test_budget_is_type_ii(self):
        f = extract_features(str(CSV_TYPE_II_BUDGET))
        result = classify(f)
        self.assertEqual(result, TYPE_II,
                         f"annualbudget_sc.csv should be Type II, got: {result}")

    def test_food_is_type_iii(self):
        f = extract_features(str(CSV_TYPE_III_FOOD))
        result = classify(f)
        self.assertEqual(result, TYPE_III,
                         f"stat_foodSafty_publicHealth.csv should be Type III, got: {result}")

    def test_health_is_type_ii(self):
        f = extract_features(str(CSV_TYPE_II_HEALTH))
        result = classify(f)
        self.assertEqual(result, TYPE_II,
                         f"healthstat_table1.csv should be Type II, got: {result}")


class TestMetaSchema(unittest.TestCase):
    """Verify that the Meta-Schema output has the required structure and values."""

    def _required_keys(self):
        return {
            "table_type", "primary_subject_columns", "time_dimension",
            "value_columns", "metadata_columns", "composite_key",
            "extraction_rule_summary",
        }

    def _check_schema_structure(self, schema: dict):
        """Assert all required top-level keys are present."""
        for key in self._required_keys():
            self.assertIn(key, schema, f"Missing key in meta-schema: {key}")
        # time_dimension sub-keys
        td = schema["time_dimension"]
        for sub in ("is_present", "location", "columns"):
            self.assertIn(sub, td, f"Missing time_dimension.{sub}")

    def test_budget_schema(self):
        f = extract_features(str(CSV_TYPE_II_BUDGET))
        t = classify(f)
        schema = build_meta_schema(f, t)
        self._check_schema_structure(schema)
        self.assertEqual(schema["table_type"], TYPE_II)
        self.assertTrue(schema["time_dimension"]["is_present"])
        self.assertEqual(schema["time_dimension"]["location"], "headers")
        self.assertGreater(len(schema["value_columns"]), 0)

    def test_food_schema(self):
        f = extract_features(str(CSV_TYPE_III_FOOD))
        t = classify(f)
        schema = build_meta_schema(f, t)
        self._check_schema_structure(schema)
        self.assertEqual(schema["table_type"], TYPE_III)
        self.assertTrue(schema["composite_key"])
        self.assertFalse(schema["time_dimension"]["is_present"])

    def test_health_schema(self):
        f = extract_features(str(CSV_TYPE_II_HEALTH))
        t = classify(f)
        schema = build_meta_schema(f, t)
        self._check_schema_structure(schema)
        self.assertEqual(schema["table_type"], TYPE_II)
        self.assertTrue(schema["time_dimension"]["is_present"])

    def test_summary_is_nonempty_string(self):
        """extraction_rule_summary must be a non-empty string for all datasets."""
        for csv_path in [CSV_TYPE_II_BUDGET, CSV_TYPE_III_FOOD, CSV_TYPE_II_HEALTH]:
            f = extract_features(str(csv_path))
            t = classify(f)
            schema = build_meta_schema(f, t)
            self.assertIsInstance(schema["extraction_rule_summary"], str)
            self.assertGreater(len(schema["extraction_rule_summary"]), 0)


if __name__ == "__main__":
    unittest.main()
