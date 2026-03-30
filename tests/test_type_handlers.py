"""
test_type_handlers.py — Unit tests for stage2/type_handlers.py.

Tests each handler (handle_type_i, handle_type_ii, handle_type_iii) with
minimal mock FeatureSet and meta_schema dicts, verifying returned schema
structure and required keys.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python -m pytest tests/test_type_handlers.py -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1.features import FeatureSet
from stage2.type_handlers import (
    handle_type_i,
    handle_type_ii,
    handle_type_iii,
    _to_attr_name,
    _dedupe,
    _get_key_cols,
)


# ---------------------------------------------------------------------------
# Helpers: minimal FeatureSet factories
# ---------------------------------------------------------------------------

def _make_features(
    columns: list[str],
    data: dict | None = None,
    raw_columns: list[str] | None = None,
    remarks_cols: list[str] | None = None,
    n_rows: int = 5,
) -> FeatureSet:
    """Build a minimal FeatureSet from column names and optional data dict."""
    if data is None:
        data = {col: [1.0, 2.0, 3.0] for col in columns}

    df = pd.DataFrame(data)
    raw = raw_columns if raw_columns is not None else columns

    return FeatureSet(
        df=df,
        header_strings=raw,
        time_cols_in_headers=[],
        time_cols_in_first_col=False,
        time_in_data_body=False,
        n_text_cols=0,
        n_numeric_cols=len(columns),
        leading_text_col_count=0,
        remarks_cols=remarks_cols if remarks_cols is not None else [],
        raw_columns=raw,
        headerless=False,
        n_rows=n_rows,
    )


def _make_meta_schema(
    subject_cols: list[str] | None = None,
    value_cols: list[str] | None = None,
    time_cols: list[str] | None = None,
    time_location: str = "headers",
    metadata_cols: list[str] | None = None,
) -> dict:
    """Build a minimal meta_schema dict."""
    schema = {
        "primary_subject_columns": subject_cols or [],
        "value_columns": value_cols or [],
        "metadata_columns": metadata_cols or [],
        "time_dimension": {
            "columns": time_cols or [],
            "location": time_location,
        },
    }
    return schema


# ---------------------------------------------------------------------------
# handle_type_i tests
# ---------------------------------------------------------------------------

class TestHandleTypeI(unittest.TestCase):

    def _run(self, subject_cols=None, value_cols=None):
        features = _make_features(
            columns=(subject_cols or []) + (value_cols or []),
        )
        meta = _make_meta_schema(
            subject_cols=subject_cols,
            value_cols=value_cols,
        )
        return handle_type_i(features, meta)

    def test_required_top_level_keys(self):
        result = self._run(subject_cols=["Country"], value_cols=["GDP", "Pop"])
        for key in ("entity_types", "relation_types", "attribute_mapping", "extraction_rules"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_relation_types_always_has_attribute(self):
        result = self._run(subject_cols=["Country"], value_cols=["GDP"])
        self.assertIn("HAS_ATTRIBUTE", result["relation_types"])

    def test_entity_types_derived_from_subject_col(self):
        result = self._run(subject_cols=["Country"], value_cols=["GDP"])
        # "Country" → PascalCase → "Country"
        self.assertEqual(result["entity_types"], ["Country"])

    def test_entity_types_fallback_when_no_subject_cols(self):
        result = self._run(subject_cols=[], value_cols=["GDP", "Pop"])
        self.assertEqual(result["entity_types"], ["Entity"])

    def test_attribute_mapping_keys_match_value_cols(self):
        value_cols = ["GDP", "Population", "Area"]
        result = self._run(subject_cols=["Country"], value_cols=value_cols)
        self.assertEqual(set(result["attribute_mapping"].keys()), set(value_cols))

    def test_attribute_mapping_values_are_strings(self):
        result = self._run(subject_cols=["Name"], value_cols=["Score (pct)", "Total"])
        for k, v in result["attribute_mapping"].items():
            self.assertIsInstance(v, str, f"Attribute name for '{k}' must be a string")

    def test_extraction_rules_has_expected_subkeys(self):
        result = self._run(subject_cols=["Name"], value_cols=["Age"])
        rules = result["extraction_rules"]
        for subkey in ("subject_extraction", "value_extraction", "time_handling", "hierarchy"):
            self.assertIn(subkey, rules)

    def test_no_mutation_of_meta_schema(self):
        meta = _make_meta_schema(subject_cols=["Country"], value_cols=["GDP"])
        original_subject = list(meta["primary_subject_columns"])
        features = _make_features(columns=["Country", "GDP"])
        handle_type_i(features, meta)
        self.assertEqual(meta["primary_subject_columns"], original_subject)

    def test_chinese_subject_col_uses_lookup(self):
        result = self._run(subject_cols=["疾病"], value_cols=["Count"])
        # "疾病" maps to "Disease" via _CN_TO_EN
        self.assertIn("Disease", result["entity_types"])

    def test_multiple_subject_cols_deduplicated(self):
        result = self._run(subject_cols=["Item", "Item"], value_cols=["Val"])
        self.assertEqual(len(result["entity_types"]), 1)


# ---------------------------------------------------------------------------
# handle_type_ii tests
# ---------------------------------------------------------------------------

class TestHandleTypeII(unittest.TestCase):

    def _make_type_ii_features(self, columns, data=None, time_col_names=None):
        """Build FeatureSet with optional time columns flagged."""
        if data is None:
            data = {col: [1.0, 2.0, 3.0] for col in columns}
        df = pd.DataFrame(data)
        raw = columns
        return FeatureSet(
            df=df,
            header_strings=raw,
            time_cols_in_headers=time_col_names or [],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=1,
            n_numeric_cols=len(columns) - 1,
            leading_text_col_count=1,
            remarks_cols=[],
            raw_columns=raw,
            headerless=False,
            n_rows=len(next(iter(data.values()))),
        )

    def test_required_top_level_keys(self):
        features = self._make_type_ii_features(
            ["Country", "2020", "2021"],
            time_col_names=["2020", "2021"],
        )
        meta = _make_meta_schema(
            subject_cols=["Country"],
            value_cols=["2020", "2021"],
            time_cols=["2020", "2021"],
        )
        result = handle_type_ii(features, meta)
        for key in ("entity_types", "relation_types", "time_parsing_rules", "extraction_rules"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_relation_types_contains_has_metric(self):
        features = self._make_type_ii_features(["Country", "2020"])
        meta = _make_meta_schema(subject_cols=["Country"], value_cols=["2020"])
        result = handle_type_ii(features, meta)
        self.assertIn("HAS_METRIC", result["relation_types"])

    def test_relation_name_budget_keyword(self):
        features = self._make_type_ii_features(["Dept", "2022-23预算"])
        meta = _make_meta_schema(
            subject_cols=["Dept"],
            value_cols=["2022-23预算"],
            time_cols=["2022-23预算"],
        )
        result = handle_type_ii(features, meta)
        # budget keyword triggers HAS_BUDGET
        self.assertIn("HAS_BUDGET", result["relation_types"])

    def test_entity_types_derived_from_subject_col(self):
        features = self._make_type_ii_features(
            ["Country", "2020", "2021"],
            time_col_names=["2020", "2021"],
        )
        meta = _make_meta_schema(
            subject_cols=["Country"],
            value_cols=["2020", "2021"],
        )
        result = handle_type_ii(features, meta)
        self.assertTrue(len(result["entity_types"]) > 0)

    def test_transposed_layout(self):
        # Transposed: first col = year, rest = metrics
        data = {
            "Year": ["2020", "2021", "2022"],
            "GDP": [100.0, 110.0, 120.0],
            "Pop": [7.0, 7.1, 7.2],
        }
        df = pd.DataFrame(data)
        features = FeatureSet(
            df=df,
            header_strings=["Year", "GDP", "Pop"],
            time_cols_in_headers=[],
            time_cols_in_first_col=True,
            time_in_data_body=False,
            n_text_cols=1,
            n_numeric_cols=2,
            leading_text_col_count=1,
            remarks_cols=[],
            raw_columns=["Year", "GDP", "Pop"],
            headerless=False,
            n_rows=3,
        )
        meta = _make_meta_schema(
            subject_cols=[],
            value_cols=["GDP", "Pop"],
            time_cols=[],
            time_location="rows",
        )
        result = handle_type_ii(features, meta)
        # Transposed should include pivot_instruction
        self.assertIn("pivot_instruction", result)
        self.assertIn("rows=time_periods", result["pivot_instruction"])

    def test_non_transposed_has_no_pivot_instruction(self):
        features = self._make_type_ii_features(
            ["Country", "2020", "2021"],
            time_col_names=["2020", "2021"],
        )
        meta = _make_meta_schema(
            subject_cols=["Country"],
            value_cols=["2020", "2021"],
            time_cols=["2020", "2021"],
            time_location="headers",
        )
        result = handle_type_ii(features, meta)
        self.assertNotIn("pivot_instruction", result)

    def test_extraction_rules_has_expected_subkeys(self):
        features = self._make_type_ii_features(["Region", "2020"])
        meta = _make_meta_schema(subject_cols=["Region"], value_cols=["2020"])
        result = handle_type_ii(features, meta)
        for subkey in ("subject_extraction", "value_extraction", "time_handling"):
            self.assertIn(subkey, result["extraction_rules"])


# ---------------------------------------------------------------------------
# handle_type_iii tests
# ---------------------------------------------------------------------------

class TestHandleTypeIII(unittest.TestCase):

    def _make_type_iii_features(self, key_cols, value_cols):
        """Build FeatureSet suitable for Type III (composite key + numeric values)."""
        text_data = {col: ["A", "B", "C"] for col in key_cols}
        num_data = {col: [1.0, 2.0, 3.0] for col in value_cols}
        data = {**text_data, **num_data}
        df = pd.DataFrame(data)
        raw = key_cols + value_cols
        return FeatureSet(
            df=df,
            header_strings=raw,
            time_cols_in_headers=[],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=len(key_cols),
            n_numeric_cols=len(value_cols),
            leading_text_col_count=len(key_cols),
            remarks_cols=[],
            raw_columns=raw,
            headerless=False,
            n_rows=3,
        )

    def test_required_top_level_keys(self):
        features = self._make_type_iii_features(
            key_cols=["Category", "Sub"],
            value_cols=["2021", "2022"],
        )
        meta = _make_meta_schema(
            subject_cols=["Category", "Sub"],
            value_cols=["2021", "2022"],
        )
        result = handle_type_iii(features, meta)
        for key in (
            "entity_types", "relation_types", "hierarchy_relations",
            "value_mapping", "extraction_rules",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_relation_types_contains_hierarchy_and_value(self):
        features = self._make_type_iii_features(["Cat", "Sub"], ["2020"])
        meta = _make_meta_schema(subject_cols=["Cat", "Sub"], value_cols=["2020"])
        result = handle_type_iii(features, meta)
        self.assertIn("HAS_SUB_ITEM", result["relation_types"])
        self.assertIn("HAS_VALUE", result["relation_types"])

    def test_hierarchy_relations_built_between_consecutive_key_levels(self):
        features = self._make_type_iii_features(
            key_cols=["Level1", "Level2", "Level3"],
            value_cols=["2021"],
        )
        meta = _make_meta_schema(
            subject_cols=["Level1", "Level2", "Level3"],
            value_cols=["2021"],
        )
        result = handle_type_iii(features, meta)
        hr = result["hierarchy_relations"]
        # 3 key cols → 2 parent-child relations
        self.assertEqual(len(hr), 2)
        self.assertEqual(hr[0]["relation"], "HAS_SUB_ITEM")
        self.assertEqual(hr[1]["relation"], "HAS_SUB_ITEM")

    def test_value_mapping_year_columns(self):
        features = self._make_type_iii_features(["Cat"], ["2021", "2022"])
        meta = _make_meta_schema(subject_cols=["Cat"], value_cols=["2021", "2022"])
        result = handle_type_iii(features, meta)
        vm = result["value_mapping"]
        self.assertEqual(vm["2021"]["type"], "year_value")
        self.assertEqual(vm["2021"]["year"], "2021")
        self.assertEqual(vm["2022"]["type"], "year_value")

    def test_value_mapping_non_year_numeric_column(self):
        features = self._make_type_iii_features(["Cat"], ["Total"])
        meta = _make_meta_schema(subject_cols=["Cat"], value_cols=["Total"])
        result = handle_type_iii(features, meta)
        vm = result["value_mapping"]
        self.assertEqual(vm["Total"]["type"], "numeric_attribute")

    def test_entity_types_derived_from_key_cols(self):
        features = self._make_type_iii_features(["疾病", "病种"], ["2020"])
        meta = _make_meta_schema(subject_cols=["疾病", "病种"], value_cols=["2020"])
        result = handle_type_iii(features, meta)
        # Both "疾病" and "病种" map to "Disease"; dedupe → just one entry
        self.assertIn("Disease", result["entity_types"])
        self.assertEqual(len(result["entity_types"]), 1)

    def test_remarks_handling_with_metadata_cols(self):
        features = self._make_type_iii_features(["Cat"], ["2020"])
        meta = _make_meta_schema(
            subject_cols=["Cat"],
            value_cols=["2020"],
            metadata_cols=["备注"],
        )
        result = handle_type_iii(features, meta)
        rules = result["extraction_rules"]
        self.assertIn("备注", rules["remarks"])

    def test_no_key_cols_fallback(self):
        # If no leading text columns, entity_types falls back to ["Category"]
        data = {"2021": [1.0], "2022": [2.0]}
        df = pd.DataFrame(data)
        features = FeatureSet(
            df=df,
            header_strings=["2021", "2022"],
            time_cols_in_headers=["2021", "2022"],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=0,
            n_numeric_cols=2,
            leading_text_col_count=0,
            remarks_cols=[],
            raw_columns=["2021", "2022"],
            headerless=False,
            n_rows=1,
        )
        meta = _make_meta_schema(subject_cols=[], value_cols=["2021", "2022"])
        result = handle_type_iii(features, meta)
        self.assertEqual(result["entity_types"], ["Category"])

    def test_extraction_rules_has_expected_subkeys(self):
        features = self._make_type_iii_features(["Cat", "Sub"], ["2020"])
        meta = _make_meta_schema(subject_cols=["Cat", "Sub"], value_cols=["2020"])
        result = handle_type_iii(features, meta)
        rules = result["extraction_rules"]
        for subkey in ("subject_extraction", "value_extraction", "time_handling", "hierarchy", "remarks"):
            self.assertIn(subkey, rules)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestToAttrName(unittest.TestCase):

    def test_plain_ascii(self):
        self.assertEqual(_to_attr_name("GDP"), "GDP")

    def test_spaces_become_underscores(self):
        result = _to_attr_name("Total Population")
        self.assertIn("Total", result)
        self.assertIn("Population", result)

    def test_strips_parentheses(self):
        result = _to_attr_name("Score (pct)")
        self.assertNotIn("(", result)
        self.assertNotIn(")", result)

    def test_newlines_become_underscores(self):
        result = _to_attr_name("2022\n实际")
        self.assertNotIn("\n", result)

    def test_empty_string_returns_value(self):
        result = _to_attr_name("")
        self.assertEqual(result, "value")

    def test_chinese_preserved(self):
        result = _to_attr_name("疾病数量")
        self.assertIn("疾病", result)

    def test_multiple_underscores_collapsed(self):
        result = _to_attr_name("A  B")
        self.assertNotIn("__", result)


class TestDedupe(unittest.TestCase):

    def test_removes_duplicates(self):
        self.assertEqual(_dedupe(["a", "b", "a", "c"]), ["a", "b", "c"])

    def test_preserves_order(self):
        self.assertEqual(_dedupe(["c", "b", "a"]), ["c", "b", "a"])

    def test_empty_list(self):
        self.assertEqual(_dedupe([]), [])

    def test_no_duplicates_unchanged(self):
        lst = ["x", "y", "z"]
        self.assertEqual(_dedupe(lst), lst)


class TestGetKeyCols(unittest.TestCase):

    def test_returns_leading_text_cols(self):
        data = {"Name": ["Alice", "Bob"], "Age": [30, 40], "Score": [80.0, 90.0]}
        df = pd.DataFrame(data)
        features = FeatureSet(
            df=df,
            header_strings=["Name", "Age", "Score"],
            time_cols_in_headers=[],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=1,
            n_numeric_cols=2,
            leading_text_col_count=1,
            remarks_cols=[],
            raw_columns=["Name", "Age", "Score"],
            headerless=False,
            n_rows=2,
        )
        key_cols = _get_key_cols(features)
        self.assertEqual(key_cols, ["Name"])

    def test_returns_empty_for_all_numeric(self):
        data = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
        df = pd.DataFrame(data)
        features = FeatureSet(
            df=df,
            header_strings=["A", "B"],
            time_cols_in_headers=[],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=0,
            n_numeric_cols=2,
            leading_text_col_count=0,
            remarks_cols=[],
            raw_columns=["A", "B"],
            headerless=False,
            n_rows=2,
        )
        key_cols = _get_key_cols(features)
        self.assertEqual(key_cols, [])

    def test_multiple_text_cols_before_numeric(self):
        data = {
            "Cat": ["X", "Y", "Z"],
            "Sub": ["a", "b", "c"],
            "Val": [1.0, 2.0, 3.0],
        }
        df = pd.DataFrame(data)
        features = FeatureSet(
            df=df,
            header_strings=["Cat", "Sub", "Val"],
            time_cols_in_headers=[],
            time_cols_in_first_col=False,
            time_in_data_body=False,
            n_text_cols=2,
            n_numeric_cols=1,
            leading_text_col_count=2,
            remarks_cols=[],
            raw_columns=["Cat", "Sub", "Val"],
            headerless=False,
            n_rows=3,
        )
        key_cols = _get_key_cols(features)
        self.assertEqual(key_cols, ["Cat", "Sub"])


if __name__ == "__main__":
    unittest.main()
