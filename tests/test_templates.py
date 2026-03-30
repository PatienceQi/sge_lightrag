"""
test_templates.py — Unit tests for stage2/templates.py.

Tests entity/relation name derivation functions and per-type template
generators (type_i_templates, type_ii_templates, type_iii_templates).

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python -m pytest tests/test_templates.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2.templates import (
    derive_entity_name,
    detect_relation_name,
    type_i_templates,
    type_ii_templates,
    type_iii_templates,
)
from stage2.header_parser import ParsedHeader


# ---------------------------------------------------------------------------
# derive_entity_name tests
# ---------------------------------------------------------------------------

class TestDeriveEntityName(unittest.TestCase):

    # --- Direct Chinese→English lookup ---

    def test_disease_col_maps_to_disease(self):
        self.assertEqual(derive_entity_name("疾病"), "Disease")

    def test_program_col_maps_to_policy_program(self):
        self.assertEqual(derive_entity_name("纲领"), "Policy_Program")

    def test_department_col_maps_to_department(self):
        self.assertEqual(derive_entity_name("部门"), "Department")

    def test_region_col_maps_to_region(self):
        self.assertEqual(derive_entity_name("地区"), "Region")

    def test_year_col_maps_to_year(self):
        self.assertEqual(derive_entity_name("年份"), "Year")

    # --- Partial match (keyword contained in longer name) ---

    def test_partial_match_disease_keyword(self):
        result = derive_entity_name("主要疾病分类")
        self.assertEqual(result, "Disease")

    def test_partial_match_service_takes_priority_over_department(self):
        # "服务" appears before "部门" in _CN_TO_EN iteration order → Service wins
        result = derive_entity_name("服务部门")
        self.assertEqual(result, "Service")

    # --- ASCII fallback ---

    def test_ascii_single_word_title_cased(self):
        result = derive_entity_name("Country")
        self.assertEqual(result, "Country")

    def test_ascii_multiple_words_joined_with_underscore(self):
        result = derive_entity_name("life expectancy")
        self.assertIn("Life", result)
        self.assertIn("Expectancy", result)

    def test_ascii_preserves_capitalisation_logic(self):
        result = derive_entity_name("gdp growth")
        self.assertEqual(result, "Gdp_Growth")

    # --- Edge cases ---

    def test_empty_string_returns_entity(self):
        result = derive_entity_name("")
        self.assertEqual(result, "Entity")

    def test_whitespace_only_returns_entity(self):
        result = derive_entity_name("   ")
        self.assertEqual(result, "Entity")

    def test_numeric_only_returns_safe_fallback(self):
        # Pure digits → no ASCII words, no Chinese keywords → safe fallback
        result = derive_entity_name("2021")
        # Should not crash; returns some non-empty string or "Entity"
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_strips_leading_trailing_whitespace(self):
        result = derive_entity_name("  Country  ")
        self.assertEqual(result, "Country")

    def test_does_not_return_empty_string(self):
        for name in ["", " ", "---", "!!!"]:
            result = derive_entity_name(name)
            self.assertTrue(len(result) > 0, f"Got empty string for input: {name!r}")


# ---------------------------------------------------------------------------
# detect_relation_name tests
# ---------------------------------------------------------------------------

class TestDetectRelationName(unittest.TestCase):

    def test_default_is_has_value(self):
        result = detect_relation_name(["2020", "2021"], ["GDP", "Pop"])
        self.assertEqual(result, "HAS_VALUE")

    def test_budget_keyword_in_time_cols_triggers_has_budget(self):
        result = detect_relation_name(["2022-23预算", "2023-24预算"], [])
        self.assertEqual(result, "HAS_BUDGET")

    def test_budget_english_keyword_triggers_has_budget(self):
        result = detect_relation_name([], ["budget_2022", "budget_2023"])
        self.assertEqual(result, "HAS_BUDGET")

    def test_expenditure_keyword_triggers_has_budget(self):
        result = detect_relation_name([], ["total_expenditure"])
        self.assertEqual(result, "HAS_BUDGET")

    def test_allocation_keyword_triggers_has_budget(self):
        result = detect_relation_name([], ["allocation_2021"])
        self.assertEqual(result, "HAS_BUDGET")

    def test_case_insensitive_budget_match(self):
        result = detect_relation_name([], ["BUDGET_2022"])
        self.assertEqual(result, "HAS_BUDGET")

    def test_empty_cols_defaults_to_has_value(self):
        result = detect_relation_name([], [])
        self.assertEqual(result, "HAS_VALUE")

    def test_partial_budget_keyword_in_value_cols(self):
        result = detect_relation_name(["2020"], ["拨款总额"])
        self.assertEqual(result, "HAS_BUDGET")


# ---------------------------------------------------------------------------
# type_i_templates tests
# ---------------------------------------------------------------------------

class TestTypeITemplates(unittest.TestCase):

    def _run(self, subject="Country", value_cols=None, entity_name="Country"):
        value_cols = value_cols or ["GDP", "Pop"]
        return type_i_templates(subject, value_cols, entity_name)

    def test_returns_three_keys(self):
        result = self._run()
        for key in ("entity_extraction_template", "relation_extraction_template", "extraction_constraints"):
            self.assertIn(key, result)

    def test_entity_template_mentions_subject_col(self):
        result = self._run(subject="Disease", entity_name="Disease")
        self.assertIn("Disease", result["entity_extraction_template"])

    def test_relation_template_mentions_has_attribute(self):
        result = self._run()
        self.assertIn("HAS_ATTRIBUTE", result["relation_extraction_template"])

    def test_extraction_constraints_is_list(self):
        result = self._run()
        self.assertIsInstance(result["extraction_constraints"], list)
        self.assertGreater(len(result["extraction_constraints"]), 0)

    def test_value_cols_truncated_to_four_in_template(self):
        many_cols = [f"Col{i}" for i in range(10)]
        result = type_i_templates("Name", many_cols, "Item")
        self.assertIn("and 6 more", result["relation_extraction_template"])

    def test_exactly_four_value_cols_no_more_annotation(self):
        four_cols = ["A", "B", "C", "D"]
        result = type_i_templates("Name", four_cols, "Item")
        self.assertNotIn("more", result["relation_extraction_template"])

    def test_entity_name_in_constraints(self):
        result = self._run(entity_name="Patient")
        constraints_text = " ".join(result["extraction_constraints"])
        self.assertIn("Patient", constraints_text)


# ---------------------------------------------------------------------------
# type_ii_templates tests (non-transposed)
# ---------------------------------------------------------------------------

def _make_parsed_header(raw, year, status=None, unit=None, is_time=True) -> ParsedHeader:
    return ParsedHeader(
        raw=raw,
        year=year,
        status=status,
        unit=unit,
        label=None,
        is_time_column=is_time,
    )


class TestTypeIITemplatesNonTransposed(unittest.TestCase):

    def _run(self, subject="Country", time_cols=None, entity_name="Country",
             relation_name="HAS_VALUE"):
        time_cols = time_cols or ["2020", "2021"]
        parsed = [_make_parsed_header(c, c) for c in time_cols]
        return type_ii_templates(
            subject_col=subject,
            time_cols=time_cols,
            parsed_headers=parsed,
            entity_name=entity_name,
            relation_name=relation_name,
            transposed=False,
        )

    def test_returns_three_keys(self):
        result = self._run()
        for key in ("entity_extraction_template", "relation_extraction_template", "extraction_constraints"):
            self.assertIn(key, result)

    def test_entity_template_mentions_subject_col(self):
        result = self._run(subject="Region")
        self.assertIn("Region", result["entity_extraction_template"])

    def test_relation_template_mentions_relation_name(self):
        result = self._run(relation_name="HAS_BUDGET")
        self.assertIn("HAS_BUDGET", result["relation_extraction_template"])

    def test_constraints_mention_entity_name(self):
        result = self._run(entity_name="Nation")
        text = " ".join(result["extraction_constraints"])
        self.assertIn("Nation", text)

    def test_constraints_is_list(self):
        result = self._run()
        self.assertIsInstance(result["extraction_constraints"], list)


class TestTypeIITemplatesTransposed(unittest.TestCase):

    def _run(self, metric_cols=None, entity_name="Metric", relation_name="HAS_VALUE"):
        metric_cols = metric_cols or ["GDP", "Pop"]
        return type_ii_templates(
            subject_col="Year",
            time_cols=metric_cols,
            parsed_headers=[],
            entity_name=entity_name,
            relation_name=relation_name,
            transposed=True,
        )

    def test_transposed_flag_changes_entity_template(self):
        result = self._run()
        # Transposed template describes rows-as-time-periods pattern
        self.assertIn("TRANSPOSED", result["extraction_constraints"][0])

    def test_transposed_entity_template_mentions_metric(self):
        result = self._run(entity_name="Indicator")
        self.assertIn("Indicator", result["entity_extraction_template"])

    def test_transposed_constraints_warn_about_critical_extraction(self):
        result = self._run()
        self.assertIn("CRITICAL", result["extraction_constraints"][0])

    def test_transposed_relation_template_references_year_attribute(self):
        result = self._run(relation_name="HAS_VALUE")
        self.assertIn("year", result["relation_extraction_template"])


# ---------------------------------------------------------------------------
# type_iii_templates tests
# ---------------------------------------------------------------------------

class TestTypeIIITemplates(unittest.TestCase):

    def _run(self, key_cols=None, value_cols=None, remarks_cols=None, entity_names=None):
        key_cols = key_cols or ["Category", "Subcategory"]
        value_cols = value_cols or ["2021", "2022"]
        remarks_cols = remarks_cols or []
        entity_names = entity_names or ["Category", "Subcategory"]
        return type_iii_templates(key_cols, value_cols, remarks_cols, entity_names)

    def test_returns_three_keys(self):
        result = self._run()
        for key in ("entity_extraction_template", "relation_extraction_template", "extraction_constraints"):
            self.assertIn(key, result)

    def test_entity_template_mentions_top_level_key(self):
        result = self._run(key_cols=["Program", "Item"])
        self.assertIn("Program", result["entity_extraction_template"])

    def test_relation_template_mentions_has_sub_item(self):
        result = self._run()
        self.assertIn("HAS_SUB_ITEM", result["relation_extraction_template"])

    def test_relation_template_mentions_has_value(self):
        result = self._run()
        self.assertIn("HAS_VALUE", result["relation_extraction_template"])

    def test_remarks_cols_mentioned_in_relation_template(self):
        result = self._run(remarks_cols=["备注"])
        self.assertIn("备注", result["relation_extraction_template"])

    def test_no_remarks_no_remarks_in_template(self):
        result = self._run(remarks_cols=[])
        self.assertNotIn("Attach remarks", result["relation_extraction_template"])

    def test_constraints_is_list(self):
        result = self._run()
        self.assertIsInstance(result["extraction_constraints"], list)
        self.assertGreater(len(result["extraction_constraints"]), 0)

    def test_constraints_mention_top_level_key_col(self):
        result = self._run(key_cols=["纲领", "项目"], entity_names=["Policy_Program", "Program_Item"])
        text = " ".join(result["extraction_constraints"])
        self.assertIn("纲领", text)

    def test_three_level_hierarchy_produces_leaf_constraint(self):
        result = self._run(
            key_cols=["L1", "L2", "L3"],
            entity_names=["Top", "Mid", "Leaf"],
        )
        text = " ".join(result["extraction_constraints"])
        self.assertIn("Leaf", text)

    def test_value_cols_truncated_to_four(self):
        many_cols = [str(y) for y in range(2015, 2025)]
        result = type_iii_templates(
            key_cols=["Cat"],
            value_cols=many_cols,
            remarks_cols=[],
            entity_names=["Category"],
        )
        self.assertIn("and 6 more", result["relation_extraction_template"])

    def test_single_key_col_no_second_level_constraint(self):
        result = type_iii_templates(
            key_cols=["OnlyKey"],
            value_cols=["2020"],
            remarks_cols=[],
            entity_names=["OnlyKey"],
        )
        text = " ".join(result["extraction_constraints"])
        # No second-level or leaf constraints for a single key column
        self.assertNotIn("Second-level key", text)

    def test_fallback_entity_names_for_short_list(self):
        # entity_names shorter than key_cols — sub/leaf fall back to "Item"/"Item"
        result = type_iii_templates(
            key_cols=["A", "B", "C"],
            value_cols=["2020"],
            remarks_cols=[],
            entity_names=["Alpha"],  # only one name provided
        )
        # Should not raise; templates should include the fallback label
        self.assertIn("Item", result["entity_extraction_template"])


if __name__ == "__main__":
    unittest.main()
