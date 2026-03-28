"""
test_compact.py — Unit tests for compact_representation module.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python3 -m unittest discover -s tests -p "test_compact*" -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage3.compact_representation import (
    COMPACT_THRESHOLD,
    MAX_YEARS_PER_CHUNK,
    should_use_compact,
    compact_serialize_type_ii,
    build_compact_system_prompt,
)


# ---------------------------------------------------------------------------
# should_use_compact
# ---------------------------------------------------------------------------

class TestShouldUseCompact(unittest.TestCase):

    def _schema(self, table_type="Time-Series-Matrix", baseline=False):
        return {
            "table_type": table_type,
            "use_baseline_mode": baseline,
        }

    def test_large_type_ii_returns_true(self):
        schema = self._schema("Time-Series-Matrix")
        self.assertTrue(should_use_compact(schema, COMPACT_THRESHOLD + 1))

    def test_exactly_threshold_returns_false(self):
        """n_rows == COMPACT_THRESHOLD is NOT > threshold."""
        schema = self._schema("Time-Series-Matrix")
        self.assertFalse(should_use_compact(schema, COMPACT_THRESHOLD))

    def test_small_table_returns_false(self):
        schema = self._schema("Time-Series-Matrix")
        self.assertFalse(should_use_compact(schema, 50))

    def test_type_i_returns_false(self):
        schema = self._schema("Flat-Entity")
        self.assertFalse(should_use_compact(schema, 500))

    def test_type_iii_returns_false(self):
        schema = self._schema("Hierarchical-Hybrid")
        self.assertFalse(should_use_compact(schema, 500))

    def test_baseline_mode_returns_false(self):
        schema = self._schema("Time-Series-Matrix", baseline=True)
        self.assertFalse(should_use_compact(schema, 500))


# ---------------------------------------------------------------------------
# compact_serialize_type_ii
# ---------------------------------------------------------------------------

def _make_wb_df(n_countries=5, years=None):
    """Build a minimal World-Bank-style DataFrame for testing."""
    if years is None:
        years = [str(y) for y in range(2000, 2006)]
    import random
    rows = []
    codes = ["CHN", "IND", "USA", "GBR", "DEU", "FRA", "JPN", "BRA", "CAN", "AUS"]
    names = ["China", "India", "United States", "United Kingdom", "Germany",
             "France", "Japan", "Brazil", "Canada", "Australia"]
    for i in range(n_countries):
        row = {
            "Country Code": codes[i % len(codes)],
            "Country Name": names[i % len(names)],
            "Indicator Name": "UNDER5_MORTALITY_RATE",
        }
        for yr in years:
            row[yr] = round(random.uniform(5.0, 50.0), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_schema(years=None):
    """Build a minimal extraction schema matching the test DataFrame."""
    if years is None:
        years = [str(y) for y in range(2000, 2006)]
    column_roles = {
        "Country Code": "subject",
        "Country Name": "metadata",
        "Indicator Name": "metadata",
    }
    for yr in years:
        column_roles[yr] = "time_value"
    return {
        "table_type": "Time-Series-Matrix",
        "column_roles": column_roles,
        "unit": "per 1,000 live births",
        "entity_types": ["Country_Code", "StatValue"],
        "relation_types": ["HAS_VALUE"],
    }


class TestCompactSerializeTypeII(unittest.TestCase):

    def setUp(self):
        self.years = [str(y) for y in range(2000, 2006)]
        self.df = _make_wb_df(n_countries=5, years=self.years)
        self.schema = _make_schema(self.years)

    def test_returns_one_chunk_per_entity(self):
        chunks = compact_serialize_type_ii(self.df, self.schema)
        self.assertEqual(len(chunks), 5)

    def test_chunk_has_entity_prefix(self):
        chunks = compact_serialize_type_ii(self.df, self.schema)
        for chunk in chunks:
            self.assertIn("Entity:", chunk)

    def test_chunk_has_timeseries_line(self):
        chunks = compact_serialize_type_ii(self.df, self.schema)
        for chunk in chunks:
            self.assertIn("Timeseries", chunk)

    def test_timeseries_format_year_equals_value(self):
        """Timeseries data must use 'year=value; year=value' format."""
        chunks = compact_serialize_type_ii(self.df, self.schema)
        import re
        ts_pattern = re.compile(r"\d{4}=[\d.]+")
        for chunk in chunks:
            ts_line = [l for l in chunk.split("\n") if l.startswith("Timeseries")]
            self.assertTrue(len(ts_line) == 1, f"Expected exactly one Timeseries line: {chunk}")
            self.assertRegex(ts_line[0], ts_pattern)

    def test_unit_appears_in_timeseries_header(self):
        chunks = compact_serialize_type_ii(self.df, self.schema)
        for chunk in chunks:
            self.assertIn("per 1,000 live births", chunk)

    def test_country_name_in_chunk(self):
        """Country Name metadata column should appear in the chunk header."""
        chunks = compact_serialize_type_ii(self.df, self.schema)
        # First row is CHN / China
        first_chunk = chunks[0]
        self.assertIn("Name:", first_chunk)

    def test_indicator_in_chunk(self):
        """Indicator Name should appear in the chunk."""
        chunks = compact_serialize_type_ii(self.df, self.schema)
        first_chunk = chunks[0]
        self.assertIn("Indicator:", first_chunk)

    def test_years_sorted_ascending(self):
        """Year-value pairs must be sorted chronologically."""
        chunks = compact_serialize_type_ii(self.df, self.schema)
        import re
        for chunk in chunks:
            ts_line = [l for l in chunk.split("\n") if l.startswith("Timeseries")][0]
            pairs = re.findall(r"(\d{4})=[\d.]+", ts_line)
            self.assertEqual(pairs, sorted(pairs))

    def test_no_nan_values(self):
        """Chunks must not contain NaN / nan strings."""
        chunks = compact_serialize_type_ii(self.df, self.schema)
        for chunk in chunks:
            self.assertNotIn("nan", chunk.lower())

    def test_truncation_at_max_years(self):
        """When more than MAX_YEARS_PER_CHUNK year columns exist, truncate to last N."""
        many_years = [str(y) for y in range(1950, 2026)]  # 76 years
        df = _make_wb_df(n_countries=3, years=many_years)
        schema = _make_schema(many_years)
        chunks = compact_serialize_type_ii(df, schema)
        import re
        for chunk in chunks:
            ts_line = [l for l in chunk.split("\n") if l.startswith("Timeseries")][0]
            pairs = re.findall(r"\d{4}=[\d.]+", ts_line)
            self.assertLessEqual(len(pairs), MAX_YEARS_PER_CHUNK)

    def test_missing_values_skipped(self):
        """Rows with NaN year values should not produce year=nan entries."""
        df = self.df.copy()
        df.loc[0, "2003"] = float("nan")
        chunks = compact_serialize_type_ii(df, self.schema)
        first_chunk = chunks[0]
        self.assertNotIn("2003=", first_chunk)


# ---------------------------------------------------------------------------
# build_compact_system_prompt
# ---------------------------------------------------------------------------

class TestBuildCompactSystemPrompt(unittest.TestCase):

    def setUp(self):
        self.schema = {
            "entity_types": ["Country_Code", "StatValue"],
            "relation_types": ["HAS_VALUE"],
        }

    def test_returns_nonempty_string(self):
        prompt = build_compact_system_prompt(self.schema)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_statvalue_removed_from_entity_types(self):
        """Compact mode should NOT instruct LightRAG to create StatValue nodes."""
        prompt = build_compact_system_prompt(self.schema)
        self.assertNotIn("StatValue", prompt)

    def test_country_code_in_prompt(self):
        prompt = build_compact_system_prompt(self.schema)
        self.assertIn("Country_Code", prompt)

    def test_has_value_relation_in_prompt(self):
        prompt = build_compact_system_prompt(self.schema)
        self.assertIn("HAS_VALUE", prompt)

    def test_one_entity_per_country_instruction(self):
        """Prompt must instruct LightRAG to create one node per country."""
        prompt = build_compact_system_prompt(self.schema)
        self.assertIn("ONE", prompt)

    def test_language_override(self):
        prompt_en = build_compact_system_prompt(self.schema, language="English")
        self.assertIn("English", prompt_en)

    def test_no_statvalue_when_not_in_types(self):
        """Even if StatValue not in entity_types, output should not mention it."""
        schema = {
            "entity_types": ["Country_Code"],
            "relation_types": ["HAS_VALUE"],
        }
        prompt = build_compact_system_prompt(schema)
        self.assertNotIn("StatValue", prompt)


# ---------------------------------------------------------------------------
# Integration: serializer uses compact_representation for large Type-II
# ---------------------------------------------------------------------------

class TestSerializerUsesCompact(unittest.TestCase):
    """
    Verify that stage3.serializer.serialize_csv dispatches to compact mode
    when the in-memory flag _n_rows > COMPACT_THRESHOLD.

    We test the compact_serialize_type_ii function directly (not via
    serialize_csv, which requires a real CSV path) to avoid filesystem
    fixtures, but verify that should_use_compact agrees with the schema.
    """

    def test_compact_mode_triggered_for_large_type_ii(self):
        schema = {
            "table_type": "Time-Series-Matrix",
            "use_baseline_mode": False,
            "_n_rows": COMPACT_THRESHOLD + 50,
        }
        from stage3.compact_representation import should_use_compact
        self.assertTrue(should_use_compact(schema, COMPACT_THRESHOLD + 50))

    def test_compact_mode_not_triggered_for_small_table(self):
        schema = {
            "table_type": "Time-Series-Matrix",
            "use_baseline_mode": False,
            "_n_rows": 30,
        }
        from stage3.compact_representation import should_use_compact
        self.assertFalse(should_use_compact(schema, 30))

    def test_compact_chunks_have_correct_structure(self):
        years = [str(y) for y in range(2000, 2010)]
        df = _make_wb_df(n_countries=10, years=years)
        schema = _make_schema(years)

        chunks = compact_serialize_type_ii(df, schema)
        self.assertEqual(len(chunks), 10)

        for chunk in chunks:
            lines = chunk.strip().split("\n")
            self.assertEqual(len(lines), 2, f"Expected 2 lines per chunk: {chunk!r}")
            self.assertTrue(lines[0].startswith("Entity:"))
            self.assertTrue(lines[1].startswith("Timeseries"))


if __name__ == "__main__":
    unittest.main()
