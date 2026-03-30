"""
test_preprocessor.py — Unit tests for stage1/preprocessor.py.

Tests encoding detection, World Bank metadata row skipping, and basic CSV
reading using tempfile-based test CSVs.

Run with:
    cd ~/Desktop/SGE/sge_lightrag
    python -m pytest tests/test_preprocessor.py -v
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1.features import _detect_encoding, _detect_skiprows
from stage1.preprocessor import (
    preprocess_csv,
    _is_title_row,
    _find_header_row,
    _forward_fill_merged_cells,
    _drop_empty_rows,
)


# ---------------------------------------------------------------------------
# Helpers: write temp CSV files with specific encodings
# ---------------------------------------------------------------------------

def _write_temp_csv(content: str, encoding: str, suffix: str = ".csv") -> str:
    """Write content to a temp file with the given encoding. Returns path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="sge_test_")
    os.close(fd)
    with open(path, "w", encoding=encoding) as fh:
        fh.write(content)
    return path


def _write_temp_csv_bytes(content: bytes, suffix: str = ".csv") -> str:
    """Write raw bytes to a temp file. Returns path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="sge_test_")
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Encoding detection tests (_detect_encoding)
# ---------------------------------------------------------------------------

class TestDetectEncoding(unittest.TestCase):

    def test_utf8_plain_detected(self):
        path = _write_temp_csv("name,age\nAlice,30\n", "utf-8")
        try:
            enc = _detect_encoding(path)
            # Plain UTF-8 → reported as "utf-8-sig" (the fallback)
            self.assertIn("utf-8", enc)
        finally:
            os.unlink(path)

    def test_utf8_bom_detected(self):
        # Write file with UTF-8 BOM manually
        content_bytes = b"\xef\xbb\xbfname,age\nAlice,30\n"
        path = _write_temp_csv_bytes(content_bytes)
        try:
            enc = _detect_encoding(path)
            self.assertEqual(enc, "utf-8-sig")
        finally:
            os.unlink(path)

    def test_utf16_le_bom_detected(self):
        content_bytes = b"\xff\xfe" + "name,age\n".encode("utf-16-le")
        path = _write_temp_csv_bytes(content_bytes)
        try:
            enc = _detect_encoding(path)
            self.assertEqual(enc, "utf-16-le")
        finally:
            os.unlink(path)

    def test_utf16_be_bom_detected(self):
        content_bytes = b"\xfe\xff" + "name,age\n".encode("utf-16-be")
        path = _write_temp_csv_bytes(content_bytes)
        try:
            enc = _detect_encoding(path)
            self.assertEqual(enc, "utf-16-be")
        finally:
            os.unlink(path)

    def test_gbk_file_detected(self):
        # GBK-encoded content (simplified Chinese)
        content = "名称,年龄\n张三,30\n"
        path = _write_temp_csv(content, "gbk")
        try:
            enc = _detect_encoding(path)
            # Should detect gbk or fall back gracefully
            self.assertIn(enc, ("gbk", "utf-8-sig", "big5hkscs"))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# World Bank metadata skip tests (via features._detect_skiprows)
# ---------------------------------------------------------------------------

class TestWorldBankMetadataSkip(unittest.TestCase):
    """
    Test that _detect_skiprows (in stage1.features) correctly identifies
    the 4-row metadata prefix in World Bank API CSV files.

    Note: preprocess_csv (in stage1.preprocessor) uses its own _find_header_row
    heuristic and does not call _detect_skiprows; the WB skip logic lives in
    features._read_csv which is the Stage 1 entry point.
    """

    # Real WB format: 4 metadata rows (0-3), header at row 4, data from row 5
    _WB_CSV = (
        '"Data Source","World Development Indicators","","",""\n'
        '"Last Updated Date","2024-01-01","","",""\n'
        '"","","","",""\n'
        '"","","","",""\n'
        '"Country Name","Country Code","Indicator Name","2020","2021"\n'
        '"China","CHN","Life expectancy at birth","76.4","76.9"\n'
        '"India","IND","Life expectancy at birth","69.7","67.2"\n'
    )

    def test_detect_skiprows_returns_4_for_world_bank(self):
        """_detect_skiprows should return 4 for a WB 'Data Source' header."""
        path = _write_temp_csv(self._WB_CSV, "utf-8")
        try:
            skip = _detect_skiprows(path, "utf-8")
            self.assertEqual(skip, 4)
        finally:
            os.unlink(path)

    def test_detect_skiprows_returns_0_for_plain_csv(self):
        path = _write_temp_csv("name,score\nAlice,90\n", "utf-8")
        try:
            skip = _detect_skiprows(path, "utf-8")
            self.assertEqual(skip, 0)
        finally:
            os.unlink(path)

    def test_world_bank_pandas_read_with_skiprows(self):
        """After skipping 4 rows, pandas should see 'Country Name' as the header."""
        path = _write_temp_csv(self._WB_CSV, "utf-8")
        try:
            import pandas as pd
            df = pd.read_csv(path, skiprows=4, encoding="utf-8")
            self.assertIn("Country Name", df.columns)
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Basic CSV reading tests
# ---------------------------------------------------------------------------

class TestBasicCSVReading(unittest.TestCase):

    _SIMPLE_CSV = "name,score,year\nAlice,90,2020\nBob,85,2021\n"

    def test_returns_dataframe_and_metadata(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            result = preprocess_csv(path)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            df, meta = result
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIsInstance(meta, dict)
        finally:
            os.unlink(path)

    def test_columns_preserved(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            df, _ = preprocess_csv(path)
            self.assertIn("name", df.columns)
            self.assertIn("score", df.columns)
            self.assertIn("year", df.columns)
        finally:
            os.unlink(path)

    def test_data_rows_correct_count(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            df, _ = preprocess_csv(path)
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_metadata_contains_encoding(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            _, meta = preprocess_csv(path)
            self.assertIn("original_encoding", meta)
            self.assertIn("utf-8", meta["original_encoding"])
        finally:
            os.unlink(path)

    def test_metadata_was_utf16_false_for_utf8(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            _, meta = preprocess_csv(path)
            self.assertFalse(meta["was_utf16"])
        finally:
            os.unlink(path)

    def test_empty_rows_dropped(self):
        csv_with_empty = "name,score\nAlice,90\n,\nBob,85\n"
        path = _write_temp_csv(csv_with_empty, "utf-8")
        try:
            df, _ = preprocess_csv(path, drop_empty=True)
            # The row with only empty/NaN values should be gone
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_no_strip_titles_rows_stripped_is_zero(self):
        # A standard CSV with strip_titles=False should report 0 stripped rows
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            df, meta = preprocess_csv(path, strip_titles=False)
            self.assertEqual(meta["rows_stripped"], 0)
        finally:
            os.unlink(path)

    def test_clean_shape_in_metadata(self):
        path = _write_temp_csv(self._SIMPLE_CSV, "utf-8")
        try:
            df, meta = preprocess_csv(path)
            self.assertEqual(meta["clean_shape"], df.shape)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Title row detection tests (_is_title_row)
# ---------------------------------------------------------------------------

class TestIsTitleRow(unittest.TestCase):

    def test_blank_row_is_title(self):
        row = pd.Series([None, None, None])
        self.assertTrue(_is_title_row(row, n_cols=3))

    def test_single_cell_row_is_title_when_sparse_enough(self):
        # 1 non-null out of 10 cols = 0.1 fill ratio < 0.3 threshold → title
        row = pd.Series(["Table 1: Summary"] + [None] * 9)
        self.assertTrue(_is_title_row(row, n_cols=10))

    def test_single_cell_row_not_title_when_few_cols(self):
        # 1 non-null out of 3 cols = 0.333 ≥ 0.3 threshold → NOT a title
        row = pd.Series(["Name", None, None])
        self.assertFalse(_is_title_row(row, n_cols=3))

    def test_dense_row_is_not_title(self):
        row = pd.Series(["Name", "Age", "Score"])
        self.assertFalse(_is_title_row(row, n_cols=3))

    def test_title_pattern_row_is_title_when_very_sparse(self):
        # 1 non-null out of 8 cols = 0.125 < 0.3 AND matches "表" pattern
        row = pd.Series(["表 1 统计数据"] + [None] * 7)
        self.assertTrue(_is_title_row(row, n_cols=8))

    def test_year_prefix_row_is_title_when_sparse(self):
        # 1 non-null out of 8 cols = 0.125 < 0.3 AND single-cell rule
        row = pd.Series(["2024年统计报告"] + [None] * 7)
        self.assertTrue(_is_title_row(row, n_cols=8))


# ---------------------------------------------------------------------------
# _find_header_row tests
# ---------------------------------------------------------------------------

class TestFindHeaderRow(unittest.TestCase):

    def test_finds_first_dense_row(self):
        df = pd.DataFrame([
            ["Table 1", None, None, None],     # row 0: title
            ["Name", "Age", "Score", "Year"],   # row 1: real header
            ["Alice", 30, 90, 2020],
        ])
        idx = _find_header_row(df)
        self.assertEqual(idx, 1)

    def test_returns_zero_if_first_row_is_header(self):
        df = pd.DataFrame([
            ["Name", "Age", "Score"],
            ["Alice", 30, 90],
        ])
        idx = _find_header_row(df)
        self.assertEqual(idx, 0)

    def test_returns_zero_for_empty_df(self):
        df = pd.DataFrame()
        idx = _find_header_row(df)
        self.assertEqual(idx, 0)


# ---------------------------------------------------------------------------
# _forward_fill_merged_cells tests
# ---------------------------------------------------------------------------

class TestForwardFillMergedCells(unittest.TestCase):

    def test_fills_text_column_nan(self):
        df = pd.DataFrame({
            "Category": ["A", None, None, "B"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        })
        result = _forward_fill_merged_cells(df)
        self.assertEqual(list(result["Category"]), ["A", "A", "A", "B"])

    def test_does_not_fill_numeric_column(self):
        df = pd.DataFrame({
            "Name": ["X", None, "Y"],
            "Score": [100.0, None, 80.0],
        })
        result = _forward_fill_merged_cells(df)
        # Numeric column NaN should remain NaN
        self.assertTrue(pd.isna(result["Score"].iloc[1]))

    def test_returns_new_dataframe(self):
        df = pd.DataFrame({"Cat": ["A", None]})
        result = _forward_fill_merged_cells(df)
        # Original should be unmodified
        self.assertTrue(pd.isna(df["Cat"].iloc[1]))

    def test_no_change_if_no_nulls(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [1.0, 2.0, 3.0]})
        result = _forward_fill_merged_cells(df)
        self.assertEqual(list(result["Cat"]), ["A", "B", "C"])


# ---------------------------------------------------------------------------
# _drop_empty_rows tests
# ---------------------------------------------------------------------------

class TestDropEmptyRows(unittest.TestCase):

    def test_removes_all_null_row(self):
        df = pd.DataFrame({
            "A": ["x", None, "y"],
            "B": [1.0, None, 2.0],
        })
        result = _drop_empty_rows(df)
        self.assertEqual(len(result), 2)

    def test_removes_all_empty_string_row(self):
        df = pd.DataFrame({
            "A": ["x", "", "y"],
            "B": ["a", "", "b"],
        })
        result = _drop_empty_rows(df)
        self.assertEqual(len(result), 2)

    def test_keeps_partial_rows(self):
        df = pd.DataFrame({
            "A": ["x", None, "y"],
            "B": [1.0, 2.0, None],
        })
        result = _drop_empty_rows(df)
        # No fully-empty rows here
        self.assertEqual(len(result), 3)

    def test_resets_index(self):
        df = pd.DataFrame({"A": ["x", None, "y"]})
        result = _drop_empty_rows(df)
        self.assertEqual(list(result.index), list(range(len(result))))


if __name__ == "__main__":
    unittest.main()
