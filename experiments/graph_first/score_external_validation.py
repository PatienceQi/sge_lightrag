"""
Score external validation answers against CSV ground truth.

For each of the 80 questions with has_any_context=True, this script:
1. Extracts the reference answer from the source CSV
2. Scores SGE and Baseline LLM answers against the reference
3. Writes a detailed scored JSON
"""

import json
import re
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# sge_lightrag/ is the project root (parent of experiments/)
SGE_ROOT = PROJECT_ROOT  # /…/sge_lightrag
DATASET_ROOT = SGE_ROOT / "dataset"

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_who() -> pd.DataFrame:
    path = DATASET_ROOT / "WHO/API_WHO_WHOSIS_000001_life_expectancy.csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def load_wb(indicator: str) -> pd.DataFrame:
    paths = {
        "wb_cm": "世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "wb_pop": "世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "wb_mat": "世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
    }
    return pd.read_csv(DATASET_ROOT / paths[indicator], skiprows=4)


def load_inpatient() -> pd.DataFrame:
    path = DATASET_ROOT / "住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv"
    df = pd.read_csv(path)
    # Row 1 (index 1) has the real column headers
    headers = df.iloc[1].tolist()
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = headers
    return df


def load_fortune500() -> pd.DataFrame:
    path = DATASET_ROOT / "non_gov/fortune500_revenue.csv"
    return pd.read_csv(path)


def load_the() -> pd.DataFrame:
    path = DATASET_ROOT / "non_gov/the_university_ranking.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Value lookup helpers
# ---------------------------------------------------------------------------

def get_who_value(df: pd.DataFrame, country_code: str, year: str):
    row = df[df["Country Code"] == country_code]
    if row.empty:
        return None
    col = str(year)
    if col not in df.columns:
        return None
    val = row[col].values[0]
    return None if pd.isna(val) else float(val)


def get_wb_value(df: pd.DataFrame, country_code: str, year: str):
    row = df[df["Country Code"] == country_code]
    if row.empty:
        return None
    col = str(year)
    if col not in df.columns:
        return None
    val = row[col].values[0]
    return None if pd.isna(val) else float(val)


def get_inpatient_col_value(df: pd.DataFrame, disease_keyword: str, col_idx: int):
    """Search for disease row by keyword in 疾病类别 column, return value from col_idx."""
    disease_col = df.columns[1]  # 疾病类别
    mask = df[disease_col].str.contains(disease_keyword, na=False)
    rows = df[mask]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    col_name = df.columns[col_idx]
    try:
        val = int(str(row.iloc[col_idx]).replace(",", ""))
    except (ValueError, TypeError):
        return None, str(row.iloc[col_idx])
    return val, col_name


def get_fortune500_value(df: pd.DataFrame, company: str, year: str):
    row = df[df["Company_Name"] == company]
    if row.empty:
        return None
    col = str(year)
    if col not in df.columns:
        return None
    val = row[col].values[0]
    return None if pd.isna(val) else float(val)


def get_the_value(df: pd.DataFrame, university: str, year: str):
    # Exact match first, then partial
    row = df[df["University_Name"] == university]
    if row.empty:
        row = df[df["University_Name"].str.contains(university, case=False, na=False)]
    if row.empty:
        return None
    col = str(year)
    if col not in df.columns:
        return None
    val = row[col].values[0]
    return None if pd.isna(val) else float(val)


# ---------------------------------------------------------------------------
# Number extraction from answer text
# ---------------------------------------------------------------------------

def extract_numbers_from_text(text: str) -> list:
    """Extract all numeric values from an answer string.

    Removes commas and markdown bold markers (**) so that values like
    **3.61325808** and 1,337,705,000 are correctly extracted.
    """
    cleaned = text.replace(",", "").replace("*", "")
    # Match decimal numbers first, then plain integers
    nums = re.findall(r"\d+\.\d+|\d+", cleaned)
    return [float(n) for n in nums]


def numbers_close(ref: float, val: float, tol: float = 0.01) -> bool:
    """Check if val is within tol (1%) of ref, or within 0.1 absolute for small values."""
    if ref == 0:
        return abs(val) < 0.01
    rel_err = abs(val - ref) / abs(ref)
    abs_err = abs(val - ref)
    return rel_err <= tol or abs_err <= 0.1


def answer_contains_value(answer: str, ref_value: float, tol: float = 0.01) -> bool:
    """Check if any number extracted from answer is close to ref_value."""
    if not answer or not answer.strip():
        return False
    nums = extract_numbers_from_text(answer)
    return any(numbers_close(ref_value, n, tol) for n in nums)


def answer_contains_direction(answer: str, direction: str) -> bool:
    """Check if answer mentions the correct direction of change."""
    text = answer.lower()
    if direction == "increase":
        return any(w in text for w in ["increase", "increas", "grew", "grow", "higher", "up", "rose", "gain", "improv", "上升", "增加", "增长"])
    elif direction == "decrease":
        return any(w in text for w in ["decrease", "decreas", "fell", "fall", "lower", "down", "declin", "drop", "reduc", "下降", "减少", "降低", "下跌"])
    return False



# Map ISO codes to common name variants used in LLM answers
_ISO_TO_NAMES: dict = {
    "CHN": ["china", "chn", "chinese"],
    "USA": ["usa", "united states", "america", "u.s.", "u.s.a."],
    "IND": ["india", "ind", "indian"],
    "JPN": ["japan", "jpn", "japanese"],
    "GBR": ["gbr", "uk", "united kingdom", "britain", "great britain"],
    "DEU": ["germany", "deu", "german"],
    "FRA": ["france", "fra", "french"],
    "BRA": ["brazil", "bra", "brazilian"],
    "AUS": ["australia", "aus", "australian"],
    "KOR": ["korea", "kor", "south korea"],
    "NGA": ["nigeria", "nga", "nigerian"],
    "ETH": ["ethiopia", "eth", "ethiopian"],
    "PAK": ["pakistan", "pak", "pakistani"],
    "BGD": ["bangladesh", "bgd", "bangladeshi"],
    "IDN": ["indonesia", "idn", "indonesian"],
    "CHE": ["switzerland", "che", "swiss"],
    "COL": ["colombia", "col", "colombian"],
    "RUS": ["russia", "rus", "russian"],
    "CVD": ["cerebrovascular", "cvd", "脑血管"],
    "malignant": ["malignant", "tumor", "cancer", "恶性肿瘤"],
    "diabetes": ["diabetes", "diabetic", "糖尿病"],
    "ischemic": ["ischemic", "缺血性心脏病"],
    "pneumonia": ["pneumonia", "肺炎"],
    "renal": ["renal", "kidney", "肾衰竭"],
    "discharge_larger": ["discharge", "出院"],
}


def answer_mentions_entity(answer: str, entity: str) -> bool:
    """Check if the answer mentions the entity (ISO code or full name)."""
    if not answer or not entity:
        return False
    text = answer.lower()
    entity_lower = entity.lower()
    if entity_lower in text:
        return True
    # Try mapped variants
    variants = _ISO_TO_NAMES.get(entity, [])
    return any(v in text for v in variants)


# ---------------------------------------------------------------------------
# Per-question ground truth extraction and scoring
# ---------------------------------------------------------------------------

def build_reference_answer(qid: str, dataset: str, question: str, query_type: str,
                            who_df, wb_dfs, inpatient_df, f500_df, the_df) -> dict:
    """
    Return dict with keys: reference_answer, reference_source, ref_type
    ref_type: 'numeric', 'direction', 'comparison_winner', 'list', 'unavailable'
    """

    # -----------------------------------------------------------------------
    # WHO
    # -----------------------------------------------------------------------
    if dataset == "who":
        df = who_df

        # Lookup queries
        if qid == "ext_001":
            v = get_who_value(df, "JPN", "2019")
            return {"reference_answer": f"{v:.5f}", "reference_source": "WHO JPN 2019",
                    "ref_type": "numeric", "ref_value": v}

        if qid == "ext_002":
            v2000 = get_who_value(df, "CHE", "2000")
            v2021 = get_who_value(df, "CHE", "2021")
            diff = v2021 - v2000
            return {"reference_answer": f"{diff:.5f}", "reference_source": "WHO CHE 2021-2000",
                    "ref_type": "numeric", "ref_value": diff}

        if qid == "ext_029":
            v2000 = get_who_value(df, "JPN", "2000")
            v2021 = get_who_value(df, "JPN", "2021")
            direction = "increase" if v2021 > v2000 else "decrease"
            diff = v2021 - v2000
            return {"reference_answer": f"Increase by {diff:.2f} years ({v2000:.2f}→{v2021:.2f})",
                    "reference_source": "WHO JPN 2000+2021",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        if qid == "ext_030":
            usa = get_who_value(df, "USA", "2021")
            chn = get_who_value(df, "CHN", "2021")
            winner = "CHN" if chn > usa else "USA"
            return {"reference_answer": f"CHN={chn:.2f}, USA={usa:.2f}; CHN higher",
                    "reference_source": "WHO CHN+USA 2021",
                    "ref_type": "two_values", "ref_value_a": chn, "ref_value_b": usa,
                    "winner": winner, "entity_a": "CHN", "entity_b": "USA"}

        # ext_v2 WHO lookups
        lookup_map = {
            "ext_v2_001": ("USA", "2015"),
            "ext_v2_002": ("CHN", "2010"),
            "ext_v2_003": ("IND", "2005"),
            "ext_v2_004": ("BRA", "2018"),
        }
        if qid in lookup_map:
            code, year = lookup_map[qid]
            v = get_who_value(df, code, year)
            return {"reference_answer": f"{v:.5f}", "reference_source": f"WHO {code} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions
        trend_map = {
            "ext_v2_005": ("JPN", "2000", "2021"),
            "ext_v2_006": ("GBR", "2010", "2021"),
            "ext_v2_007": ("NGA", "2000", "2021"),
        }
        if qid in trend_map:
            code, y1, y2 = trend_map[qid]
            v1 = get_who_value(df, code, y1)
            v2 = get_who_value(df, code, y2)
            diff = v2 - v1
            direction = "increase" if diff > 0 else "decrease"
            return {"reference_answer": f"{direction} by {abs(diff):.2f} ({v1:.2f}→{v2:.2f})",
                    "reference_source": f"WHO {code} {y1}+{y2}",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        # Comparison questions
        comp_map = {
            "ext_v2_008": ("AUS", "USA", "2019"),
            "ext_v2_009": ("JPN", "KOR", "2021"),
            "ext_v2_010": ("DEU", "FRA", "2015"),
        }
        if qid in comp_map:
            code_a, code_b, year = comp_map[qid]
            va = get_who_value(df, code_a, year)
            vb = get_who_value(df, code_b, year)
            winner = code_a if va > vb else code_b
            return {"reference_answer": f"{code_a}={va:.2f}, {code_b}={vb:.2f}; {winner} higher",
                    "reference_source": f"WHO {code_a}+{code_b} {year}",
                    "ref_type": "two_values", "ref_value_a": va, "ref_value_b": vb,
                    "winner": winner, "entity_a": code_a, "entity_b": code_b}

    # -----------------------------------------------------------------------
    # WB Child Mortality
    # -----------------------------------------------------------------------
    if dataset == "wb_cm":
        df = wb_dfs["wb_cm"]

        if qid == "ext_005":
            v = get_wb_value(df, "IND", "2015")
            return {"reference_answer": f"{v:.1f}", "reference_source": "WB CM IND 2015",
                    "ref_type": "numeric", "ref_value": v}

        if qid == "ext_006":
            v2000 = get_wb_value(df, "BRA", "2000")
            v2021 = get_wb_value(df, "BRA", "2021")
            diff = v2000 - v2021
            return {"reference_answer": f"Decrease by {diff:.1f} ({v2000:.1f}→{v2021:.1f})",
                    "reference_source": "WB CM BRA 2000+2021",
                    "ref_type": "direction_and_value", "ref_value": -diff, "direction": "decrease"}

        if qid == "ext_008":
            usa = get_wb_value(df, "USA", "2021")
            chn = get_wb_value(df, "CHN", "2021")
            winner = "CHN" if chn > usa else "USA"
            return {"reference_answer": f"CHN={chn:.1f}, USA={usa:.1f}; {winner} higher",
                    "reference_source": "WB CM CHN+USA 2021",
                    "ref_type": "two_values", "ref_value_a": chn, "ref_value_b": usa,
                    "winner": winner, "entity_a": "CHN", "entity_b": "USA"}

        if qid == "ext_033":
            v2000 = get_wb_value(df, "IND", "2000")
            v2021 = get_wb_value(df, "IND", "2021")
            diff = v2021 - v2000
            direction = "decrease"  # mortality goes down
            return {"reference_answer": f"Decrease by {abs(diff):.1f} ({v2000:.1f}→{v2021:.1f})",
                    "reference_source": "WB CM IND 2000+2021",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        if qid == "ext_034":
            # Germany consistently lower than Brazil
            # Check key years: 2000, 2010, 2022
            years_to_check = ["2000", "2010", "2022"]
            always_lower = all(
                get_wb_value(df, "DEU", y) < get_wb_value(df, "BRA", y)
                for y in years_to_check
                if get_wb_value(df, "DEU", y) is not None and get_wb_value(df, "BRA", y) is not None
            )
            return {"reference_answer": f"Yes, Germany consistently lower than Brazil",
                    "reference_source": "WB CM DEU+BRA multiple years",
                    "ref_type": "boolean", "ref_bool": True}

        # ext_v2 WB CM lookups
        lookup_map = {
            "ext_v2_011": ("IND", "2010"),
            "ext_v2_012": ("NGA", "2005"),
            "ext_v2_013": ("CHN", "2015"),
        }
        if qid in lookup_map:
            code, year = lookup_map[qid]
            v = get_wb_value(df, code, year)
            return {"reference_answer": f"{v:.1f}", "reference_source": f"WB CM {code} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions
        trend_map = {
            "ext_v2_014": ("ETH", "2000", "2022"),
            "ext_v2_015": ("PAK", "2005", "2020"),
            "ext_v2_016": ("BGD", "2000", "2022"),
        }
        if qid in trend_map:
            code, y1, y2 = trend_map[qid]
            v1 = get_wb_value(df, code, y1)
            v2 = get_wb_value(df, code, y2)
            diff = v2 - v1
            direction = "increase" if diff > 0 else "decrease"
            return {"reference_answer": f"{direction} by {abs(diff):.1f} ({v1:.1f}→{v2:.1f})",
                    "reference_source": f"WB CM {code} {y1}+{y2}",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        # Comparison questions
        comp_map = {
            "ext_v2_017": ("JPN", "DEU", "2020"),
            "ext_v2_018": ("IND", "CHN", "2015"),
            "ext_v2_019": ("BRA", "COL", "2022"),
        }
        if qid in comp_map:
            code_a, code_b, year = comp_map[qid]
            va = get_wb_value(df, code_a, year)
            vb = get_wb_value(df, code_b, year)
            winner_lower = code_a if va < vb else code_b  # lower is better for mortality
            return {"reference_answer": f"{code_a}={va:.1f}, {code_b}={vb:.1f}; {winner_lower} lower",
                    "reference_source": f"WB CM {code_a}+{code_b} {year}",
                    "ref_type": "two_values", "ref_value_a": va, "ref_value_b": vb,
                    "winner": winner_lower, "entity_a": code_a, "entity_b": code_b}

        if qid == "ext_v2_020":
            usa = get_wb_value(df, "USA", "2018")
            gbr = get_wb_value(df, "GBR", "2018")
            winner = "USA" if usa > gbr else "GBR"
            return {"reference_answer": f"USA={usa:.1f}, GBR={gbr:.1f}; USA higher",
                    "reference_source": "WB CM USA+GBR 2018",
                    "ref_type": "two_values", "ref_value_a": usa, "ref_value_b": gbr,
                    "winner": "USA", "entity_a": "USA", "entity_b": "GBR"}

    # -----------------------------------------------------------------------
    # WB Population
    # -----------------------------------------------------------------------
    if dataset == "wb_pop":
        df = wb_dfs["wb_pop"]

        if qid == "ext_036":
            v = get_wb_value(df, "CHN", "2010")
            return {"reference_answer": f"{v:.0f}", "reference_source": "WB Pop CHN 2010",
                    "ref_type": "numeric", "ref_value": v}

        if qid == "ext_037":
            ind = get_wb_value(df, "IND", "2021")
            usa = get_wb_value(df, "USA", "2021")
            winner = "IND" if ind > usa else "USA"
            return {"reference_answer": f"IND={ind:.0f}, USA={usa:.0f}; IND larger",
                    "reference_source": "WB Pop IND+USA 2021",
                    "ref_type": "two_values", "ref_value_a": ind, "ref_value_b": usa,
                    "winner": winner, "entity_a": "IND", "entity_b": "USA"}

        # ext_v2 WB Pop lookups
        lookup_map = {
            "ext_v2_021": ("IND", "2020"),
            "ext_v2_022": ("USA", "2000"),
            "ext_v2_023": ("IDN", "2010"),
        }
        if qid in lookup_map:
            code, year = lookup_map[qid]
            v = get_wb_value(df, code, year)
            return {"reference_answer": f"{v:.0f}" if v else "N/A",
                    "reference_source": f"WB Pop {code} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions
        trend_map = {
            "ext_v2_024": ("NGA", "2000", "2023"),
            "ext_v2_025": ("BRA", "2010", "2023"),
        }
        if qid in trend_map:
            code, y1, y2 = trend_map[qid]
            v1 = get_wb_value(df, code, y1)
            v2 = get_wb_value(df, code, y2)
            if v1 is None or v2 is None:
                return {"reference_answer": "N/A", "reference_source": f"WB Pop {code} {y1}+{y2}",
                        "ref_type": "unavailable"}
            diff = v2 - v1
            direction = "increase" if diff > 0 else "decrease"
            return {"reference_answer": f"{direction} by {abs(diff):.0f} ({v1:.0f}→{v2:.0f})",
                    "reference_source": f"WB Pop {code} {y1}+{y2}",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        if qid == "ext_v2_026":
            # Ethiopia 1990-2020
            v1990 = get_wb_value(df, "ETH", "1990")
            v2020 = get_wb_value(df, "ETH", "2020")
            if v1990 is None or v2020 is None:
                return {"reference_answer": "N/A", "reference_source": "WB Pop ETH 1990+2020",
                        "ref_type": "unavailable"}
            diff = v2020 - v1990
            direction = "increase" if diff > 0 else "decrease"
            return {"reference_answer": f"increase by {diff:.0f} ({v1990:.0f}→{v2020:.0f})",
                    "reference_source": "WB Pop ETH 1990+2020",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        # Comparison questions
        if qid == "ext_v2_027":
            chn = get_wb_value(df, "CHN", "2010")
            ind = get_wb_value(df, "IND", "2010")
            winner = "CHN" if chn > ind else "IND"
            return {"reference_answer": f"CHN={chn:.0f}, IND={ind:.0f}; {winner} larger",
                    "reference_source": "WB Pop CHN+IND 2010",
                    "ref_type": "two_values", "ref_value_a": chn, "ref_value_b": ind,
                    "winner": winner, "entity_a": "CHN", "entity_b": "IND"}

        if qid == "ext_v2_028":
            usa = get_wb_value(df, "USA", "2023")
            idn = get_wb_value(df, "IDN", "2023")
            if usa is None or idn is None:
                return {"reference_answer": "N/A", "reference_source": "WB Pop USA+IDN 2023",
                        "ref_type": "unavailable"}
            winner = "USA" if usa > idn else "IDN"
            return {"reference_answer": f"USA={usa:.0f}, IDN={idn:.0f}; {winner} larger",
                    "reference_source": "WB Pop USA+IDN 2023",
                    "ref_type": "two_values", "ref_value_a": usa, "ref_value_b": idn,
                    "winner": winner, "entity_a": "USA", "entity_b": "IDN"}

        if qid == "ext_v2_029":
            pak = get_wb_value(df, "PAK", "2000")
            rus = get_wb_value(df, "RUS", "2000")
            if pak is None or rus is None:
                return {"reference_answer": "N/A", "reference_source": "WB Pop PAK+RUS 2000",
                        "ref_type": "unavailable"}
            winner = "PAK" if pak > rus else "RUS"
            return {"reference_answer": f"PAK={pak:.0f}, RUS={rus:.0f}; {winner} larger",
                    "reference_source": "WB Pop PAK+RUS 2000",
                    "ref_type": "two_values", "ref_value_a": pak, "ref_value_b": rus,
                    "winner": winner, "entity_a": "PAK", "entity_b": "RUS"}

        if qid == "ext_v2_030":
            jpn2023 = get_wb_value(df, "JPN", "2023")
            jpn2000 = get_wb_value(df, "JPN", "2000")
            if jpn2023 is None or jpn2000 is None:
                return {"reference_answer": "N/A", "reference_source": "WB Pop JPN 2023+2000",
                        "ref_type": "unavailable"}
            exceeded = jpn2023 > jpn2000
            return {"reference_answer": f"No, JPN 2023={jpn2023:.0f} < JPN 2000={jpn2000:.0f}" if not exceeded
                    else f"Yes, JPN 2023={jpn2023:.0f} > JPN 2000={jpn2000:.0f}",
                    "reference_source": "WB Pop JPN 2023+2000",
                    "ref_type": "boolean", "ref_bool": exceeded}

    # -----------------------------------------------------------------------
    # WB Maternal Mortality
    # -----------------------------------------------------------------------
    if dataset == "wb_mat":
        df = wb_dfs["wb_mat"]

        # ext_v2 WB Mat lookups
        lookup_map = {
            "ext_v2_031": ("ETH", "2010"),
            "ext_v2_032": ("NGA", "2015"),
            "ext_v2_033": ("IND", "2005"),
        }
        if qid in lookup_map:
            code, year = lookup_map[qid]
            v = get_wb_value(df, code, year)
            return {"reference_answer": f"{v:.0f}" if v else "N/A",
                    "reference_source": f"WB Mat {code} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions
        trend_map = {
            "ext_v2_034": ("BGD", "2000", "2020"),
            "ext_v2_035": ("IDN", "2000", "2020"),
            "ext_v2_036": ("PAK", "2000", "2020"),
        }
        if qid in trend_map:
            code, y1, y2 = trend_map[qid]
            v1 = get_wb_value(df, code, y1)
            v2 = get_wb_value(df, code, y2)
            if v1 is None or v2 is None:
                return {"reference_answer": "N/A", "reference_source": f"WB Mat {code} {y1}+{y2}",
                        "ref_type": "unavailable"}
            diff = v2 - v1
            direction = "increase" if diff > 0 else "decrease"
            return {"reference_answer": f"{direction} by {abs(diff):.0f} ({v1:.0f}→{v2:.0f})",
                    "reference_source": f"WB Mat {code} {y1}+{y2}",
                    "ref_type": "direction_and_value", "ref_value": diff, "direction": direction}

        # Comparison questions
        comp_map = {
            "ext_v2_037": ("FRA", "DEU", "2015"),
            "ext_v2_038": ("NGA", "ETH", "2020"),
            "ext_v2_039": ("CHN", "IND", "2020"),
        }
        if qid in comp_map:
            code_a, code_b, year = comp_map[qid]
            va = get_wb_value(df, code_a, year)
            vb = get_wb_value(df, code_b, year)
            if va is None or vb is None:
                return {"reference_answer": "N/A", "reference_source": f"WB Mat {code_a}+{code_b} {year}",
                        "ref_type": "unavailable"}
            winner_lower = code_a if va < vb else code_b
            return {"reference_answer": f"{code_a}={va:.0f}, {code_b}={vb:.0f}; {winner_lower} lower",
                    "reference_source": f"WB Mat {code_a}+{code_b} {year}",
                    "ref_type": "two_values", "ref_value_a": va, "ref_value_b": vb,
                    "winner": winner_lower, "entity_a": code_a, "entity_b": code_b}

        if qid == "ext_v2_040":
            # UK (GBR) vs Japan (JPN) in 2010
            gbr = get_wb_value(df, "GBR", "2010")
            jpn = get_wb_value(df, "JPN", "2010")
            if gbr is None or jpn is None:
                return {"reference_answer": "N/A", "reference_source": "WB Mat GBR+JPN 2010",
                        "ref_type": "unavailable"}
            exceeded = gbr > jpn
            return {"reference_answer": f"GBR={gbr:.0f}, JPN={jpn:.0f}; GBR {'>' if exceeded else '<='} JPN",
                    "reference_source": "WB Mat GBR+JPN 2010",
                    "ref_type": "two_values", "ref_value_a": gbr, "ref_value_b": jpn,
                    "winner": "GBR" if exceeded else "JPN", "entity_a": "GBR", "entity_b": "JPN"}

    # -----------------------------------------------------------------------
    # Inpatient
    # -----------------------------------------------------------------------
    if dataset == "inpatient":
        df = inpatient_df
        # Column indices (0-indexed from renamed columns):
        # 0: ICD code, 1: 疾病类别, 2: HA discharge, 3: prison discharge, 4: private discharge
        # 5: total discharge, 6: death male, 7: death female, 8: death unknown, 9: total death

        # Pneumonia (肺炎) - lookup total discharge
        if qid == "ext_v2_041":
            val, col = get_inpatient_col_value(df, "肺炎", 5)
            if val is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 肺炎 discharge",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{val}", "reference_source": "Inpatient 肺炎 total discharge",
                    "ref_type": "numeric", "ref_value": float(val)}

        # Diabetes (糖尿病) - deaths
        if qid == "ext_v2_042":
            val, col = get_inpatient_col_value(df, "糖尿病", 9)
            if val is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 糖尿病 deaths",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{val}", "reference_source": "Inpatient 糖尿病 total deaths",
                    "ref_type": "numeric", "ref_value": float(val)}

        # Ischemic heart disease (缺血性心脏病) - total discharge
        if qid == "ext_v2_043":
            val, col = get_inpatient_col_value(df, "缺血性心脏病", 5)
            if val is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 缺血性心脏病 discharge",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{val}", "reference_source": "Inpatient 缺血性心脏病 total discharge",
                    "ref_type": "numeric", "ref_value": float(val)}

        # Renal failure (肾衰竭) - deaths vs discharge
        if qid == "ext_v2_044":
            deaths, _ = get_inpatient_col_value(df, "肾衰竭", 9)
            discharge, _ = get_inpatient_col_value(df, "肾衰竭", 5)
            if deaths is None or discharge is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 肾衰竭",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"deaths={deaths}, discharge={discharge}",
                    "reference_source": "Inpatient 肾衰竭 deaths+discharge",
                    "ref_type": "two_values", "ref_value_a": float(deaths), "ref_value_b": float(discharge),
                    "winner": "discharge_larger", "entity_a": "deaths", "entity_b": "discharge"}

        # Cerebrovascular vs renal failure - discharge comparison
        if qid == "ext_v2_045":
            cvd, _ = get_inpatient_col_value(df, "脑血管疾病", 5)
            renal, _ = get_inpatient_col_value(df, "肾衰竭", 5)
            if cvd is None or renal is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 脑血管+肾衰竭",
                        "ref_type": "unavailable"}
            higher = "CVD" if cvd > renal else "renal"
            return {"reference_answer": f"CVD_discharge={cvd}, renal_discharge={renal}; {'CVD higher' if cvd > renal else 'renal higher'}",
                    "reference_source": "Inpatient 脑血管+肾衰竭 discharge",
                    "ref_type": "two_values", "ref_value_a": float(cvd), "ref_value_b": float(renal),
                    "winner": "CVD" if cvd > renal else "renal", "entity_a": "CVD", "entity_b": "renal"}

        # Malignant tumor vs diabetes - deaths comparison
        # "恶性肿瘤" matches many rows — aggregation is ambiguous; mark unavailable
        if qid == "ext_v2_046":
            diab, _ = get_inpatient_col_value(df, "糖尿病", 9)
            return {"reference_answer": f"diabetes_deaths=570; malignant total deaths >> diabetes (ambiguous aggregation)",
                    "reference_source": "Inpatient ambiguous: malignant has many sub-rows",
                    "ref_type": "unavailable"}

        # Cholera (霍乱) - total discharge
        if qid == "ext_v2_047":
            val, _ = get_inpatient_col_value(df, "霍乱", 5)
            if val is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 霍乱 discharge",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{val}", "reference_source": "Inpatient 霍乱 total discharge",
                    "ref_type": "numeric", "ref_value": float(val)}

        # Tuberculosis (结核病) - deaths
        if qid == "ext_v2_048":
            # Combine A15-A16 and A17-A19
            tb1, _ = get_inpatient_col_value(df, "呼吸道结核病", 9)
            tb2, _ = get_inpatient_col_value(df, "其他结核病", 9)
            if tb1 is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 结核病 deaths",
                        "ref_type": "unavailable"}
            total = (tb1 or 0) + (tb2 or 0)
            return {"reference_answer": f"{total} (呼吸道={tb1}, 其他={tb2})",
                    "reference_source": "Inpatient 结核病 deaths (A15-A19)",
                    "ref_type": "numeric", "ref_value": float(total)}

        # Pneumonia deaths vs CVD deaths
        if qid == "ext_v2_049":
            pneu, _ = get_inpatient_col_value(df, "肺炎", 9)
            cvd, _ = get_inpatient_col_value(df, "脑血管疾病", 9)
            if pneu is None or cvd is None:
                return {"reference_answer": "N/A", "reference_source": "Inpatient 肺炎+脑血管 deaths",
                        "ref_type": "unavailable"}
            winner = "pneumonia" if pneu > cvd else "CVD"
            return {"reference_answer": f"pneumonia_deaths={pneu}, CVD_deaths={cvd}; {winner} higher",
                    "reference_source": "Inpatient 肺炎+脑血管 deaths",
                    "ref_type": "two_values", "ref_value_a": float(pneu), "ref_value_b": float(cvd),
                    "winner": winner, "entity_a": "pneumonia", "entity_b": "CVD"}

        # Ischemic vs malignant - discharge
        # "恶性肿瘤" matches many sub-categories; the comparison is ambiguous
        # (one ischemic row: 26785 vs many malignant rows summing to much more).
        # Mark as unavailable to avoid misleading scoring.
        if qid == "ext_v2_050":
            return {"reference_answer": "Ambiguous: 缺血性心脏病 has one row (26785) but 恶性肿瘤 has many rows; total malignant >> ischemic",
                    "reference_source": "Inpatient ambiguous aggregation",
                    "ref_type": "unavailable"}

    # -----------------------------------------------------------------------
    # Fortune 500 — NOTE: CSV only has years 2015–2019
    # -----------------------------------------------------------------------
    if dataset == "fortune500":
        df = f500_df

        if qid == "ext_046":
            # Amazon revenue trend 2019-2023 — only 2019 available
            v2019 = get_fortune500_value(df, "Amazon.com", "2019")
            return {"reference_answer": f"2019={v2019:.0f}; 2020-2023 not in dataset",
                    "reference_source": "Fortune500 Amazon.com 2019 (only year available > 2018)",
                    "ref_type": "unavailable"}

        lookup_map = {
            "ext_v2_051": ("Walmart", "2021"),
            "ext_v2_052": ("Exxon Mobil", "2019"),
            "ext_v2_053": ("Apple", "2022"),
        }
        if qid in lookup_map:
            company, year = lookup_map[qid]
            v = get_fortune500_value(df, company, year)
            if v is None:
                return {"reference_answer": f"Year {year} not in dataset (only 2015-2019 available)",
                        "reference_source": f"Fortune500 {company} {year}",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{v:.0f}", "reference_source": f"Fortune500 {company} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions — years beyond 2019 not available
        trend_unavail = {"ext_v2_054", "ext_v2_055", "ext_v2_056"}
        if qid in trend_unavail:
            return {"reference_answer": "N/A — years beyond 2019 not in dataset",
                    "reference_source": "Fortune500 (2015-2019 only)",
                    "ref_type": "unavailable"}

        # Comparison questions
        comp_map = {
            "ext_v2_057": ("Walmart", "Amazon.com", "2021"),
            "ext_v2_058": ("Apple", "Alphabet", "2022"),
            "ext_v2_059": ("Exxon Mobil", "Berkshire Hathaway", "2023"),
            "ext_v2_060": ("UnitedHealth Group", "McKesson", "2022"),
        }
        if qid in comp_map:
            ca, cb, year = comp_map[qid]
            va = get_fortune500_value(df, ca, year)
            vb = get_fortune500_value(df, cb, year)
            if va is None or vb is None:
                return {"reference_answer": f"Year {year} not in dataset (only 2015-2019 available)",
                        "reference_source": f"Fortune500 {ca}+{cb} {year}",
                        "ref_type": "unavailable"}
            winner = ca if va > vb else cb
            return {"reference_answer": f"{ca}={va:.0f}, {cb}={vb:.0f}; {winner} higher",
                    "reference_source": f"Fortune500 {ca}+{cb} {year}",
                    "ref_type": "two_values", "ref_value_a": va, "ref_value_b": vb,
                    "winner": winner, "entity_a": ca, "entity_b": cb}

    # -----------------------------------------------------------------------
    # THE University Ranking — NOTE: CSV only has years 2011–2016
    # -----------------------------------------------------------------------
    if dataset == "the":
        df = the_df

        lookup_map = {
            "ext_v2_061": ("University of Oxford", "2021"),
            "ext_v2_063": ("ETH Zurich – Swiss Federal Institute of Technology Zurich", "2020"),
        }
        if qid in lookup_map:
            university, year = lookup_map[qid]
            v = get_the_value(df, university, year)
            if v is None:
                return {"reference_answer": f"Year {year} not in dataset (only 2011-2016 available)",
                        "reference_source": f"THE {university} {year}",
                        "ref_type": "unavailable"}
            return {"reference_answer": f"{v}", "reference_source": f"THE {university} {year}",
                    "ref_type": "numeric", "ref_value": v}

        # Trend questions — years 2017-2023 not available
        trend_unavail = {"ext_v2_064", "ext_v2_065", "ext_v2_066"}
        if qid in trend_unavail:
            return {"reference_answer": "N/A — years beyond 2016 not in dataset",
                    "reference_source": "THE (2011-2016 only)",
                    "ref_type": "unavailable"}

        # Comparison questions
        comp_map = {
            "ext_v2_067": ("University of Cambridge", "Princeton University", "2022"),
            "ext_v2_069": ("University of Chicago", "ETH Zurich – Swiss Federal Institute of Technology Zurich", "2023"),
            "ext_v2_070": ("University of Oxford", "Harvard University", "2020"),
        }
        if qid in comp_map:
            ua, ub, year = comp_map[qid]
            va = get_the_value(df, ua, year)
            vb = get_the_value(df, ub, year)
            if va is None or vb is None:
                return {"reference_answer": f"Year {year} not in dataset (only 2011-2016 available)",
                        "reference_source": f"THE {ua[:20]}+{ub[:20]} {year}",
                        "ref_type": "unavailable"}
            winner = ua if va > vb else ub
            return {"reference_answer": f"{ua[:30]}={va}, {ub[:30]}={vb}; {winner[:30]} higher",
                    "reference_source": f"THE comparison {year}",
                    "ref_type": "two_values", "ref_value_a": va, "ref_value_b": vb,
                    "winner": winner, "entity_a": ua, "entity_b": ub}

    # Default: no reference available
    return {"reference_answer": "N/A", "reference_source": "not mapped",
            "ref_type": "unavailable"}


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

_INSUFFICIENT_PHRASES = [
    "insufficient context",
    "cannot provide",
    "cannot answer",
    "not included in the",
    "not provided in",
    "not contain",
    "not available in",
    "not in the provided",
    "not explicitly stated",
    "i cannot",
    "i would need",
    "not enough information",
]


def _answer_claims_insufficient(answer: str) -> bool:
    """Return True if the answer explicitly claims it cannot answer the question."""
    text = answer.lower()
    return any(phrase in text for phrase in _INSUFFICIENT_PHRASES)


def score_answer(answer: str, ref: dict) -> bool:
    """Return True if the answer is factually correct based on the reference."""
    if not answer or not answer.strip():
        return False

    # If the answer claims insufficient context, treat it as incorrect even if
    # it accidentally contains a matching number.
    if _answer_claims_insufficient(answer):
        return False

    ref_type = ref.get("ref_type", "unavailable")

    if ref_type == "unavailable":
        return False

    if ref_type == "numeric":
        rv = ref.get("ref_value")
        if rv is None:
            return False
        return answer_contains_value(answer, rv, tol=0.01)

    if ref_type == "direction_and_value":
        direction = ref.get("direction", "")
        rv = ref.get("ref_value")
        direction_ok = answer_contains_direction(answer, direction)
        if rv is not None and abs(rv) > 0.001:
            value_ok = answer_contains_value(answer, abs(rv), tol=0.05)
            return direction_ok and value_ok
        return direction_ok

    if ref_type == "two_values":
        va = ref.get("ref_value_a")
        vb = ref.get("ref_value_b")
        winner = ref.get("winner", "")
        loser = ref.get("entity_b") if winner == ref.get("entity_a") else ref.get("entity_a")

        # Check if answer mentions the winner entity
        winner_mentioned = answer_mentions_entity(answer, winner)

        # Check that the answer does not explicitly claim the loser is higher/more
        # This prevents false positives when winner is mentioned incidentally.
        if loser and loser != winner:
            loser_mentioned = answer_mentions_entity(answer, loser)
            if loser_mentioned:
                text_lower = answer.lower()
                # Find context around loser mentions — if the answer says loser is higher, reject
                higher_words = ["更高", "更多", "higher", "larger", "more", "exceed", "greater", "surpass"]
                # Simple heuristic: if answer says "X is higher" and X is the loser, it's wrong
                # We check this by seeing if "higher" follows closer to the loser than the winner
                # For simplicity: if the answer doesn't use winner as the claimed higher, reject
                # Use the presence of winner + "higher" near each other as proxy
                winner_names = [winner.lower()] + _ISO_TO_NAMES.get(winner, [])
                winner_is_affirmed = any(
                    f"{w}" in text_lower and any(h in text_lower[max(0, text_lower.find(w) - 50):text_lower.find(w) + 200] for h in higher_words)
                    for w in winner_names if w in text_lower
                )
                loser_names = [loser.lower()] + _ISO_TO_NAMES.get(loser, [])
                loser_is_affirmed = any(
                    f"{w}" in text_lower and any(h in text_lower[max(0, text_lower.find(w) - 50):text_lower.find(w) + 200] for h in higher_words)
                    for w in loser_names if w in text_lower
                )
                # If loser is explicitly affirmed as higher and winner is not, reject
                if loser_is_affirmed and not winner_is_affirmed:
                    return False

        # For large numbers (population), check if at least one reference value is close
        has_correct_value = False
        if va is not None:
            has_correct_value = has_correct_value or answer_contains_value(answer, va, tol=0.02)
        if vb is not None:
            has_correct_value = has_correct_value or answer_contains_value(answer, vb, tol=0.02)

        return winner_mentioned and has_correct_value

    if ref_type == "boolean":
        ref_bool = ref.get("ref_bool", True)
        text = answer.lower()
        if ref_bool:
            # Answer should confirm the positive claim
            yes_words = ["yes", "higher", "larger", "exceeded", "greater", "true", "是", "超过",
                         "consistently lower", "were lower", "was lower", "has been lower",
                         "confirmed", "correct", "indeed"]
            # Only explicit negations of the claim
            deny_words = ["no,", "incorrect", "not higher", "not larger", "not exceeded",
                          "not consistently", "false", "否", "并非", "没有", "不是"]
            has_yes = any(w in text for w in yes_words)
            has_deny = any(w in text for w in deny_words)
            return has_yes and not has_deny
        else:
            no_words = ["no", "did not", "didn't", "not exceed", "not higher", "not larger",
                        "smaller", "lower", "false", "否", "没有超过"]
            confirm_words = ["yes, exceeded", "yes, higher", "yes, larger", "exceeded", "surpassed"]
            has_no = any(w in text for w in no_words)
            has_confirm = any(w in text for w in confirm_words)
            return has_no and not has_confirm

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_path = SGE_ROOT / "experiments/results/external_validation_v2_results.json"
    output_path = SGE_ROOT / "experiments/results/external_validation_v2_scored.json"

    with open(results_path) as f:
        data = json.load(f)

    # Load all CSV datasets once
    who_df = load_who()
    wb_dfs = {
        "wb_cm": load_wb("wb_cm"),
        "wb_pop": load_wb("wb_pop"),
        "wb_mat": load_wb("wb_mat"),
    }
    inpatient_df = load_inpatient()
    f500_df = load_fortune500()
    the_df = load_the()

    scored_results = []
    sge_correct_total = 0
    baseline_correct_total = 0
    per_dataset: dict = {}

    context_results = [r for r in data["results"] if r.get("has_any_context", False)]

    for r in context_results:
        qid = r["id"]
        dataset = r["dataset"]
        question = r["question"]
        query_type = r.get("query_type", "unknown")
        sge_answer = r.get("sge_answer", "")
        base_answer = r.get("base_answer", "")

        ref = build_reference_answer(
            qid, dataset, question, query_type,
            who_df, wb_dfs, inpatient_df, f500_df, the_df
        )

        sge_correct = score_answer(sge_answer, ref)
        base_correct = score_answer(base_answer, ref)

        if dataset not in per_dataset:
            per_dataset[dataset] = {
                "n_with_context": 0, "sge_correct": 0, "baseline_correct": 0,
                "unavailable": 0
            }

        per_dataset[dataset]["n_with_context"] += 1
        if ref["ref_type"] == "unavailable":
            per_dataset[dataset]["unavailable"] += 1
        if sge_correct:
            sge_correct_total += 1
            per_dataset[dataset]["sge_correct"] += 1
        if base_correct:
            baseline_correct_total += 1
            per_dataset[dataset]["baseline_correct"] += 1

        scored_results.append({
            "id": qid,
            "dataset": dataset,
            "question": question,
            "query_type": query_type,
            "reference_answer": ref["reference_answer"],
            "reference_source": ref["reference_source"],
            "ref_type": ref["ref_type"],
            "sge_answer": sge_answer,
            "base_answer": base_answer,
            "sge_correct": sge_correct,
            "base_correct": base_correct,
        })

    # Count scoreable questions (those with ref_type != unavailable)
    scoreable = [r for r in scored_results if r["ref_type"] != "unavailable"]
    n_scoreable = len(scoreable)
    sge_correct_scoreable = sum(1 for r in scoreable if r["sge_correct"])
    base_correct_scoreable = sum(1 for r in scoreable if r["base_correct"])

    for ds in per_dataset:
        ds_results = [r for r in scored_results if r["dataset"] == ds]
        ds_scoreable = [r for r in ds_results if r["ref_type"] != "unavailable"]
        n_ds = len(ds_scoreable)
        per_dataset[ds]["n_scoreable"] = n_ds
        per_dataset[ds]["sge_accuracy"] = round(per_dataset[ds]["sge_correct"] / n_ds, 3) if n_ds > 0 else None
        per_dataset[ds]["baseline_accuracy"] = round(per_dataset[ds]["baseline_correct"] / n_ds, 3) if n_ds > 0 else None

    output = {
        "total_with_context": len(context_results),
        "total_scoreable": n_scoreable,
        "sge_correct": sge_correct_scoreable,
        "baseline_correct": base_correct_scoreable,
        "sge_accuracy": round(sge_correct_scoreable / n_scoreable, 3) if n_scoreable > 0 else None,
        "baseline_accuracy": round(base_correct_scoreable / n_scoreable, 3) if n_scoreable > 0 else None,
        "per_dataset": per_dataset,
        "scored_results": scored_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Scored {len(context_results)} questions with context ({n_scoreable} scoreable)")
    print(f"SGE correct: {sge_correct_scoreable}/{n_scoreable} = {output['sge_accuracy']:.1%}")
    print(f"Baseline correct: {base_correct_scoreable}/{n_scoreable} = {output['baseline_accuracy']:.1%}")
    print()
    print("Per-dataset breakdown:")
    for ds, stats in per_dataset.items():
        print(f"  {ds}: scoreable={stats['n_scoreable']} unavail={stats['unavailable']} "
              f"SGE={stats['sge_correct']} ({stats['sge_accuracy']}) "
              f"Base={stats['baseline_correct']} ({stats['baseline_accuracy']})")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
