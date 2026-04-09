#!/usr/bin/env python3
"""
Step 1: Multi-indicator data preparation.

Loads 6 CSV datasets (5 WB + 1 WHO), generates compact chunks, computes
reference answers, and saves benchmark questions. Zero LLM calls.

Usage:
    python3 experiments/ggcr/multi_indicator_prepare.py [--dry-run]

Outputs:
    output/ggcr_cache/multi_indicator_data.json   — chunks + entity_map + benchmark
"""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WB_BASE = PROJECT_ROOT.parent / "dataset" / "世界银行数据"
WHO_BASE = PROJECT_ROOT / "dataset" / "WHO"
CACHE_DIR = PROJECT_ROOT / "output" / "ggcr_cache"

YEARS = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]

G7 = ["USA", "GBR", "FRA", "DEU", "JPN", "ITA", "CAN"]
G20 = [
    "ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU",
    "IND", "IDN", "ITA", "JPN", "KOR", "MEX", "RUS",
    "SAU", "ZAF", "TUR", "GBR", "USA",
]
BRICS = ["BRA", "RUS", "IND", "CHN", "ZAF"]
# Approximate EU-27 (countries with EU membership as of 2021, using ISO-3 codes present in WB data)
EU27 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]

INDICATOR_CONFIGS = {
    "Population": {
        "path": WB_BASE / "population" / "API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
        "format": "wb", "unit": "people", "skip_lines": 4,
    },
    "ChildMortality": {
        "path": WB_BASE / "child_mortality" / "API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
        "format": "wb", "unit": "per 1,000 live births", "skip_lines": 4,
    },
    "MaternalMortality": {
        "path": WB_BASE / "maternal_mortality" / "API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
        "format": "wb", "unit": "per 100,000 live births", "skip_lines": 4,
    },
    "HealthExpenditure": {
        "path": WB_BASE / "health_expenditure" / "API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_558.csv",
        "format": "wb", "unit": "% of GDP", "skip_lines": 4,
    },
    "HospitalBeds": {
        "path": WB_BASE / "hospital_beds" / "API_SH.MED.BEDS.ZS_DS2_en_csv_v2_4542.csv",
        "format": "wb", "unit": "per 1,000 people", "skip_lines": 4,
    },
    "LifeExpectancy": {
        "path": WHO_BASE / "API_WHO_WHOSIS_000001_life_expectancy.csv",
        "format": "who", "unit": "years", "skip_lines": 0,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wb_metadata() -> dict[str, dict]:
    """Load {code: {region, income_group, name}} for countries with non-empty Region."""
    meta = {}
    meta_path = WB_BASE / "population" / "Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"
    with open(meta_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = (row.get("Country Code") or "").strip()
            region = (row.get("Region") or "").strip()
            name = (row.get("TableName") or "").strip()
            income_group = (row.get("IncomeGroup") or "").strip()
            if code and region:
                meta[code] = {"region": region, "income_group": income_group, "name": name}
    return meta


def load_indicator_data(
    indicator_name: str, country_meta: dict[str, dict]
) -> dict[str, dict[str, float]]:
    """Returns {country_code: {year: float}}."""
    cfg = INDICATOR_CONFIGS[indicator_name]
    data: dict[str, dict[str, float]] = {}
    with open(cfg["path"], encoding="utf-8-sig") as f:
        for _ in range(cfg["skip_lines"]):
            f.readline()
        for row in csv.DictReader(f):
            code = (row.get("Country Code") or "").strip()
            if code not in country_meta:
                continue
            vals: dict[str, float] = {}
            for y in YEARS:
                raw = (row.get(y) or "").strip()
                if raw:
                    try:
                        vals[y] = float(raw)
                    except ValueError:
                        pass
            if vals:
                data[code] = vals
    return data


# ---------------------------------------------------------------------------
# Compact chunk generation
# ---------------------------------------------------------------------------

def build_all_chunks(
    indicator_data: dict[str, dict[str, dict[str, float]]],
    country_meta: dict[str, dict],
) -> tuple[list[str], dict[str, int]]:
    """Returns (chunks, entity_map) where entity_map = {key: chunk_index}."""
    chunks: list[str] = []
    entity_map: dict[str, int] = {}

    for ind_name in sorted(INDICATOR_CONFIGS.keys()):
        ind_data = indicator_data.get(ind_name, {})
        unit = INDICATOR_CONFIGS[ind_name]["unit"]
        for code in sorted(ind_data.keys()):
            name = country_meta.get(code, {}).get("name", code)
            ts = "; ".join(f"{y}={ind_data[code][y]}" for y in YEARS if y in ind_data[code])
            chunk = (
                f"Entity: {code} | Name: {name} | Indicator: {ind_name}\n"
                f"Unit: {unit} | Years: {ts}"
            )
            idx = len(chunks)
            entity_map[f"{code}:{ind_name}"] = idx
            entity_map[f"{code}:{ind_name}".lower()] = idx
            chunks.append(chunk)

    return chunks, entity_map


# ---------------------------------------------------------------------------
# Benchmark generation (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _sorted_median(values: list[float]) -> float:
    """Return median of a list (deterministic, no random sampling)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return round((s[mid - 1] + s[mid]) / 2, 4)


def compute_reference_answers(
    indicator_data: dict[str, dict[str, dict[str, float]]],
    country_meta: dict[str, dict],
) -> dict:
    pop = indicator_data.get("Population", {})
    cm = indicator_data.get("ChildMortality", {})
    mm = indicator_data.get("MaternalMortality", {})
    he = indicator_data.get("HealthExpenditure", {})
    hb = indicator_data.get("HospitalBeds", {})
    le = indicator_data.get("LifeExpectancy", {})

    def get_region(code: str) -> str:
        return country_meta.get(code, {}).get("region", "")

    def get_income(code: str) -> str:
        return country_meta.get(code, {}).get("income_group", "")

    ssa_codes = {c for c in country_meta if "Sub-Saharan Africa" in get_region(c)}
    eap_codes = {c for c in country_meta if "East Asia & Pacific" in get_region(c)}
    high_income_codes = {c for c in country_meta if get_income(c) == "High income"}
    low_income_codes = {c for c in country_meta if get_income(c) == "Low income"}

    # -----------------------------------------------------------------------
    # L3 questions 01-10 (UNCHANGED — backward compatibility)
    # -----------------------------------------------------------------------
    pop50m = {c for c in pop if pop[c].get("2021", 0) > 50_000_000}
    cm_2021_pop50m = {c: cm[c]["2021"] for c in pop50m if c in cm and "2021" in cm[c]}
    l3_01 = [c for c, _ in sorted(cm_2021_pop50m.items(), key=lambda x: x[1])[:5]]

    he_high = {c for c in he if he[c].get("2021", 0) > 10}
    le_2021_he_high = {c: le[c]["2021"] for c in he_high if c in le and "2021" in le[c]}
    l3_02 = [c for c, _ in sorted(le_2021_he_high.items(), key=lambda x: x[1], reverse=True)[:5]]

    hb_low2 = {c for c in hb if hb[c].get("2020", 99) < 2}
    mm_2020_hb_low = {c: mm[c]["2020"] for c in hb_low2 if c in mm and "2020" in mm[c]}
    l3_03 = [c for c, _ in sorted(mm_2020_hb_low.items(), key=lambda x: x[1], reverse=True)[:5]]

    le_high75 = {c for c in le if le[c].get("2019", 0) > 75}
    he_2019_le75 = {c: he[c]["2019"] for c in le_high75 if c in he and "2019" in he[c]}
    l3_04 = [c for c, _ in sorted(he_2019_le75.items(), key=lambda x: x[1], reverse=True)[:5]]

    pop10m = {c for c in pop if pop[c].get("2021", 0) > 10_000_000}
    le_2021_pop10m = {c: le[c]["2021"] for c in pop10m if c in le and "2021" in le[c]}
    l3_05 = [c for c, _ in sorted(le_2021_pop10m.items(), key=lambda x: x[1])[:5]]

    cm_low20 = {c for c in cm if cm[c].get("2019", 99) < 20}
    hb_2019_cm20 = {c: hb[c]["2019"] for c in cm_low20 if c in hb and "2019" in hb[c]}
    l3_06 = [c for c, _ in sorted(hb_2019_cm20.items(), key=lambda x: x[1], reverse=True)[:5]]

    cm_g20_2020 = {c: cm[c]["2020"] for c in G20 if c in cm and "2020" in cm[c]}
    l3_07 = [c for c, _ in sorted(cm_g20_2020.items(), key=lambda x: x[1], reverse=True)[:5]]

    mm_low50 = {c for c in mm if mm[c].get("2021", 9999) < 50}
    pop_2021_mm50 = {c: pop[c]["2021"] for c in mm_low50 if c in pop and "2021" in pop[c]}
    l3_08 = [c for c, _ in sorted(pop_2021_mm50.items(), key=lambda x: x[1], reverse=True)[:5]]

    le_ssa_2021 = {c: le[c]["2021"] for c in ssa_codes if c in le and "2021" in le[c]}
    l3_09 = [c for c, _ in sorted(le_ssa_2021.items(), key=lambda x: x[1], reverse=True)[:5]]

    he_g20_2018 = {c: he[c]["2018"] for c in G20 if c in he and "2018" in he[c]}
    l3_10 = [c for c, _ in sorted(he_g20_2018.items(), key=lambda x: x[1])[:5]]

    # -----------------------------------------------------------------------
    # L3 questions 11-25 (NEW)
    # -----------------------------------------------------------------------

    # L3_11: Pop > 25M — lowest child mortality 2021
    pop25m = {c for c in pop if pop[c].get("2021", 0) > 25_000_000}
    cm_2021_pop25m = {c: cm[c]["2021"] for c in pop25m if c in cm and "2021" in cm[c]}
    l3_11 = [c for c, _ in sorted(cm_2021_pop25m.items(), key=lambda x: x[1])[:5]]

    # L3_12: Pop > 75M — highest life expectancy 2021
    pop75m = {c for c in pop if pop[c].get("2021", 0) > 75_000_000}
    le_2021_pop75m = {c: le[c]["2021"] for c in pop75m if c in le and "2021" in le[c]}
    l3_12 = [c for c, _ in sorted(le_2021_pop75m.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_13: Pop > 100M — highest maternal mortality 2020
    pop100m_set = {c for c in pop if pop[c].get("2021", 0) > 100_000_000}
    mm_2020_pop100m = {c: mm[c]["2020"] for c in pop100m_set if c in mm and "2020" in mm[c]}
    l3_13 = [c for c, _ in sorted(mm_2020_pop100m.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_14: (MaternalMort, Population) — MM > 200, highest population 2020
    mm_high200 = {c for c in mm if mm[c].get("2020", 0) > 200}
    pop_2020_mm200 = {c: pop[c]["2020"] for c in mm_high200 if c in pop and "2020" in pop[c]}
    l3_14 = [c for c, _ in sorted(pop_2020_mm200.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_15: (HospitalBeds, LifeExpectancy) — HB > 4, highest LE 2019
    hb_high4 = {c for c in hb if hb[c].get("2019", 0) > 4}
    le_2019_hb4 = {c: le[c]["2019"] for c in hb_high4 if c in le and "2019" in le[c]}
    l3_15 = [c for c, _ in sorted(le_2019_hb4.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_16: (HealthExp, ChildMortality) — HE > 8% GDP, lowest child mortality 2019
    he_high8 = {c for c in he if he[c].get("2019", 0) > 8}
    cm_2019_he8 = {c: cm[c]["2019"] for c in he_high8 if c in cm and "2019" in cm[c]}
    l3_16 = [c for c, _ in sorted(cm_2019_he8.items(), key=lambda x: x[1])[:5]]

    # L3_17: East Asia & Pacific — highest life expectancy 2021
    le_eap_2021 = {c: le[c]["2021"] for c in eap_codes if c in le and "2021" in le[c]}
    l3_17 = [c for c, _ in sorted(le_eap_2021.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_18: High income countries — lowest child mortality 2021
    cm_high_income_2021 = {
        c: cm[c]["2021"] for c in high_income_codes if c in cm and "2021" in cm[c]
    }
    l3_18 = [c for c, _ in sorted(cm_high_income_2021.items(), key=lambda x: x[1])[:5]]

    # L3_19: Low income countries — highest maternal mortality 2020
    mm_low_income_2020 = {
        c: mm[c]["2020"] for c in low_income_codes if c in mm and "2020" in mm[c]
    }
    l3_19 = [c for c, _ in sorted(mm_low_income_2020.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_20: Low income countries — lowest health expenditure 2019
    he_low_income_2019 = {
        c: he[c]["2019"] for c in low_income_codes if c in he and "2019" in he[c]
    }
    l3_20 = [c for c, _ in sorted(he_low_income_2019.items(), key=lambda x: x[1])[:5]]

    # L3_21: Time-series ranking — top 5 LE improvement 2015→2021
    le_improvement = {}
    for c in le:
        if "2015" in le[c] and "2021" in le[c]:
            le_improvement[c] = le[c]["2021"] - le[c]["2015"]
    l3_21 = [c for c, _ in sorted(le_improvement.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_22: BRICS — highest health expenditure 2021
    he_brics_2021 = {c: he[c]["2021"] for c in BRICS if c in he and "2021" in he[c]}
    l3_22 = [c for c, _ in sorted(he_brics_2021.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_23: East Asia & Pacific — highest hospital beds 2019
    hb_eap_2019 = {c: hb[c]["2019"] for c in eap_codes if c in hb and "2019" in hb[c]}
    l3_23 = [c for c, _ in sorted(hb_eap_2019.items(), key=lambda x: x[1], reverse=True)[:5]]

    # L3_24: HospitalBeds > 3 per 1000, lowest maternal mortality 2020
    hb_high3 = {c for c in hb if hb[c].get("2019", 0) > 3}
    mm_2020_hb3 = {c: mm[c]["2020"] for c in hb_high3 if c in mm and "2020" in mm[c]}
    l3_24 = [c for c, _ in sorted(mm_2020_hb3.items(), key=lambda x: x[1])[:5]]

    # L3_25: EU27 — highest life expectancy 2021
    le_eu27_2021 = {c: le[c]["2021"] for c in EU27 if c in le and "2021" in le[c]}
    l3_25 = [c for c, _ in sorted(le_eu27_2021.items(), key=lambda x: x[1], reverse=True)[:5]]

    # -----------------------------------------------------------------------
    # L4 questions 01-10 (UNCHANGED — backward compatibility)
    # -----------------------------------------------------------------------
    le_g20 = [le[c]["2021"] for c in G20 if c in le and "2021" in le[c]]
    l4_01 = round(sum(le_g20) / len(le_g20), 4) if le_g20 else 0.0

    cm_low10 = {c for c in cm if cm[c].get("2021", 99) < 10}
    pop_cm10 = [pop[c]["2021"] for c in cm_low10 if c in pop and "2021" in pop[c]]
    l4_02 = round(sum(pop_cm10), 0)

    le_above80 = {c for c in le if le[c].get("2021", 0) > 80}
    he_le80 = [he[c]["2021"] for c in le_above80 if c in he and "2021" in he[c]]
    l4_03 = round(sum(he_le80) / len(he_le80), 4) if he_le80 else 0.0

    cm_ssa = [cm[c]["2021"] for c in ssa_codes if c in cm and "2021" in cm[c]]
    l4_04 = round(sum(cm_ssa) / len(cm_ssa), 4) if cm_ssa else 0.0

    pop_g20 = [pop[c]["2021"] for c in G20 if c in pop and "2021" in pop[c]]
    l4_05 = round(sum(pop_g20), 0)

    mm_g7 = [mm[c]["2020"] for c in G7 if c in mm and "2020" in mm[c]]
    l4_06 = round(sum(mm_g7) / len(mm_g7), 4) if mm_g7 else 0.0

    l4_07 = sum(1 for c in hb if hb[c].get("2019", 0) > 5)

    he_g20_2021 = [he[c]["2021"] for c in G20 if c in he and "2021" in he[c]]
    l4_08 = round(sum(he_g20_2021) / len(he_g20_2021), 4) if he_g20_2021 else 0.0

    pop_le80 = [pop[c]["2021"] for c in le_above80 if c in pop and "2021" in pop[c]]
    l4_09 = round(sum(pop_le80), 0)

    pop100m = {c for c in pop if pop[c].get("2021", 0) > 100_000_000}
    cm_pop100m = [cm[c]["2021"] for c in pop100m if c in cm and "2021" in cm[c]]
    l4_10 = round(sum(cm_pop100m) / len(cm_pop100m), 4) if cm_pop100m else 0.0

    # -----------------------------------------------------------------------
    # L4 questions 11-25 (NEW)
    # -----------------------------------------------------------------------

    # L4_11: Median life expectancy in 2021 for G20 countries
    le_g20_vals = [le[c]["2021"] for c in G20 if c in le and "2021" in le[c]]
    l4_11 = round(_sorted_median(le_g20_vals), 4)

    # L4_12: Min child mortality in 2021 (all countries)
    cm_all_2021 = [cm[c]["2021"] for c in cm if "2021" in cm[c]]
    l4_12 = round(min(cm_all_2021), 4) if cm_all_2021 else 0.0

    # L4_13: Max maternal mortality in 2020 (all countries)
    mm_all_2020 = [mm[c]["2020"] for c in mm if "2020" in mm[c]]
    l4_13 = round(max(mm_all_2020), 4) if mm_all_2020 else 0.0

    # L4_14: Average health expenditure in 2021 for BRICS countries
    he_brics = [he[c]["2021"] for c in BRICS if c in he and "2021" in he[c]]
    l4_14 = round(sum(he_brics) / len(he_brics), 4) if he_brics else 0.0

    # L4_15: Total population in 2021 for Low income countries
    pop_low_income = [pop[c]["2021"] for c in low_income_codes if c in pop and "2021" in pop[c]]
    l4_15 = round(sum(pop_low_income), 0)

    # L4_16: Average health expenditure among countries with child mortality < 15 in 2021
    cm_low15 = {c for c in cm if cm[c].get("2021", 99) < 15}
    he_cm15 = [he[c]["2021"] for c in cm_low15 if c in he and "2021" in he[c]]
    l4_16 = round(sum(he_cm15) / len(he_cm15), 4) if he_cm15 else 0.0

    # L4_17: Total population change in G20 from 2015 to 2021
    pop_g20_2015 = sum(pop[c]["2015"] for c in G20 if c in pop and "2015" in pop[c])
    pop_g20_2021 = sum(pop[c]["2021"] for c in G20 if c in pop and "2021" in pop[c])
    l4_17 = round(pop_g20_2021 - pop_g20_2015, 0)

    # L4_18: Average life expectancy in 2021 for EU27 countries
    le_eu27 = [le[c]["2021"] for c in EU27 if c in le and "2021" in le[c]]
    l4_18 = round(sum(le_eu27) / len(le_eu27), 4) if le_eu27 else 0.0

    # L4_19: Median child mortality in 2021 for Sub-Saharan Africa countries
    cm_ssa_vals = [cm[c]["2021"] for c in ssa_codes if c in cm and "2021" in cm[c]]
    l4_19 = round(_sorted_median(cm_ssa_vals), 4)

    # L4_20: Count of countries with life expectancy above 75 in 2021
    l4_20 = sum(1 for c in le if le[c].get("2021", 0) > 75)

    # L4_21: Average maternal mortality in 2020 for Low income countries
    mm_low_income = [mm[c]["2020"] for c in low_income_codes if c in mm and "2020" in mm[c]]
    l4_21 = round(sum(mm_low_income) / len(mm_low_income), 4) if mm_low_income else 0.0

    # L4_22: Max hospital beds per 1000 in 2019 (all countries)
    hb_all_2019 = [hb[c]["2019"] for c in hb if "2019" in hb[c]]
    l4_22 = round(max(hb_all_2019), 4) if hb_all_2019 else 0.0

    # L4_23: Median health expenditure in 2021 for High income countries
    he_high_income = [he[c]["2021"] for c in high_income_codes if c in he and "2021" in he[c]]
    l4_23 = round(_sorted_median(he_high_income), 4)

    # L4_24: Average hospital beds in 2019 for G7 countries
    hb_g7 = [hb[c]["2019"] for c in G7 if c in hb and "2019" in hb[c]]
    l4_24 = round(sum(hb_g7) / len(hb_g7), 4) if hb_g7 else 0.0

    # L4_25: Total population in 2021 for East Asia & Pacific countries
    pop_eap = [pop[c]["2021"] for c in eap_codes if c in pop and "2021" in pop[c]]
    l4_25 = round(sum(pop_eap), 0)

    return {
        "l3_01": l3_01, "l3_02": l3_02, "l3_03": l3_03, "l3_04": l3_04, "l3_05": l3_05,
        "l3_06": l3_06, "l3_07": l3_07, "l3_08": l3_08, "l3_09": l3_09, "l3_10": l3_10,
        "l3_11": l3_11, "l3_12": l3_12, "l3_13": l3_13, "l3_14": l3_14, "l3_15": l3_15,
        "l3_16": l3_16, "l3_17": l3_17, "l3_18": l3_18, "l3_19": l3_19, "l3_20": l3_20,
        "l3_21": l3_21, "l3_22": l3_22, "l3_23": l3_23, "l3_24": l3_24, "l3_25": l3_25,
        "l4_01": l4_01, "l4_02": l4_02, "l4_03": l4_03, "l4_04": l4_04, "l4_05": l4_05,
        "l4_06": l4_06, "l4_07": l4_07, "l4_08": l4_08, "l4_09": l4_09, "l4_10": l4_10,
        "l4_11": l4_11, "l4_12": l4_12, "l4_13": l4_13, "l4_14": l4_14, "l4_15": l4_15,
        "l4_16": l4_16, "l4_17": l4_17, "l4_18": l4_18, "l4_19": l4_19, "l4_20": l4_20,
        "l4_21": l4_21, "l4_22": l4_22, "l4_23": l4_23, "l4_24": l4_24, "l4_25": l4_25,
    }


def build_benchmark(ref: dict) -> list[dict]:
    """Build 50 benchmark questions (25 L3 + 25 L4) with precomputed reference answers."""
    questions = [
        # ------------------------------------------------------------------
        # L3: Multi-indicator ranking (25 questions)
        # ------------------------------------------------------------------
        {"id": "multi_L3_01", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["Population", "ChildMortality"],
         "reference_answer": ref["l3_01"],
         "question": "Among countries with population over 50 million in 2021, which 5 countries have the lowest child mortality rate in 2021? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_02", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HealthExpenditure", "LifeExpectancy"],
         "reference_answer": ref["l3_02"],
         "question": "Among countries with health expenditure above 10% of GDP in 2021, which 5 countries have the highest life expectancy in 2021? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_03", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HospitalBeds", "MaternalMortality"],
         "reference_answer": ref["l3_03"],
         "question": "Among countries with fewer than 2 hospital beds per 1000 people in 2020, which 5 countries have the highest maternal mortality ratio in 2020? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_04", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["LifeExpectancy", "HealthExpenditure"],
         "reference_answer": ref["l3_04"],
         "question": "Among countries with life expectancy above 75 years in 2019, which 5 countries have the highest current health expenditure (% of GDP) in 2019? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_05", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["Population", "LifeExpectancy"],
         "reference_answer": ref["l3_05"],
         "question": "Among countries with population over 10 million in 2021, which 5 countries have the lowest life expectancy in 2021? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_06", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["ChildMortality", "HospitalBeds"],
         "reference_answer": ref["l3_06"],
         "question": "Among countries with under-5 child mortality rate below 20 per 1000 in 2019, which 5 countries have the highest number of hospital beds per 1000 people in 2019? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_07", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["ChildMortality"],
         "reference_answer": ref["l3_07"],
         "question": "Among G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA), which 5 have the highest under-5 child mortality rate in 2020? List country codes in order from highest to lowest."},
        {"id": "multi_L3_08", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["MaternalMortality", "Population"],
         "reference_answer": ref["l3_08"],
         "question": "Among countries with maternal mortality ratio below 50 per 100,000 live births in 2021, which 5 countries have the largest population in 2021? List country codes (ISO 3-letter) in order from largest to smallest."},
        {"id": "multi_L3_09", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": ref["l3_09"],
         "question": "Among Sub-Saharan Africa countries, which 5 have the highest life expectancy in 2021? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_10", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": ref["l3_10"],
         "question": "Among G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA), which 5 have the lowest current health expenditure as % of GDP in 2018? List country codes in order from lowest to highest."},
        # New L3 questions (11-25)
        {"id": "multi_L3_11", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["Population", "ChildMortality"],
         "reference_answer": ref["l3_11"],
         "question": "Among countries with population over 25 million in 2021, which 5 countries have the lowest child mortality rate in 2021? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_12", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["Population", "LifeExpectancy"],
         "reference_answer": ref["l3_12"],
         "question": "Among countries with population over 75 million in 2021, which 5 countries have the highest life expectancy in 2021? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_13", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["Population", "MaternalMortality"],
         "reference_answer": ref["l3_13"],
         "question": "Among countries with population over 100 million in 2021, which 5 countries have the highest maternal mortality ratio in 2020? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_14", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["MaternalMortality", "Population"],
         "reference_answer": ref["l3_14"],
         "question": "Among countries with maternal mortality ratio above 200 per 100,000 live births in 2020, which 5 countries have the largest population in 2020? List country codes (ISO 3-letter) in order from largest to smallest."},
        {"id": "multi_L3_15", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HospitalBeds", "LifeExpectancy"],
         "reference_answer": ref["l3_15"],
         "question": "Among countries with more than 4 hospital beds per 1000 people in 2019, which 5 countries have the highest life expectancy in 2019? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_16", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HealthExpenditure", "ChildMortality"],
         "reference_answer": ref["l3_16"],
         "question": "Among countries with health expenditure above 8% of GDP in 2019, which 5 countries have the lowest under-5 child mortality rate in 2019? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_17", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": ref["l3_17"],
         "question": "Among East Asia & Pacific countries, which 5 have the highest life expectancy in 2021? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_18", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["ChildMortality"],
         "reference_answer": ref["l3_18"],
         "question": "Among High income countries, which 5 have the lowest under-5 child mortality rate in 2021? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_19", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["MaternalMortality"],
         "reference_answer": ref["l3_19"],
         "question": "Among Low income countries, which 5 have the highest maternal mortality ratio in 2020? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_20", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": ref["l3_20"],
         "question": "Among Low income countries, which 5 have the lowest current health expenditure as % of GDP in 2019? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_21", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": ref["l3_21"],
         "question": "Which 5 countries had the largest improvement in life expectancy from 2015 to 2021 (i.e., highest increase in years)? List country codes (ISO 3-letter) in order from largest to smallest improvement."},
        {"id": "multi_L3_22", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": ref["l3_22"],
         "question": "Among BRICS countries (BRA, RUS, IND, CHN, ZAF), which 5 have the highest current health expenditure as % of GDP in 2021? List country codes in order from highest to lowest."},
        {"id": "multi_L3_23", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HospitalBeds"],
         "reference_answer": ref["l3_23"],
         "question": "Among East Asia & Pacific countries, which 5 have the highest number of hospital beds per 1000 people in 2019? List country codes (ISO 3-letter) in order from highest to lowest."},
        {"id": "multi_L3_24", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["HospitalBeds", "MaternalMortality"],
         "reference_answer": ref["l3_24"],
         "question": "Among countries with more than 3 hospital beds per 1000 people in 2019, which 5 countries have the lowest maternal mortality ratio in 2020? List country codes (ISO 3-letter) in order from lowest to highest."},
        {"id": "multi_L3_25", "level": "L3", "scoring": "set_overlap",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": ref["l3_25"],
         "question": "Among EU27 countries (AUT, BEL, BGR, HRV, CYP, CZE, DNK, EST, FIN, FRA, DEU, GRC, HUN, IRL, ITA, LVA, LTU, LUX, MLT, NLD, POL, PRT, ROU, SVK, SVN, ESP, SWE), which 5 have the highest life expectancy in 2021? List country codes in order from highest to lowest."},
        # ------------------------------------------------------------------
        # L4: Cross-indicator aggregation (25 questions)
        # ------------------------------------------------------------------
        {"id": "multi_L4_01", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": str(ref["l4_01"]), "reference_numeric": ref["l4_01"],
         "question": "What is the average life expectancy in 2021 for G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA)? Report the numeric value."},
        {"id": "multi_L4_02", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality", "Population"],
         "reference_answer": str(int(ref["l4_02"])), "reference_numeric": ref["l4_02"],
         "question": "What is the total population in 2021 of countries with under-5 child mortality rate below 10 per 1000 live births? Report the numeric value."},
        {"id": "multi_L4_03", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy", "HealthExpenditure"],
         "reference_answer": str(ref["l4_03"]), "reference_numeric": ref["l4_03"],
         "question": "What is the mean current health expenditure (% of GDP) in 2021 for countries with life expectancy above 80 years in 2021? Report the numeric value."},
        {"id": "multi_L4_04", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality"],
         "reference_answer": str(ref["l4_04"]), "reference_numeric": ref["l4_04"],
         "question": "What is the average under-5 child mortality rate in 2021 for Sub-Saharan Africa countries? Report the numeric value."},
        {"id": "multi_L4_05", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["Population"],
         "reference_answer": str(int(ref["l4_05"])), "reference_numeric": ref["l4_05"],
         "question": "What is the combined total population in 2021 of all G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA)? Report the numeric value."},
        {"id": "multi_L4_06", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["MaternalMortality"],
         "reference_answer": str(ref["l4_06"]), "reference_numeric": ref["l4_06"],
         "question": "What is the average maternal mortality ratio in 2020 for G7 countries (USA, GBR, FRA, DEU, JPN, ITA, CAN)? Report the numeric value."},
        {"id": "multi_L4_07", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HospitalBeds"],
         "reference_answer": str(ref["l4_07"]), "reference_numeric": float(ref["l4_07"]),
         "question": "How many countries have more than 5 hospital beds per 1000 people in 2019? Report the count."},
        {"id": "multi_L4_08", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": str(ref["l4_08"]), "reference_numeric": ref["l4_08"],
         "question": "What is the average current health expenditure (% of GDP) in 2021 for G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA)? Report the numeric value."},
        {"id": "multi_L4_09", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy", "Population"],
         "reference_answer": str(int(ref["l4_09"])), "reference_numeric": ref["l4_09"],
         "question": "What is the total population in 2021 of all countries with life expectancy above 80 years in 2021? Report the numeric value."},
        {"id": "multi_L4_10", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality", "Population"],
         "reference_answer": str(ref["l4_10"]), "reference_numeric": ref["l4_10"],
         "question": "What is the average under-5 child mortality rate in 2021 for countries with population over 100 million people in 2021? Report the numeric value."},
        # New L4 questions (11-25)
        {"id": "multi_L4_11", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": str(ref["l4_11"]), "reference_numeric": ref["l4_11"],
         "question": "What is the median life expectancy in 2021 for G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA)? Report the numeric value."},
        {"id": "multi_L4_12", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality"],
         "reference_answer": str(ref["l4_12"]), "reference_numeric": ref["l4_12"],
         "question": "What is the minimum under-5 child mortality rate in 2021 across all countries? Report the numeric value."},
        {"id": "multi_L4_13", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["MaternalMortality"],
         "reference_answer": str(ref["l4_13"]), "reference_numeric": ref["l4_13"],
         "question": "What is the maximum maternal mortality ratio in 2020 across all countries? Report the numeric value."},
        {"id": "multi_L4_14", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": str(ref["l4_14"]), "reference_numeric": ref["l4_14"],
         "question": "What is the average current health expenditure (% of GDP) in 2021 for BRICS countries (BRA, RUS, IND, CHN, ZAF)? Report the numeric value."},
        {"id": "multi_L4_15", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["Population"],
         "reference_answer": str(int(ref["l4_15"])), "reference_numeric": ref["l4_15"],
         "question": "What is the combined total population in 2021 of all Low income countries? Report the numeric value."},
        {"id": "multi_L4_16", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality", "HealthExpenditure"],
         "reference_answer": str(ref["l4_16"]), "reference_numeric": ref["l4_16"],
         "question": "What is the average current health expenditure (% of GDP) in 2021 for countries with under-5 child mortality rate below 15 per 1000 live births in 2021? Report the numeric value."},
        {"id": "multi_L4_17", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["Population"],
         "reference_answer": str(int(ref["l4_17"])), "reference_numeric": ref["l4_17"],
         "question": "What is the total population change (2021 minus 2015) across all G20 countries (ARG, AUS, BRA, CAN, CHN, FRA, DEU, IND, IDN, ITA, JPN, KOR, MEX, RUS, SAU, ZAF, TUR, GBR, USA)? Report the numeric value."},
        {"id": "multi_L4_18", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": str(ref["l4_18"]), "reference_numeric": ref["l4_18"],
         "question": "What is the average life expectancy in 2021 for EU27 countries (AUT, BEL, BGR, HRV, CYP, CZE, DNK, EST, FIN, FRA, DEU, GRC, HUN, IRL, ITA, LVA, LTU, LUX, MLT, NLD, POL, PRT, ROU, SVK, SVN, ESP, SWE)? Report the numeric value."},
        {"id": "multi_L4_19", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["ChildMortality"],
         "reference_answer": str(ref["l4_19"]), "reference_numeric": ref["l4_19"],
         "question": "What is the median under-5 child mortality rate in 2021 for Sub-Saharan Africa countries? Report the numeric value."},
        {"id": "multi_L4_20", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["LifeExpectancy"],
         "reference_answer": str(ref["l4_20"]), "reference_numeric": float(ref["l4_20"]),
         "question": "How many countries have life expectancy above 75 years in 2021? Report the count."},
        {"id": "multi_L4_21", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["MaternalMortality"],
         "reference_answer": str(ref["l4_21"]), "reference_numeric": ref["l4_21"],
         "question": "What is the average maternal mortality ratio in 2020 for Low income countries? Report the numeric value."},
        {"id": "multi_L4_22", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HospitalBeds"],
         "reference_answer": str(ref["l4_22"]), "reference_numeric": ref["l4_22"],
         "question": "What is the maximum number of hospital beds per 1000 people in 2019 across all countries? Report the numeric value."},
        {"id": "multi_L4_23", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HealthExpenditure"],
         "reference_answer": str(ref["l4_23"]), "reference_numeric": ref["l4_23"],
         "question": "What is the median current health expenditure (% of GDP) in 2021 for High income countries? Report the numeric value."},
        {"id": "multi_L4_24", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["HospitalBeds"],
         "reference_answer": str(ref["l4_24"]), "reference_numeric": ref["l4_24"],
         "question": "What is the average number of hospital beds per 1000 people in 2019 for G7 countries (USA, GBR, FRA, DEU, JPN, ITA, CAN)? Report the numeric value."},
        {"id": "multi_L4_25", "level": "L4", "scoring": "numeric_tolerance",
         "required_indicators": ["Population"],
         "reference_answer": str(int(ref["l4_25"])), "reference_numeric": ref["l4_25"],
         "question": "What is the combined total population in 2021 of all East Asia & Pacific countries? Report the numeric value."},
    ]
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-indicator data preparation")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Multi-Indicator Data Preparation")
    print("=" * 60)

    # Load metadata
    country_meta = load_wb_metadata()
    print(f"\nCountry metadata: {len(country_meta)} countries")

    # Load all indicators
    indicator_data: dict[str, dict[str, dict[str, float]]] = {}
    for ind_name in INDICATOR_CONFIGS:
        data = load_indicator_data(ind_name, country_meta)
        indicator_data[ind_name] = data
        print(f"  {ind_name}: {len(data)} countries")

    n_entities = sum(len(d) for d in indicator_data.values())
    print(f"  Total entities: {n_entities}")

    # Build chunks
    chunks, entity_map = build_all_chunks(indicator_data, country_meta)
    total_tokens = sum(len(c) for c in chunks) // 4
    print(f"\nCompact chunks: {len(chunks)}")
    print(f"Estimated total tokens: {total_tokens:,}")

    # Build benchmark
    ref = compute_reference_answers(indicator_data, country_meta)
    questions = build_benchmark(ref)
    print(f"\nBenchmark: {len(questions)} questions")
    print(f"  L3: {sum(1 for q in questions if q['level'] == 'L3')}")
    print(f"  L4: {sum(1 for q in questions if q['level'] == 'L4')}")

    # Show sample reference answers
    for key in ["l3_01", "l3_05", "l4_01", "l4_05"]:
        print(f"  {key}: {ref[key]}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "multi_indicator_data.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_entities": n_entities,
        "n_indicators": len(INDICATOR_CONFIGS),
        "n_chunks": len(chunks),
        "total_tokens_estimate": total_tokens,
        "chunks": chunks,
        "entity_map": {k: v for k, v in entity_map.items() if not k.islower() or ":" not in k},
        "indicator_data": {
            ind: {code: vals for code, vals in data.items()}
            for ind, data in indicator_data.items()
        },
        "country_meta": country_meta,
        "benchmark": questions,
        "reference_answers": ref,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
