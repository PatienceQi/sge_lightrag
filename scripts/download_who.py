#!/usr/bin/env python3
"""
download_who.py — Download WHO GHO life expectancy data and save as wide-format CSV.

The output mimics World Bank format:
  Country Name, Country Code, Indicator Name, Indicator Code, 2000, 2001, ..., 2022

Usage:
    python3 scripts/download_who.py
    python3 scripts/download_who.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

GHO_BASE = "https://ghoapi.azureedge.net/api"

# Life expectancy at birth (both sexes) — analogous to World Bank SP.POP.TOTL
INDICATOR_CODE = "WHOSIS_000001"
INDICATOR_NAME = "Life expectancy at birth (years)"

# Country code → name mapping (ISO 3-letter → display name)
_COUNTRY_URL = f"{GHO_BASE}/DIMENSION/COUNTRY/DimensionValues"


def fetch_all_pages(url: str, params: dict | None = None) -> list[dict]:
    """Fetch all OData pages (follow @odata.nextLink)."""
    records: list[dict] = []
    next_url: str | None = url
    page = 0
    while next_url:
        page += 1
        try:
            r = requests.get(next_url, params=params if page == 1 else None,
                             timeout=30, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [warn] Page {page} failed: {e}")
            time.sleep(2)
            continue
        records.extend(data.get("value", []))
        next_url = data.get("@odata.nextLink")
        params = None  # params only on first request
        print(f"  Fetched page {page}: {len(records)} records total")
    return records


def build_country_map() -> dict[str, str]:
    """Return {CountryCode: CountryName} from WHO dimension API."""
    try:
        r = requests.get(_COUNTRY_URL, timeout=20,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        items = r.json().get("value", [])
        return {item["Code"]: item["Title"] for item in items if "Code" in item}
    except Exception as e:
        print(f"  [warn] Could not load country map: {e}")
        return {}


def download_life_expectancy(output_dir: Path) -> Path:
    """Download WHOSIS_000001 (both sexes) and save as wide-format CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "API_WHO_WHOSIS_000001_life_expectancy.csv"

    print(f"Downloading WHO GHO: {INDICATOR_NAME}")
    print(f"  Indicator: {INDICATOR_CODE}, Filter: both sexes (SEX_BTSX)")

    # Fetch data (both sexes only, to keep it clean)
    records = fetch_all_pages(
        f"{GHO_BASE}/{INDICATOR_CODE}",
        params={"$filter": "Dim1 eq 'SEX_BTSX'",
                "$select": "SpatialDim,TimeDim,NumericValue"},
    )

    if not records:
        raise RuntimeError("No records returned from WHO GHO API")

    df_long = pd.DataFrame(records)
    df_long.columns = ["Country Code", "Year", "Value"]
    df_long = df_long.dropna(subset=["Value"])
    df_long["Year"] = df_long["Year"].astype(int)

    # Pivot to wide format: rows = countries, columns = years
    df_wide = df_long.pivot_table(
        index="Country Code", columns="Year", values="Value", aggfunc="first"
    ).reset_index()
    df_wide.columns.name = None

    # Add country name column
    country_map = build_country_map()
    df_wide.insert(0, "Country Name", df_wide["Country Code"].map(country_map).fillna(""))
    df_wide.insert(2, "Indicator Name", INDICATOR_NAME)
    df_wide.insert(3, "Indicator Code", INDICATOR_CODE)

    df_wide.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {output_path}")
    print(f"  Rows (countries): {len(df_wide)}, Year columns: {df_long['Year'].nunique()}")

    # Also save Metadata_Country in same format as World Bank
    meta_path = output_dir / "Metadata_Country_WHO_WHOSIS_000001.csv"
    meta_records = [
        {"Country Code": code, "Country Name": name, "Region": "", "IncomeGroup": ""}
        for code, name in country_map.items()
    ]
    pd.DataFrame(meta_records).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"  Metadata saved: {meta_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WHO GHO life expectancy data")
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).parent.parent / "dataset" / "WHO"),
        help="Directory to save output CSV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    download_life_expectancy(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
