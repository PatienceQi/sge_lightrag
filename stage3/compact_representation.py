"""
compact_representation.py — Compact serialization for large Type-II time-series datasets.

Problem
-------
Current SGE strategy creates one StatValue node per (entity, year) pair.
For WB Child Mortality (244 countries × ~23 years) this yields ~5218 nodes vs
Baseline's 383 nodes — a 13.6× inflation that increases storage and computation
cost without proportionally improving retrieval quality.

Solution
--------
When n_rows > COMPACT_THRESHOLD:
  - Collapse all year-value observations into the entity's description as a compact
    time-series string:  "year=value; year=value; ..."
  - LightRAG creates ONE entity node per country (not one per year-value pair)
  - All year-specific data is stored in the entity description → substring-matched
    by the evaluator, same as before

Expected impact (WB CM):
  - Independent strategy: 244 × ~23 = 5218 nodes, 5509 edges
  - Compact strategy:     ~244 country nodes + 1 indicator node ≈ 245 nodes
  - Reduction: ~21×  (from 5218 → ~245)

Tradeoff
--------
The compact entity description contains "2000=45.2; 2005=35.1; ..." as a flat
string rather than separate linked nodes. Multi-hop graph traversal (country→year
→value) is replaced by substring search within the entity description. For the
EC/FC evaluation (substring matching), both approaches produce equivalent results.
For complex structural reasoning over specific year relationships, the independent
approach provides richer graph topology.
"""

from __future__ import annotations

import re
import pandas as pd

# Tables with more than this many data rows use compact representation
COMPACT_THRESHOLD = 100

# Maximum number of years to include in a compact chunk (to stay within context)
MAX_YEARS_PER_CHUNK = 64


def should_use_compact(schema: dict, n_rows: int) -> bool:
    """
    Return True if compact representation should be used.

    Conditions:
    - Table is Type-II (Time-Series-Matrix)
    - n_rows > COMPACT_THRESHOLD
    - Not adaptive baseline mode
    """
    if schema.get("use_baseline_mode", False):
        return False
    if schema.get("table_type") != "Time-Series-Matrix":
        return False
    return n_rows > COMPACT_THRESHOLD


def compact_serialize_type_ii(df: pd.DataFrame, schema: dict) -> list[str]:
    """
    Serialize a large Type-II table using compact timeseries representation.

    Format per entity:
        Entity: CHN (China) | Indicator: UNDER5_MORTALITY_RATE
        Timeseries (per 1,000 live births): 2000=45.2; 2001=42.1; ...; 2022=7.1

    Each country gets ONE chunk regardless of how many years it has.

    Parameters
    ----------
    df     : Full DataFrame (all rows, not sampled)
    schema : Stage 2 extraction schema

    Returns
    -------
    list[str] — text chunks, one per entity
    """
    column_roles = schema.get("column_roles", {})

    subject_cols   = [c for c, r in column_roles.items() if r == "subject"]
    time_value_cols = [c for c, r in column_roles.items() if r == "time_value"]
    metadata_cols  = [c for c, r in column_roles.items() if r == "metadata"]

    # Detect indicator name from metadata columns
    indicator_col = _find_indicator_col(df, metadata_cols)
    country_name_col = _find_country_name_col(df, metadata_cols, subject_cols)

    # Parse time column headers
    year_cols = _extract_year_cols(time_value_cols)

    chunks = []

    for _, row in df.iterrows():
        # Build entity identifier (subject columns)
        entity_parts = []
        for col in subject_cols:
            if col in row and _is_valid(row[col]):
                entity_parts.append(str(row[col]).strip())
        if not entity_parts:
            continue
        entity_id = " / ".join(entity_parts)

        # Optional: long country name
        country_name = ""
        if country_name_col and country_name_col in row:
            v = str(row[country_name_col]).strip()
            if _is_valid(v) and v != entity_id:
                country_name = v

        # Optional: indicator name
        indicator_name = ""
        if indicator_col and indicator_col in row:
            v = str(row[indicator_col]).strip()
            if _is_valid(v):
                indicator_name = v

        # Build compact timeseries string
        year_values = []
        for col in time_value_cols:
            if col not in row or not _is_valid(row[col]):
                continue
            val = row[col]
            try:
                fval = float(str(val))
            except (ValueError, TypeError):
                continue
            # Get year from column name
            year = year_cols.get(col, str(col))
            year_values.append((year, fval))

        if not year_values:
            continue

        # Sort by year
        year_values.sort(key=lambda x: x[0])

        # Truncate if too many years
        if len(year_values) > MAX_YEARS_PER_CHUNK:
            year_values = year_values[-MAX_YEARS_PER_CHUNK:]

        ts_str = "; ".join(f"{yr}={val}" for yr, val in year_values)

        # Assemble chunk
        header_parts = [f"Entity: {entity_id}"]
        if country_name:
            header_parts.append(f"Name: {country_name}")
        if indicator_name:
            header_parts.append(f"Indicator: {indicator_name}")

        unit = schema.get("unit", "")
        unit_str = f" ({unit})" if unit else ""
        ts_line = f"Timeseries{unit_str}: {ts_str}"

        chunk = " | ".join(header_parts) + "\n" + ts_line
        chunks.append(chunk)

    return chunks


def build_compact_system_prompt(schema: dict, language: str = "Chinese") -> str:
    """
    Build a schema-aware system prompt for compact timeseries representation.

    Key difference from standard prompt:
    - Instructs LightRAG to create ONE entity node per country
    - Year-value data stored in entity DESCRIPTION, NOT as separate nodes
    - Prevents StatValue node explosion
    """
    entity_types = schema.get("entity_types", ["Country_Code"])
    # Remove StatValue from entity types — compact mode stores values in description
    entity_types_compact = [t for t in entity_types if t != "StatValue"]
    if not entity_types_compact:
        entity_types_compact = ["Country_Code"]

    relation_types = schema.get("relation_types", ["HAS_VALUE"])

    entity_types_str = ", ".join(entity_types_compact)
    relation_types_str = ", ".join(relation_types)

    prompt = f"""---Role---
You are a Knowledge Graph Specialist extracting structured data from time-series CSV records.
Follow the compact extraction rules below EXACTLY.

---Compact Extraction Rules---
1. Create ONE entity node per country/subject (entity type: {entity_types_str})
2. The entity DESCRIPTION must include ALL year-value observations as a compact string:
   "year=value; year=value; ..." (e.g., "2000=45.2; 2005=35.1; 2010=25.3")
3. Do NOT create separate entity nodes for individual years or numeric values
4. Relationship types allowed: {relation_types_str}
5. If an indicator/metric is present, create ONE indicator node and link to it
6. Output language: {language}

---Output Format---
entity<|#|>entity_name<|#|>entity_type<|#|>entity_description_with_timeseries
relation<|#|>source<|#|>target<|#|>keywords<|#|>description
<|COMPLETE|>

---Example---
entity<|#|>CHN<|#|>Country_Code<|#|>China. UNDER5_MORTALITY_RATE timeseries: 2000=45.2; 2005=35.1; 2010=25.3; 2015=18.1; 2020=8.3 per 1,000 live births
relation<|#|>CHN<|#|>UNDER5_MORTALITY_RATE<|#|>HAS_VALUE<|#|>China's under-5 mortality rate time series
<|COMPLETE|>
"""
    return prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_year_cols(time_value_cols: list) -> dict:
    """Map column name → 4-digit year string."""
    year_re = re.compile(r"\b((?:19|20)\d{2})\b")
    result = {}
    for col in time_value_cols:
        m = year_re.search(str(col))
        if m:
            result[col] = m.group(1)
        else:
            result[col] = str(col)
    return result


def _find_indicator_col(df: pd.DataFrame, metadata_cols: list) -> str | None:
    """Find the column most likely to hold the indicator/metric name."""
    for col in metadata_cols:
        if col not in df.columns:
            continue
        name_lower = str(col).lower()
        if any(kw in name_lower for kw in ("indicator", "metric", "series")):
            return col
    return None


def _find_country_name_col(
    df: pd.DataFrame, metadata_cols: list, subject_cols: list
) -> str | None:
    """Find a column holding the full country name (complementing the code)."""
    for col in metadata_cols:
        if col not in df.columns:
            continue
        name_lower = str(col).lower()
        if any(kw in name_lower for kw in ("country name", "country", "name", "nation")):
            return col
    return None


def _is_valid(val) -> bool:
    """Return True if val is non-null and non-empty."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() not in ("nan", "n/a", "..", "")
