"""
templates.py — Extraction template generators per table type.

Each function receives the parsed schema components and returns
human-readable template strings that Stage 3 (entity/relation extraction)
can use as prompts or rule specifications.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Name derivation helpers
# ---------------------------------------------------------------------------

# Simple Chinese→English concept mappings for common column names
_CN_TO_EN: dict[str, str] = {
    "疾病": "Disease",
    "病种": "Disease",
    "纲领": "Policy_Program",
    "项目": "Program_Item",
    "计划": "Plan",
    "服务": "Service",
    "机构": "Institution",
    "部门": "Department",
    "地区": "Region",
    "年份": "Year",
    "年度": "Fiscal_Year",
    "数据内容": "Data_Category",
    "内容分类": "Content_Type",
    "食物安全": "Food_Safety",
    "医疗": "Healthcare",
    "卫生": "Health",
    "统计": "Statistics",
}


def derive_entity_name(col_name: str) -> str:
    """
    Derive a PascalCase entity type name from a column name.

    Strategy:
    1. Direct lookup in known Chinese→English map
    2. Strip punctuation/numbers, title-case remaining ASCII
    3. Fall back to "Entity"
    """
    clean = col_name.strip()

    # Direct lookup
    if clean in _CN_TO_EN:
        return _CN_TO_EN[clean]

    # Partial match (column name contains a known keyword)
    for cn, en in _CN_TO_EN.items():
        if cn in clean:
            return en

    # ASCII fallback: title-case words
    import re
    ascii_words = re.findall(r"[A-Za-z]+", clean)
    if ascii_words:
        return "_".join(w.capitalize() for w in ascii_words)

    # Last resort: use the raw name with spaces replaced
    safe = re.sub(r"[^\w\u4e00-\u9fff]", "_", clean)
    return safe if safe else "Entity"


def detect_relation_name(time_cols: list[str], value_cols: list[str]) -> str:
    """
    Heuristically pick the best relation type name for time-value links.

    Looks at column names for budget/financial keywords → HAS_BUDGET
    Otherwise defaults to HAS_VALUE.
    """
    import re
    budget_kw = re.compile(r"预算|budget|expenditure|支出|拨款|allocation", re.IGNORECASE)
    all_cols = time_cols + value_cols
    for col in all_cols:
        if budget_kw.search(str(col)):
            return "HAS_BUDGET"
    return "HAS_VALUE"


# ---------------------------------------------------------------------------
# Per-type template generators
# ---------------------------------------------------------------------------

def type_i_templates(
    subject_col: str,
    value_cols: list[str],
    entity_name: str,
) -> dict:
    """Generate extraction templates for Type I (Flat-Entity) tables."""
    attr_list = ", ".join(f'"{c}"' for c in value_cols[:4])
    more = f" (and {len(value_cols) - 4} more)" if len(value_cols) > 4 else ""

    entity_tmpl = (
        f"For each row, extract the value of '{subject_col}' "
        f"as a {entity_name} entity node."
    )
    relation_tmpl = (
        f"For each {entity_name} entity, create HAS_ATTRIBUTE relations "
        f"to numeric values in columns: {attr_list}{more}."
    )
    constraints = [
        f"Each row represents one distinct {entity_name} entity.",
        f"Use '{subject_col}' as the entity identifier.",
        "Numeric columns become attribute values linked via HAS_ATTRIBUTE.",
        "Skip rows where the subject column is empty or null.",
    ]
    return {
        "entity_extraction_template": entity_tmpl,
        "relation_extraction_template": relation_tmpl,
        "extraction_constraints": constraints,
    }


def type_ii_templates(
    subject_col: str,
    time_cols: list[str],
    parsed_headers: list,   # list[ParsedHeader]
    entity_name: str,
    relation_name: str,
    transposed: bool = False,
) -> dict:
    """Generate extraction templates for Type II (Time-Series-Matrix) tables."""
    if transposed:
        entity_tmpl = (
            f"Each column (except the first) represents a {entity_name} metric. "
            f"The first column contains time periods (years)."
        )
        relation_tmpl = (
            f"For each (metric, year) cell, create: "
            f"(metric:{entity_name}) -[{relation_name} {{year: <row_year>}}]-> (value)."
        )
        constraints = [
            "Table is transposed: rows are time periods, columns are metrics.",
            "First column contains year labels.",
            f"Each non-year column is a {entity_name} metric entity.",
            f"Create {relation_name} relations with year attribute from the row label.",
        ]
    else:
        # Build a sample time-column description
        sample_time = []
        for ph in parsed_headers[:3]:
            if ph.is_time_column:
                parts = [ph.year]
                if ph.status:
                    parts.append(ph.status)
                if ph.unit:
                    parts.append(ph.unit)
                sample_time.append("{" + ", ".join(parts) + "}")
        sample_str = "; ".join(sample_time) if sample_time else str(time_cols[:3])

        entity_tmpl = (
            f"For each row, extract the value of '{subject_col}' "
            f"as a {entity_name} entity node."
        )
        relation_tmpl = (
            f"For each time column, create: "
            f"({entity_name}) -[{relation_name} {{year: X, status: Y, unit: Z}}]-> (value). "
            f"Example time columns: {sample_str}."
        )
        constraints = [
            f"Each row represents one {entity_name} entity identified by '{subject_col}'.",
            f"Time columns encode fiscal year, status, and unit in compound headers.",
            f"Parse compound headers (newline-separated) into year/status/unit attributes.",
            f"Create one {relation_name} relation per (entity, time_column) pair.",
            "Skip cells with null or non-numeric values.",
        ]

    return {
        "entity_extraction_template": entity_tmpl,
        "relation_extraction_template": relation_tmpl,
        "extraction_constraints": constraints,
    }


def type_iii_templates(
    key_cols: list[str],
    value_cols: list[str],
    remarks_cols: list[str],
    entity_names: list[str],
) -> dict:
    """Generate extraction templates for Type III (Hierarchical-Hybrid) tables."""
    top_entity = entity_names[0] if entity_names else "Category"
    sub_entity = entity_names[1] if len(entity_names) > 1 else "Item"
    leaf_entity = entity_names[2] if len(entity_names) > 2 else sub_entity

    key_desc = " → ".join(f"'{c}'" for c in key_cols)
    val_list = ", ".join(f'"{c}"' for c in value_cols[:4])
    more = f" (and {len(value_cols) - 4} more)" if len(value_cols) > 4 else ""

    entity_tmpl = (
        f"Composite key columns: {key_desc}. "
        f"Empty cells inherit the last non-empty value from the same column above. "
        f"Top-level key '{key_cols[0]}' → {top_entity} entity. "
        + (f"Second-level key '{key_cols[1]}' → {sub_entity} entity. " if len(key_cols) > 1 else "")
        + (f"Leaf key '{key_cols[-1]}' → {leaf_entity} entity." if len(key_cols) > 2 else "")
    )
    relation_tmpl = (
        f"Create HAS_SUB_ITEM relations between hierarchy levels: "
        f"({top_entity}) -[HAS_SUB_ITEM]-> ({sub_entity}). "
        f"Create HAS_VALUE relations from leaf entities to numeric columns: {val_list}{more}."
        + (f" Attach remarks columns {remarks_cols} as metadata properties on the leaf entity." if remarks_cols else "")
    )
    constraints = [
        "Detect hierarchy by sparse fill: empty cells in key columns inherit the parent value.",
        f"Top-level key '{key_cols[0]}' defines the root entity ({top_entity}).",
    ]
    if len(key_cols) > 1:
        constraints.append(
            f"Second-level key '{key_cols[1]}' defines child entities ({sub_entity}) "
            f"linked to their parent via HAS_SUB_ITEM."
        )
    if len(key_cols) > 2:
        constraints.append(
            f"Leaf key '{key_cols[-1]}' defines the most granular entity ({leaf_entity})."
        )
    constraints += [
        "Numeric columns become HAS_VALUE relations from the leaf entity.",
        "Remarks/notes columns become metadata properties (not separate entities).",
        "Skip rows that are entirely empty or contain only section headers.",
    ]

    return {
        "entity_extraction_template": entity_tmpl,
        "relation_extraction_template": relation_tmpl,
        "extraction_constraints": constraints,
    }
