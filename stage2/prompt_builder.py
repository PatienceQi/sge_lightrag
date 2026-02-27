"""
prompt_builder.py — Generate natural-language prompt_context strings.

Produces a Chinese paragraph (with English technical terms) that describes
how to read and extract knowledge from a given table. This paragraph is
injected into LightRAG's extraction prompt in Stage 3.
"""

from stage1.classifier import TYPE_I, TYPE_II, TYPE_III


def build_prompt_context(
    table_type: str,
    entity_types: list[str],
    relation_types: list[str],
    extraction_rules: dict,
    type_specific: dict,
) -> str:
    """
    Generate a Chinese natural-language paragraph describing the table structure
    and extraction strategy.

    Parameters
    ----------
    table_type       : one of TYPE_I / TYPE_II / TYPE_III
    entity_types     : list of entity type names
    relation_types   : list of relation type names
    extraction_rules : dict with keys subject_extraction, value_extraction,
                       time_handling, hierarchy, remarks
    type_specific    : the type-handler output dict (for extra context)

    Returns
    -------
    A Chinese string suitable for injection into an LLM extraction prompt.
    """
    if table_type == TYPE_I:
        return _prompt_type_i(entity_types, relation_types, extraction_rules)
    elif table_type == TYPE_II:
        return _prompt_type_ii(entity_types, relation_types, extraction_rules, type_specific)
    elif table_type == TYPE_III:
        return _prompt_type_iii(entity_types, relation_types, extraction_rules, type_specific)
    else:
        return _prompt_generic(table_type, entity_types, relation_types, extraction_rules)


# ---------------------------------------------------------------------------
# Type-specific prompt builders
# ---------------------------------------------------------------------------

def _prompt_type_i(entity_types, relation_types, rules) -> str:
    et = "、".join(entity_types) if entity_types else "Entity"
    return (
        f"本表为 Flat-Entity 类型（扁平实体属性表）。"
        f"表中每一行代表一个独立的实体，实体类型为 {et}。"
        f"请将每行的主键列作为实体节点的标识符（entity identifier），"
        f"并将各数值列作为该实体的属性（attribute），"
        f"通过 HAS_ATTRIBUTE 关系连接到对应的数值。"
        f"提取时请跳过主键列为空的行。"
        f"所有实体节点和关系均应以结构化三元组（subject, relation, object）的形式输出。"
    )


def _prompt_type_ii(entity_types, relation_types, rules, type_specific) -> str:
    et = "、".join(entity_types) if entity_types else "Entity"
    rel = relation_types[0] if relation_types else "HAS_VALUE"
    transposed = "pivot_instruction" in type_specific
    time_rules = type_specific.get("time_parsing_rules", {})
    statuses = time_rules.get("known_statuses", [])
    units = time_rules.get("known_units", [])

    if transposed:
        return (
            f"本表为 Time-Series-Matrix 类型（时间序列矩阵表），且为转置布局（transposed layout）。"
            f"表中行代表时间周期（time period），列代表指标实体（metric entity），实体类型为 {et}。"
            f"第一列包含年份标签，请将其解析为时间维度（time dimension）。"
            f"对每个（指标, 年份）单元格，提取三元组："
            f"（{et} 实体）-[{rel} {{year: <行年份>}}]->（数值）。"
            f"请跳过数值为空或非数字的单元格。"
        )
    else:
        status_str = "、".join(statuses) if statuses else "实际/预算/修订"
        unit_str = "、".join(units) if units else "百万元"
        return (
            f"本表为 Time-Series-Matrix 类型（时间序列矩阵表）。"
            f"表中每一行代表一个实体（entity），实体类型为 {et}，由主键列唯一标识。"
            f"列标题为复合标题（compound header），以换行符分隔，"
            f"包含财政年度（fiscal year）、状态（status：{status_str}）和单位（unit：{unit_str}）三个组成部分。"
            f"请将每个复合标题解析为 {{year, status, unit}} 结构。"
            f"对每个（实体, 时间列）组合，提取三元组："
            f"（{et} 实体）-[{rel} {{year: X, status: Y, unit: Z}}]->（数值）。"
            f"请跳过数值为空或非数字的单元格。"
        )


def _prompt_type_iii(entity_types, relation_types, rules, type_specific) -> str:
    et_list = entity_types if entity_types else ["Category", "Item"]
    et_str = "、".join(et_list)
    hierarchy_rels = type_specific.get("hierarchy_relations", [])
    value_mapping = type_specific.get("value_mapping", {})
    remarks_rule = rules.get("remarks", "")

    # Build hierarchy description
    if hierarchy_rels:
        hier_parts = []
        for hr in hierarchy_rels:
            hier_parts.append(
                f"（{hr['parent_entity']}）-[{hr['relation']}]->（{hr['child_entity']}）"
            )
        hier_str = "，".join(hier_parts)
    else:
        hier_str = f"（{et_list[0]}）-[HAS_SUB_ITEM]->（子实体）"

    # Count year columns
    year_cols = [k for k, v in value_mapping.items() if v.get("type") == "year_value"]
    year_str = "、".join(str(y) for y in year_cols[:4])
    if len(year_cols) > 4:
        year_str += f"等共 {len(year_cols)} 个年份列"

    has_remarks = "无备注列" not in remarks_rule and remarks_rule and "No remarks" not in remarks_rule

    return (
        f"本表为 Hierarchical-Hybrid 类型（层级混合表）。"
        f"表中包含多级复合主键（composite key），形成层级结构，实体类型依次为 {et_str}。"
        f"复合主键列中的空单元格应通过向上继承（sparse fill）填充父级值。"
        f"层级关系为：{hier_str}。"
        f"年份列（{year_str}）包含各层级叶节点（leaf entity）的数值，"
        f"请通过 HAS_VALUE 关系将叶节点与对应年份的数值相连，并将年份作为关系属性。"
        + (
            f"备注列（remarks column）的内容应作为叶节点的元数据属性（metadata property）附加，"
            f"不需要创建独立的实体节点。"
            if has_remarks else ""
        )
        + f"所有实体节点和关系均应以结构化三元组（subject, relation, object）的形式输出。"
    )


def _prompt_generic(table_type, entity_types, relation_types, rules) -> str:
    et = "、".join(entity_types) if entity_types else "Entity"
    rel = "、".join(relation_types) if relation_types else "HAS_VALUE"
    return (
        f"本表类型为 {table_type}。"
        f"实体类型（entity types）包括：{et}。"
        f"关系类型（relation types）包括：{rel}。"
        f"请按照表格结构提取实体和关系，以三元组（subject, relation, object）形式输出。"
    )
