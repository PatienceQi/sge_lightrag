# SGE-LightRAG 黄金标准标注指南

## 概述

本目录包含用于评估 LightRAG 知识图谱抽取质量的黄金标准三元组（Gold Standard Triples）。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `annotation_schema.json` | 三元组标注的 JSON Schema，定义字段格式与枚举值 |
| `gold_budget.jsonl` | 年度预算数据集的黄金三元组（4个纲领 × 4个时期 = 20条） |
| `gold_food_sample.jsonl` | 食物安全数据集前5行的黄金三元组（含层级关系） |
| `evaluate.py` | 评估脚本，计算 P/R/F1 及图拓扑指标 |

---

## 标注格式

每行一个 JSON 对象，字段如下：

```json
{
  "source_file": "数据来源文件名",
  "row_index": 0,
  "triple": {
    "subject": "主体实体",
    "subject_type": "实体类型",
    "relation": "关系类型",
    "object": "客体实体或数值",
    "object_type": "实体类型",
    "attributes": {"year": "年份", "status": "状态", "unit": "单位"}
  },
  "annotator": "标注者姓名",
  "confidence": "high | medium | low",
  "notes": "备注"
}
```

---

## 实体类型（subject_type / object_type）

| 类型 | 说明 |
|------|------|
| `Policy_Program` | 政策纲领（如：防止贪污、执法工作） |
| `Category` | 数据大类（如：食物安全、防治虫鼠） |
| `SubCategory` | 数据子类 |
| `Metric` | 统计指标名称 |
| `BudgetAmount` | 预算金额（数值） |
| `StatValue` | 统计数值 |
| `Year` | 年份实体 |
| `Organization` | 机构名称 |
| `Literal` | 原始字面量（如编号） |

---

## 关系类型（relation）

| 关系 | 说明 |
|------|------|
| `HAS_BUDGET` | 纲领拥有某年度预算金额 |
| `HAS_VALUE` | 指标在某年的统计值 |
| `HAS_SUB_ITEM` | 类别包含子指标（层级关系） |
| `BELONGS_TO` | 实体归属于某类别 |
| `HAS_PROGRAM_ID` | 纲领编号 |
| `IN_YEAR` | 数值对应的年份 |
| `HAS_STATUS` | 预算状态（实际/原来预算/修订/预算） |

---

## 标注置信度

- `high`：三元组明确来自原始数据，无歧义
- `medium`：需要一定推断，但合理
- `low`：存在歧义或不确定性，需复核

---

## 运行评估

```bash
cd ~/Desktop/SGE/sge_lightrag

# 评估预算数据集
python3 evaluate.py \
  --graph output/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml \
  --gold evaluation/gold_budget.jsonl

# 保存结果到 JSON 文件
python3 evaluate.py \
  --graph output/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml \
  --gold evaluation/gold_budget.jsonl \
  --output evaluation/results_budget.json
```

---

## 新增标注说明

1. 复制 `annotation_schema.json` 中的 `example` 字段作为模板
2. 填写 `source_file`、`row_index`、`triple` 字段
3. 填写 `annotator`（你的姓名）和 `confidence`
4. 追加到对应的 `.jsonl` 文件末尾（每行一个 JSON 对象）
5. 运行 `evaluate.py` 验证结果变化

---

## 注意事项

- JSONL 格式：每行必须是合法的 JSON，不能有多余逗号
- 实体名称应与原始数据保持一致（包括空格、标点）
- 数值类型（BudgetAmount、StatValue）的 `object` 字段填写字符串形式的数字
- 评估脚本默认使用**模糊匹配**（子字符串包含），以应对 LightRAG 实体名称变体
