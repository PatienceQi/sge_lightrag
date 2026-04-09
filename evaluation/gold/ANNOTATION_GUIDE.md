# Gold Standard 双人交叉标注指南

## 任务定义

对 4 个香港本地数据集（年度预算、食物安全、健康统计、住院统计）的 102 条 Gold Standard 事实进行独立交叉验证。

## 标注对象

每条事实为一个 (subject, relation, object, attributes) 四元组，从对应 CSV 文件中提取。标注员需判断每条事实是否**在原始 CSV 中可验证**。

## 标注流程

1. **独立标注**：两位标注员分别对 102 条执行判断，互不沟通
2. **判断标准**：对每条事实标注 `CORRECT` 或 `ERROR`
   - `CORRECT`：subject + relation + object + year 在 CSV 中可找到对应单元格
   - `ERROR`：任意字段与 CSV 不匹配（标注具体错误类型）
3. **错误类型**：`wrong_value` / `wrong_subject` / `wrong_year` / `not_in_csv`

## 输入文件

| 数据集 | Gold Standard | 原始 CSV | 事实数 |
|--------|--------------|---------|--------|
| 年度预算 | `gold_budget.jsonl` | `dataset/年度预算/` | 20 |
| 食物安全 | `gold_food_sample.jsonl` | `dataset/食物安全/` | 52 |
| 健康统计 | `gold_health.jsonl` | `dataset/健康统计/` | 14 |
| 住院统计 | `gold_inpatient_2023.jsonl` | `dataset/住院统计/` | 16 |

## 输出格式

每位标注员生成一个 JSONL 文件：

```json
{"gold_file": "gold_budget.jsonl", "line": 1, "judgment": "CORRECT", "annotator": "A"}
{"gold_file": "gold_budget.jsonl", "line": 2, "judgment": "ERROR", "error_type": "wrong_value", "note": "CSV says 91.5 not 92.0", "annotator": "A"}
```

## 一致性计算

标注完成后运行：
```python
# 计算 Cohen's Kappa
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator_a_judgments, annotator_b_judgments)
agreement = sum(a == b for a, b in zip(a_judgments, b_judgments)) / len(a_judgments)
print(f"Agreement: {agreement:.1%}, Cohen's κ: {kappa:.3f}")
```

## 预期结果

由于标注对象为确定性数值提取（从 CSV 复制 subject+time+value），预期一致率接近 100%。任何不一致需逐条讨论并裁决。
