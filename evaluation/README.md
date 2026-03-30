# SGE-LightRAG 评估框架

## 概述

本目录包含 SGE-LightRAG 的完整评估框架，支持：
- **EC/FC 信息覆盖率评估**（子串匹配 + 2-hop 邻居搜索）
- **v2 Gold Standard**（4 个国际数据集 × 25 国 × 150 事实 = 600 事实 + 102 本地价值事实 = 702 总事实）
- **Bootstrap 95% 置信区间**（n=1000 次重采样）
- **Wilcoxon signed-rank 检验**（Bonferroni 校正 k=4 + 效应量 r）
- **下游 QA 评估**（100 题：67 直接 + 33 推理，含直接图谱上下文 + E2E LightRAG 查询两种模式）
- **Direct LLM 基线对比**（非等量评估，仅供定性参考）

---

## 文件说明

### 评估脚本

| 文件 | 说明 |
|------|------|
| `evaluate.py` | 传统三元组匹配评估（P/R/F1 + 图拓扑指标） |
| `evaluate_coverage.py` | EC/FC 信息覆盖率评估（子串匹配 + 2-hop） |
| `generate_gold_standards.py` | v2 Gold Standard 自动生成（从 CSV 直接提取） |
| `run_evaluations_v2.py` | v2 全量评估 + Bootstrap CI + 结果汇总 |
| `run_qa_eval.py` | 下游 QA 评估（100 题，直接图谱上下文检索） |
| `run_all_evaluations.py` | 批量评估所有数据集 |
| `direct_llm_baseline.py` | Direct LLM 基线（直接喂 CSV 给 LLM 提取三元组） |
| `graph_loaders.py` | GraphML/JSON 图谱加载与解析 |

### Gold Standard 文件

#### 本地数据集（手动标注）

| 文件 | 数据集 | 类型 | 规模 |
|------|--------|------|------|
| `gold_budget.jsonl` | 年度预算 | Type-II | 4 实体 × 20 事实 |
| `gold_food_sample.jsonl` | 食物安全 | Type-III | 17 实体 × 52 事实 |
| `gold_health.jsonl` | 健康统计 | Type-II-T | 3 实体 × 14 事实 |
| `gold_inpatient_2023.jsonl` | 住院统计 | Type-III | 8 实体 × 16 事实 |

#### 国际数据集 v2（自动生成，25 国 × 6 年 × 150 事实）

| 文件 | 数据集 | 来源 |
|------|--------|------|
| `gold_who_life_expectancy_v2.jsonl` | WHO 全球预期寿命（196 国） | WHO GHO |
| `gold_wb_child_mortality_v2.jsonl` | WB 儿童死亡率（244 国） | World Bank |
| `gold_wb_population_v2.jsonl` | WB 人口（265 国） | World Bank |
| `gold_wb_maternal_v2.jsonl` | WB 产妇死亡率 | World Bank |

v2 Gold Standard 直接从实验用 CSV 生成（`generate_gold_standards.py`），25 个目标国家按 GDP 排名 + 地理分布均衡选取，6 个目标年份（2000/2005/2010/2015/2020/2022），数值直接从 CSV 单元格读取，无截断或外部来源。

### QA 评估

| 文件 | 说明 |
|------|------|
| `qa_questions.jsonl` | 100 题 QA 问题集（67 直接 + 19 比较 + 14 趋势） |
| `qa_results_v2.json` | v2 评估结果（60 题，已被 v3 取代） |
| `qa_results_v3_100q.json` | **权威** v3 评估结果（100 题：SGE 93%, Baseline 59%） |

### 评估结果

| 文件 | 说明 |
|------|------|
| `all_results_v2.json` | v2 全量 EC/FC 评估结果 |
| `direct_llm_results.json` | Direct LLM 基线结果 |

---

## 标注格式

每行一个 JSON 对象：

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

## 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **EC（实体覆盖率）** | Gold Standard 实体在图谱中的子串匹配比例 | 实体识别能力 |
| **FC（事实覆盖率）** | Gold Standard 三元组在图谱 2-hop 邻居中的捕获比例 | 事实完整性 |
| **η（信息密度）** | (FC_SGE/\|V_SGE\|) / (FC_base/\|V_base\|) | 单节点效率（小图适用） |
| **Bootstrap 95% CI** | n=1000 次重采样的置信区间 | 统计显著性 |
| **Fisher's exact test** | 2×2 列联表精确检验 | 小样本显著性（n<30） |
| **QA Accuracy** | 问答正确率（直接题 + 推理题） | 下游任务有效性 |

---

## 运行评估

```bash
cd ~/Desktop/SGE/sge_lightrag

# 1. 生成 v2 Gold Standard
python3 evaluation/generate_gold_standards.py

# 2. 运行 v2 全量评估（EC/FC + Bootstrap CI）
python3 evaluation/run_evaluations_v2.py

# 3. 运行 QA 评估
python3 evaluation/run_qa_eval.py

# 4. 单数据集传统评估
python3 evaluation/evaluate.py \
  --graph output/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml \
  --gold evaluation/gold_budget.jsonl
```

---

## 核心结果速览（v2）

| 数据集 | SGE FC | Base FC | SGE/Base | Bootstrap CI 重叠 |
|--------|--------|---------|----------|-------------------|
| WHO 预期寿命 | 1.000 | 0.167 | 6.0× | 无重叠 ✓ |
| WB 儿童死亡率 | 1.000 | 0.473 | 2.11× | 无重叠 ✓ |
| WB 人口 | 1.000 | 0.187 | 5.35× | 无重叠 ✓ |
| WB 产妇死亡率 | 0.967 | 0.787 | 1.23× | 无重叠 ✓ |
| 住院统计 | 0.938 | 0.438 | 2.14× | Fisher p≈0.003 ✓ |

QA（直接图谱上下文）：SGE 93%（93/100）vs Baseline 59%（59/100），趋势题 SGE 86% vs Baseline 36%。
E2E LightRAG 查询：SGE 13% vs Baseline 13%（Δ=0）——向量检索瓶颈。
Wilcoxon（Bonferroni k=4）：全部 4 个国际数据集 p_Bonf < 0.05，效应量 r ≥ 0.80（large）。
