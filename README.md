# SGE-LightRAG

Structure-Guided Extraction for LightRAG — 将异构 CSV 表格数据转化为结构化知识图谱，通过三阶段感知流水线（拓扑识别 → 模式诱导 → 约束提取）以非侵入式方式增强 LightRAG 的表格处理能力。

## 核心特性

- **三种 CSV 拓扑自动分类**：Type-I（扁平实体）、Type-II（时序矩阵）、Type-III（混合层级），Algorithm 1 基于 5 类特征信号 + 3 条优先级规则
- **双模式 Schema 诱导**：规则驱动（确定性快速）+ LLM 增强（语义丰富），失败自动降级
- **自适应降级模式**：小型 Type-III（n_rows < 20）自动切换 Baseline 模式，避免过约束
- **紧凑时序表示**：大规模 Type-II（n_rows > 100）自动启用节点压缩（预期 21× 压缩比）
- **完整评估框架**：EC/FC/η 指标 + v2 Gold Standard（716 事实）+ 60 题 QA + Bootstrap CI + Fisher's exact test

## 快速开始

### 环境要求

- Python 3.10+
- 依赖库：`pandas`, `networkx`, `openai` SDK, `lightrag-hku`
- [Ollama](https://ollama.com) 并已拉取 `mxbai-embed-large` 模型：
  ```bash
  ollama pull mxbai-embed-large
  ```

### 安装

```bash
pip3 install --break-system-packages pandas networkx openai lightrag-hku
```

## 使用方法

### Stage 1 — 拓扑模式识别

```bash
python3 run_stage1.py data/sample.csv
```

### Stage 2 — Schema 诱导（规则驱动）

```bash
python3 run_stage2.py data/sample.csv
```

### Stage 2 — Schema 诱导（LLM 增强）

```bash
python3 run_stage2_llm.py data/sample.csv
```

### 完整 Pipeline（Stage 1 → 2 → 3）

```bash
python3 run_pipeline.py data/sample.csv --output-dir output/my_run
```

输出目录结构：
```
output/my_run/
├── meta_schema.json          # Stage 1 输出（拓扑类型 + Meta-Schema）
├── extraction_schema.json    # Stage 2 输出（提取模式 Σ）
├── chunks/                   # 序列化文本块
├── prompts/                  # LightRAG 注入 prompt
└── pipeline_report.json      # 各阶段摘要
```

### LightRAG 集成（端到端）

```bash
python3 run_lightrag_integration.py data/sample.csv
```

同时运行 SGE 增强版与 Baseline 对比，结果写入 `output/` 目录。

### Baseline 单独运行

```bash
python3 run_baseline_only.py data/sample.csv
```

### 从已有输出重跑 LightRAG

```bash
python3 run_lightrag_from_output.py output/my_run/
```

### 批量处理

```bash
python3 run_batch.py
```

### 评估（v2 Gold Standard）

```bash
# 生成 v2 Gold Standard（25 国 × 6 年 × 150 事实/数据集）
python3 evaluation/generate_gold_standards.py

# 运行全量 v2 评估（EC/FC/Bootstrap CI）
python3 evaluation/run_evaluations_v2.py

# 运行下游 QA 评估（60 题）
python3 evaluation/run_qa_eval.py
```

### 测试

```bash
python3 -m pytest tests/ -v
```

## 项目结构

```
sge_lightrag/
├── stage1/                        # Stage 1：拓扑模式识别
│   ├── classifier.py              #   Algorithm 1 分类器
│   ├── features.py                #   5 类特征信号提取
│   └── schema.py                  #   Meta-Schema 生成
├── stage2/                        # Stage 2：规则驱动 Schema 诱导
│   ├── inductor.py                #   主诱导器（含自适应降级逻辑）
│   ├── inducer.py                 #   诱导辅助
│   ├── header_parser.py           #   列头解析
│   ├── type_handlers.py           #   按拓扑类型的处理器
│   ├── prompt_builder.py          #   提取约束 Prompt 构建
│   └── templates.py               #   Schema 模板
├── stage2_llm/                    # Stage 2：LLM 增强 Schema 诱导
│   ├── inductor.py                #   LLM 诱导器（entity_types ≤ 2 约束）
│   ├── llm_client.py              #   LLM API 调用
│   └── prompts.py                 #   系统/用户 Prompt 模板
├── stage3/                        # Stage 3：LightRAG 集成层
│   ├── integrator.py              #   LightRAG 注入协调器
│   ├── prompt_injector.py         #   PROMPTS 字典覆写
│   ├── serializer.py              #   行序列化（Type-I/II/III 差异化）
│   └── compact_representation.py  #   紧凑时序表示（大规模 Type-II 压缩）
├── evaluation/                    # 评估框架
│   ├── evaluate.py                #   传统三元组匹配评估
│   ├── evaluate_coverage.py       #   EC/FC 信息覆盖率评估
│   ├── generate_gold_standards.py #   v2 Gold Standard 自动生成
│   ├── run_evaluations_v2.py      #   v2 全量评估（含 Bootstrap CI）
│   ├── run_qa_eval.py             #   下游 QA 评估（60 题）
│   ├── direct_llm_baseline.py     #   Direct LLM 基线
│   ├── graph_loaders.py           #   GraphML/JSON 图谱加载器
│   ├── qa_questions.jsonl         #   QA 问题集
│   ├── gold_budget.jsonl          #   年度预算 Gold Standard
│   ├── gold_food_sample.jsonl     #   食物安全 Gold Standard
│   ├── gold_health.jsonl          #   健康统计 Gold Standard
│   ├── gold_inpatient_2023.jsonl  #   住院统计 Gold Standard
│   ├── gold_who_life_expectancy_v2.jsonl   # WHO v2（25 国 × 150 事实）
│   ├── gold_wb_child_mortality_v2.jsonl    # WB 儿童死亡率 v2
│   ├── gold_wb_population_v2.jsonl         # WB 人口 v2
│   └── gold_wb_maternal_v2.jsonl           # WB 产妇死亡率 v2
├── tests/                         # 测试套件
│   ├── test_stage1.py             #   Stage 1 分类器测试
│   ├── test_stage2.py             #   Stage 2 诱导器测试（含自适应降级）
│   ├── test_stage3.py             #   Stage 3 序列化/注入测试
│   └── test_compact.py            #   紧凑表示测试
├── run_stage1.py                  # Stage 1 单独运行
├── run_stage2.py                  # Stage 2（规则）单独运行
├── run_stage2_llm.py              # Stage 2（LLM）单独运行
├── run_pipeline.py                # 完整 Pipeline
├── run_lightrag_integration.py    # LightRAG 端到端集成
├── run_baseline_only.py           # Baseline 单独运行
├── run_lightrag_from_output.py    # 从已有输出重跑 LightRAG
├── run_batch.py                   # 批量处理
├── run_food_rerun.py              # 食物安全重跑（自适应降级验证）
├── preprocessor.py                # CSV 预处理（编码检测、元数据跳过）
└── insert_missing_gold_chunks.py  # Gold Standard chunk 补丁
```

## 实验配置

| 配置 | Stage 2 | Schema 注入 | 说明 |
|------|---------|------------|------|
| C1: Rule SGE | 规则 | 是 | 完整三阶段（规则版） |
| C2: LLM v2 SGE | LLM（约束版） | 是 | entity_types ≤ 2 |
| C3: LLM v1 SGE | LLM（无约束） | 是 | entity_types 不限 |
| C4: LLM Baseline | — | 否 | SGE chunks, 无 Schema |
| C5: Rule Baseline | — | 否 | 原生 LightRAG |

## 核心实验结果（v2 Gold Standard, 716 事实）

| 数据集 | SGE FC | Baseline FC | 倍数 |
|--------|--------|-------------|------|
| WHO 预期寿命（25 国 × 150 事实） | 1.000 | 0.167 | **6.0×** |
| WB 人口（25 国 × 150 事实） | 1.000 | 0.187 | **5.35×** |
| WB 儿童死亡率（25 国 × 150 事实） | 1.000 | 0.473 | **2.11×** |
| 住院统计（318 ICD 分类） | 0.938 | 0.438 | **2.14×** |
| WB 产妇死亡率（25 国 × 150 事实） | 0.967 | 0.787 | **1.23×** |

下游 QA：SGE 95%（57/60）vs Baseline 60%（36/60），推理题 SGE 100%（13/13）。

## API 配置

LLM 调用默认使用 OpenAI-compatible 接口，配置在 `stage2_llm/llm_client.py`：

```python
_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
```

实验环境：LightRAG `v1.3.8`、MS GraphRAG `v1.0.0`、mxbai-embed-large（1024 维）、`llm_model_max_async=5`。
