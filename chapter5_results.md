# 第五章 实验结果与分析

本章基于第四章所述的实验设计，报告 SGE-LightRAG 在三类异构政策 CSV 数据上的实验结果，并从图谱拓扑质量、信息覆盖率和消融分析三个维度进行深入讨论。

---

## 5.1 实验总览

### 5.1.1 实验配置矩阵

本实验在两个代表性数据集上进行了完整的端到端对比，每个数据集包含以下五种配置：

**表 5-1 实验配置说明**

| 配置编号 | 简称 | Stage 2 模式 | 是否注入 SGE prompt | 说明 |
|----------|------|-------------|-------------------|------|
| C1 | Rule SGE | 规则驱动 | 是 | 完整三阶段流程（规则版） |
| C2 | LLM v2 SGE | LLM 增强（约束版） | 是 | LLM 生成 Schema，entity_types 约束为 1–2 个 |
| C3 | LLM v1 SGE | LLM 增强（无约束版） | 是 | LLM 生成 Schema，entity_types 不限制数量 |
| C4 | LLM Baseline | — | 否 | 使用 SGE chunks 但不注入 Schema prompt |
| C5 | Rule Baseline | — | 否 | 原生 LightRAG，无 SGE 预处理 |

所有配置使用相同的 LLM（Claude Haiku 4.5，temperature=0）和 Embedding 模型（mxbai-embed-large via Ollama），仅在 Stage 2 策略和 prompt 注入方式上有差异。

### 5.1.2 评估方法

本实验采用信息覆盖率（Information Coverage）作为核心评估指标，而非传统的三元组精确匹配。原因在于：LightRAG 的图谱构建采用 (Entity→Year) 边承载数值的方式，与 Gold Standard 中 (Entity→Value) 的结构不同，传统三元组匹配会产生大量假阴性。

信息覆盖率评估包含两个子指标：

**实体覆盖率（Entity Coverage）**：Gold Standard 中定义的实体在图谱中被识别的比例。匹配采用模糊字符串匹配，允许同义词和缩写。

**事实覆盖率（Fact Coverage）**：Gold Standard 中定义的 (subject, value, year) 三元组在图谱中被捕获的比例。匹配规则为：图谱中存在某个节点或边的描述包含该 subject 的名称、对应的数值和年份信息。

Gold Standard 以 JSONL 格式标注，每条记录包含 source_file、triple（subject/relation/object/attributes）和 confidence 字段。

---

## 5.2 年度预算数据集（Type II: Time-Series-Matrix）

### 5.2.1 数据特征

年度预算 CSV（annualbudget_sc.csv）为典型的时序矩阵型表格，包含 4 个政策纲领（防止贪污、执法工作、倡廉教育、争取支持），每个纲领有 4 个财政年度的预算数据（2022-23 实际、2023-24 原来预算、2023-24 修订、2024-25 预算），单位为百万港元。

Gold Standard 包含 4 个实体、20 条事实（每个实体 × 5 个年度-数值对）。

### 5.2.2 实验结果

**表 5-2 年度预算数据集五组对比结果**

| 配置 | Entity Cov | Fact Cov | 节点数 | 边数 | 孤立节点 | 类型覆盖率 |
|------|-----------|----------|--------|------|---------|-----------|
| C1: Rule SGE | 100% | **100%** | 7 | 6 | 2 | 57.1% |
| C2: LLM v2 SGE | 100% | **100%** | 7 | 9 | 1 | 57.1% |
| C3: LLM v1 SGE | 100% | 30% | 19 | 14 | 6 | 100% |
| C4: LLM Baseline | 100% | 100% | 15 | 22 | 0 | 100% |
| C5: Rule Baseline | 100% | 55% | 19 | 24 | 0 | 100% |

### 5.2.3 结果分析

**SGE 的图谱压缩效果显著。** C1 和 C2 均以 7 个节点达到了 100% 的信息覆盖率，而 Baseline（C5）需要 19 个节点才能达到 55% 的覆盖率。SGE 通过结构感知的序列化，将每个政策纲领的多年度预算信息编码为紧凑的自然语言描述，使 LightRAG 能够在更少的节点中捕获更多的事实信息。

**LLM v1 的 entity_types 过多导致 parser 溢出。** C3（LLM v1 SGE）的 fact coverage 仅为 30%，远低于 C1 和 C2。分析 LightRAG 的运行日志发现，LLM v1 生成了 3 个 entity types（PolicyProgram, FiscalYear, BudgetAmount），导致 LightRAG 的 extraction LLM 输出了 6 字段的 tuple，而 LightRAG 的内部 parser 仅支持 4 字段格式 (entity_name, entity_type, description, source_id)，多余字段被截断，大量 relation 被丢弃。

这一发现揭示了 LightRAG 的一个架构限制：其 tuple parser 采用固定字段数解析，无法适应复杂的多类型 Schema。这也解释了为什么约束 entity_types 为 1–2 个（C2）是必要的——不是因为更多类型没有语义价值，而是因为下游 parser 无法处理。

**LLM v2 的优势在于关系丰富度。** C2 比 C1 多了 3 条边（9 vs 6），说明 LLM 生成的 prompt_context 帮助 LightRAG 识别了更多的实体间关系。在 entity_types 数量相同的前提下，LLM 的语义理解能力体现在关系抽取的质量上，而非实体类型的数量上。

**Baseline 的冗余节点问题。** C5（Rule Baseline）产生了 19 个节点和 24 条边，但 fact coverage 仅为 55%。多出的节点主要是年份标签（如 "2022-23年度"）和金额值（如 "91.5百万元"）被错误地提取为独立实体，而非作为关系属性。这正是 SGE 序列化所解决的核心问题。

---

## 5.3 食物安全数据集（Type III: Hierarchical-Hybrid）

### 5.3.1 数据特征

食物安全及公众卫生统计数字 CSV（stat_foodSafty_publicHealth.csv）为典型的混合层级型表格，包含三级层级结构：数据内容（一级分类）→ 内容分类（二级分类）→ 项目（具体指标），每个指标有 2021–2024 四个年度的数值。

该数据集的挑战在于：(1) 层级关系通过空白单元格的稀疏填充（Sparse Fill）隐式表达；(2) "项目"列的指标名称长度差异大（从 4 字到 30+ 字）；(3) 部分指标有备注列。

Gold Standard 包含 7 个实体、20 条事实。

### 5.3.2 实验结果

**表 5-3 食物安全数据集四组对比结果**

| 配置 | Entity Cov | Fact Cov | 节点数 | 边数 | 孤立节点 |
|------|-----------|----------|--------|------|---------|
| C2: LLM v2 SGE | 71% | **45%** | 23 | 14 | 10 |
| C4: LLM v2 Baseline | 71% | 40% | 29 | 35 | 0 |
| C1: Rule SGE | 71% | 20% | 20 | 20 | 4 |
| C5: Rule Baseline | 57% | 20% | 29 | 45 | 0 |

### 5.3.3 结果分析

**LLM v2 SGE 在 Type III 上优势最为明显。** C2 的 fact coverage（45%）是 C1（20%）的 2.25 倍，说明 LLM 生成的 prompt_context 对层级结构的语义理解有显著帮助。LLM 的 prompt_context 明确描述了三级层级关系和指标的含义，使 LightRAG 的 extraction LLM 能够正确识别 "食物安全 → 抽取作化验的食物样本 → 日本进口食物样本" 这样的层级路径。

**SGE 的图谱紧凑性优势在 Type III 上更突出。** C2 用 23 个节点 / 14 条边达到了 45% 的 fact coverage，而 C5（Rule Baseline）用 29 个节点 / 45 条边仅达到 20%。SGE 的节点效率（fact coverage / 节点数）为 C2 的 0.020 vs C5 的 0.007，提升约 2.9 倍。

**未覆盖实体的分析。** 四种配置均未能覆盖 "抽取作化验的食物样本数目" 和 "抽取猪只尿液样本进行乙类促效剂测试的数目" 两个实体（entity coverage 上限为 71%）。分析发现，这两个指标的名称过长（30+ 字），LightRAG 的 extraction LLM 倾向于将其截断或改写，导致与 Gold Standard 的模糊匹配失败。这是 LLM 抽取层面的限制，而非 SGE 序列化的问题。

**孤立节点问题。** C2 有 10 个孤立节点，高于 C1 的 4 个。分析发现，LLM-enhanced prompt 引导 LightRAG 为每个年度创建了独立的 FoodSafetyMetric 实体（如 "FoodSafetyMetric_在食物管制站对运送食物的车辆进行重点检查数目_2021"），但这些实体之间的 relation 因 parser 字段溢出被丢弃，导致孤立。这与 5.2.3 中发现的 parser 限制一致。

---

## 5.4 消融分析：Stage 2 策略对比

### 5.4.1 实验设计

消融实验聚焦于 Stage 2 的两种实现策略：规则驱动（Rule-based）和 LLM 增强（LLM-enhanced），在相同的 Stage 1 分类结果和 Stage 3 序列化逻辑下，仅替换 Stage 2 的 Schema Induction 模块。

LLM-enhanced 策略经历了两个版本迭代：
- **v1**：不限制 entity_types 数量，LLM 自由生成 Schema
- **v2**：约束 entity_types 为 1–2 个，将语义丰富度集中在 prompt_context 中

### 5.4.2 Schema 质量对比

**表 5-4 四个数据集的 Stage 2 Schema 对比**

| 数据集 | Rule-based entity_types | LLM v2 entity_types | Rule-based relation_types | LLM v2 relation_types |
|--------|------------------------|---------------------|--------------------------|----------------------|
| 年度预算 | Policy_Program | Programme | HAS_BUDGET | HAS_BUDGET |
| 食物安全 | Data_Category, Content_Type | FoodSafetyMetric | HAS_SUB_ITEM, HAS_VALUE, HAS_METADATA | HAS_ANNUAL_VALUE, HAS_REMARK |
| 医疗卫生 | Col | HealthcareResource | HAS_VALUE | HAS_METRIC_VALUE |
| 住院病人 | Unnamed | DiseaseCategory, HospitalSystem, DeathStatistic, ICD10Code | HAS_SUB_ITEM, HAS_VALUE, HAS_METADATA | HAS_DISCHARGE_COUNT, HAS_DEATH_COUNT, BELONGS_TO_ICD10, REPORTED_BY_HOSPITAL |

**关键发现：**

1. **命名质量提升显著。** Rule-based 在列名不规范时会生成无意义的 entity type（如 "Col"、"Unnamed"），而 LLM 能够理解表格语义，生成领域相关的类型名称（如 "HealthcareResource"、"DiseaseCategory"）。

2. **LLM 的 prompt_context 是核心价值。** LLM v2 的 prompt_context 包含了详细的中文语义描述，明确指示 "不要为时间段、金额或单位创建独立的实体节点，所有这些信息都应作为关系的属性存储"。这种自然语言约束比 Rule-based 的模板化指令更有效地引导了 LightRAG 的抽取行为。

3. **entity_types 数量必须受限。** 住院病人数据集的 LLM v1 生成了 4 个 entity types，虽然语义上完全正确（DiseaseCategory、HospitalSystem、DeathStatistic、ICD10Code），但会触发 LightRAG 的 parser 字段溢出问题。这是一个工程约束而非方法论缺陷。

### 5.4.3 端到端效果对比

**表 5-5 Rule-based vs LLM v2 端到端效果**

| 数据集 | 指标 | Rule SGE | LLM v2 SGE | 提升 |
|--------|------|---------|-----------|------|
| 年度预算 | Fact Cov | 100% | 100% | — |
| 年度预算 | 边数 | 6 | 9 | +50% |
| 食物安全 | Fact Cov | 20% | 45% | +125% |
| 食物安全 | 节点数 | 20 | 23 | +15% |

在 Type II（年度预算）上，两种策略的 fact coverage 相同（均为 100%），但 LLM v2 产生了更多的关系边（+50%），说明 LLM 的语义理解在关系抽取层面有增量贡献。

在 Type III（食物安全）上，LLM v2 的 fact coverage 提升了 125%（20% → 45%），这是因为 Type III 的层级结构需要语义理解才能正确解析，Rule-based 的模板化规则无法捕获 "数据内容 → 内容分类 → 项目" 的三级层级语义。

---

## 5.5 图谱拓扑质量分析

### 5.5.1 全量数据集拓扑统计

**表 5-6 全量数据集图谱拓扑对比**

| 数据集 | 方法 | 节点数 | 边数 | 孤立节点 | 平均度数 | 类型覆盖率 |
|--------|------|--------|------|---------|---------|-----------|
| 年度预算 | SGE (Rule) | 7 | 6 | 2 | 1.71 | 57.1% |
| 年度预算 | Baseline | 19 | 24 | 0 | 2.53 | 100% |
| 食物安全 v2 | SGE (Rule) | 20 | 20 | 4 | 2.00 | 100% |
| 食物安全 v2 | Baseline | 29 | 45 | 0 | 3.10 | 100% |
| 医疗卫生 | SGE (Rule) | 13 | 24 | 1 | 3.69 | 100% |
| 医疗卫生 | Baseline | 6 | 5 | 0 | 1.67 | 100% |

### 5.5.2 分析

**SGE 在 Type II 和 Type III 上实现了显著的图谱压缩。** 年度预算数据集中，SGE 将节点数从 19 压缩到 7（-63%），同时将 fact coverage 从 55% 提升到 100%。食物安全数据集中，SGE 将节点数从 29 压缩到 20（-31%），边数从 45 压缩到 20（-56%）。

**医疗卫生数据集是一个反例。** SGE 产生了 13 个节点 / 24 条边，而 Baseline 仅产生 6 个节点 / 5 条边。这是因为该数据集为转置型 Time-Series-Matrix（年份在列头，指标在行头），SGE 的 serializer 正确地将每个指标-年份组合展开为独立的 chunk，产生了更多但更精确的节点。Baseline 的 6 个节点实际上是高度聚合的，丢失了大量细粒度信息（fact coverage 仅 14%，而 SGE 为 21%）。

---

## 5.6 已知限制与讨论

### 5.6.1 LightRAG Tuple Parser 的 4 字段限制

本实验发现的最关键的工程限制是 LightRAG 的 entity extraction parser 仅支持 4 字段 tuple 格式 (entity_name, entity_type, description, source_id)。当 Schema 包含 3 个以上 entity types 时，LLM 倾向于在 tuple 中添加额外字段（如 budget_type、fiscal_year），导致 parser 截断和 relation 丢弃。

**应对策略：** 本文采用 "约束 entity_types + 丰富 prompt_context" 的方案，将语义复杂度从 Schema 结构转移到自然语言描述中。这一策略在实验中被验证为有效（LLM v2 vs v1 的对比）。

未来工作可以考虑修改 LightRAG 的 parser 以支持可变字段数，或采用 JSON 格式替代 tuple 格式进行实体抽取。

### 5.6.2 转置型 Time-Series-Matrix 的处理

医疗卫生统计数据集（healthstat_table1.csv）采用 UTF-16 编码 + Tab 分隔，且为转置型表格。当前 SGE 的 Stage 1 feature extraction 和 Stage 3 serializer 对该文件的列名解析不一致（Stage 1 使用数字索引，Stage 3 使用实际列名），导致 serializer 产出 0 chunks。

这一问题的根因是 CSV 读取逻辑未统一。修复方案为在 Stage 1 的 feature extraction 中保存原始 DataFrame 的列名映射，供 Stage 3 使用。

### 5.6.3 评估方法的局限性

本实验采用的信息覆盖率评估依赖人工标注的 Gold Standard，标注规模有限（3 个数据集，共 31 个实体、54 条事实）。未来工作应扩大标注规模，并引入多标注者一致性检验（如 Cohen's Kappa）以提高评估的可靠性。

此外，信息覆盖率仅衡量 "信息是否被捕获"，不衡量 "信息是否被正确结构化"。例如，一个事实可能被编码在节点描述中而非作为独立的关系边，这在覆盖率评估中被视为 "已覆盖"，但在图谱查询场景中可能不够理想。

---

## 本章小结

本章通过两个代表性数据集的五组对比实验和消融分析，验证了 SGE-LightRAG 框架的有效性。主要结论如下：

1. **SGE 显著提升了图谱的信息密度。** 在 Type II 数据上，SGE 以 63% 更少的节点达到了 100% 的 fact coverage（vs Baseline 的 55%）；在 Type III 数据上，LLM v2 SGE 的 fact coverage 是 Rule Baseline 的 2.25 倍。

2. **LLM-enhanced Stage 2 的核心价值在于 prompt_context 的语义质量，而非 entity_types 的数量。** 约束 entity_types 为 1–2 个是必要的工程妥协，但 LLM 生成的自然语言描述有效地引导了下游抽取行为。

3. **LightRAG 的 tuple parser 是当前架构的主要瓶颈。** 4 字段限制迫使 Schema 设计必须保持简洁，限制了 SGE 框架在复杂数据集上的表现上限。
