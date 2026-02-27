# 第四章修订说明（基于实际实验结果）

以下为第四章需要根据实际实验情况修订的内容：

## 4.1.2 软件环境
- 删除 Neo4j：实际使用 LightRAG 内置的 nano-vectordb + GraphML 文件存储
- Python 版本确认为 3.14

## 4.1.3 LLM 配置
- 修正：Stage 1 和 Stage 2 均使用 Claude Haiku 4.5（非 Sonnet 4.6）
- Stage 2 LLM-enhanced 模式也使用 Haiku（成本考虑）
- 所有 LLM 调用 temperature=0.2（非 0）

## 4.3.2 语义准确性指标
- 替换为信息覆盖率（Information Coverage）评估
- 原因：LightRAG 的图谱结构与 Gold Standard 的三元组结构不同，传统 P/R/F1 会产生大量假阴性
- 新指标：Entity Coverage + Fact Coverage（详见第五章 5.1.2）

## 4.3.3 多跳推理能力指标
- 暂未实施，可在未来工作中补充

## 4.4.1 实验系统
- 删除 Baseline-2（Manual Prompt）
- 新增 LLM v1 SGE 和 LLM v2 SGE 作为消融配置
- 实际对比系统为 5 种配置（见第五章表 5-1）

## 4.4.3 标注数据集构建
- 实际标注规模：3 个数据集，31 个实体，54 条事实
- 标注文件：gold_budget.jsonl, gold_food_sample.jsonl, gold_health.jsonl
- 未达到原设计的 400 条规模，在 limitations 中说明

## 4.4.4 统计显著性检验
- 因样本量过小（3 个数据集），未进行统计检验
- 在 limitations 中说明

## 4.5 消融实验设计
- 实际消融聚焦于 Stage 2 策略对比（Rule vs LLM v1 vs LLM v2）
- 未实施 Ablation A（Stage 1 only）和 Ablation B（Stage 1+2 无约束提取）
- 原因：时间限制 + 实际发现 Stage 2 策略差异是最有价值的消融维度
