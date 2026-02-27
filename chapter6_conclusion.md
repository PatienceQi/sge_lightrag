# 第六章 总结与展望

## 6.1 研究总结

本文针对异构政策 CSV 数据的知识图谱自动构建问题，提出了 SGE-LightRAG（Structure-Guided Extraction LightRAG）框架。该框架通过三阶段感知流水线——拓扑模式识别（Topological Pattern Recognition）、模式诱导（Schema Induction）和约束性提取（Constrained Extraction）——在 LightRAG 的上游引入结构感知能力，系统性地解决了原生 LightRAG 在处理异构 CSV 数据时的"结构盲视"问题。

在香港政府公开政策数据集（27 个 CSV 文件，涵盖 3 种拓扑类型、3 种字符编码）上的实验表明：

1. **SGE 显著提升了图谱的信息密度。** 在 Type II（时序矩阵型）数据上，SGE 以 63% 更少的节点达到了 100% 的事实覆盖率，而原生 LightRAG 仅达到 55%。在 Type III（混合层级型）数据上，LLM 增强版 SGE 的事实覆盖率是原生 LightRAG 的 2.25 倍。

2. **SGE 使原本不可处理的数据类型变得可处理。** 在转置型 Type II 数据上，原生 LightRAG 的实体覆盖率和事实覆盖率均为 0%，而 SGE 分别达到 33% 和 21%。

3. **LLM 增强的 Schema Induction 在复杂表格上优势明显。** 在 Type III 数据上，LLM 增强版的事实覆盖率比规则驱动版提升了 125%（20% → 45%），其核心价值在于生成高质量的自然语言 prompt_context，而非增加 entity_types 的数量。

## 6.2 主要贡献

本文的主要贡献包括：

1. **异构 CSV 拓扑分类学。** 提出了面向政策数据的三类 CSV 拓扑类型（Flat-Entity、Time-Series-Matrix、Hierarchical-Hybrid），并设计了基于规则的自动分类器（76 项测试全部通过）。

2. **三阶段感知框架。** 设计并实现了从拓扑识别到约束提取的完整流水线，以插件形式集成于 LightRAG，无需修改其核心代码。

3. **信息覆盖率评估方法。** 针对 LightRAG 图谱构建的特点，提出了基于实体覆盖率和事实覆盖率的评估方法，替代传统的三元组精确匹配，更准确地衡量图谱的信息捕获能力。

4. **LightRAG tuple parser 限制的发现与规避策略。** 发现 LightRAG 的 4 字段 tuple parser 是复杂 Schema 应用的主要瓶颈，并提出"约束 entity_types + 丰富 prompt_context"的有效规避方案。

## 6.3 研究局限性

1. **数据集规模有限。** 实验仅在 27 个香港政府政策 CSV 文件上进行，数据领域单一。框架在其他领域（如金融、医疗、教育）的泛化能力有待验证。

2. **Gold Standard 标注规模较小。** 当前仅标注了 3 个数据集共 14 个实体和 54 条事实，标注者为论文作者本人，缺乏多标注者一致性检验。

3. **转置型表格处理仍有不足。** 虽然修复了 serializer 的核心 bug，但 health stats 数据集的 entity coverage 仅为 33%，说明转置型表格的处理逻辑仍需优化。

4. **依赖 LightRAG 的 parser 架构。** 当前方案受限于 LightRAG 的 4 字段 tuple parser，无法充分利用 LLM 生成的复杂 Schema。

## 6.4 未来工作

1. **扩展数据集与领域。** 将框架应用于更多领域的异构 CSV 数据，验证拓扑分类学的通用性，并根据新数据类型扩展分类体系。

2. **改进 LightRAG 的 parser。** 修改 LightRAG 的 entity extraction parser 以支持可变字段数或 JSON 格式输出，解除 4 字段限制，释放 LLM-enhanced Schema 的全部潜力。

3. **引入多跳推理评估。** 在图谱构建质量评估之外，增加基于图谱的多跳问答评估，衡量 SGE 对下游 RAG 检索质量的实际影响。

4. **自动化 Gold Standard 生成。** 探索利用 LLM 辅助生成 Gold Standard 标注，降低人工标注成本，扩大评估规模。

5. **端到端优化。** 将 Stage 1-3 的串行流水线优化为可并行的架构，并探索 Stage 2 的 few-shot learning 方案，减少对 LLM 调用的依赖。
