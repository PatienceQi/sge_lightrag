# SGE-LightRAG

将统计政府表格（CSV）转化为结构化知识图谱，并集成 LightRAG 实现语义检索增强生成。

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

### Stage 1 — 元数据提取（表格结构分析）

```bash
python3 run_stage1.py data/sample.csv
```

### Stage 2 — Schema 归纳（规则驱动）

```bash
python3 run_stage2.py data/sample.csv
```

### Stage 2 — Schema 归纳（LLM 增强）

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
├── meta_schema.json          # Stage 1 输出
├── extraction_schema.json    # Stage 2 输出
├── chunks/                   # 序列化文本块
├── prompts/                  # LightRAG 注入 prompt
└── pipeline_report.json      # 各阶段摘要
```

### LightRAG 集成（端到端）

```bash
python3 run_lightrag_integration.py data/sample.csv
```

同时运行 SGE 增强版与 baseline 对比，结果写入 `output/` 目录。

### 批量处理

```bash
python3 run_batch.py
```

### 评估

```bash
python3 evaluation/evaluate.py --graph output/my_run/graph.graphml --gold evaluation/gold_budget.jsonl
```

## 项目结构

```
sge_lightrag/
├── stage1/                   # 表格特征提取与分类
│   ├── classifier.py
│   ├── features.py
│   └── schema.py
├── stage2/                   # 规则驱动 schema 归纳
│   ├── header_parser.py
│   ├── inductor.py
│   └── type_handlers.py
├── stage2_llm/               # LLM 增强 schema 归纳
│   ├── inductor.py
│   ├── llm_client.py
│   └── prompts.py
├── stage3/                   # LightRAG 集成层
│   ├── integrator.py
│   ├── prompt_injector.py
│   └── serializer.py
├── evaluation/               # 图谱质量评估
│   ├── evaluate.py
│   └── gold_*.jsonl
├── run_stage1.py
├── run_stage2.py
├── run_stage2_llm.py
├── run_pipeline.py
├── run_lightrag_integration.py
├── run_batch.py
└── preprocessor.py
```

## API 配置

LLM 调用默认使用 OpenAI-compatible 接口，配置在 `stage2_llm/llm_client.py`：

```python
_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
```

如需切换模型或 endpoint，修改上述常量，或在调用 `call_llm()` 时通过参数传入：

```python
call_llm(system_prompt, user_prompt, model="gpt-4o", base_url="https://api.openai.com/v1")
```
