"""
prompts.py — Prompt templates for LLM-enhanced schema induction.
"""

SYSTEM_PROMPT = """\
You are a Senior Data Architect specializing in Knowledge Graph construction \
from government policy datasets. Your task is to analyze a CSV snippet and \
generate a precise extraction schema in JSON format.

IMPORTANT: Respond with ONLY valid JSON — no markdown fences, no explanation text.\
"""

USER_PROMPT_TEMPLATE = """\
Analyze the provided CSV snippet (Header + First 5 rows) and the pre-classified \
table type from Stage 1 heuristics.

Table Type (from Stage 1): {table_type}

Meta-Schema (from Stage 1):
{meta_schema_json}

CSV Data:
{csv_snippet}

Generate a detailed extraction schema as a JSON object with EXACTLY these keys:

{{
  "table_type": "<same as Stage 1 table_type>",
  "entity_types": ["<PascalCase entity type names — ONLY the primary subject entities>"],
  "relation_types": ["<UPPER_SNAKE_CASE relation names, e.g. HAS_BUDGET, HAS_METRIC>"],
  "extraction_rules": {{
    "subject_extraction": "<how to identify the main entity from each row>",
    "value_extraction": "<how to handle numeric values, units, fiscal-year labels>",
    "time_handling": "<how to parse time dimensions if present, else 'N/A'>",
    "hierarchy": "<parent-child relationships if Hierarchical-Hybrid, else 'N/A'>",
    "remarks": "<how to handle remarks/notes columns, else 'N/A'>"
  }},
  "prompt_context": "<A natural language paragraph in Chinese (keep technical terms like entity type names, relation names, and column names in English) describing exactly how to read this table — this paragraph will be injected into LightRAG's extraction prompt in Stage 3>"
}}

Rules:
- entity_types: ONLY 1-2 primary subject entity types. Do NOT create separate entity types for time periods, numeric values, or units — these should be relation attributes or embedded in the entity description. LightRAG's parser only supports 4-field tuples (entity_name, entity_type, description, source_id), so complex multi-type schemas cause parsing failures.
- relation_types: 1-3 relation types that connect the primary entities to their values/attributes
- prompt_context must be written in Chinese with English technical terms preserved
- prompt_context should instruct the LLM to encode time periods and numeric values as relation attributes (e.g. year, amount, unit) rather than as separate entity nodes
- Do NOT wrap the JSON in markdown code fences
"""


def build_user_prompt(table_type: str, meta_schema_json: str, csv_snippet: str) -> str:
    """Fill the user prompt template with actual values."""
    return USER_PROMPT_TEMPLATE.format(
        table_type=table_type,
        meta_schema_json=meta_schema_json,
        csv_snippet=csv_snippet,
    )
