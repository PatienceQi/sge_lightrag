"""
prompt_injector.py — Prompt template generation for Stage 3.

Generates:
1. A schema-aware system prompt that overrides LightRAG's default
   entity_extraction_system_prompt, instructing the LLM to use the
   delimiter-based output format and respect the domain schema.
2. A user prompt template for each text chunk.

Output format MUST match LightRAG's expected delimiter format:
    entity<|#|>EntityName<|#|>entity_type<|#|>description
    relation<|#|>Source<|#|>Target<|#|>keywords<|#|>description
    <|COMPLETE|>
"""

from __future__ import annotations

import json

TUPLE_DELIMITER      = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
---Role---
You are a Knowledge Graph Specialist extracting structured data from policy CSV records.
You MUST follow the Domain Schema below. Do NOT extract entities or relationships outside the defined types.

---Domain Schema---
```json
{schema_json}
```

---Instructions---
1. Entity types: use ONLY the types listed in schema.entity_types
2. Relationship types: use ONLY the predicates listed in schema.relation_types
3. Follow ALL extraction constraints in schema.extraction_constraints exactly
4. Output Format — Entity:
   entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description
5. Output Format — Relation:
   relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}keywords{tuple_delimiter}description
6. {tuple_delimiter} is a field separator ONLY — never include it in content
7. End output with {completion_delimiter}
8. Output language: {language}
9. Do NOT output JSON, Markdown, or any other format — only the delimiter-based lines above

---Examples---
{examples}
"""

_DEFAULT_EXAMPLES = """\
entity<|#|>防止贪污<|#|>Policy_Program<|#|>廉政公署下设的政策纲领，负责防止贪污相关工作
relation<|#|>防止贪污<|#|>97.2<|#|>HAS_BUDGET, 2024-25, 预算<|#|>防止贪污纲领在2024-25财年的预算为97.2百万元
<|COMPLETE|>"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_PROMPT_TEMPLATE = """\
---Task---
Extract entities and relationships from the input text below.
Use ONLY the entity types and relation types defined in the system prompt schema.
Output ONLY delimiter-based lines. End with {completion_delimiter}.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_system_prompt(
    schema: dict,
    language: str = "Chinese",
    examples: str = "",
) -> str:
    """
    Generate the schema-aware system prompt for LightRAG injection.

    Parameters
    ----------
    schema   : Stage 2 extraction schema dict
    language : output language for the LLM
    examples : optional few-shot examples string; defaults to built-in example

    Returns
    -------
    str — the fully rendered system prompt
    """
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    used_examples = examples if examples else _DEFAULT_EXAMPLES

    return _SYSTEM_PROMPT_TEMPLATE.format(
        schema_json=schema_json,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        language=language,
        examples=used_examples,
    )


def generate_user_prompt_template(schema: dict) -> str:
    """
    Generate the user prompt template (with {input_text} placeholder).

    The caller fills in {input_text} for each chunk before sending to the LLM.

    Parameters
    ----------
    schema : Stage 2 extraction schema dict

    Returns
    -------
    str — user prompt template with {input_text} placeholder
    """
    entity_types = ", ".join(schema.get("entity_types", ["Entity"]))

    return _USER_PROMPT_TEMPLATE.format(
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=entity_types,
        input_text="{input_text}",  # keep as placeholder
    )


def render_user_prompt(template: str, input_text: str) -> str:
    """
    Fill the {input_text} placeholder in a user prompt template.

    Parameters
    ----------
    template   : output of generate_user_prompt_template()
    input_text : the serialized text chunk

    Returns
    -------
    str — the fully rendered user prompt
    """
    return template.replace("{input_text}", input_text)
