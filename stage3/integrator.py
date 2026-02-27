"""
integrator.py — LightRAG integration layer for Stage 3.

Provides two public functions:

  patch_lightrag(schema) -> dict
      Prepares the PROMPTS override and returns a context dict that can be
      merged into LightRAG's context_base. Does NOT import or run LightRAG.

  prepare_chunks(csv_path, schema) -> list[str]
      Serializes a CSV file into text chunks using the extraction schema.

NOTE: LightRAG is not installed in this environment. This module prepares
everything LightRAG would need without actually importing it.
"""

from __future__ import annotations

import json
from typing import Optional

from .serializer import serialize_csv
from .prompt_injector import (
    generate_system_prompt,
    generate_user_prompt_template,
    TUPLE_DELIMITER,
    COMPLETION_DELIMITER,
)


def patch_lightrag(schema: dict, language: str = "Chinese") -> dict:
    """
    Prepare the LightRAG injection payload for a given extraction schema.

    Returns a dict containing:
      - "system_prompt"     : the overridden entity_extraction_system_prompt
      - "context_base_extra": extra fields to merge into context_base
      - "addon_params"      : dict to pass as LightRAG(addon_params=...)
      - "entity_types"      : list of entity type strings

    Usage (when LightRAG is available):
        from lightrag.prompt import PROMPTS
        payload = patch_lightrag(schema)
        PROMPTS["entity_extraction_system_prompt"] = payload["system_prompt"]
        rag = LightRAG(
            working_dir="./rag_storage",
            addon_params=payload["addon_params"],
        )

    Parameters
    ----------
    schema   : Stage 2 extraction schema dict
    language : output language for the LLM (default: "Chinese")

    Returns
    -------
    dict with keys: system_prompt, context_base_extra, addon_params, entity_types
    """
    entity_types = schema.get("entity_types", ["Entity"])
    schema_json_str = json.dumps(schema, ensure_ascii=False)

    system_prompt = generate_system_prompt(schema, language=language)

    # context_base_extra: fields to add to operate.py's context_base dict
    context_base_extra = {
        "schema_json": schema_json_str,
    }

    # addon_params: passed to LightRAG constructor
    addon_params = {
        "language": language,
        "entity_types": entity_types,
        "schema_json": schema_json_str,
    }

    return {
        "system_prompt": system_prompt,
        "context_base_extra": context_base_extra,
        "addon_params": addon_params,
        "entity_types": entity_types,
    }


def prepare_chunks(csv_path: str, schema: dict) -> list[str]:
    """
    Serialize a CSV file into text chunks ready for LightRAG ingestion.

    Parameters
    ----------
    csv_path : path to the CSV file
    schema   : Stage 2 extraction schema dict

    Returns
    -------
    list[str] — text chunks, one per entity row (or batch for Type I)
    """
    return serialize_csv(csv_path, schema)
