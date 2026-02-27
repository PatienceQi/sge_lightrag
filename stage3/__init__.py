"""
stage3 — Constrained Extraction + LightRAG Integration

Takes Stage 2 extraction schema and:
1. Serializes CSV rows into natural-language text chunks
2. Generates schema-aware prompts for LightRAG
3. Provides the integration layer to inject schema into LightRAG
"""

from .serializer import serialize_csv
from .prompt_injector import generate_system_prompt, generate_user_prompt_template
from .integrator import patch_lightrag, prepare_chunks

__all__ = [
    "serialize_csv",
    "generate_system_prompt",
    "generate_user_prompt_template",
    "patch_lightrag",
    "prepare_chunks",
]
