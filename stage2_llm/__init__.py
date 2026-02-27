"""
stage2_llm — LLM-Enhanced Schema Induction (Stage 2)

Takes Stage 1's Meta-Schema + CSV snippet → calls Claude Haiku via
OpenRouter to generate a rich extraction schema for Stage 3.
"""

from .inductor import induce_schema

__all__ = ["induce_schema"]
