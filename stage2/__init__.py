"""
stage2 — Rule-Based Schema Induction

Takes Stage 1 output (table_type + Meta-Schema) and the raw CSV data,
and produces a full extraction schema without requiring an LLM.
"""

from .inductor import induce_schema, induce_schema_from_meta

__all__ = ["induce_schema", "induce_schema_from_meta"]
