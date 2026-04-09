"""
config.py — Centralized API configuration for evaluation scripts.

All baseline and evaluation scripts should import from here instead of
hardcoding API keys and model names.

Usage:
    from evaluation.config import API_KEY, BASE_URL, MODEL, EMBED_DIM
"""

import os

API_KEY   = os.environ.get("SGE_API_KEY", "")
BASE_URL  = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL     = os.environ.get("SGE_MODEL", "claude-haiku-4-5-20251001")
EMBED_DIM = 1024
