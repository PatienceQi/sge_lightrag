"""
llm_client.py — Thin wrapper around the OpenRouter / OpenAI-compatible API.

Uses the openai SDK pointed at https://api.packy.ai/v1.
"""

import time
from openai import OpenAI

_DEFAULT_BASE_URL = "https://wolfai.top/v1"
_DEFAULT_API_KEY  = "sk-GhswVJ825Z6sqFGlUm54n8W9jj0sJwfJOdWjyMNWJEihROlr"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"

_MAX_RETRIES = 3


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat completion request and return the assistant's text content.
    Retries up to _MAX_RETRIES times on failure.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    last_exc = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(1 * attempt)
                continue

    raise RuntimeError(f"LLM API call failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc
