"""
llm_client.py — Thin wrapper around the OpenRouter / OpenAI-compatible API.

Uses the openai SDK pointed at https://api.packy.ai/v1.
"""

from openai import OpenAI

_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
_DEFAULT_API_KEY  = "sk-WGhptV4Oz0oqHcccSyy7AfD3nFKxQLdHeJ0rFqcns188eoPX"
_DEFAULT_MODEL    = "claude-haiku-4-5-20251001"


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

    Raises RuntimeError with a descriptive message on API failure.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)

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
        raise RuntimeError(f"LLM API call failed: {exc}") from exc
