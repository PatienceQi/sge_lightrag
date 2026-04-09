#!/usr/bin/env python3
"""
query_parser.py — LLM-based statistical query parser for graph-first retrieval.

Extracts structured fields from natural language questions (English or Chinese):
  - entities: list of named entities mentioned (countries, companies, universities, diseases)
  - years: list of year constraints (e.g. [2020])
  - metric: the statistical metric being queried
  - query_type: one of lookup / comparison / ranking / trend / aggregation

Usage:
    from experiments.graph_first.query_parser import parse_statistical_query
    result = parse_statistical_query("What was China's life expectancy in 2020?")
    # -> {"entities": ["China"], "years": [2020], "metric": "life expectancy", "query_type": "lookup"}
"""

from __future__ import annotations

import os
import json
import re
import sys
import time
from pathlib import Path

import requests as _requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── LLM config (mirror of graph_guided_retriever.py pattern) ─────────────────
_API_KEY = os.environ.get("SGE_API_KEY", "")
_BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are a precise query parser for statistical data questions.
Extract structured information from the question and respond with JSON only.
No markdown, no explanation — just a single JSON object with these fields:
  "entities": list of named entities (countries, companies, universities, diseases)
  "years": list of years as integers (empty list if no year constraint)
  "metric": the statistical metric being queried (short phrase, English)
  "query_type": one of "lookup" | "comparison" | "ranking" | "trend" | "aggregation"

Definitions:
  lookup: single entity, single value (e.g. "China's population in 2020")
  comparison: two or more specific entities compared (e.g. "China vs India")
  ranking: which entities are top/bottom K (e.g. "top 5 countries by...")
  trend: change over time for one entity (e.g. "did X increase from 2000 to 2020?")
  aggregation: aggregate across all entities (e.g. "average", "how many countries above X")

Examples:
Q: "What was China's life expectancy in 2020?"
A: {"entities":["China"],"years":[2020],"metric":"life expectancy","query_type":"lookup"}

Q: "Which 3 countries had the highest child mortality in 2015?"
A: {"entities":[],"years":[2015],"metric":"child mortality","query_type":"ranking"}

Q: "Did India's maternal mortality consistently decrease from 2000 to 2019?"
A: {"entities":["India"],"years":[2000,2019],"metric":"maternal mortality","query_type":"trend"}

Q: "What was the average population across all countries in 2010?"
A: {"entities":[],"years":[2010],"metric":"population","query_type":"aggregation"}

Q: "肺炎的住院病人出院及死亡总人次是多少？"
A: {"entities":["肺炎"],"years":[],"metric":"inpatient total","query_type":"lookup"}
"""


def _call_llm_parse(question: str, max_retries: int = 3) -> str:
    """Call LLM to parse a question, returning raw text."""
    for attempt in range(max_retries):
        try:
            resp = _requests.post(
                f"{_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    "max_tokens": 256,
                    "temperature": 0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM parse failed: {exc}") from exc


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {text!r}")


def _fallback_parse(question: str) -> dict:
    """
    Regex-based fallback parser — no LLM required.
    Extracts years and guesses query type from keywords.
    """
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]

    q_lower = question.lower()
    if any(kw in q_lower for kw in ["top", "highest", "lowest", "best", "worst",
                                     "rank", "which", "最多", "最少", "前"]):
        query_type = "ranking"
    elif any(kw in q_lower for kw in ["average", "mean", "how many", "count",
                                       "total", "sum", "均", "合计", "多少种"]):
        query_type = "aggregation"
    elif any(kw in q_lower for kw in ["increase", "decrease", "trend", "change",
                                       "consistently", "增加", "减少", "是否"]):
        query_type = "trend"
    elif any(kw in q_lower for kw in ["compare", "versus", "vs", "higher than",
                                       "compared", "比较"]):
        query_type = "comparison"
    else:
        query_type = "lookup"

    return {
        "entities": [],
        "years": years,
        "metric": "",
        "query_type": query_type,
    }


def parse_statistical_query(question: str) -> dict:
    """
    Parse a natural language statistical question into structured fields.

    Args:
        question: Natural language question (English or Chinese).

    Returns:
        Dict with keys: entities (list[str]), years (list[int]),
        metric (str), query_type (str).
        On parse failure, falls back to regex-based extraction.
    """
    try:
        raw = _call_llm_parse(question)
        parsed = _extract_json(raw)
        # Normalize: ensure all required keys exist
        return {
            "entities": [str(e) for e in parsed.get("entities", [])],
            "years": [int(y) for y in parsed.get("years", [])],
            "metric": str(parsed.get("metric", "")),
            "query_type": str(parsed.get("query_type", "lookup")),
        }
    except Exception:
        return _fallback_parse(question)


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "What was China's life expectancy in 2020?",
        "Which 3 countries had the highest child mortality in 2015?",
        "Did India's maternal mortality consistently decrease from 2000 to 2019?",
        "What was the average population across all countries in 2010?",
        "肺炎的住院病人出院及死亡总人次是多少？",
        "Walmart's revenue in 2018?",
    ]
    for q in samples:
        result = parse_statistical_query(q)
        print(f"Q: {q}")
        print(f"   -> {result}")
        print()
