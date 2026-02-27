"""
header_parser.py — Parse compound column headers into structured components.

Handles headers like:
    "2022-23\n(实际)\n(百万元)"  →  {year: "2022-23", status: "实际", unit: "百万元"}
    "2024"                       →  {year: "2024", status: None, unit: None}
    "纲领"                       →  {year: None, status: None, unit: None}
"""

import re
from dataclasses import dataclass
from typing import Optional

# Matches bare 4-digit years or fiscal-year ranges like 2022-23 / 2022-2023
_YEAR_RE = re.compile(r"^(\d{4}(?:-\d{2,4})?)$")

# Parenthesised tokens: (实际), (百万元), (原来预算), etc.
_PAREN_RE = re.compile(r"\(([^)]+)\)")

# Known unit keywords (Chinese financial/statistical units)
_UNIT_KEYWORDS = re.compile(
    r"百万|千万|亿|元|万元|港元|美元|million|billion|%|percent|人|次|个|所|张|辆|公里|km",
    re.IGNORECASE,
)

# Known status/qualifier keywords
_STATUS_KEYWORDS = re.compile(
    r"实际|预算|修订|原来|估计|actual|budget|revised|estimate|forecast",
    re.IGNORECASE,
)


@dataclass
class ParsedHeader:
    """Structured representation of a (possibly compound) column header."""
    raw: str                    # original header string
    year: Optional[str]         # e.g. "2022-23" or "2024"
    status: Optional[str]       # e.g. "实际", "预算", "修订"
    unit: Optional[str]         # e.g. "百万元", "%"
    label: Optional[str]        # non-year, non-paren text (the "name" part)
    is_time_column: bool        # True if this header encodes a time period

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "year": self.year,
            "status": self.status,
            "unit": self.unit,
            "label": self.label,
            "is_time_column": self.is_time_column,
        }


def parse_header(raw: str) -> ParsedHeader:
    """
    Parse a single column header string into its components.

    Compound headers use newlines as separators (as produced by pandas when
    reading multi-line CSV cell values).  We split on newlines first, then
    classify each token.
    """
    raw_str = str(raw).strip()

    # Split on newlines; each line may be a year, a parenthesised qualifier,
    # or a plain label.
    lines = [ln.strip() for ln in raw_str.split("\n") if ln.strip()]

    year: Optional[str] = None
    status: Optional[str] = None
    unit: Optional[str] = None
    label_parts: list[str] = []

    for line in lines:
        # Extract all parenthesised tokens from this line
        paren_tokens = _PAREN_RE.findall(line)
        # The non-paren remainder
        remainder = _PAREN_RE.sub("", line).strip()

        # Classify parenthesised tokens
        for tok in paren_tokens:
            tok = tok.strip()
            if _STATUS_KEYWORDS.search(tok):
                status = tok
            elif _UNIT_KEYWORDS.search(tok):
                unit = tok
            else:
                # Unknown paren token — treat as status qualifier
                status = status or tok

        # Classify the remainder
        if remainder:
            m = _YEAR_RE.match(remainder)
            if m:
                year = m.group(1)
            else:
                label_parts.append(remainder)

    label = " ".join(label_parts) if label_parts else None
    is_time = year is not None

    return ParsedHeader(
        raw=raw_str,
        year=year,
        status=status,
        unit=unit,
        label=label,
        is_time_column=is_time,
    )


def parse_all_headers(raw_columns: list[str]) -> list[ParsedHeader]:
    """Parse every column header in a list and return ParsedHeader objects."""
    return [parse_header(c) for c in raw_columns]
