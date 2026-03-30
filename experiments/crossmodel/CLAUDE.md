# Cross-Model Validation

| Script | Paper Section | Purpose |
|--------|--------------|---------|
| `run_crossmodel_expansion.py` | §4.6 Table 9 | GPT-5-mini as extraction backend on 5 datasets |

Validates that SGE's structural constraints transfer across LLM backends. Key finding: English Type-II achieves near-parity; Chinese Type-III shows 33.4% FC drop due to GPT-5-mini's weaker Chinese parsing.
