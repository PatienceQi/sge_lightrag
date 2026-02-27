#!/usr/bin/env python3
"""
run_stage2_llm.py — CLI for LLM-enhanced schema induction (Stage 2).

Usage:
    python3 run_stage2_llm.py <csv_path>

Prints the resulting extraction schema as formatted JSON to stdout.
"""

import sys
import json
import os

# Ensure the sge_lightrag package root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_llm import induce_schema


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_stage2_llm.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {csv_path}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        schema = induce_schema(csv_path)
    except RuntimeError as exc:
        print(f"\n[API ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[PARSE ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Pretty-print the schema (hide internal _meta_schema for cleaner output)
    output = {k: v for k, v in schema.items() if k != "_meta_schema"}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n[Stage 1 Meta-Schema]")
    print(json.dumps(schema.get("_meta_schema", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
