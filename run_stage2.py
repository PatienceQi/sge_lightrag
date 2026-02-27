#!/usr/bin/env python3
"""
run_stage2.py — CLI entry point for Stage 2: Rule-Based Schema Induction.

Usage:
    python3 run_stage2.py <csv_path>

Output:
    Prints the full extraction schema JSON to stdout.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage2.inductor import induce_schema


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_stage2.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    schema = induce_schema(csv_path)

    print(json.dumps(schema, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
