#!/usr/bin/env python3
"""
run_stage1.py — CLI entry point for Stage 1: Topological Pattern Recognition.

Usage:
    python run_stage1.py <csv_path> [--preprocess]

Options:
    --preprocess    Run the CSV preprocessor before Stage 1 feature extraction.
                    Strips title rows, handles merged cells, normalises encoding.

Output:
    Prints the classified table type and the Meta-Schema JSON to stdout.
"""

import sys
import json
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_stage1.py <csv_path> [--preprocess]", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    use_preprocessor = "--preprocess" in sys.argv

    if not Path(csv_path).exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Optional preprocessing step
    temp_path = None
    if use_preprocessor:
        from stage1.preprocessor import preprocess_to_tempfile
        temp_path, meta = preprocess_to_tempfile(csv_path)
        print(f"[Preprocessor] encoding={meta['original_encoding']}, "
              f"rows_stripped={meta['rows_stripped']}, "
              f"shape {meta['original_shape']} → {meta['clean_shape']}",
              file=sys.stderr)
        pipeline_path = temp_path
    else:
        pipeline_path = csv_path

    try:
        # Stage 1 pipeline
        features    = extract_features(pipeline_path)
        table_type  = classify(features)
        meta_schema = build_meta_schema(features, table_type)

        print(f"Table Type : {table_type}")
        print("Meta-Schema:")
        print(json.dumps(meta_schema, ensure_ascii=False, indent=2))
    finally:
        # Clean up temp file if created
        if temp_path:
            import os
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
