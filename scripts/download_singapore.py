#!/usr/bin/env python3
"""
download_singapore.py — Download health datasets from data.gov.sg.

Uses the async polling pattern:
  POST initiate-download → poll GET url → download from S3 pre-signed URL.

Usage:
    python3 scripts/download_singapore.py
    python3 scripts/download_singapore.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

BASE_URL = "https://api-open.data.gov.sg/v1/public/api/datasets"

DATASETS = {
    "d_a40c83a6f36893fc4611eda91f84eb6b": "GovernmentHealthExpenditureAnnual.csv",
    "d_e7fa2cba35ccc1173d941986e07b09df": "HealthFacilitiesAnnual.csv",
}

_MAX_POLL = 30
_POLL_INTERVAL = 2  # seconds


def download_dataset(dataset_id: str, output_path: Path) -> bool:
    """Initiate download, poll until ready, then fetch CSV. Returns True on success."""
    print(f"  Initiating download: {dataset_id}")

    try:
        r = requests.post(f"{BASE_URL}/{dataset_id}/initiate-download", timeout=15)
        r.raise_for_status()
        poll_url = r.json()["data"]["url"]
    except Exception as e:
        print(f"  [error] Failed to initiate download: {e}")
        return False

    print(f"  Polling for completion...")
    for attempt in range(1, _MAX_POLL + 1):
        try:
            status_r = requests.get(poll_url, timeout=15)
            status_r.raise_for_status()
            data = status_r.json().get("data", {})

            if data.get("status") == "completed":
                csv_url = data["url"]
                print(f"  Downloading CSV from S3...")
                csv_r = requests.get(csv_url, timeout=60)
                csv_r.raise_for_status()
                output_path.write_bytes(csv_r.content)
                print(f"  Saved: {output_path} ({len(csv_r.content):,} bytes)")
                return True

            if data.get("status") == "failed":
                print(f"  [error] Server reported download failed")
                return False

            print(f"  [{attempt}/{_MAX_POLL}] status={data.get('status', '?')}, waiting...")
            time.sleep(_POLL_INTERVAL)

        except Exception as e:
            print(f"  [warn] Poll attempt {attempt} failed: {e}")
            time.sleep(_POLL_INTERVAL)

    print(f"  [error] Timed out after {_MAX_POLL} poll attempts")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Singapore data.gov.sg health datasets")
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).parent.parent / "dataset" / "新加坡"),
        help="Directory to save downloaded CSV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    results = {}
    for dataset_id, filename in DATASETS.items():
        output_path = output_dir / filename
        print(f"Dataset: {filename}")
        success = download_dataset(dataset_id, output_path)
        results[filename] = "OK" if success else "FAILED"
        print()

    print("=" * 40)
    print("Summary:")
    for name, status in results.items():
        print(f"  {status}  {name}")


if __name__ == "__main__":
    main()
