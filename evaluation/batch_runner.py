#!/usr/bin/env python3
"""
batch_runner.py — Batch baseline runner with timeout + retry.

Runs a list of baseline commands with:
  - Per-command timeout (default 30 min)
  - Auto-retry on failure (max 2 attempts)
  - Progress logging
  - Skips already-completed datasets (checks result JSON)

Usage:
    python3 evaluation/batch_runner.py --config fixed_stv
    python3 evaluation/batch_runner.py --config oecd_sge
    python3 evaluation/batch_runner.py --config type3_ood
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "fixed_stv": {
        "result_file": "evaluation/results/fixed_stv_baseline_results.json",
        "datasets": ["who", "wb_cm", "wb_pop", "wb_mat", "inpatient", "fortune500", "the"],
        "cmd_template": "python3 evaluation/fixed_stv_baseline.py --dataset {dataset} --fresh",
        "timeout_min": 40,
    },
    "row_local": {
        "result_file": "evaluation/results/row_local_baseline_results.json",
        "datasets": ["who", "wb_cm", "wb_pop", "wb_mat", "inpatient", "fortune500", "the"],
        "cmd_template": "python3 evaluation/row_local_baseline.py --dataset {dataset} --fresh",
        "timeout_min": 40,
    },
    "json_structured": {
        "result_file": "evaluation/results/json_structured_baseline_results.json",
        "datasets": ["who", "wb_cm", "wb_mat"],
        "cmd_template": "python3 evaluation/json_structured_baseline.py --dataset {dataset} --fresh",
        "timeout_min": 30,
    },
}


def dataset_done(result_file: str, dataset: str) -> bool:
    """Check if a dataset has a complete result (with stats or timestamp)."""
    path = PROJECT_ROOT / result_file
    if not path.exists():
        return False
    with open(path) as f:
        data = json.load(f)
    entry = data.get(dataset, {})
    # Consider complete if it has evaluation data with non-zero totals
    ev = entry.get("evaluation", {})
    return "ec" in ev and ("stats" in entry or "timestamp" in entry)


def run_one(cmd: str, timeout_sec: int, max_retries: int = 2) -> tuple[bool, str]:
    """Run a command with timeout and retry. Returns (success, output)."""
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}: {cmd}")
        try:
            env = {**os.environ, "no_proxy": "*", "NO_PROXY": "*"}
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout_sec, cwd=str(PROJECT_ROOT), env=env,
            )
            if result.returncode == 0:
                return True, result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            print(f"  FAILED (exit {result.returncode}): {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {timeout_sec}s")
            # Kill any orphan python processes for this command
            subprocess.run(
                f"ps aux | grep '{cmd.split()[-1]}' | grep -v grep | awk '{{print $2}}' | xargs kill 2>/dev/null",
                shell=True, capture_output=True,
            )
        if attempt < max_retries:
            print(f"  Retrying in 30s...")
            time.sleep(30)

    return False, "All attempts failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--timeout", type=int, help="Override timeout (minutes)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    timeout_sec = (args.timeout or cfg["timeout_min"]) * 60

    print(f"{'='*60}")
    print(f"Batch Runner: {args.config}")
    print(f"Datasets: {cfg['datasets']}")
    print(f"Timeout: {timeout_sec // 60} min per dataset")
    print(f"{'='*60}")

    results_summary = {}

    for ds in cfg["datasets"]:
        print(f"\n--- {ds} ---")

        if dataset_done(cfg["result_file"], ds):
            print(f"  SKIP (already complete)")
            results_summary[ds] = "skipped"
            continue

        cmd = cfg["cmd_template"].format(dataset=ds)
        success, output = run_one(cmd, timeout_sec)

        if success:
            # Read the result
            result_path = PROJECT_ROOT / cfg["result_file"]
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                ev = data.get(ds, {}).get("evaluation", {})
                fc = ev.get("fc", "?")
                print(f"  OK: FC={fc}")
                results_summary[ds] = f"FC={fc}"
            else:
                print(f"  OK but result file not found")
                results_summary[ds] = "ok_no_file"
        else:
            print(f"  FAILED after retries")
            results_summary[ds] = "FAILED"

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.config}")
    print(f"{'='*60}")
    for ds, status in results_summary.items():
        print(f"  {ds}: {status}")

    failed = [ds for ds, s in results_summary.items() if s == "FAILED"]
    if failed:
        print(f"\nFAILED datasets: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll datasets complete or skipped.")


if __name__ == "__main__":
    main()
