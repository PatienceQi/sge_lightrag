"""
Threshold Sensitivity Analysis for Stage 1 Classification.

Sweeps _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE (C_T threshold) from 3 to 9
and SMALL_TABLE_THRESHOLD from 10 to 30.

Reports: which files change classification at each threshold.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from stage1.features import extract_features
from stage1 import classifier

# Find all non-metadata CSV files
DATASET_ROOT = Path(__file__).resolve().parent.parent.parent / "dataset"

def find_csv_files():
    """Find all non-metadata CSV files."""
    csvs = []
    for p in sorted(DATASET_ROOT.rglob("*.csv")):
        if "Metadata" in p.name:
            continue
        csvs.append(p)
    return csvs


def classify_with_threshold(features_cache, ct_threshold):
    """Classify all files with a given C_T threshold."""
    original = classifier._MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE
    classifier._MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE = ct_threshold
    results = {}
    for path, feat in features_cache.items():
        results[path] = classifier.classify(feat)
    classifier._MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE = original
    return results


def main():
    csv_files = find_csv_files()
    print(f"Found {len(csv_files)} CSV files\n")

    # Extract features once (expensive part)
    features_cache = {}
    for p in csv_files:
        try:
            feat = extract_features(str(p))
            features_cache[p] = feat
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    print(f"Successfully loaded {len(features_cache)} files\n")

    # --- Experiment 1: C_T threshold sweep ---
    print("=" * 80)
    print("EXPERIMENT 1: C_T Threshold Sensitivity (|C_T| <= threshold)")
    print("=" * 80)

    baseline_threshold = 6
    baseline_results = classify_with_threshold(features_cache, baseline_threshold)

    # Print baseline distribution
    type_counts = {}
    for t in baseline_results.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nBaseline (threshold={baseline_threshold}): {type_counts}")

    # Show per-file year columns for context
    print(f"\n{'File':<50} {'|C_T|':>5} {'Type(6)':>18}")
    print("-" * 78)
    for p, feat in sorted(features_cache.items(), key=lambda x: x[0].name):
        n_ct = len(feat.time_cols_in_headers)
        t = baseline_results[p]
        short = p.name[:48]
        print(f"{short:<50} {n_ct:>5} {t:>18}")

    # Sweep thresholds
    print(f"\n{'Threshold':>10}", end="")
    for t in ["Type-I", "Type-II", "Type-III"]:
        print(f"  {t:>8}", end="")
    print("  Changes vs baseline")
    print("-" * 70)

    for thresh in range(3, 10):
        results = classify_with_threshold(features_cache, thresh)
        counts = {"Flat-Entity": 0, "Time-Series-Matrix": 0, "Hierarchical-Hybrid": 0}
        for t in results.values():
            counts[t] = counts.get(t, 0) + 1

        changes = []
        for p in features_cache:
            if results[p] != baseline_results[p]:
                changes.append(f"{p.name[:30]}:{baseline_results[p][:5]}->{results[p][:5]}")

        marker = " ← current" if thresh == baseline_threshold else ""
        print(f"{thresh:>10}  {counts.get('Flat-Entity', 0):>8}  "
              f"{counts.get('Time-Series-Matrix', 0):>8}  "
              f"{counts.get('Hierarchical-Hybrid', 0):>8}  "
              f"{len(changes)} change(s){marker}")
        for c in changes:
            print(f"{'':>42}{c}")

    # --- Experiment 2: SMALL_TABLE_THRESHOLD sweep ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Adaptive Degradation Threshold (n_rows < threshold)")
    print("=" * 80)

    print(f"\n{'File':<50} {'n_rows':>6} {'Type':>18}")
    print("-" * 78)

    type3_files = [(p, feat) for p, feat in features_cache.items()
                   if baseline_results[p] == "Hierarchical-Hybrid"]
    for p, feat in sorted(type3_files, key=lambda x: x[1].n_rows):
        print(f"{p.name[:48]:<50} {feat.n_rows:>6} {baseline_results[p]:>18}")

    print(f"\n{'Threshold':>10}  {'Files degraded':>15}  Affected files")
    print("-" * 70)
    for thresh in [10, 15, 20, 25, 30]:
        degraded = [(p, feat) for p, feat in type3_files if feat.n_rows < thresh]
        marker = " ← current" if thresh == 20 else ""
        names = ", ".join(p.name[:25] for p, _ in degraded)
        print(f"{thresh:>10}  {len(degraded):>15}  {names}{marker}")


if __name__ == "__main__":
    main()
