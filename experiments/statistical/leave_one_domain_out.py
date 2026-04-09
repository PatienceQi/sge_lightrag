#!/usr/bin/env python3
"""
leave_one_domain_out.py — Leave-one-domain-out cross-validation for Stage 1 classifier.

Validates that Algorithm 1's classification rules generalize across domains
by holding out each domain in turn and verifying classification accuracy.

Since Algorithm 1 is rule-based (not learned), "holding out" means:
1. Verify that the threshold _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE=6 would still
   be a reasonable choice based on the remaining domains' |C_T| distribution
2. Classify all held-out files and check accuracy
3. Report per-fold accuracy and feature distributions

Domains (5):
  1. HK Medical Expenditure Accounts (8 files, Type-I)
  2. HK Inpatient Statistics (12 files, Type-III)
  3. HK Surveys & Safety (food safety + HES, 4 files, Type-III)
  4. HK Admin & Stats (budget + health stats, 3 files, Type-II)
  5. International (WB + WHO, 6 files, Type-II)
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from stage1.features import extract_features
from stage1.classifier import classify, _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE

# ── All 33 files with ground truth types ──────────────────────────────────
BASE = str(_PROJECT_ROOT.parent / "dataset")

DOMAINS = {
    "HK Medical Expenditure": {
        "expected_type": "Flat-Entity",
        "files": [f"{BASE}/香港本地医疗卫生总开支账目 /Table_{i}.csv" for i in range(1, 9)],
    },
    "HK Inpatient": {
        "expected_type": "Hierarchical-Hybrid",
        "files": [
            f"{BASE}/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease {y} (SC).csv"
            for y in range(2012, 2024)
        ],
    },
    "HK Surveys & Safety": {
        "expected_type": "Hierarchical-Hybrid",
        "files": [
            f"{BASE}/食物安全及公众卫生统计数字/stat_foodSafty_publicHealth.csv",
            f"{BASE}/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 1.CSV",
            f"{BASE}/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 2.CSV",
            f"{BASE}/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 3.CSV",
        ],
    },
    "HK Admin & Stats": {
        "expected_type": "Time-Series-Matrix",
        "files": [
            f"{BASE}/年度预算/annualbudget_sc.csv",
            f"{BASE}/香港主要医疗卫生统计数字/healthstat_table1.csv",
            f"{BASE}/香港主要医疗卫生统计数字/healthstat_table2.csv",
        ],
    },
    "International (WB + WHO)": {
        "expected_type": "Time-Series-Matrix",
        "files": [
            f"{BASE}/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
            f"{BASE}/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv",
            f"{BASE}/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv",
            f"{BASE}/世界银行数据/health_expenditure/API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_558.csv",
            f"{BASE}/世界银行数据/hospital_beds/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_4542.csv",
            str(_PROJECT_ROOT.parent / "dataset" / "WHO" / "API_WHO_WHOSIS_000001_life_expectancy.csv"),
        ],
    },
}


def extract_file_features(csv_path):
    """Extract features and classify a single file, returning detailed info."""
    features = extract_features(csv_path)
    predicted_type = classify(features)
    return {
        "file": Path(csv_path).name,
        "path": csv_path,
        "predicted_type": predicted_type,
        "|C_T|": len(features.time_cols_in_headers),
        "C_key": features.leading_text_col_count,
        "n_numeric": features.n_numeric_cols,
        "transposed": features.time_cols_in_first_col,
        "yearInBody": features.time_in_data_body,
        "fiscal": any("-" in c for c in features.time_cols_in_headers),
        "n_rows": features.n_rows,
    }


def run_leave_one_domain_out():
    """Run the full leave-one-domain-out experiment."""
    # Step 1: Classify all 33 files and collect features
    print("=" * 70)
    print("LEAVE-ONE-DOMAIN-OUT CROSS-VALIDATION")
    print("=" * 70)

    all_results = {}
    for domain_name, domain_info in DOMAINS.items():
        domain_results = []
        for csv_path in domain_info["files"]:
            try:
                info = extract_file_features(csv_path)
                info["expected_type"] = domain_info["expected_type"]
                info["correct"] = info["predicted_type"] == info["expected_type"]
                domain_results.append(info)
            except Exception as e:
                domain_results.append({
                    "file": Path(csv_path).name,
                    "path": csv_path,
                    "error": str(e),
                    "expected_type": domain_info["expected_type"],
                    "correct": False,
                })
        all_results[domain_name] = domain_results

    # Step 2: Full-set accuracy
    total_files = sum(len(v) for v in all_results.values())
    total_correct = sum(
        sum(1 for r in results if r.get("correct", False))
        for results in all_results.values()
    )
    print(f"\n{'─' * 50}")
    print(f"Full-set accuracy: {total_correct}/{total_files} ({total_correct/total_files:.1%})")
    print(f"Threshold: _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE = {_MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE}")

    # Step 3: Show |C_T| distribution by type
    print(f"\n{'─' * 50}")
    print("|C_T| DISTRIBUTION BY PREDICTED TYPE")
    type_ct = defaultdict(list)
    for domain_results in all_results.values():
        for r in domain_results:
            if "error" not in r:
                type_ct[r["predicted_type"]].append(r["|C_T|"])

    for t in sorted(type_ct.keys()):
        vals = sorted(type_ct[t])
        print(f"  {t}: |C_T| ∈ {{{', '.join(str(v) for v in sorted(set(vals)))}}}")
        print(f"    count={len(vals)}, min={min(vals)}, max={max(vals)}")

    # Step 4: Leave-one-domain-out
    print(f"\n{'=' * 70}")
    print("PER-FOLD RESULTS")
    print("=" * 70)

    fold_results = []
    for held_out_domain in DOMAINS.keys():
        # Held-out files
        held_out_files = all_results[held_out_domain]
        n_held = len(held_out_files)
        n_correct_held = sum(1 for r in held_out_files if r.get("correct", False))
        accuracy = n_correct_held / n_held if n_held > 0 else 0.0

        # Remaining domains' |C_T| distribution (for threshold analysis)
        remaining_ct_by_type = defaultdict(list)
        for other_domain, results in all_results.items():
            if other_domain == held_out_domain:
                continue
            for r in results:
                if "error" not in r:
                    remaining_ct_by_type[r["predicted_type"]].append(r["|C_T|"])

        # Check if threshold=6 still creates separation
        type2_ct = remaining_ct_by_type.get("Time-Series-Matrix", [])
        type3_ct = remaining_ct_by_type.get("Hierarchical-Hybrid", [])
        type1_ct = remaining_ct_by_type.get("Flat-Entity", [])

        # Find min separation gap
        type3_max_ct = max(type3_ct) if type3_ct else 0
        type2_with_ct = [v for v in type2_ct if v > 0]  # Type-II files triggered by |C_T|>0
        type2_min_ct = min(type2_with_ct) if type2_with_ct else float('inf')

        print(f"\n{'─' * 50}")
        print(f"FOLD: Hold out '{held_out_domain}' ({n_held} files)")
        print(f"  Held-out accuracy: {n_correct_held}/{n_held} ({accuracy:.1%})")
        print(f"  Remaining |C_T| distribution:")
        for t in sorted(remaining_ct_by_type.keys()):
            vals = remaining_ct_by_type[t]
            print(f"    {t}: |C_T| ∈ {{{', '.join(str(v) for v in sorted(set(vals)))}}}")
        print(f"  Type-III max |C_T|={type3_max_ct}, Type-II min |C_T|={type2_min_ct}")
        print(f"  Threshold=6 still valid: {'YES' if type3_max_ct <= 6 else 'NO'}")

        # Misclassified files (if any)
        misclassified = [r for r in held_out_files if not r.get("correct", False)]
        if misclassified:
            print(f"  MISCLASSIFIED:")
            for r in misclassified:
                if "error" in r:
                    print(f"    {r['file']}: ERROR - {r['error']}")
                else:
                    print(f"    {r['file']}: expected={r['expected_type']}, got={r['predicted_type']}")
                    print(f"      |C_T|={r['|C_T|']}, C_key={r['C_key']}, n_numeric={r['n_numeric']}")
                    print(f"      transposed={r['transposed']}, yearInBody={r['yearInBody']}, fiscal={r['fiscal']}")

        fold_results.append({
            "domain": held_out_domain,
            "n_files": n_held,
            "n_correct": n_correct_held,
            "accuracy": accuracy,
            "misclassified": [
                {k: v for k, v in r.items() if k != "path"}
                for r in misclassified
            ],
            "remaining_ct_distribution": {
                t: sorted(set(vals))
                for t, vals in remaining_ct_by_type.items()
            },
            "threshold_valid": type3_max_ct <= _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE,
        })

    # Step 5: Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    all_correct = all(f["accuracy"] == 1.0 for f in fold_results)
    all_threshold_valid = all(f["threshold_valid"] for f in fold_results)

    print(f"  Domains: {len(fold_results)}")
    print(f"  Total files: {total_files}")
    print(f"  Per-fold accuracies: {', '.join(f'{f['accuracy']:.0%}' for f in fold_results)}")
    print(f"  All folds 100%: {'YES ✓' if all_correct else 'NO ✗'}")
    print(f"  Threshold valid in all folds: {'YES ✓' if all_threshold_valid else 'NO ✗'}")

    # Detailed feature table for all files
    print(f"\n{'=' * 70}")
    print("DETAILED FEATURE TABLE (ALL 33 FILES)")
    print("=" * 70)
    print(f"{'Domain':<25} {'File':<45} {'Type':<25} |C_T| C_key n_num trans yearB fiscal")
    for domain_name, results in all_results.items():
        for r in results:
            if "error" not in r:
                print(
                    f"{domain_name:<25} {r['file'][:44]:<45} {r['predicted_type']:<25} "
                    f"{r['|C_T|']:>4} {r['C_key']:>5} {r['n_numeric']:>5} "
                    f"{'T' if r['transposed'] else 'F':>5} {'T' if r['yearInBody'] else 'F':>5} "
                    f"{'T' if r['fiscal'] else 'F':>6}"
                )

    # Save results
    output = {
        "experiment": "leave-one-domain-out",
        "n_domains": len(fold_results),
        "n_files": total_files,
        "threshold": _MAX_YEAR_COLS_FOR_HYBRID_OVERRIDE,
        "full_set_accuracy": total_correct / total_files,
        "all_folds_perfect": all_correct,
        "all_threshold_valid": all_threshold_valid,
        "folds": fold_results,
        "type_distribution": dict(Counter(
            r["predicted_type"]
            for results in all_results.values()
            for r in results
            if "error" not in r
        )),
    }

    out_path = Path(__file__).parent / "leave_one_domain_out_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_leave_one_domain_out()
