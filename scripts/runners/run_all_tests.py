#!/usr/bin/env python3
"""
run_all_tests.py — Run Stage 1 on all CSV files and collect results.
"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema

CSV_FILES = [
    # 香港本地医疗卫生总开支账目
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_1.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_2.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_3.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_4.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_5.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_6.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_7.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港本地医疗卫生总开支账目 /Table_8.csv",
    # 食物安全及公众卫生统计数字
    "/Users/qipatience/Desktop/SGE/dataset/食物安全及公众卫生统计数字/stat_foodSafty_publicHealth.csv",
    # 香港主要医疗卫生统计数字
    "/Users/qipatience/Desktop/SGE/dataset/香港主要医疗卫生统计数字/healthstat_table1.csv",
    "/Users/qipatience/Desktop/SGE/dataset/香港主要医疗卫生统计数字/healthstat_table2.csv",
    # 年度预算
    "/Users/qipatience/Desktop/SGE/dataset/年度预算/annualbudget_sc.csv",
    # 住院病人统计
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2012 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2013 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2014 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2015 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2016 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2017 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2018 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2019 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2020 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2021 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2022 (SC).csv",
    "/Users/qipatience/Desktop/SGE/dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv",
    # 有关人口的专题文章 - 住户开支统计调查结果
    "/Users/qipatience/Desktop/SGE/dataset/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 1.CSV",
    "/Users/qipatience/Desktop/SGE/dataset/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 2.CSV",
    "/Users/qipatience/Desktop/SGE/dataset/有关人口的专题文章 - 住户开支统计调查结果/B71608FB2016XXXXB01/2014_15_HES_Table 3.CSV",
]

results = []

for csv_path in CSV_FILES:
    p = Path(csv_path)
    fname = p.name
    folder = p.parent.name
    try:
        features = extract_features(csv_path)
        table_type = classify(features)
        meta_schema = build_meta_schema(features, table_type)
        
        # Inspect first few rows for correctness assessment
        df = features.df
        first_rows_preview = df.head(3).to_string(max_cols=6, max_colwidth=30)
        
        results.append({
            "file": fname,
            "folder": folder,
            "path": csv_path,
            "type": table_type,
            "headerless": features.headerless,
            "n_cols": len(features.raw_columns),
            "time_cols_in_headers": features.time_cols_in_headers[:5],
            "time_in_first_col": features.time_cols_in_first_col,
            "time_in_data_body": features.time_in_data_body,
            "leading_text_cols": features.leading_text_col_count,
            "first_rows": first_rows_preview,
            "error": None,
        })
    except Exception as e:
        results.append({
            "file": fname,
            "folder": folder,
            "path": csv_path,
            "type": "ERROR",
            "error": traceback.format_exc(),
        })

# Print summary
print("=" * 80)
print("STAGE 1 FULL CSV TEST RESULTS")
print("=" * 80)
for r in results:
    print(f"\n{'─'*60}")
    print(f"File   : {r['file']}")
    print(f"Folder : {r['folder']}")
    print(f"Type   : {r['type']}")
    if r.get("error"):
        print(f"ERROR  : {r['error'][:500]}")
    else:
        print(f"Headerless: {r.get('headerless')} | Cols: {r.get('n_cols')} | Leading text cols: {r.get('leading_text_cols')}")
        print(f"Time in headers: {r.get('time_cols_in_headers')} | In first col: {r.get('time_in_first_col')} | In body: {r.get('time_in_data_body')}")
        print(f"First rows preview:")
        print(r.get("first_rows", ""))

# Save JSON for report generation
with open("/tmp/stage1_results.json", "w", encoding="utf-8") as f:
    # Remove df preview for JSON (not serializable as-is)
    json_results = [{k: v for k, v in r.items() if k != "first_rows"} for r in results]
    json.dump(json_results, f, ensure_ascii=False, indent=2)

print("\n\nSUMMARY")
print("=" * 40)
from collections import Counter
type_counts = Counter(r["type"] for r in results)
for t, c in type_counts.items():
    print(f"  {t}: {c}")
print(f"  Total: {len(results)}")
errors = [r for r in results if r.get("error")]
print(f"  Errors: {len(errors)}")
