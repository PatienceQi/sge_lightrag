#!/bin/bash
# Run LLM-enhanced SGE pipeline on 3 failed OOD datasets
set -e
cd "$(dirname "$0")/../.."

OOD_DIR="../dataset/ood_blind_test"
OUTPUT_BASE="output/ood_llm"

DATASETS=(
    "wb_unemployment.csv"
    "wb_immunization_dpt.csv"
    "wb_immunization_measles.csv"
)

echo "=== OOD LLM SGE Pipeline (3 failed datasets) ==="
echo "Start time: $(date)"

for csv_file in "${DATASETS[@]}"; do
    name="${csv_file%.csv}"
    out_dir="${OUTPUT_BASE}/${name}"

    echo ""
    echo ">>> Processing ${csv_file} with LLM Stage 2..."
    python3 scripts/runners/run_lightrag_integration_llm.py "${OOD_DIR}/${csv_file}" --output-dir "${out_dir}" 2>&1
    echo "    Done: ${csv_file} at $(date)"
done

echo ""
echo "=== All 3 datasets complete: $(date) ==="

# Run evaluation on LLM results
echo ""
echo ">>> Running FC evaluation on LLM SGE results..."
python3 -c "
import json, sys
sys.path.insert(0, '.')
from evaluation.evaluate_coverage import evaluate_coverage
from pathlib import Path

GOLD_DIR = Path('evaluation/gold')
OUTPUT_BASE = Path('output/ood_llm')
results = []

for name in ['wb_unemployment', 'wb_immunization_dpt', 'wb_immunization_measles']:
    gold_file = GOLD_DIR / f'gold_ood_{name}.jsonl'
    sge_storage = OUTPUT_BASE / name / 'sge_budget' / 'lightrag_storage'
    baseline_storage = OUTPUT_BASE / name / 'baseline_budget' / 'lightrag_storage'

    if not gold_file.exists():
        print(f'  SKIP {name}: no gold file')
        continue

    row = {'dataset': name}
    for label, storage in [('sge_llm', sge_storage), ('baseline', baseline_storage)]:
        if not storage.exists():
            print(f'  SKIP {name}/{label}: no storage')
            row[f'{label}_fc'] = None
            continue
        try:
            ec, fc, details = evaluate_coverage(graph_dir=str(storage), gold_path=str(gold_file))
            row[f'{label}_ec'] = round(ec, 3)
            row[f'{label}_fc'] = round(fc, 3)
            row[f'{label}_nodes'] = details.get('n_nodes', 0)
            print(f'  {name}/{label}: EC={ec:.3f} FC={fc:.3f} nodes={details.get(\"n_nodes\", 0)}')
        except Exception as e:
            print(f'  ERROR {name}/{label}: {e}')
            row[f'{label}_fc'] = None
    results.append(row)

# Also compare with Rule SGE results
print()
print('=== Comparison: Rule SGE vs LLM SGE vs Baseline ===')
RULE_BASE = Path('output/ood')
for row in results:
    name = row['dataset']
    rule_storage = RULE_BASE / name / 'sge_budget' / 'lightrag_storage'
    gold_file = GOLD_DIR / f'gold_ood_{name}.jsonl'
    if rule_storage.exists() and gold_file.exists():
        try:
            ec, fc, details = evaluate_coverage(graph_dir=str(rule_storage), gold_path=str(gold_file))
            row['sge_rule_fc'] = round(fc, 3)
        except:
            row['sge_rule_fc'] = None
    print(f'{name}: Rule={row.get(\"sge_rule_fc\",\"?\")}, LLM={row.get(\"sge_llm_fc\",\"?\")}, Base={row.get(\"baseline_fc\",\"?\")}')

out_path = Path('evaluation/results/ood_llm_results.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nResults saved to {out_path}')
"
