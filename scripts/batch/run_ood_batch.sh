#!/bin/bash
# Run full SGE pipeline + Baseline on all OOD Type-II datasets
# Usage: bash run_ood_batch.sh

set -e
cd "$(dirname "$0")/../.."

OOD_DIR="../dataset/ood_blind_test"
OUTPUT_BASE="output/ood"

# Type-II datasets (Country×Year matrix)
TYPE2_FILES=(
    "wb_gdp_growth.csv"
    "wb_unemployment.csv"
    "wb_education_spending.csv"
    "wb_health_expenditure.csv"
    "wb_co2_emissions.csv"
    "wb_cereal_production.csv"
    "wb_literacy_rate.csv"
    "wb_population_growth.csv"
    "wb_immunization_dpt.csv"
    "wb_immunization_measles.csv"
)

echo "=== OOD Batch Pipeline ==="
echo "Start time: $(date)"
echo ""

for csv_file in "${TYPE2_FILES[@]}"; do
    name="${csv_file%.csv}"
    out_dir="${OUTPUT_BASE}/${name}"

    if [ -d "${out_dir}" ] && [ -f "${out_dir}/sge_*/graph_chunk_entity_relation.graphml" 2>/dev/null ]; then
        echo "SKIP ${csv_file} (output exists)"
        continue
    fi

    echo ">>> Processing ${csv_file}..."
    python3 scripts/runners/run_lightrag_integration.py "${OOD_DIR}/${csv_file}" --output-dir "${out_dir}" 2>&1 | tail -5
    echo "    Done: ${csv_file}"
    echo ""
done

echo "=== Batch complete: $(date) ==="
echo ""
echo "Next step: Run evaluation"
echo "  python3 evaluation/run_ood_evaluation.py"
