#!/bin/bash
# Run full SGE pipeline with LLM-enhanced Stage 2 on OOD Type-II datasets
#
# Usage: bash run_ood_llm_batch.sh
#
# Target datasets:
#   - wb_unemployment.csv
#   - wb_immunization_dpt.csv
#   - wb_immunization_measles.csv

set -e
cd "$(dirname "$0")/../.."

OOD_DIR="../dataset/ood_blind_test"
OUTPUT_BASE="output_llm"

# Three target OOD datasets
TARGET_FILES=(
    "wb_unemployment.csv"
    "wb_immunization_dpt.csv"
    "wb_immunization_measles.csv"
)

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         OOD Batch Pipeline (LLM-Enhanced Stage 2)                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo "Output dir: ${OUTPUT_BASE}/"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for csv_file in "${TARGET_FILES[@]}"; do
    name="${csv_file%.csv}"
    out_dir="${OUTPUT_BASE}/${name}"

    # Check if output already exists
    if [ -d "${out_dir}" ] && [ -f "${out_dir}/sge_budget/lightrag_storage/graph_chunk_entity_relation.graphml" ]; then
        echo "[SKIP] ${csv_file}"
        echo "       Output already exists: ${out_dir}/"
        echo ""
        continue
    fi

    echo "[RUN] ${csv_file}"
    echo "      CSV: ${OOD_DIR}/${csv_file}"
    echo "      Output: ${out_dir}/"

    # Run integration pipeline
    if python3 scripts/runners/run_lightrag_integration_llm.py \
        "${OOD_DIR}/${csv_file}" \
        --output-dir "${out_dir}" \
        2>&1 | tail -20; then
        echo "      ✓ Done: ${csv_file}"
        ((SUCCESS_COUNT++))
    else
        echo "      ✗ Failed: ${csv_file}"
        ((FAIL_COUNT++))
    fi

    echo ""
done

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        Batch Summary                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Success: $SUCCESS_COUNT"
echo "Failed:  $FAIL_COUNT"
echo ""

# Print summary of results
echo "Comparison reports:"
for csv_file in "${TARGET_FILES[@]}"; do
    name="${csv_file%.csv}"
    report="${OUTPUT_BASE}/${name}/comparison_report.json"
    if [ -f "$report" ]; then
        echo ""
        echo "  ${name}:"
        jq '.sge | {stage2: "llm_enhanced", nodes: .node_count, edges: .edge_count}' "$report" 2>/dev/null || echo "    (could not parse)"
    fi
done

echo ""
echo "Next step: Run evaluation"
echo "  python3 evaluation/run_ood_evaluation.py --mode llm"
echo ""
echo "Or compare C1 vs C2:"
echo "  for csv in wb_unemployment wb_immunization_dpt wb_immunization_measles; do"
echo "    echo \"=== \$csv ===\""
echo "    jq '.sge | {nodes: .node_count, edges: .edge_count}' output/\$csv/comparison_report.json"
echo "    jq '.sge | {nodes: .node_count, edges: .edge_count}' output_llm/\$csv/comparison_report.json"
echo "  done"
