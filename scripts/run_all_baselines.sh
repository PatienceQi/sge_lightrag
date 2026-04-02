#!/bin/bash
# run_all_baselines.sh — Sequential execution of all multi-host baselines
# Avoids Ollama resource contention by running one at a time

set -e
cd /Users/qipatience/Desktop/SGE/sge_lightrag

PYTHON3="/opt/homebrew/opt/python@3.14/bin/python3.14"
HIPPO_PYTHON="/Users/qipatience/miniconda3/envs/hipporag/bin/python"

echo "============================================"
echo "Phase 1: Evaluate existing GraphRAG outputs"
echo "============================================"

# WHO already done during indexing
echo ">> GraphRAG WHO evaluation..."
$PYTHON3 evaluation/evaluate_coverage.py \
    --graph output/graphrag_who/output/graph.graphml \
    --gold evaluation/gold/gold_who_life_expectancy_v2.jsonl 2>&1 | tail -5

# Wait for WB_CM and Inpatient indexing if still running
echo ">> Checking for running GraphRAG processes..."
while pgrep -f "graphrag index" > /dev/null 2>&1; do
    echo "   GraphRAG still indexing... waiting 30s"
    sleep 30
done
echo "   All GraphRAG indexing complete."

# Evaluate WB_CM
if [ -f "output/graphrag_wb_cm/output/graph.graphml" ]; then
    echo ">> GraphRAG WB_CM evaluation..."
    $PYTHON3 evaluation/evaluate_coverage.py \
        --graph output/graphrag_wb_cm/output/graph.graphml \
        --gold evaluation/gold/gold_wb_child_mortality_v2.jsonl 2>&1 | tail -5
fi

# Evaluate Inpatient (full)
if [ -f "output/graphrag_inpatient_full/output/graph.graphml" ]; then
    echo ">> GraphRAG Inpatient evaluation..."
    $PYTHON3 evaluation/evaluate_coverage.py \
        --graph output/graphrag_inpatient_full/output/graph.graphml \
        --gold evaluation/gold/gold_inpatient_2023.jsonl 2>&1 | tail -5
fi

echo ""
echo "============================================"
echo "Phase 2: Naive RAG baseline (QA evaluation)"
echo "============================================"
echo ">> Building embedding indices + running QA..."
$PYTHON3 evaluation/naive_rag_baseline.py \
    --questions evaluation/gold/qa_questions.jsonl \
    --output evaluation/results/naive_rag_results.json

echo ""
echo "============================================"
echo "Phase 3: HippoRAG indexing + QA"
echo "============================================"

for ds in who wb_cm inpatient; do
    echo ">> HippoRAG $ds indexing..."
    $HIPPO_PYTHON evaluation/hipporag_baseline.py \
        --dataset $ds --mode both \
        --output output/hipporag_$ds \
        --results evaluation/results/hipporag_${ds}_qa.json \
        2>&1 | tail -10
    echo ""
done

echo ""
echo "============================================"
echo "All baselines complete!"
echo "============================================"
