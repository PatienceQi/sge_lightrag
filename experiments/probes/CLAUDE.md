# Downstream Probes & E2E Experiments

## Graph-Native Probes (§4.5 Table 6)

| Script | Dataset | Method |
|--------|---------|--------|
| `graph_native_probe.py` | WHO Life Expectancy | BFS 2-hop + regex + deterministic rules |
| `graph_native_probe_wb_pop.py` | WB Population | Same method |
| `graph_native_probe_inpatient.py` | HK Inpatient | Same method (Type-III) |

No LLM involved — pure graph traversal + rule-based computation.

## E2E Experiments (§4.5 Table 7)

| Script | Purpose |
|--------|---------|
| `run_e2e_lightrag_qa.py` | Full LightRAG hybrid query (100 questions) |
| `run_e2e_compact.py` | SGE compact graph construction + E2E query |
| `run_e2e_compact_query_only.py` | Query-only on existing compact graph |
| `run_e2e_baseline_compact.py` | Baseline compact graph construction |
| `run_e2e_baseline_compact_query_only.py` | Query-only on baseline compact graph |

## Analysis

| Script | Purpose |
|--------|---------|
| `who_fidelity_compact_analysis.py` | FC comparison: independent vs compact nodes |
