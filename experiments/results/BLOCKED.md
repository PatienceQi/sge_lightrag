# Blocked Experiments

## Status: 0 items pending (4 resolved: 3 from Round 1 + 1 from Round 2)

### 1. ~~Phase 2A: Markdown No-Schema cell~~ **RESOLVED**
- Result: EC=0.96, FC=0.127 (34 nodes, 37 edges)
- Schema HURTS Markdown: +Schema FC=0.000 vs -Schema FC=0.127 (interaction = -0.167)

### 2. ~~Phase 2B: Context-exceeding GGCR~~ **RESOLVED**
- Result: GGCR 78% vs Concat-All-Truncated 20% (McNemar p<0.001)
- 217 total questions across 3 scale tiers
- GGCR = Concat-All-Full (78% vs 72%, p=0.55) when window is sufficient

### 3. ~~Phase 3: MS GraphRAG cross-host port~~ **RESOLVED**
- **Result**: SGE-GraphRAG WHO FC = 1.000 (150/150), matching baseline FC = 1.000
- **SGE entity structure**: 2 types (Country_Code: 196, YearValue: 4312) vs baseline 4 types
- **Script**: `experiments/sge_graphrag_who_eval.py`
- **Output**: `output/graphrag_sge_who/` (196 chunks, 4508 entities, 4312 relationships)
- **Results**: `experiments/results/sge_graphrag_results.json`
- **Note**: WHO is a clean dataset where both SGE and baseline achieve FC=1.000. The key
  contribution is structural fidelity — SGE produces exactly 2 entity types matching the
  HAS_MEASUREMENT schema, while baseline produces 4 types including noise (YEAR: 22 nodes).
  For cross-host generalizability argument, cite MS_GraphRAG WHO=1.0, WB_Pop=0.6, Inpatient=0.0
  from existing unified_cross_system.json.

### 4. ~~SGE-GraphRAG Inpatient (Type-III)~~ **RESOLVED**
- **Result**: SGE-GraphRAG Inpatient FC = 1.000 (16/16), vs baseline FC = 0.000
- **SGE entity structure**: 2 types (Disease_Category: 308, StatValue: 924) vs baseline 0 recoverable facts
- **Output**: `output/graphrag_sge_inpatient/` (308 chunks, 1232 entities, 924 relationships)
- **Results**: `experiments/results/sge_graphrag_inpatient_results.json`
