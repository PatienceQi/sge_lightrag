# Type-III Datasets: Triple F1 Applicability Analysis

Generated: 2026-04-03

## Conclusion

**Triple F1 (canonical (subject, relation, object) recall) is NOT applicable to Type-III datasets.**
Both Inpatient and THE Ranking report recall=0 for both SGE and Baseline — this is structurally expected,
not a system failure. The metric should only be reported for Type-II datasets in the paper.

---

## Dataset 1: Inpatient 2023 (HK ICD Discharges)

### Graph structure (SGE output)

The SGE graph uses a two-level hierarchy:

```
Disease_Category (e.g., 肺炎)
  └─[HAS_SUB_ITEM]─> ICD_code_node (e.g., J12-J18)
      ├─[HAS_VALUE,住院病人出院及死亡人次,合计]─> 60614.0
      ├─[HAS_VALUE,住院病人出院及死亡人次,医院管理局辖下医院]─> 55487.0
      ├─[HAS_VALUE,住院病人出院及死亡人次,私家医院]─> 5127.0
      ├─[HAS_VALUE,全港登记死亡人数,合计]─> 11334.0
      └─ ...
```

Graph stats: 1190 nodes, 996 edges. Edge keywords encode column semantics.

### Gold Standard format

Gold triples use the disease category name as subject:

```json
{"triple": {"subject": "肺炎", "relation": "INPATIENT_TOTAL", "object": "60614"}}
```

### Why triple recall = 0

The gold triple `(肺炎, INPATIENT_TOTAL, 60614)` requires a direct edge
`肺炎 → 60614`. In the SGE graph, this path is two hops:
`肺炎 →[HAS_SUB_ITEM]→ J12-J18 →[HAS_VALUE,合计]→ 60614.0`.

The subject in the gold triple (`肺炎`) does not appear as the direct source
of the value edge — the ICD code node (`J12-J18`) does. This two-level
intermediary is a correct and intentional graph design for Type-III
hierarchical data, but it breaks the (subject, relation, object) triple
matching assumption.

Additionally, the object format differs: gold uses integer `"60614"`,
graph stores float `"60614.0"`.

---

## Dataset 2: THE University Ranking 2024

### Graph structure (SGE output)

```
University_node (e.g., California Institute of Technology)
  └─[HAS_PERFORMANCE_METRIC,score: 96.0,year: 2011]─> year_node (e.g., "2011")
  └─[HAS_PERFORMANCE_METRIC,score: 94.8,year: 2012]─> year_node (e.g., "2012")
  └─ ...
```

Graph stats: 72 nodes, 196 edges. The score value is encoded in the edge
keyword string, not as a separate value node.

### Gold Standard format

```json
{"triple": {"subject": "California Institute of Technology",
            "relation": "RANKING_SCORE", "object": "96.0",
            "attributes": {"year": "2011"}}}
```

### Why triple recall = 0

The gold triple `(Caltech, RANKING_SCORE, 96.0)` requires the object to be
the score value `"96.0"`. In the SGE graph, the edge target is the year node
`"2011"`, and the score `96.0` is embedded in the keyword field
(`HAS_PERFORMANCE_METRIC,score: 96.0,year: 2011`).

The object in the gold triple (`96.0`) is not a standalone node — it is an
edge attribute. Standard triple matching compares nodes, not edge keywords,
so no match is found.

---

## Type-II vs Type-III Comparison

| Dataset | Type | SGE Recall | Baseline Recall | Triple F1 applicable? |
|---------|------|-----------|----------------|----------------------|
| WHO Life Expectancy | Type-II | 1.000 | 0.000 | Yes |
| WB Child Mortality | Type-II | 1.000 | 0.000 | Yes |
| WB Population | Type-II | 1.000 | 0.040 | Yes |
| WB Maternal | Type-II | 0.967 | 0.000 | Yes |
| Fortune 500 Revenue | Type-II | 1.000 | 0.000 | Yes |
| **Inpatient 2023** | **Type-III** | **0.000** | **0.000** | **No** |
| **THE Ranking** | **Type-III** | **0.000** | **0.000** | **No** |

Type-II average SGE recall = 0.993; Baseline = 0.008.

---

## Recommendation for Paper

1. Report triple F1 (recall) only for the 5 Type-II datasets.
2. For Type-III datasets, use FC (Fact Coverage) as the primary metric —
   the existing FC evaluation correctly handles multi-hop and attribute-embedded
   graph structures via partial-match scoring.
3. Add a footnote: "Triple recall is not computed for Type-III datasets
   (Inpatient, THE Ranking) because their hierarchical graph structures
   encode values as edge attributes or through intermediary ICD-code nodes,
   which are incompatible with flat (subject, relation, object) triple matching."
4. The zero recall for both SGE and Baseline on Type-III is symmetric —
   it does not unfairly disadvantage either system.
