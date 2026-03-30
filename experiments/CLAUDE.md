# Experiments

Paper experiment scripts grouped by type. All scripts use `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` to locate the project root.

## Subdirectories

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| `ablation/` | Ablation experiments (decoupled, C4, threshold, misclassify) | 4 scripts |
| `statistical/` | Statistical tests (Wilcoxon, McNemar, LODO, Bootstrap) | 5 scripts |
| `probes/` | Downstream probes (graph-native, E2E, compact) | 9 scripts |
| `crossmodel/` | Cross-model validation (GPT-5-mini) | 1 script |
| `results/` | Experiment output JSONs (authoritative) | JSON files |

## Running Experiments

```bash
# From project root (sge_lightrag/)
python3 experiments/ablation/run_decoupled_ablation.py
python3 experiments/statistical/wilcoxon_effect_sizes.py
python3 experiments/probes/graph_native_probe.py
```

## Results

All results save to `experiments/results/*.json`. These are referenced by the paper but are NOT authoritative evaluation numbers — those live in `evaluation/results/`.
