# Scripts

Utility and runner scripts. Not part of the core pipeline — these orchestrate pipeline execution.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `runners/` | Pipeline runner scripts (batch, integration, OOD, individual stages) |
| `batch/` | Shell batch scripts for multi-dataset runs |
| `_archive/` | One-time data fixes and deprecated scripts |

## Usage

All runners should be called from the project root:

```bash
python3 scripts/runners/run_lightrag_integration.py data/sample.csv
```
