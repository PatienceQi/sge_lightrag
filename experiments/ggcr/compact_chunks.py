#!/usr/bin/env python3
"""
compact_chunks.py — Compact chunk generation + vector index for GGCR evaluation.

Generates one compact text chunk per entity:
  - Type-II (WHO/WB_CM): "Entity: CHN | Name: China | ...\nTimeseries: 2000=70.83; 2005=72.44; ..."
  - Type-III (Inpatient): "Disease: 肺炎 (J12-J18)\nTotal: 60614 | Deaths: 11334 | HA Hospital: 55487"

Provides two retrieval modes:
  1. vector_retrieve(query, top_k) — cosine similarity on embeddings
  2. entity_retrieve(entity_names) — exact match by entity name
"""

from __future__ import annotations

import json
import hashlib
import numpy as np
import urllib3
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "ggcr_cache"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "mxbai-embed-large"


# ---------------------------------------------------------------------------
# Embedding — uses urllib3 directly to bypass macOS system proxy (Clash)
# ---------------------------------------------------------------------------

_http = urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)


def _embed_text_sync(text: str, max_retries: int = 3) -> np.ndarray:
    """Embed a single text using Ollama mxbai-embed-large, with retry."""
    import time
    # Truncate to fit mxbai-embed-large context window (~512 tokens)
    # Naive RAG chunks with repeated numeric patterns tokenize densely; 1000 chars is safe
    if len(text) > 1000:
        text = text[:1000]
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    for attempt in range(max_retries):
        try:
            resp = _http.urlopen(
                "POST", "/api/embeddings",
                body=body,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
            if resp.status != 200:
                raise RuntimeError(f"Ollama {resp.status}: {resp.data[:200]}")
            return np.array(json.loads(resp.data)["embedding"], dtype=np.float32)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise


def _embed_batch_sync(texts: list[str], batch_size: int = 4) -> np.ndarray:
    """Embed a batch of texts sequentially with pacing to avoid Ollama overload."""
    import time
    embeddings = []
    for i, t in enumerate(texts):
        embeddings.append(_embed_text_sync(t))
        if (i + 1) % 20 == 0 or (i + 1) == len(texts):
            print(f"    Embedded {i + 1}/{len(texts)} chunks")
        # Small delay every 10 requests to let Ollama breathe
        if (i + 1) % 10 == 0:
            time.sleep(0.3)
    return np.stack(embeddings)


def cosine_similarity(query_emb: np.ndarray, chunk_embs: np.ndarray) -> np.ndarray:
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    return chunk_norms @ query_norm


# ---------------------------------------------------------------------------
# Gold data loaders (same as benchmark_generator.py)
# ---------------------------------------------------------------------------

def _load_type_ii_gold(gold_path: str) -> tuple[dict, dict]:
    """Returns ({code: {year: val}}, {code: country_name})."""
    data, names = {}, {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["triple"]
            code = t["subject"]
            year = t["attributes"]["year"]
            value = float(t["object"])
            name = t["attributes"].get("country_name", code)
            if code not in data:
                data[code] = {}
                names[code] = name
            data[code][year] = value
    return data, names


def _load_type_iii_gold(gold_path: str) -> tuple[dict, dict]:
    """Returns ({entity: {relation: val}}, {entity: icd_code})."""
    data, icd_codes = {}, {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["triple"]
            entity = t["subject"]
            relation = t["relation"]
            value = float(t["object"])
            icd = t["attributes"].get("icd_code", "")
            if entity not in data:
                data[entity] = {}
                icd_codes[entity] = icd
            data[entity][relation] = value
    return data, icd_codes


# ---------------------------------------------------------------------------
# Compact chunk generation
# ---------------------------------------------------------------------------

def generate_type_ii_chunks(
    gold_path: str, metric_name: str, unit: str
) -> tuple[list[str], dict[str, int]]:
    """
    Generate compact chunks for Type-II dataset.
    Returns (chunks, entity_map) where entity_map maps entity code → chunk index.
    """
    data, names = _load_type_ii_gold(gold_path)
    chunks = []
    entity_map = {}

    for code in sorted(data.keys()):
        name = names[code]
        years_sorted = sorted(data[code].keys())
        timeseries = "; ".join(f"{y}={data[code][y]}" for y in years_sorted)

        chunk = (
            f"Entity: {code} | Name: {name} | Metric: {metric_name}\n"
            f"Timeseries ({unit}): {timeseries}"
        )
        entity_map[code] = len(chunks)
        # Also map by country name (case-insensitive)
        entity_map[name.lower()] = len(chunks)
        chunks.append(chunk)

    return chunks, entity_map


def generate_type_iii_chunks(
    gold_path: str,
) -> tuple[list[str], dict[str, int]]:
    """
    Generate compact chunks for Type-III dataset (Inpatient).
    Returns (chunks, entity_map) where entity_map maps entity name → chunk index.
    """
    data, icd_codes = _load_type_iii_gold(gold_path)
    chunks = []
    entity_map = {}

    rel_labels = {
        "INPATIENT_TOTAL": "住院病人出院及死亡总人次",
        "REGISTERED_DEATHS": "在医院登记死亡人数",
        "INPATIENT_HA_HOSPITAL": "医院管理局辖下医院住院人次",
    }

    for entity in sorted(data.keys()):
        icd = icd_codes.get(entity, "")
        lines = [f"Disease: {entity} ({icd})"]
        for rel in ["INPATIENT_TOTAL", "INPATIENT_HA_HOSPITAL", "REGISTERED_DEATHS"]:
            if rel in data[entity]:
                label = rel_labels.get(rel, rel)
                val = int(data[entity][rel])
                lines.append(f"  {label}: {val}")
        chunk = "\n".join(lines)
        entity_map[entity] = len(chunks)
        entity_map[entity.lower()] = len(chunks)
        chunks.append(chunk)

    return chunks, entity_map


# ---------------------------------------------------------------------------
# CompactIndex
# ---------------------------------------------------------------------------

class CompactIndex:
    """Per-dataset compact chunk index with vector search + entity lookup."""

    def __init__(self, dataset: str, chunks: list[str], entity_map: dict[str, int]):
        self.dataset = dataset
        self.chunks = chunks
        self.entity_map = entity_map  # entity_name_or_code → chunk_index
        self.embeddings: np.ndarray | None = None

    def build_embeddings(self, force: bool = False):
        """Build or load cached embeddings."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Cache key based on chunk content hash
        content_hash = hashlib.md5(
            "\n".join(self.chunks).encode()
        ).hexdigest()[:12]
        cache_path = CACHE_DIR / f"compact_emb_{self.dataset}_{content_hash}.npy"

        if cache_path.exists() and not force:
            print(f"  Loading cached embeddings: {cache_path.name}")
            self.embeddings = np.load(cache_path)
            return

        print(f"  Embedding {len(self.chunks)} compact chunks for {self.dataset}...")
        self.embeddings = _embed_batch_sync(self.chunks)
        np.save(cache_path, self.embeddings)
        print(f"  Cached to {cache_path.name}")

    def vector_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Cosine similarity retrieval."""
        if self.embeddings is None:
            raise RuntimeError("Call build_embeddings() first")
        query_emb = _embed_text_sync(query)
        scores = cosine_similarity(query_emb, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def entity_retrieve(self, entity_keys: list[str]) -> list[str]:
        """Direct lookup by entity name/code. Returns matched chunks."""
        result = []
        seen_indices = set()
        for key in entity_keys:
            idx = self.entity_map.get(key) or self.entity_map.get(key.lower())
            if idx is not None and idx not in seen_indices:
                result.append(self.chunks[idx])
                seen_indices.add(idx)
        return result

    def get_all_chunks(self) -> list[str]:
        """Return ALL chunks (for Concat-All baseline)."""
        return list(self.chunks)

    def get_all_entity_keys(self) -> list[str]:
        """Return all unique entity keys (codes/names)."""
        seen = set()
        keys = []
        for key, idx in self.entity_map.items():
            if idx not in seen:
                seen.add(idx)
                keys.append(key)
        return keys


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

GOLD_DIR = PROJECT_ROOT / "evaluation" / "gold"

DATASET_CONFIGS = {
    "who": {
        "gold_path": GOLD_DIR / "gold_who_life_expectancy_v2.jsonl",
        "type": "II",
        "metric_name": "Life expectancy at birth",
        "unit": "years",
    },
    "wb_cm": {
        "gold_path": GOLD_DIR / "gold_wb_child_mortality_v2.jsonl",
        "type": "II",
        "metric_name": "Under-5 mortality rate",
        "unit": "per 1,000 live births",
    },
    "inpatient": {
        "gold_path": GOLD_DIR / "gold_inpatient_2023.jsonl",
        "type": "III",
    },
    "who_full": {
        "gold_path": GOLD_DIR / "gold_who_life_expectancy_full.jsonl",
        "type": "II",
        "metric_name": "Life expectancy at birth",
        "unit": "years",
    },
}


def build_compact_index(dataset: str, embed: bool = True) -> CompactIndex:
    """Build a CompactIndex for a dataset."""
    cfg = DATASET_CONFIGS[dataset]

    if cfg["type"] == "II":
        chunks, entity_map = generate_type_ii_chunks(
            str(cfg["gold_path"]), cfg["metric_name"], cfg["unit"]
        )
    else:
        chunks, entity_map = generate_type_iii_chunks(str(cfg["gold_path"]))

    index = CompactIndex(dataset, chunks, entity_map)
    if embed:
        index.build_embeddings()
    return index


# ---------------------------------------------------------------------------
# CLI: preview compact chunks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["who", "wb_cm", "inpatient"]

    for ds in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")
        index = build_compact_index(ds, embed=False)
        print(f"  Chunks: {len(index.chunks)}")
        print(f"  Entity map entries: {len(index.entity_map)}")
        print(f"\n  Sample chunk:")
        print(f"  {index.chunks[0]}")
        print()
