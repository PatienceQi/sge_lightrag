#!/usr/bin/env python3
"""
Step 2: Build embeddings for multi-indicator Pure Compact system.

Reads chunks from Step 1 output, embeds via Ollama mxbai-embed-large,
and caches to .npy file. Requires Ollama running locally.

Usage:
    python3 experiments/ggcr/multi_indicator_embed.py

Reads:  output/ggcr_cache/multi_indicator_data.json
Writes: output/ggcr_cache/multi_indicator_emb_*.npy
"""
from __future__ import annotations

import json
import time
import hashlib
import numpy as np
import urllib3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "ggcr_cache"
DATA_PATH = CACHE_DIR / "multi_indicator_data.json"

EMBED_MODEL = "mxbai-embed-large"
_http = urllib3.HTTPConnectionPool("127.0.0.1", port=11434, maxsize=4)


def _embed_text_sync(text: str, max_retries: int = 3) -> np.ndarray:
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


def main():
    print("=" * 60)
    print("Step 2: Build Multi-Indicator Embeddings")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run multi_indicator_prepare.py first.")
        return

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    n = len(chunks)
    print(f"Chunks to embed: {n}")

    # Cache path
    content_hash = hashlib.md5("\n".join(chunks[:20]).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"multi_indicator_emb_{n}_{content_hash}.npy"

    if cache_path.exists():
        emb = np.load(cache_path)
        if emb.shape[0] == n:
            print(f"Embeddings already cached: {cache_path.name} ({emb.shape})")
            return
        print("Cache size mismatch — rebuilding...")

    print(f"Embedding {n} chunks (this may take a few minutes)...")
    embeddings = []
    for i, t in enumerate(chunks):
        embeddings.append(_embed_text_sync(t))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}")
        if (i + 1) % 10 == 0:
            time.sleep(0.3)

    emb = np.stack(embeddings)
    np.save(cache_path, emb)
    print(f"Saved: {cache_path.name} (shape={emb.shape})")


if __name__ == "__main__":
    main()
