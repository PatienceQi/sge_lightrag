#!/usr/bin/env python3
"""
Step 3: Run multi-indicator evaluation (3 systems × 20 questions).

Reads prepared data + embeddings from Steps 1-2, runs LLM queries for
each (system, question) pair, scores answers, and saves results.

Can run a single system or all systems. Systems are independent and
can be run in parallel across terminal sessions.

Usage:
    python3 experiments/ggcr/multi_indicator_run.py                    # all systems
    python3 experiments/ggcr/multi_indicator_run.py --system sge_ggcr  # one system
    python3 experiments/ggcr/multi_indicator_run.py --system concat_all
    python3 experiments/ggcr/multi_indicator_run.py --system pure_compact
    python3 experiments/ggcr/multi_indicator_run.py --verbose
"""
from __future__ import annotations

import os
import re
import json
import time
import argparse
import hashlib
import threading
import numpy as np
import urllib3
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "ggcr_cache"
DATA_PATH = CACHE_DIR / "multi_indicator_data.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ggcr_results_multi_indicator.json"

# LLM config (same as run_ggcr_eval.py)
API_KEY = os.environ.get("SGE_API_KEY", "")
BASE_URL = os.environ.get("SGE_API_BASE", "https://api.openai.com/v1")
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a precise QA assistant for statistical data. Answer based ONLY on "
    "the provided data context. When the question asks for a ranking, list country "
    "codes (3-letter ISO) in order separated by commas. When it asks for a numeric "
    "value (average, total, count), compute precisely from the data and report the "
    "number. Be concise. If the data is insufficient, say so."
)

EMBED_MODEL = "mxbai-embed-large"
MAX_CONCAT_TOKENS = 180_000

ALL_SYSTEMS = ["sge_ggcr", "concat_all", "pure_compact"]


# ---------------------------------------------------------------------------
# Data container (loaded from Step 1 JSON)
# ---------------------------------------------------------------------------

class MultiIndicatorData:
    def __init__(self, data_path: str):
        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks: list[str] = raw["chunks"]
        self.entity_map: dict[str, int] = raw["entity_map"]
        self.benchmark: list[dict] = raw["benchmark"]
        self.n_entities: int = raw["n_entities"]
        self.n_indicators: int = raw["n_indicators"]
        self.total_tokens: int = raw["total_tokens_estimate"]
        self.embeddings: np.ndarray | None = None

    def load_embeddings(self):
        """Load cached embeddings from Step 2."""
        n = len(self.chunks)
        content_hash = hashlib.md5("\n".join(self.chunks[:20]).encode()).hexdigest()[:12]
        cache_path = CACHE_DIR / f"multi_indicator_emb_{n}_{content_hash}.npy"
        if cache_path.exists():
            self.embeddings = np.load(cache_path)
            print(f"  Embeddings loaded: {cache_path.name}")
        else:
            print(f"  WARNING: Embeddings not found at {cache_path.name}")
            print(f"  Run multi_indicator_embed.py first for Pure Compact system.")

    def retrieve_by_indicators(self, indicator_names: list[str]) -> list[str]:
        """GGCR path: retrieve ALL chunks for given indicators."""
        result = []
        seen = set()
        for indicator in indicator_names:
            suffix = f":{indicator}".lower()
            for key, idx in self.entity_map.items():
                if key.lower().endswith(suffix) and idx not in seen:
                    result.append(self.chunks[idx])
                    seen.add(idx)
        return result

    def vector_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Pure Compact path: cosine similarity."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded")
        query_emb = _embed_text_sync(query)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        c_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        scores = c_norms @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Embedding helper (for vector_retrieve query)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _call_llm(context: str, question: str, max_retries: int = 3) -> str:
    _llm_http = urllib3.PoolManager()
    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Data context:\n{context}\n\nQuestion: {question}"},
                ],
                "max_tokens": 512,
                "temperature": 0,
            }).encode("utf-8")
            resp = _llm_http.request(
                "POST", f"{BASE_URL}/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=urllib3.Timeout(connect=10.0, read=60.0),
            )
            if resp.status != 200:
                raise RuntimeError(f"API {resp.status}: {resp.data[:200]}")
            return json.loads(resp.data)["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return f"[LLM Error: {e}]"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize_number(s: str) -> str:
    return re.sub(r"\.0+\b", "", s.replace(",", "").replace("，", ""))


def score_answer(answer: str, question: dict) -> bool:
    scoring = question["scoring"]
    ref = question["reference_answer"]

    if scoring == "set_overlap":
        ref_set = ref if isinstance(ref, list) else [ref]
        matched = sum(1 for code in ref_set if code.upper() in answer.upper())
        return matched >= max(1, int(len(ref_set) * 0.6))

    elif scoring == "numeric_tolerance":
        ref_num = question.get("reference_numeric")
        if ref_num is None:
            return _normalize_number(str(ref)) in _normalize_number(answer)
        numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        for num_str in numbers:
            try:
                predicted = float(num_str)
                if ref_num == 0:
                    if abs(predicted) < 0.1:
                        return True
                elif abs(predicted - ref_num) / abs(ref_num) < 0.05:
                    return True
            except ValueError:
                continue
        return _normalize_number(str(ref)) in _normalize_number(answer)

    return False


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------

def run_sge_ggcr(q: dict, data: MultiIndicatorData) -> tuple[str, int, bool]:
    indicators = q.get("required_indicators", [])
    chunks = data.retrieve_by_indicators(indicators) if indicators else data.chunks
    context = "\n---\n".join(chunks)
    return _call_llm(context, q["question"]), len(context) // 4, False


def run_concat_all(q: dict, data: MultiIndicatorData) -> tuple[str, int, bool]:
    context = "\n---\n".join(data.chunks)
    tokens = len(context) // 4
    truncated = False
    if tokens > MAX_CONCAT_TOKENS:
        context = context[:MAX_CONCAT_TOKENS * 4]
        truncated = True
    return _call_llm(context, q["question"]), tokens, truncated


def run_pure_compact(q: dict, data: MultiIndicatorData) -> tuple[str, int, bool]:
    chunks = data.vector_retrieve(q["question"], top_k=5)
    context = "\n---\n".join(chunks)
    return _call_llm(context, q["question"]), len(context) // 4, False


SYSTEM_RUNNERS = {
    "sge_ggcr": run_sge_ggcr,
    "concat_all": run_concat_all,
    "pure_compact": run_pure_compact,
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(systems: list[str], verbose: bool = False):
    print("=" * 60)
    print("Step 3: Multi-Indicator Evaluation")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run multi_indicator_prepare.py first.")
        return

    data = MultiIndicatorData(str(DATA_PATH))
    print(f"\nLoaded: {data.n_entities} entities, {len(data.chunks)} chunks, {data.total_tokens:,} tokens")
    print(f"Benchmark: {len(data.benchmark)} questions")

    need_embed = "pure_compact" in systems
    if need_embed:
        data.load_embeddings()
        if data.embeddings is None:
            print("Skipping pure_compact (no embeddings).")
            systems = [s for s in systems if s != "pure_compact"]

    questions = data.benchmark
    total_tasks = len(questions) * len(systems)
    print(f"\nRunning {total_tasks} evaluations ({systems})...")

    lock = threading.Lock()
    results_by_system: dict[str, dict[str, list[bool]]] = {
        s: {"L3": [], "L4": []} for s in systems
    }
    details: list[dict] = []
    done = [0]

    def _run_one(q: dict, system: str):
        runner = SYSTEM_RUNNERS[system]
        try:
            answer, ctx_tokens, truncated = runner(q, data)
        except Exception as e:
            answer, ctx_tokens, truncated = f"[Error: {e}]", 0, False
        correct = score_answer(answer, q)
        return q, system, answer, ctx_tokens, truncated, correct

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(_run_one, q, s)
            for q in questions
            for s in systems
        ]
        for future in as_completed(futures):
            q, system, answer, ctx_tokens, truncated, correct = future.result()
            with lock:
                results_by_system[system][q["level"]].append(correct)
                done[0] += 1
                entry = {"answer": answer[:500], "correct": correct, "context_tokens": ctx_tokens}
                if truncated:
                    entry["truncated"] = True
                existing = next((d for d in details if d["id"] == q["id"]), None)
                if existing is None:
                    details.append({
                        "id": q["id"], "level": q["level"],
                        "question": q["question"],
                        "reference_answer": q["reference_answer"],
                        "scoring": q["scoring"],
                        "systems": {system: entry},
                    })
                else:
                    existing["systems"][system] = entry
                if done[0] % 5 == 0:
                    print(f"  Progress: {done[0]}/{total_tasks}")
            if verbose:
                mark = "+" if correct else "-"
                print(f"  [{mark}] {q['id']} [{system}]")

    details.sort(key=lambda d: d["id"])

    # Summary
    print("\n" + "=" * 60)
    print(f"{'System':<16} {'L3':>8} {'L4':>8} {'Overall':>10}")
    print("-" * 48)

    summary = {}
    for system in systems:
        l3 = results_by_system[system]["L3"]
        l4 = results_by_system[system]["L4"]
        all_r = l3 + l4
        s = {}
        if l3:
            s["L3"] = {"correct": sum(l3), "total": len(l3)}
        if l4:
            s["L4"] = {"correct": sum(l4), "total": len(l4)}
        if all_r:
            s["overall"] = {"correct": sum(all_r), "total": len(all_r), "accuracy": round(sum(all_r)/len(all_r), 4)}
        summary[system] = s
        l3s = f"{sum(l3)}/{len(l3)}" if l3 else "—"
        l4s = f"{sum(l4)}/{len(l4)}" if l4 else "—"
        ovs = f"{s['overall']['accuracy']:.0%}" if "overall" in s else "—"
        print(f"{system:<16} {l3s:>8} {l4s:>8} {ovs:>10}")

    # Save (merge with existing results if present)
    output = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            output = json.load(f)

    output.update({
        "timestamp": datetime.now().isoformat(),
        "experiment": "multi_indicator_1200_entities",
        "n_entities": data.n_entities,
        "n_indicators": data.n_indicators,
        "n_questions": len(questions),
        "total_chunks_tokens_estimate": data.total_tokens,
    })
    if "systems" not in output:
        output["systems"] = {}
    output["systems"].update(summary)
    output["details"] = details

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Multi-indicator evaluation runner")
    parser.add_argument("--system", choices=ALL_SYSTEMS, help="Run a single system (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    systems = [args.system] if args.system else ALL_SYSTEMS
    run_evaluation(systems, verbose=args.verbose)


if __name__ == "__main__":
    main()
