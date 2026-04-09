#!/usr/bin/env python3
"""
run_external_eval_v2.py — External validation evaluation with improved entity matching.

Improvements over v1:
  1. Country name → ISO code mapping (e.g., "Japan" → "JPN" for WHO)
  2. Stop word filtering (removes question words as entities)
  3. Fuzzy matching fallback (SequenceMatcher ≥ 0.85)
  4. Combined v1 + v2 benchmarks

Usage:
    python3 experiments/graph_first/run_external_eval_v2.py
    python3 experiments/graph_first/run_external_eval_v2.py --benchmark v2-only
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.graph_first.graph_context import (
    _get_node_name, _bfs_2hop, _format_context, _estimate_tokens,
)
from evaluation.config import API_KEY, BASE_URL, MODEL

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# ---------------------------------------------------------------------------
# Country name ↔ ISO code mapping (for WHO which uses ISO codes)
# ---------------------------------------------------------------------------

COUNTRY_TO_ISO = {
    "afghanistan": "AFG", "albania": "ALB", "argentina": "ARG", "australia": "AUS",
    "bangladesh": "BGD", "brazil": "BRA", "china": "CHN", "colombia": "COL",
    "germany": "DEU", "egypt": "EGY", "ethiopia": "ETH", "france": "FRA",
    "united kingdom": "GBR", "uk": "GBR", "indonesia": "IDN", "india": "IND",
    "iran": "IRN", "japan": "JPN", "kenya": "KEN", "korea": "KOR",
    "korea, rep.": "KOR", "south korea": "KOR", "mexico": "MEX",
    "nigeria": "NGA", "pakistan": "PAK", "philippines": "PHL",
    "russia": "RUS", "russian federation": "RUS",
    "united states": "USA", "usa": "USA", "us": "USA",
    "thailand": "THA", "turkey": "TUR", "viet nam": "VNM", "vietnam": "VNM",
    "south africa": "ZAF",
    # Chinese names
    "日本": "JPN", "中国": "CHN", "印度": "IND", "巴西": "BRA",
    "美国": "USA", "德国": "DEU", "法国": "FRA", "英国": "GBR",
    "俄罗斯": "RUS", "韩国": "KOR", "墨西哥": "MEX", "埃及": "EGY",
    "阿根廷": "ARG", "澳大利亚": "AUS", "瑞士": "CHE", "挪威": "NOR",
    "瑞典": "SWE", "尼日利亚": "NGA", "肯尼亚": "KEN", "孟加拉国": "BGD",
    "巴基斯坦": "PAK", "菲律宾": "PHL", "泰国": "THA", "越南": "VNM",
    "南非": "ZAF", "土耳其": "TUR", "哥伦比亚": "COL", "伊朗": "IRN",
    "印度尼西亚": "IDN", "埃塞俄比亚": "ETH",
}

# Stop words to filter from entity extraction
STOP_WORDS = {
    "what", "which", "how", "who", "when", "where", "why", "does", "did", "is",
    "are", "was", "were", "the", "a", "an", "in", "of", "for", "to", "and",
    "between", "from", "many", "much",
    "哪个", "哪些", "什么", "多少", "怎么", "如何", "是否", "有多少",
    "在", "的", "了", "和", "与", "从", "到", "中", "上", "下",
}

# ---------------------------------------------------------------------------
# Graph paths per dataset
# ---------------------------------------------------------------------------

GRAPH_PATHS = {
    "who": {
        "sge": PROJECT_ROOT / "output" / "who_life_expectancy" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_who_life" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "wb_cm": {
        "sge": PROJECT_ROOT / "output" / "wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_wb_child_mortality" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "wb_pop": {
        "sge": PROJECT_ROOT / "output" / "wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_wb_population" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "wb_mat": {
        "sge": PROJECT_ROOT / "output" / "wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_wb_maternal" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "inpatient": {
        "sge": PROJECT_ROOT / "output" / "inpatient_2023" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_inpatient23" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "fortune500": {
        "sge": PROJECT_ROOT / "output" / "fortune500_revenue" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_fortune500_revenue" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
    "the": {
        "sge": PROJECT_ROOT / "output" / "the_university_ranking" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
        "baseline": PROJECT_ROOT / "output" / "baseline_the_university_ranking" / "lightrag_storage" / "graph_chunk_entity_relation.graphml",
    },
}


# ---------------------------------------------------------------------------
# Improved entity extraction from question text
# ---------------------------------------------------------------------------

def extract_entities_from_question(question: str, dataset: str) -> list[str]:
    """Extract entity names from question text with improved parsing."""
    entities = []

    # Pattern 1: Quoted entities
    quoted = re.findall(r'[""\'](.*?)[""\'"]', question)
    entities.extend(quoted)

    # Pattern 2: ISO country codes (3 uppercase letters)
    iso_codes = re.findall(r'\b([A-Z]{3})\b', question)
    entities.extend(iso_codes)

    # Pattern 3: Known country names (case-insensitive)
    # WHO and WB CM use ISO codes; WB Pop and WB Mat use full country names
    ISO_DATASETS = {"who", "wb_cm"}
    q_lower = question.lower()
    for name, code in COUNTRY_TO_ISO.items():
        if name in q_lower:
            if dataset in ISO_DATASETS:
                entities.append(code)
            elif dataset in ("wb_pop", "wb_mat"):
                # These use full country names — find proper casing
                # Special cases
                name_map = {
                    "korea, rep.": "Korea, Rep.", "south korea": "Korea, Rep.",
                    "united states": "United States", "usa": "United States",
                    "us": "United States", "united kingdom": "United Kingdom",
                    "uk": "United Kingdom", "south africa": "South Africa",
                    "viet nam": "Viet Nam", "vietnam": "Viet Nam",
                    "russian federation": "Russian Federation", "russia": "Russian Federation",
                }
                proper = name_map.get(name, name.title())
                entities.append(proper)
            else:
                entities.append(name.title() if len(name) > 3 else name)

    # Pattern 4: Chinese disease names (for inpatient)
    if dataset == "inpatient":
        disease_names = ["肺炎", "肾衰竭", "糖尿病", "恶性肿瘤", "缺血性心脏病",
                         "脑血管疾病", "霍乱", "结核病", "肝炎", "败血症",
                         "流行性感冒", "伤寒", "痢疾", "白血病"]
        for d in disease_names:
            if d in question:
                entities.append(d)

    # Pattern 5: Company names (for fortune500)
    if dataset == "fortune500":
        companies = ["Walmart", "Amazon", "Apple", "CVS Health", "UnitedHealth",
                     "ExxonMobil", "Berkshire Hathaway", "Alphabet", "McKesson",
                     "AmerisourceBergen", "Costco", "Cigna", "Microsoft"]
        for c in companies:
            if c.lower() in q_lower:
                entities.append(c)

    # Pattern 6: University names (for THE)
    if dataset == "the":
        universities = ["Oxford", "Stanford", "MIT", "Harvard", "Cambridge",
                        "Caltech", "Princeton", "Chicago", "Imperial", "ETH Zurich",
                        "Yale", "Columbia", "Johns Hopkins", "UCL", "Berkeley",
                        "Peking", "Tsinghua", "Tokyo", "NUS"]
        for u in universities:
            if u.lower() in q_lower:
                entities.append(u)

    # Filter stop words and short tokens
    filtered = []
    for e in entities:
        e_clean = e.strip()
        if len(e_clean) < 2:
            continue
        if e_clean.lower() in STOP_WORDS:
            continue
        if e_clean not in filtered:
            filtered.append(e_clean)

    return filtered


# ---------------------------------------------------------------------------
# Improved entity matching with fuzzy fallback
# ---------------------------------------------------------------------------

def match_entities_improved(
    G: nx.Graph, entity_queries: list[str], dataset: str
) -> list[str]:
    """Match entities with substring + fuzzy matching + name-code mapping."""
    id_to_name = {
        node_id: _get_node_name(G, node_id)
        for node_id in G.nodes()
    }

    matched = []
    for query in entity_queries:
        q_lower = query.lower().strip()
        if not q_lower or len(q_lower) < 2:
            continue

        # Try exact substring match first
        for node_id, name in id_to_name.items():
            name_lower = name.lower()
            if q_lower in name_lower or name_lower in q_lower:
                if len(q_lower) >= 3 or q_lower == name_lower:
                    if node_id not in matched:
                        matched.append(node_id)

        # If no match, try country name → ISO code mapping (WHO/WB_CM use ISO codes)
        if not matched and dataset in ("who", "wb_cm"):
            iso = COUNTRY_TO_ISO.get(q_lower)
            if iso:
                for node_id, name in id_to_name.items():
                    if iso.lower() == name.lower().strip():
                        if node_id not in matched:
                            matched.append(node_id)

        # Fuzzy matching fallback (SequenceMatcher ≥ 0.85)
        if not matched:
            best_score = 0.0
            best_node = None
            for node_id, name in id_to_name.items():
                name_lower = name.lower()
                # Only compare with short node names (entity nodes, not value nodes)
                if len(name_lower) > 50:
                    continue
                score = SequenceMatcher(None, q_lower, name_lower).ratio()
                if score > best_score and score >= 0.85:
                    best_score = score
                    best_node = node_id
            if best_node and best_node not in matched:
                matched.append(best_node)

    return matched


# ---------------------------------------------------------------------------
# Build context from graph
# ---------------------------------------------------------------------------

def build_context(graph_path: Path, entities: list[str], dataset: str,
                  max_tokens: int = 3000) -> tuple[str, list[str]]:
    """Build graph context with improved matching. Returns (context_text, matched_nodes)."""
    if not graph_path.exists():
        return "", []

    G = nx.read_graphml(str(graph_path))
    matched = match_entities_improved(G, entities, dataset)

    if not matched:
        return "", []

    hop_map = _bfs_2hop(G, matched)
    context = _format_context(G, matched, hop_map, max_tokens)
    return context, matched


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

_SEM = asyncio.Semaphore(5)

async def ask_llm(question: str, context: str) -> str:
    """Generate answer using graph context."""
    import httpx
    system = "You are a statistical data analyst. Answer the question using ONLY the provided graph context. If the context doesn't contain enough information, say 'Insufficient context'."
    user = f"Graph context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"

    async with _SEM:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": MODEL, "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def is_informative(answer: str) -> bool:
    """Check if answer is informative (not a refusal or error)."""
    a = answer.lower().strip()
    refusal_markers = [
        "insufficient context", "no information", "not found",
        "cannot determine", "not available", "i don't have",
        "[error", "no data", "无法", "没有找到", "不足",
    ]
    return not any(m in a for m in refusal_markers)


async def evaluate_question(q: dict) -> dict:
    """Evaluate one question on both SGE and Baseline graphs."""
    dataset = q["dataset"]
    question = q["question"]
    qid = q["id"]

    if dataset not in GRAPH_PATHS:
        return {"id": qid, "dataset": dataset, "status": "unknown_dataset"}

    entities = extract_entities_from_question(question, dataset)

    if not entities:
        return {
            "id": qid, "dataset": dataset, "question": question,
            "entities_extracted": [],
            "status": "no_entities_extracted",
        }

    paths = GRAPH_PATHS[dataset]

    # SGE
    sge_ctx, sge_matched = build_context(paths["sge"], entities, dataset)
    sge_answer = ""
    if sge_ctx:
        sge_answer = await ask_llm(question, sge_ctx)

    # Baseline
    base_ctx, base_matched = build_context(paths["baseline"], entities, dataset)
    base_answer = ""
    if base_ctx:
        base_answer = await ask_llm(question, base_ctx)

    sge_has_ctx = bool(sge_ctx)
    base_has_ctx = bool(base_ctx)
    has_any_ctx = sge_has_ctx or base_has_ctx

    sge_info = is_informative(sge_answer) if sge_answer else False
    base_info = is_informative(base_answer) if base_answer else False

    return {
        "id": qid,
        "dataset": dataset,
        "question": question,
        "query_type": q.get("query_type"),
        "entities_extracted": entities,
        "sge_matched_nodes": len(sge_matched),
        "base_matched_nodes": len(base_matched),
        "sge_has_context": sge_has_ctx,
        "base_has_context": base_has_ctx,
        "has_any_context": has_any_ctx,
        "sge_answer": sge_answer,
        "base_answer": base_answer,
        "sge_informative": sge_info,
        "base_informative": base_info,
        "sge_wins": sge_info and not base_info,
        "base_wins": base_info and not sge_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="combined",
                        choices=["v1", "v2-only", "combined"],
                        help="Which benchmark to use")
    args = parser.parse_args()

    # Load questions
    questions = []
    v1_path = PROJECT_ROOT / "experiments" / "graph_first" / "benchmark_external.jsonl"
    v2_path = PROJECT_ROOT / "experiments" / "graph_first" / "benchmark_external_v2.jsonl"

    if args.benchmark in ("v1", "combined") and v1_path.exists():
        with open(v1_path) as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"Loaded {len(questions)} v1 questions")

    if args.benchmark in ("v2-only", "combined") and v2_path.exists():
        with open(v2_path) as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"Total after v2: {len(questions)} questions")

    if not questions:
        print("No questions found!")
        return

    print(f"\nEvaluating {len(questions)} questions...", flush=True)

    # Run evaluation
    results = []
    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0 or i == len(questions) - 1:
            print(f"  [{i+1}/{len(questions)}] {q['dataset']} | {q.get('query_type', '?')}", flush=True)
        result = await evaluate_question(q)
        results.append(result)
        # Print intermediate stats every 20
        if (i + 1) % 20 == 0:
            ctx_so_far = sum(1 for r in results if r.get("has_any_context"))
            print(f"    -> {ctx_so_far}/{i+1} with context", flush=True)

    # Aggregate
    with_ctx = [r for r in results if r.get("has_any_context")]
    no_ctx = [r for r in results if not r.get("has_any_context")]
    sge_info = sum(1 for r in with_ctx if r.get("sge_informative"))
    base_info = sum(1 for r in with_ctx if r.get("base_informative"))
    sge_wins = sum(1 for r in with_ctx if r.get("sge_wins"))
    base_wins = sum(1 for r in with_ctx if r.get("base_wins"))

    # Per-dataset breakdown
    by_dataset = defaultdict(lambda: {"n": 0, "ctx": 0, "sge_info": 0, "base_info": 0, "sge_wins": 0})
    for r in results:
        ds = r["dataset"]
        by_dataset[ds]["n"] += 1
        if r.get("has_any_context"):
            by_dataset[ds]["ctx"] += 1
            if r.get("sge_informative"):
                by_dataset[ds]["sge_info"] += 1
            if r.get("base_informative"):
                by_dataset[ds]["base_info"] += 1
            if r.get("sge_wins"):
                by_dataset[ds]["sge_wins"] += 1

    # Print summary
    print(f"\n{'='*60}")
    print("EXTERNAL VALIDATION v2 SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(questions)}")
    print(f"With graph context: {len(with_ctx)} ({len(with_ctx)/len(questions)*100:.0f}%)")
    print(f"No context: {len(no_ctx)}")
    print(f"\nAmong {len(with_ctx)} questions with context:")
    print(f"  SGE informative: {sge_info} ({sge_info/len(with_ctx)*100:.1f}%)" if with_ctx else "")
    print(f"  Baseline informative: {base_info} ({base_info/len(with_ctx)*100:.1f}%)" if with_ctx else "")
    print(f"  SGE wins: {sge_wins}, Baseline wins: {base_wins}")
    print(f"\nPer dataset:")
    for ds in sorted(by_dataset):
        d = by_dataset[ds]
        print(f"  {ds:12s}: {d['ctx']}/{d['n']} ctx | SGE {d['sge_info']} | Base {d['base_info']} | SGE-wins {d['sge_wins']}")
    print(f"{'='*60}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "External validation v2 with improved entity matching",
        "total_questions": len(questions),
        "questions_with_context": len(with_ctx),
        "match_rate": round(len(with_ctx) / len(questions), 3),
        "sge_informative": sge_info,
        "baseline_informative": base_info,
        "sge_informative_rate": round(sge_info / len(with_ctx), 3) if with_ctx else 0,
        "baseline_informative_rate": round(base_info / len(with_ctx), 3) if with_ctx else 0,
        "sge_wins": sge_wins,
        "baseline_wins": base_wins,
        "per_dataset": {ds: dict(d) for ds, d in sorted(by_dataset.items())},
        "results": results,
    }

    out_path = RESULTS_DIR / "external_validation_v2_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
