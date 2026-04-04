#!/bin/bash
# Run Gemini cross-model for ONE dataset at a time to avoid OOM
# Usage: ./run_gemini_single.sh who|wb_cm|wb_pop|wb_mat|inpatient

DATASET=$1
if [ -z "$DATASET" ]; then echo "Usage: $0 <dataset>"; exit 1; fi

cd /Users/qipatience/Desktop/SGE/sge_lightrag

python3 -c "
import sys, asyncio, json, shutil
from pathlib import Path

sys.path.insert(0, '.')
DATASET = '$DATASET'

# Config
API_KEY = 'sk-wMPHAJ5PsHQpf06bD1ki2N581rx83y5qdtcDOSjZGONIoDVu'
BASE_URL = 'https://wolfai.top/v1'
MODEL = 'gemini-2.5-flash'

DATASETS = {
    'who': {'csv': 'dataset/WHO/API_WHO_WHOSIS_000001_life_expectancy.csv', 'gold': 'evaluation/gold/gold_who_life_expectancy_v2.jsonl'},
    'wb_cm': {'csv': 'dataset/世界银行数据/child_mortality/API_SH.DYN.MORT_DS2_en_csv_v2_632.csv', 'gold': 'evaluation/gold/gold_wb_child_mortality_v2.jsonl'},
    'wb_pop': {'csv': 'dataset/世界银行数据/population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv', 'gold': 'evaluation/gold/gold_wb_population_v2.jsonl'},
    'wb_mat': {'csv': 'dataset/世界银行数据/maternal_mortality/API_SH.STA.MMRT_DS2_en_csv_v2_708.csv', 'gold': 'evaluation/gold/gold_wb_maternal_v2.jsonl'},
    'inpatient': {'csv': 'dataset/住院病人统计/Inpatient Discharges and Deaths in Hospitals and Registered Deaths in Hong Kong by Disease 2023 (SC).csv', 'gold': 'evaluation/gold/gold_inpatient_2023.jsonl'},
}
import glob  # keep import for compatibility

cfg = DATASETS[DATASET]
csv_path = cfg['csv']
gold_path = cfg['gold']
output_dir = Path(f'output/crossmodel_gemini_{DATASET}')

print(f'=== Gemini {DATASET} ===')
print(f'CSV: {csv_path}')

# Stage 1-3
from stage1.features import extract_features
from stage1.classifier import classify
from stage1.schema import build_meta_schema
from stage2.inducer import induce_schema as induce_schema_rule
from stage3.serializer import serialize_csv
from stage3.prompt_injector import generate_system_prompt

features = extract_features(csv_path)
table_type = classify(features)
meta = build_meta_schema(features, table_type)
schema = induce_schema_rule(meta, features)
chunks = serialize_csv(csv_path, schema)
sys_prompt = generate_system_prompt(schema)
print(f'Type: {table_type}, Chunks: {len(chunks)}')

# Clean output
storage = output_dir / 'lightrag_storage'
import shutil as _sh
if output_dir.exists(): _sh.rmtree(str(output_dir))
if storage.exists():
    shutil.rmtree(str(storage))
storage.mkdir(parents=True, exist_ok=True)

# LightRAG setup
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS
import numpy as np
import urllib3

PROMPTS['entity_extraction_system_prompt'] = sys_prompt

async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL, prompt, system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL, **kwargs)

pool = urllib3.HTTPConnectionPool('127.0.0.1', port=11434, maxsize=1)
def embed_sync(texts):
    embeddings = []
    for t in texts:
        r = pool.request('POST', '/api/embeddings', json={'model': 'mxbai-embed-large', 'prompt': t})
        data = json.loads(r.data)
        embeddings.append(data['embedding'])
    return np.array(embeddings)

async def embed_func(texts):
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, embed_sync, texts)

rag = LightRAG(
    working_dir=str(storage),
    llm_model_func=llm_func,
    embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed_func),
    addon_params={'schema_json': json.dumps(schema, ensure_ascii=False)},
    llm_model_max_async=5,
    embedding_func_max_async=1,
    entity_extract_max_gleaning=0,
)

async def run():
    await rag.initialize_storages()
    for i, chunk in enumerate(chunks, 1):
        if i % 50 == 0 or i == len(chunks):
            print(f'  [{i}/{len(chunks)}]')
        await rag.ainsert(chunk)
    await rag.finalize_storages()

asyncio.run(run())

# Evaluate
from evaluation.evaluate_coverage import load_graph, load_gold, check_entity_coverage, check_fact_coverage
G, nodes, et2h = load_graph(str(storage / 'graph_chunk_entity_relation.graphml'))
entities, facts = load_gold(gold_path)
matched = check_entity_coverage(entities, nodes)
covered, _ = check_fact_coverage(facts, nodes, et2h)
ec = len(matched)/len(entities) if entities else 0
fc = len(covered)/len(facts) if facts else 0
print(f'EC={ec:.3f} FC={fc:.3f} ({len(G.nodes())}n/{len(G.edges())}e)')

# Save result
result = {'dataset': DATASET, 'model': MODEL, 'ec': ec, 'fc': fc, 'nodes': len(G.nodes()), 'edges': len(G.edges())}
result_file = Path('experiments/results') / f'crossmodel_gemini_{DATASET}.json'
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to {result_file}')
"
