"""
Offline Evaluation for Animal Text-to-Image Retrieval
- CLIP text embedding
- KG Query Expansion
- FAISS Search
- Precision@K, Recall@K, MRR
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import numpy as np
import pandas as pd
import torch
import clip
import faiss
from rdflib import Graph

from eval_engine import evaluate_system

# =========================
# CONFIG
# =========================
CACHE_DIR = "cache100"

CAPTIONS_PATH = os.path.join(CACHE_DIR, "captions.csv")
EMBEDS_PATH   = os.path.join(CACHE_DIR, "image_embeds.npy")
INDEX_PATH    = os.path.join(CACHE_DIR, "faiss.index")
KG_PATH       = os.path.join(CACHE_DIR, "kg.ttl")

TOP_K  = 10
K_EVAL = 5

ANIMALS   = ['dog', 'cat']
BEHAVIORS = ['lying', 'running', 'standing', 'sitting', 'eating', 'sleeping', 'swimming']
ENVS      = ['grass', 'barn', 'field', 'road', 'forest', 'indoor', 'outdoor', 'laptop']

# =========================
# LOAD MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

# =========================
# LOAD DATA
# =========================
assert os.path.exists(INDEX_PATH), "FAISS index not found"
assert os.path.exists(EMBEDS_PATH), "Embeddings not found"
assert os.path.exists(CAPTIONS_PATH), "Captions not found"

index = faiss.read_index(INDEX_PATH)
df = pd.read_csv(CAPTIONS_PATH)

if os.path.exists(KG_PATH):
    kg_graph = Graph().parse(KG_PATH, format="turtle")
else:
    kg_graph = None

print(f"[INFO] Loaded {len(df)} images")

# =========================
# FUNCTIONS
# =========================
def clip_embed_text(text: str) -> np.ndarray:
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def kg_reasoning(query: str) -> str:
    if kg_graph is None:
        return query

    words = re.findall(r'\b\w+\b', query.lower())
    animal = next((w for w in words if w in ANIMALS), None)
    if not animal:
        return query

    sparql = f"""
    PREFIX ex: <http://animal.org/>
    SELECT DISTINCT ?b ?e
    WHERE {{
        ex:{animal} ex:hasBehavior ?beh .
        BIND(STRAFTER(STR(?beh), "#") AS ?b)
        OPTIONAL {{
            ex:{animal} ex:inEnvironment ?env .
            BIND(STRAFTER(STR(?env), "#") AS ?e)
        }}
    }}
    """

    try:
        results = kg_graph.query(sparql)
    except:
        return query

    expansions = set()
    for row in results:
        if row.b:
            expansions.add(str(row.b))
        if row.e:
            expansions.add(str(row.e))

    if expansions:
        return query + " " + " ".join(expansions)

    return query


def search_faiss(query_emb: np.ndarray, top_k: int = TOP_K):
    q = np.array([query_emb], dtype="float32")
    _, I = index.search(q, top_k)
    return I[0].tolist()

# =========================
# TEST QUERIES & GROUND TRUTH
# =========================
# test_queries = [
#     "a black dog laying on the grass",
#     "a dog swimming in a pool",
#     "a grey cat sleeping on a laptop"
# ]

# # Ground truth = index trong captions.csv
# ground_truth = [
#     [33, 35, 44],
#     [42, 41, 40],
#     [21, 22, 19]
# ]

# test_queries = [
#     "a dog lying on the grass",
#     "a dog running in water",
#     "a cat sitting in the grass",
#     "a cat sleeping on a laptop",
#     "a cat playing with a toy",
#     "a dog swimming in a pool",
#     "a dog playing with a ball",
#     "a cat sleeping on a bed",
#     "a dog sitting with a person",
#     "two dogs playing together"
# ]

# # =========================
# # GROUND TRUTH
# # id lấy trực tiếp từ caption_id50.csv
# # =========================
# ground_truth = [
#     # 1. dog lying on the grass
#     [32, 33, 34],

#     # 2. dog running in water
#     [35],

#     # 3. cat sitting in the grass
#     [3, 4],

#     # 4. cat sleeping on a laptop
#     [20, 21],

#     # 5. cat playing with a toy
#     [12, 13],

#     # 6. dog swimming in a pool
#     [39, 40, 41],

#     # 7. dog playing with a ball
#     [43, 44],

#     # 8. cat sleeping on a bed
#     [18, 19, 25],

#     # 9. dog sitting with a person
#     [37, 38],

#     # 10. two dogs playing together
#     [47, 49]
# ]

#caption_id100.csv
test_queries= [
    "a cat sitting in the garden",
    "a cat sleeping on a bed",
    "a cat sleeping on a laptop",
    "a cat playing with a toy",
    "a cat sitting near a window",
    "a dog drinking water",
    "a dog lying on the grass",
    "a dog running on the beach",
    "a dog swimming in a pool",
    "two dogs playing together"
]
ground_truth = [
    [5, 6, 7, 9, 10, 11, 12],          # cat in garden / grass
    [55, 56, 58, 59, 60, 62],          # cat sleeping on bed
    [65, 66, 68, 69],                  # cat sleeping on laptop
    [34, 35, 36, 37, 42, 43],          # cat playing with toy
    [49, 50, 51, 52, 53],              # cat sitting near window
    [103, 107, 109],                   # dog drinking water
    [113, 114, 115, 116, 118, 120],    # dog lying on grass
    [139, 140, 141, 143],              # dog running on beach
    [162, 163, 164, 165, 166],         # dog swimming
    [188, 189, 191, 192, 195, 196]     # two dogs playing together
]

# =========================
# RUN EVALUATION
# =========================
all_retrieved = []

for i, query in enumerate(test_queries):
    print(f"[INFO] Query {i+1}/{len(test_queries)}: {query}")

    # 1. Text embedding
    emb = clip_embed_text(query)

    # 2. KG expansion (giống code Streamlit)
    try:
        expanded = kg_reasoning(query)
        emb2 = clip_embed_text(expanded)
        emb = 0.75 * emb + 0.25 * emb2
        emb = emb / np.linalg.norm(emb)
    except:
        pass

    # 3. FAISS search
    retrieved_ids = search_faiss(emb, TOP_K)
    all_retrieved.append(retrieved_ids)

# =========================
# METRICS
# =========================
df_eval = evaluate_system(
    query_results=all_retrieved,
    ground_truth=ground_truth,
    k=K_EVAL,
    save_path="eval_results100.csv"
)

# =========================
# OUTPUT
# =========================
print("\n===== EVALUATION RESULTS =====")
print(df_eval.round(4).to_string(index=False))

print("\n[INFO] Saved results to eval_results.csv")
