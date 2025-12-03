# eval_engine.py
import numpy as np
import pandas as pd
from typing import List

# ==============================
# PRECISION / RECALL / MRR
# ==============================

def precision_at_k(retrieved: List[int], relevant: List[int], k: int):
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    tp = sum(1 for i in retrieved_k if i in rel_set)
    return tp / k

def recall_at_k(retrieved: List[int], relevant: List[int], k: int):
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    tp = sum(1 for i in retrieved_k if i in rel_set)
    return tp / len(relevant) if len(relevant) > 0 else 0

def mean_reciprocal_rank(results: List[List[int]], relevants: List[List[int]]):
    rr = []
    for res, rel in zip(results, relevants):
        rank = 0
        for idx, img in enumerate(res):
            if img in rel:
                rank = idx + 1
                break
        rr.append(1 / rank if rank > 0 else 0)
    return np.mean(rr)

# ==============================
# FULL EVALUATION PIPELINE
# ==============================

def evaluate_system(
        query_results: List[List[int]],
        ground_truth: List[List[int]],
        k=5,
        save_path="eval_results.csv"
    ):
    rows = []
    all_precision = []
    all_recall = []

    for i, (retrieved, relevant) in enumerate(zip(query_results, ground_truth)):
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        all_precision.append(p)
        all_recall.append(r)
        rows.append({
            "Query_ID": i,
            "Precision@K": p,
            "Recall@K": r
        })

    mrr = mean_reciprocal_rank(query_results, ground_truth)

    df = pd.DataFrame(rows)
    df.loc["AVG"] = ["AVG", np.mean(all_precision), np.mean(all_recall)]
    df["MRR"] = mrr
    df.to_csv(save_path, index=False)
    return df
