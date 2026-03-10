import numpy as np


def ndcg_at_k(ranked_df, relevant_ids, k=10):
    top_k = ranked_df.head(k)["id"].tolist()
    relevant = set(relevant_ids)
    dcg = sum(1 / np.log2(i + 2) for i, v in enumerate(top_k) if v in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0
