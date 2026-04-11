import numpy as np

def ndcg_at_k(ranked_df, relevant_ids, k=50):
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).

    Measures how well the top-k ranked candidates match the ground-truth
    relevant set (recruiter-starred IDs). A score of 1.0 means all relevant
    candidates appear at the very top; 0.0 means none appear in the top k.

    Args:
        ranked_df:    DataFrame sorted by score descending (best candidate first).
        relevant_ids: Collection of recruiter-starred IDs.
        k:            Rank cutoff — only the top-k rows are evaluated.

    Returns:
        Float in [0.0, 1.0].
    """
    top_k = ranked_df.head(k)["id"].tolist()
    relevant = set(relevant_ids)

    dcg = sum(1 / np.log2(i + 2) for i, v in enumerate(top_k) if v in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0
