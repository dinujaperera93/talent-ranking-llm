from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Using the same sentence transformer model as feature_engineering for consistency.
# Cosine similarity between all candidate titles and the starred candidate titles
# gives a "similarity to what the recruiter liked" score, which is blended into fit.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def rerank(df, starred_ids, alpha=0.9):  # alpha=0.9 confirmed optimal by grid search
    df = df.copy()
    starred_titles = df.loc[df["id"].isin(starred_ids), "job_title_clean"].tolist()

    candidate_embs = _MODEL.encode(df["job_title_clean"].tolist(), show_progress_bar=False)
    starred_embs   = _MODEL.encode(starred_titles,                 show_progress_bar=False)

    sims = cosine_similarity(candidate_embs, starred_embs)
    df["starred_sim"] = sims.mean(axis=1)
    df["fit"] = (1 - alpha) * df["fit"] + alpha * df["starred_sim"]
    return df.sort_values("fit", ascending=False).reset_index(drop=True)
