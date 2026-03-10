from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rerank(df, starred_ids, alpha=0.5):
    df = df.copy()
    starred_titles = df.loc[df["id"].isin(starred_ids), "job_title_clean"].tolist()

    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(df["job_title_clean"])
    sims = cosine_similarity(
        vec.transform(df["job_title_clean"]),
        vec.transform(starred_titles),
    )
    df["starred_sim"] = sims.mean(axis=1)
    df["fit"] = (1 - alpha) * df["fit"] + alpha * df["starred_sim"]
    return df.sort_values("fit", ascending=False).reset_index(drop=True)
