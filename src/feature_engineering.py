from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TARGET_KEYWORDS


def add_features(df):
    df = df.copy()
    corpus = list(df["job_title_clean"]) + TARGET_KEYWORDS
    tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(corpus)
    sims = cosine_similarity(tfidf[: len(df)], tfidf[len(df) :])
    df["fit"] = sims.max(axis=1)
    return df
