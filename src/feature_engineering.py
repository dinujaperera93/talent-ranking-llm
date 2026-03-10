from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import TARGET_KEYWORDS

# Design decision: sentence transformers instead of TF-IDF
#
# TF-IDF is a bag-of-words method — it matches exact words and counts them.
# This causes two problems:
#   1. Long titles get penalized: a title like
#      "Aspiring HR Manager | Graduating May 2020 | Seeking Entry-Level Position"
#      has its "aspiring" and "human resources" signal diluted by the extra words,
#      so cosine similarity with the short target keyword comes out lower even though
#      the candidate is clearly relevant.
#   2. Vocabulary mismatch: TF-IDF treats "HR" and "Human Resources" as unrelated,
#      and "seek" and "seeking" as different even after lemmatization.
#
# Sentence transformers (all-MiniLM-L6-v2) fix both issues:
#   - They produce a fixed-size embedding regardless of title length, so long titles
#     are not penalized.
#   - The model was pre-trained on millions of sentences and understands that
#     "HR Manager" and "Human Resources Professional" are semantically close.
#   - No need to clean/lemmatize the target keywords — the model handles raw text.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def add_features(df):
    df = df.copy()
    candidate_embs = _MODEL.encode(df["job_title_clean"].tolist(), show_progress_bar=False)
    target_embs    = _MODEL.encode(TARGET_KEYWORDS,                 show_progress_bar=False)
    sims = cosine_similarity(candidate_embs, target_embs)
    df["fit"] = sims.max(axis=1)
    return df
