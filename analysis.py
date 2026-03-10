# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Talent Fit Scoring Pipeline
# Rank candidates by fit score using job title similarity and connection count.

# %%
from pathlib import Path
from config import DATA_FILE
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import add_features
from src.ranking import rank_candidates
from src.reranking import rerank
from src.evaluation import ndcg_at_k

try:
    ROOT = Path(__file__).resolve().parent  # running as a script
except NameError:
    ROOT = Path().resolve()                 # running as a notebook
df = load_data(ROOT / DATA_FILE)
df.head()

# %% [markdown]
# ## Inspect raw data

# %%
print(df.shape)
print(df.dtypes)
df["connection"].value_counts()

# %% [markdown]
# ## Preprocess

# %%
df = preprocess(df)
# connections_norm: LinkedIn connections is a proxy for how networked/active someone is in their field.
# A well-connected person is likely more experienced or engaged professionally.
# Raw values go from 0 to "500+" so we normalize (divide by 500) to get a 0-1 scale.
# Without this, we can't combine it with fit (also 0-1) — they'd be on completely different
# scales and connections would dominate unfairly.
df[["job_title", "job_title_clean", "connections_raw", "connections_norm"]].head(10)

# %% [markdown]
# ## Feature engineering (TF-IDF cosine similarity)

# %%
df = add_features(df)
df[["job_title_clean", "fit"]].sort_values("fit", ascending=False).head(10)

# %% [markdown]
# ## Rank candidates

# %%
ranked = rank_candidates(df)
ranked[["id", "job_title", "location", "connections_raw", "fit"]].head(10)

# %% [markdown]
# ## Re-rank after human feedback

# %%
# Full reviewed list (43 candidates) — kept for reference:
# STARRED_IDS = [1, 3, 6, 7, 9, 10, 14, 15, 17, 19, 21, 24, 25, 27, 28, 29, 30, 31, 33, 36, 37, 39, 40, 44, 46, 49, 50, 52, 53, 57, 58, 60, 62, 66, 72, 73, 75, 76, 79, 82, 97, 99, 100]
STARRED_IDS = [3, 6, 17, 19, 31, 100]

reranked = rerank(ranked, STARRED_IDS)
reranked[["id", "job_title", "location", "connections_raw", "fit"]].head(10)

# %% [markdown]
# ## Hyperparameter tuning
# Grid search over w (title weight) and alpha (re-rank blend) to maximise NDCG@10.
# df_feats is saved before ranking so we can restart cleanly on each iteration.

# %%
import numpy as np

df_feats = df.copy()  # snapshot after add_features, before any ranking modifies fit

best_ndcg, best_w, best_alpha = -1, None, None
results = []

for w in np.arange(0.1, 1.0, 0.1):
    for alpha in np.arange(0.1, 1.0, 0.1):
        r  = rank_candidates(df_feats, w=round(w, 1))
        rr = rerank(r, STARRED_IDS, alpha=round(alpha, 1))
        score = ndcg_at_k(rr, STARRED_IDS)
        results.append((round(w, 1), round(alpha, 1), round(score, 4)))
        if score > best_ndcg:
            best_ndcg, best_w, best_alpha = score, round(w, 1), round(alpha, 1)

print(f"Best w={best_w}, alpha={best_alpha} → NDCG@10={best_ndcg:.4f}")

# %% [markdown]
# ## Evaluate

# %%
# @10 is the standard in information retrieval — users realistically only look at the first page of results.
print(f"NDCG@10 before re-ranking: {ndcg_at_k(ranked,   STARRED_IDS):.4f}")
print(f"NDCG@10 after  re-ranking: {ndcg_at_k(reranked, STARRED_IDS):.4f}")
