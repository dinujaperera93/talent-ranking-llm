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

ROOT = Path(__file__).resolve().parent
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
STARRED_IDS = [1, 3, 6]

reranked = rerank(ranked, STARRED_IDS)
reranked[["id", "job_title", "location", "connections_raw", "fit"]].head(10)

# %% [markdown]
# ## Evaluate

# %%
print(f"NDCG@10 before re-ranking: {ndcg_at_k(ranked,   STARRED_IDS):.4f}")
print(f"NDCG@10 after  re-ranking: {ndcg_at_k(reranked, STARRED_IDS):.4f}")
