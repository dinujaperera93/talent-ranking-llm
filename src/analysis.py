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
# # Embedding Techniques Comparison
# Compare six embedding methods for candidate ranking by semantic similarity to target keywords.

# %%
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent      # running as script: src/ -> project root
except NameError:
    try:
        ROOT = Path(__vsc_ipynb_file__).resolve().parent.parent  # VS Code notebook: notebooks/ -> project root
    except NameError:
        ROOT = Path().resolve().parent                 # fallback: Jupyter launched from notebooks/

sys.path.insert(0, str(ROOT))

from IPython.display import Image, display
from src.config import DATA_FILE
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.compare_embeddings import METHODS, visualize
from src.visualisation import build_table

df = load_data(ROOT / DATA_FILE)
df = preprocess(df)
df.head()

# %% [markdown]
# ## Inspect raw data

# %%
print(df.shape)
print(df.dtypes)
df["connection"].value_counts()

# %% [markdown]
# ## Preprocess
#
# Three steps:
# 1. `job_title_clean` — title kept as-is (transformer models handle raw text better than stripped tokens)
# 2. `connections_raw` — parse "500+" → 500, invalid → 0
# 3. `connections_norm` — divide by 500, clip to [0, 1] so it can be blended with cosine similarity scores

# %%
df[["job_title", "job_title_clean", "connections_raw", "connections_norm"]].head(10)

# %% [markdown]
# ## Embedding Methods Overview
#
# Each method embeds candidate job titles and target keywords into vector space,
# then computes cosine similarity. The candidate's score = max similarity across all targets.
#
# | Method       | Type               | Trained on                                   | Key property |
# |--------------|--------------------|--------------------------------------------- |--------------|
# | TF-IDF       | Bag-of-words       | This corpus                                  | Exact word match, weighted by rarity |
# | Word2Vec     | Word embeddings    | This corpus                                  | Context-based, avg pooling |
# | FastText     | Subword embeddings | This corpus                                  | Handles rare/misspelled words |
# | GloVe-FT     | Word embeddings    | Wikipedia + Gigaword → fine-tuned (Mittens)  | Pretrained vectors adapted to job-title vocab |
# | BERT-FT      | Transformer        | Large pretraining → fine-tuned (SimCSE)      | Full sentence context, domain-adapted |
# | E5-small-FT  | Transformer        | Large pretraining → fine-tuned (SimCSE)      | Retrieval-optimized, asymmetric query/passage |

# %% [markdown]
# ## TF-IDF
#
# **What it is:** Bag-of-words method. Gives each word a weight based on how often it appears
# in a document (TF) and how rare it is across all documents (IDF). No understanding of meaning.
#
# **Advantage:** Simple, fast, interpretable.
#
# **Disadvantages:**
# - Long titles get penalized: "Aspiring HR Manager | Graduating May 2020 | Seeking Entry-Level Position"
#   has its signal diluted by extra words, so cosine similarity with the short target is lower
#   even though the candidate is clearly relevant.
# - Vocabulary mismatch: "HR" and "Human Resources" are treated as completely unrelated.
#   "seek" and "seeking" are different tokens even after lemmatization.

# %%
df["fit_tfidf"] = METHODS["TF-IDF"](df)
df[["id", "job_title", "fit_tfidf"]].sort_values("fit_tfidf", ascending=False).head(10)

# %% [markdown]
# ## Word2Vec
#
# **What it is:** Trains a shallow neural network to predict surrounding words (skip-gram)
# or the current word from context (CBOW). Words appearing in similar contexts get similar vectors.
# A candidate's title vector = average of its word vectors.
#
# **Advantage over TF-IDF:** Words don't need to match exactly.
# "HR" and "human resources" may end up nearby if they appear in similar contexts.
#
# **Disadvantages:**
# - **Negation failure:** Averaging ignores negation.
#   "not interested in management" and "interested in management" produce nearly the same vector
#   because "not" is a tiny, direction-agnostic word that barely shifts the mean.
# - **Small corpus:** With only ~100 job titles, Word2Vec can't learn reliable relationships.
#   "aspiring" and "seeking" will have near-random vectors relative to each other.
# - **Averaging loses word order:** "Human Resources Manager" and "Manager Human Resources"
#   produce identical vectors.

# %%
df["fit_w2v"] = METHODS["Word2Vec"](df)
df[["id", "job_title", "fit_w2v"]].sort_values("fit_w2v", ascending=False).head(10)

# %% [markdown]
# ## FastText
#
# **What it is:** Extends Word2Vec with character-level n-gram embeddings.
# "resources" → "res", "reso", "resou", ... The word vector = average of all its n-gram vectors.
#
# **Advantage over Word2Vec:** Handles rare and unseen words.
# "HR-Specialist" or "Resourcing" can still get a meaningful vector by composing
# subword pieces shared with known words like "resource" and "resources".
#
# **Disadvantages:**
# - Still trained on a small corpus (~100 titles) — n-gram patterns are poorly calibrated.
# - Averaging still loses word order, same as Word2Vec.
# - **Cannot distinguish intent from surface similarity:**
#   "looking for HR manager role" (candidate) and "hiring HR manager role" (employer)
#   share the same words → nearly identical vectors. FastText has no way to tell them apart.
#   BERT-style models handle this better by encoding full sentence context.

# %%
df["fit_ft"] = METHODS["FastText"](df)
df[["id", "job_title", "fit_ft"]].sort_values("fit_ft", ascending=False).head(10)

# %% [markdown]
# ## GloVe
#
# **What it is:** Pretrained word embeddings trained on 6 billion tokens from Wikipedia
# and Gigaword by factorizing a global word co-occurrence matrix.
#
# **Advantage over Word2Vec/FastText on our corpus:** Trained on massive, diverse data —
# "aspiring", "seeking", "human", "resources" all have well-calibrated vectors.
# "HR" and "human resources" are likely close because they co-occur in the same Wikipedia contexts.
#
# **Disadvantages:**
# - **Domain gap:** Not trained on job postings or LinkedIn data.
#   Jargon like "HRBP" or "talent acquisition" may be absent (OOV → zero vector).
# - **Still word-level averaging:** Same word-order and negation blindness as Word2Vec.

# %%
df["fit_glove"] = METHODS["GloVe-FT"](df)
df[["id", "job_title", "fit_glove"]].sort_values("fit_glove", ascending=False).head(10)

# %% [markdown]
# ## BERT (all-MiniLM-L6-v2)
#
# **What it is:** Transformer model that encodes the entire sentence at once using self-attention.
# Every token attends to every other token, so context is captured — "resources" in
# "human resources" is represented differently than "resources" in "natural resources".
#
# **Advantages:**
# - Semantic understanding: matches "aspiring human resources" with "HR enthusiast looking for a role"
#   even with no word overlap.
# - Handles long titles: the full title is encoded as one vector — extra words don't dilute the signal.
# - No preprocessing needed: handles punctuation, casing, and stopwords internally.
#
# **Disadvantage — symmetric model:**
# all-MiniLM-L6-v2 encodes both the candidate title and the target keyword the same way.
# It does not distinguish between a "document to be retrieved" and a "search query".
# For asymmetric retrieval (short query vs. longer document), E5 is better suited.

# %%
df["fit_bert"] = METHODS["BERT-FT"](df)
df[["id", "job_title", "fit_bert"]].sort_values("fit_bert", ascending=False).head(10)

# %% [markdown]
# ## E5-small (intfloat/e5-small-v2)
#
# **What it is:** Transformer model designed specifically for text retrieval.
# Trained to distinguish between **queries** (what you search for) and **passages** (documents being searched).
# The "passage: " and "query: " prefixes are mandatory — they are part of the model's training protocol.
# Omitting them produces generic embeddings and significantly degrades similarity scores.
#
# **How it differs from BERT:**
# - BERT is **symmetric**: both sides encoded the same way.
# - E5 is **asymmetric**: query and passage have separate learned representations,
#   which better matches real retrieval — a short keyword and a long job title are different in nature.
#
# **Advantages:**
# - Purpose-built for retrieval: "query: aspiring human resources" pulls close to
#   "passage: Aspiring HR Professional | Entry-Level Candidate" even without exact word overlap.
# - Strong out-of-the-box performance — no fine-tuning needed on our data.
#
# **Disadvantage:** e5-small-v2 is a compressed model (12M parameters).
# e5-large-v2 would be more accurate but slower — for ~100 candidates the small variant is sufficient.

# %%
df["fit_e5"] = METHODS["E5-small-FT"](df)
df[["id", "job_title", "fit_e5"]].sort_values("fit_e5", ascending=False).head(10)

# %% [markdown]
# ## Score Comparison Across All Methods

# %%
score_cols = ["id", "job_title", "fit_tfidf", "fit_w2v", "fit_ft", "fit_glove", "fit_bert", "fit_e5"]
df[score_cols].sort_values("fit_e5", ascending=False).head(20)

# %% [markdown]
# ## Visualize Embedding Space
#
# Each method's embeddings are reduced to 2D using PCA and t-SNE.
# - **Green dots** = candidates starred by the recruiter (relevant)
# - **Light blue dots** = non-relevant candidates
# - **Red stars** = target keywords (T1, T2)
#
# A good embedding method should cluster green dots close to the red stars.

# %%
visualize(df)

# %% [markdown]
# ### PCA — Linear projection (fast, shows global structure)

# %%
display(Image(filename=str(ROOT / "outputs" / "embedding_space_pca.png")))

# %% [markdown]
# ### t-SNE — Non-linear projection (slower, better cluster separation)
#
# Note: distances *between* clusters in t-SNE are not meaningful — only local neighbourhood structure is.

# %%
display(Image(filename=str(ROOT / "outputs" / "embedding_space_tsne.png")))

# %% [markdown]
# ## Candidate Comparison Table
#
# Top-10 candidate IDs selected by each method — 14 columns total.
#
# **Embedding methods (6):** ranked by similarity score blended with `connections_norm` (w=0.7).
#
# | Column | Method |
# |--------|--------|
# | TF-IDF | Bag-of-words similarity |
# | Word2Vec | Corpus-trained word embeddings |
# | FastText | Subword embeddings |
# | GloVe-FT | Pretrained GloVe + fine-tuned |
# | BERT-FT | Transformer sentence embeddings |
# | E5-small-FT | Retrieval-optimised transformer |
#
# **LLM methods (8):** top-10 from 4 prompting techniques × 2 models (Qwen-0.5B, Gemma-4-E2B-it).
#
# | Technique | Description |
# |-----------|-------------|
# | Zero-shot | Plain instruction, no examples |
# | Few-shot | Small worked example shown before the question |
# | Chat | Framed as a user ↔ assistant conversation |
# | CoT | Model reasons step-by-step before answering |

# %%
tbl = build_table(df)
tbl
