"""
Compare embedding techniques for candidate ranking.

Methods:
  - TF-IDF       (sklearn, trained on corpus)
  - Word2Vec     (gensim, trained on corpus, avg pooling)
  - FastText     (gensim, trained on corpus, avg pooling + subword)
  - GloVe        (gensim downloader: glove-wiki-gigaword-100, pretrained)
  - BERT         (sentence-transformers: all-MiniLM-L6-v2)
  - E5-small     (sentence-transformers: intfloat/e5-small-v2, query/passage prefixes)

For each method:
  1. Compute cosine similarity between candidate job titles and target keywords
  2. Rank candidates purely by max similarity score
  3. Visualize the vector space via PCA and t-SNE
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from .config import TARGET_KEYWORDS, STARRED_IDS
from .data_loader import load_data
from .preprocessing import preprocess, TARGETS_CLEAN

def _avg_word_vecs(texts: list[str], model, vector_size: int) -> np.ndarray:
    # Get averaged word vectors for a sentence for Word2Vec / FastText
    # Split sentence into words, Get vector for each word (if it exists)
    # Average all word vectors, If no words exist → return zero vector
    result = []
    for text in texts:
        vecs = [model.wv[tok] for tok in text.split() if tok in model.wv]
        result.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(result)


def _avg_keyed_vecs(texts: list[str], kv, vector_size: int) -> np.ndarray:
    # Get averaged keyed vectors for a sentence from pretrained embeddings for Glove
    result = []
    for text in texts:
        vecs = [kv[tok] for tok in text.split() if tok in kv]
        result.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(result)


# Ranking methods
# Each embed_* function takes the full candidate DataFrame and returns a 1D
# array of shape (n_candidates,) — the cosine similarity score of each
# candidate against the closest target keyword. This becomes the "fit" score
# before blending with connections in ranking.py.
#
# All methods follow the same pattern:
#   1. Embed candidates  → cand_embs  (n_candidates × dim)
#   2. Embed targets     → tgt_embs   (n_targets × dim)
#   3. cosine_similarity → matrix     (n_candidates × n_targets)
#   4. .max(axis=1)      → scores     (n_candidates,)


# TF-IDF is a bag-of-words method — it matches exact words and counts them.
# TF = Term Frequency
# IDF = Inverse Document Frequency
# It gives a weight to each word based on, how often it appears in a sentence/document or how rare it is across all documents but it cannot understand the meaning.
# This causes two problems:
#   1. Long titles get penalized: a title like
#      "Aspiring HR Manager | Graduating May 2020 | Seeking Entry-Level Position"
#      has its "aspiring" and "human resources" signal diluted by the extra words,
#      so cosine similarity with the short target keyword comes out lower even though
#      the candidate is clearly relevant.
#   2. Vocabulary mismatch: TF-IDF treats "HR" and "Human Resources" as unrelated,
#      and "seek" and "seeking" as different even after lemmatization.

def embed_tfidf(df: pd.DataFrame) -> np.ndarray:
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    vec = TfidfVectorizer()
    vec.fit(all_texts)
    cand_embs = vec.transform(df["job_title_clean"].tolist()).toarray()
    tgt_embs  = vec.transform(TARGETS_CLEAN).toarray()
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


# Word2Vec learns dense word embeddings by training a shallow neural network
# to predict surrounding words or the current word from context (CBOW).
# Words that appear in similar contexts end up with similar vectors, so the model
# captures semantic relationships — e.g. "manager" and "director" will be close
# in vector space because they co-occur with the same words (team, report, lead).
#
# Advantage over TF-IDF:
#   Words don't need to match exactly. "HR" and "human resources" may end up nearby
#   if they appear in similar sentences, which TF-IDF would treat as unrelated.
#
# Disadvantage — averaging cannot handle negation:
#   A sentence vector is just the average of its word vectors, so negation words
#   like "not" have almost no effect on the final vector direction.
#   Example:
#     Sentence A: "not interested in management"
#     Sentence B: "interested in management"
#   Both produce nearly the same average vector because "not" is a short, common word
#   with a small, direction-agnostic vector that barely shifts the mean.
#   Word2Vec treats these two sentences as semantically similar — which is wrong.
#
# Disadvantage — small corpus problem:
#   Word2Vec needs a large corpus to learn meaningful relationships.
#   With only ~100 candidate job titles, there isn't enough context for the model
#   to learn that "aspiring" and "seeking" are semantically related —
#   it will just assign them random nearby vectors based on their few occurrences.
#
# Disadvantage — averaging destroys word order:
#   The title "Human Resources Manager" is averaged into one vector, losing
#   the structure of the phrase. "Manager Human Resources" would give the same vector.

def embed_word2vec(df: pd.DataFrame) -> np.ndarray:
    from gensim.models import Word2Vec
    DIM = 100
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    corpus = [t.split() for t in all_texts]
    model = Word2Vec(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    cand_embs = _avg_word_vecs(df["job_title_clean"].tolist(), model, DIM)
    tgt_embs  = _avg_word_vecs(TARGETS_CLEAN, model, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


# FastText is built on top of Word2Vec but adds character-level n-gram embeddings.
# Each word is represented as the sum of its subword pieces.
# For example, "resources" is broken into: "res", "reso", "resou", "esour", ... etc.
# The final word vector = average of all its n-gram vectors.
#
# Advantage over Word2Vec — handles rare and unseen words:
#   If a candidate writes "HR-Specialist" (hyphenated) or "Resourcing" (uncommon form),
#   Word2Vec would return OOV (zero vector). FastText can still produce a meaningful
#   vector by composing the subword pieces it has seen in other words.
#   Example: even if "Resourcing" was never seen, it shares subwords with "resources"
#   and "resource", so FastText places it close to those words.
#
# Disadvantage — still trained on a small corpus:
#   The subword advantage is most useful for morphological variations.
#   On a corpus of only ~100 titles, the n-gram representations are
#   still poorly calibrated — not enough data to learn reliable subword patterns.
#
# Disadvantage — averaging still loses word order, same as Word2Vec.
#
# Disadvantage — cannot distinguish intent from surface similarity:
#   FastText (like Word2Vec) embeds words based on co-occurrence, not meaning in context.
#   Two sentences that share the same keywords can end up very close in vector space
#   even when they mean completely different things.
#   Example:
#     Sentence A: "looking for HR manager role"   ← candidate seeking a job
#     Sentence B: "hiring HR manager role"        ← employer posting a job
#   Both sentences contain "HR", "manager", "role" — so their averaged vectors
#   will be nearly identical. FastText has no way to tell that one describes a job seeker
#   and the other describes a recruiter. BERT-style models handle this better because
#   they encode the full sentence context, not just a bag of word vectors.

def embed_fasttext(df: pd.DataFrame) -> np.ndarray:
    from gensim.models import FastText
    DIM = 100
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    corpus = [t.split() for t in all_texts]
    model = FastText(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    cand_embs = _avg_word_vecs(df["job_title_clean"].tolist(), model, DIM)
    tgt_embs  = _avg_word_vecs(TARGETS_CLEAN, model, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)

# GloVe (Global Vectors for Word Representation) is a pretrained word embedding model
# trained on 6 billion tokens from Wikipedia and Gigaword news articles.
# It learns vectors by factorizing a global word co-occurrence matrix —
# words that frequently appear together get similar vectors.
#
# Advantage over Word2Vec/FastText trained on our corpus:
#   GloVe was trained on a massive, diverse corpus so its vectors are
#   much more reliable. "aspiring", "seeking", "human", "resources" all have
#   well-calibrated vectors that reflect real-world language use.
#   Example: "HR" and "human resources" are likely close in GloVe space
#   because they appear in the same contexts across millions of Wikipedia articles.
#
# Disadvantage — domain gap:
#   GloVe was not trained on job postings or LinkedIn data.
#   Professional jargon like "HRBP" (HR Business Partner) or "talent acquisition"
#   may be rare or absent in Wikipedia, giving them poor or zero vectors.
#
# Disadvantage — still word-level averaging:
#   Like Word2Vec, the title embedding is just the average of its word vectors.
#   "Human Resources Manager" and "Manager of Human Resources" produce the same vector.
#   Uncommon tokens not in GloVe's vocabulary are silently dropped (OOV → skipped).
#
# Note: loaded via gensim.downloader — uses _avg_keyed_vecs instead of _avg_word_vecs
# because the downloaded model returns a bare KeyedVectors object (no .wv wrapper).
def embed_glove(df: pd.DataFrame) -> np.ndarray:
    DIM = 100
    kv = gensim_dl.load("glove-wiki-gigaword-100")
    cand_embs = _avg_keyed_vecs(df["job_title_clean"].tolist(), kv, DIM)
    tgt_embs  = _avg_keyed_vecs(TARGETS_CLEAN, kv, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


# BERT (Bidirectional Encoder Representations from Transformers) is a transformer
# model pretrained on a huge text corpus. Here we use all-MiniLM-L6-v2, a distilled
# version fine-tuned specifically for semantic sentence similarity tasks.
#
# Unlike TF-IDF, Word2Vec, FastText, and GloVe — which all produce one vector per word
# and then average — BERT encodes the entire title as one sentence vector using
# self-attention across all tokens simultaneously. This means it understands context:
# the word "resources" in "human resources" contributes differently than "resources"
# in "natural resources".
#
# Advantage — semantic understanding:
#   BERT can match "aspiring human resources" with "HR enthusiast looking for a role"
#   even though no words overlap, because it has learned the semantic meaning of
#   sentences from massive pretraining data.
#
# Advantage — handles full titles well:
#   Unlike TF-IDF, long titles like "Aspiring HR Manager | Seeking Entry-Level Position"
#   are encoded as a whole, so the relevant signal isn't diluted by extra words.
#
# Note: we encode raw TARGET_KEYWORDS (not TARGETS_CLEAN) because BERT handles
# punctuation and casing internally — no preprocessing is needed.
def embed_bert(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cand_embs = model.encode(df["job_title_clean"].tolist(), show_progress_bar=False)
    tgt_embs  = model.encode(TARGET_KEYWORDS, show_progress_bar=False)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


# E5 (intfloat/e5-small-v2) is a transformer model designed specifically for
# text retrieval tasks (embedding-based search). It is trained with a distinction
# between queries (what you're searching for) and passages (documents being searched).
#
# This makes it fundamentally different from BERT (all-MiniLM-L6-v2):
#   - BERT is symmetric: both sides are encoded the same way.
#   - E5 is asymmetric: queries and passages have separate roles and are encoded
#     with different representations, which better matches real retrieval scenarios.
#
# Advantage — designed for retrieval:
#   Our task is exactly retrieval: given a short query ("aspiring human resources"),
#   find the most relevant candidate titles (passages). E5 was trained to excel at this.
#   Example: "query: aspiring human resources" will pull close to
#   "passage: Aspiring HR Professional | Entry-Level Candidate" even without
#   exact word overlap, because E5 learned asymmetric retrieval from contrastive training.
#
# Advantage — strong out-of-the-box performance:
#   E5 does not need to be fine-tuned on our data. The pretrained model already
#   generalizes well to job title retrieval without any domain adaptation.
def embed_e5(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer("intfloat/e5-small-v2")
    cand_texts = ["passage: " + t for t in df["job_title_clean"].tolist()]
    tgt_texts  = ["query: "   + t for t in TARGET_KEYWORDS]
    cand_embs = model.encode(cand_texts, show_progress_bar=False)
    tgt_embs  = model.encode(tgt_texts,  show_progress_bar=False)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


# Registry used by main.py / analysis.py to iterate over all methods.
METHODS = {
    "TF-IDF":   embed_tfidf,
    "Word2Vec": embed_word2vec,
    "FastText": embed_fasttext,
    "GloVe":    embed_glove,
    "BERT":     embed_bert,
    "E5-small": embed_e5,
}


# Full-embedding methods (for visualization only)
# These mirror the ranking methods above but return the full normalized embedding
# matrix for candidates + targets concatenated, rather than scalar similarity
# scores. Used exclusively by visualize() to feed PCA and t-SNE.
# Normalization (L2) ensures that PCA/t-SNE distances reflect direction (angle)
# rather than magnitude, which matches how cosine similarity works.
def _get_tfidf(cands, targets):
    vec = TfidfVectorizer()
    vec.fit(cands + targets)
    return normalize(vec.transform(cands + targets).toarray())


def _get_word2vec(cands, targets):
    DIM = 100
    corpus = [t.split() for t in cands + targets]
    model = Word2Vec(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    return normalize(_avg_word_vecs(cands + targets, model, DIM))


def _get_fasttext(cands, targets):
    DIM = 100
    corpus = [t.split() for t in cands + targets]
    model = FastText(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    return normalize(_avg_word_vecs(cands + targets, model, DIM))


def _get_glove(cands, targets):
    DIM = 100
    kv = gensim_dl.load("glove-wiki-gigaword-100")
    return normalize(_avg_keyed_vecs(cands + targets, kv, DIM))


def _get_bert(cands, targets):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return normalize(model.encode(cands + targets, show_progress_bar=False))


def _get_e5(cands, targets):
    model = SentenceTransformer("intfloat/e5-small-v2")
    # Apply asymmetric prefixes: candidates are passages, targets are queries.
    texts = ["passage: " + t for t in cands] + ["query: " + t for t in targets]
    return normalize(model.encode(texts, show_progress_bar=False))


# Internal registry for the visualization path — parallel to METHODS above.
_VIZ_METHODS = {
    "TF-IDF":   _get_tfidf,
    "Word2Vec": _get_word2vec,
    "FastText": _get_fasttext,
    "GloVe":    _get_glove,
    "BERT":     _get_bert,
    "E5-small": _get_e5,
}


# Green = recruiter-starred (relevant), light blue = non-relevant, red star = target keyword.
_COLORS = {
    "relevant":     "#2ca02c",
    "non-relevant": "#aec7e8",
    "target":       "#d62728",
}


def _scatter(ax, coords, ids, n_targets, title):
    """Plot one embedding method's 2D projection onto ax.

    Candidates come first in coords (rows 0..n_cands-1).
    Target keywords occupy the last n_targets rows.
    """
    n_cands = len(ids)
    # Plot each candidate colored by whether the recruiter starred them.
    for i, cid in enumerate(ids):
        color = _COLORS["relevant"] if cid in STARRED_IDS else _COLORS["non-relevant"]
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=30, alpha=0.8, zorder=2)
    # Plot target keywords as large red stars with T1/T2 labels.
    for j in range(n_targets):
        idx = n_cands + j
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=_COLORS["target"], s=120, marker="*", zorder=3)
        ax.annotate(f"T{j+1}", (coords[idx, 0], coords[idx, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color=_COLORS["target"], fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _make_legend():
    """Build legend handles shared across all subplots."""
    return [
        mpatches.Patch(color=_COLORS["relevant"],     label="Relevant candidate"),
        mpatches.Patch(color=_COLORS["non-relevant"], label="Non-relevant candidate"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=_COLORS["target"], markersize=12, label="Target keyword"),
    ]


def _plot_grid(all_coords_2d, ids, n_targets, reducer_name, out_path):
    """Arrange one subplot per embedding method in a 3-column grid and save."""
    n = len(all_coords_2d)
    ncols = 3
    nrows = -(-n // ncols)  # ceiling division — e.g. 6 methods → 2 rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"Embedding Vector Space — {reducer_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    for ax, (name, coords) in zip(axes.flat, all_coords_2d.items()):
        _scatter(ax, coords, ids, n_targets, name)
    # Hide any unused subplot cells when n is not a multiple of ncols.
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.legend(handles=_make_legend(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def visualize(df: pd.DataFrame) -> None:
    """Produce PCA and t-SNE plots for all embedding methods.

    Saves two files to the project root:
      - embedding_space_pca.png   (fast, linear projection)
      - embedding_space_tsne.png  (slower, non-linear — better cluster separation)

    PCA is deterministic and shows global structure.
    t-SNE reveals local clusters but distances between clusters are not meaningful.
    """
    cands = df["job_title_clean"].tolist()
    ids   = df["id"].tolist()
    n_targets = len(TARGET_KEYWORDS)

    # Compute high-dimensional embeddings once; reuse for both PCA and t-SNE.
    print("Computing embeddings for visualization...")
    high_dim = {}
    for name, fn in _VIZ_METHODS.items():
        print(f"  {name}...", end=" ", flush=True)
        high_dim[name] = fn(cands, TARGET_KEYWORDS)
        print("done")

    print("\nReducing with PCA...")
    pca_coords = {
        name: PCA(n_components=2, random_state=42).fit_transform(embs)
        for name, embs in high_dim.items()
    }
    _plot_grid(pca_coords, ids, n_targets, "PCA (2D)", ROOT / "embedding_space_pca.png")

    print("Reducing with t-SNE (slower)...")
    # Perplexity must be < n_samples; cap at 30 (the typical default).
    perp = min(30, len(cands) // 3)
    tsne_coords = {
        name: TSNE(n_components=2, perplexity=perp, random_state=42,
                   max_iter=1000, verbose=0).fit_transform(embs)
        for name, embs in high_dim.items()
    }
    _plot_grid(tsne_coords, ids, n_targets, "t-SNE (2D)", ROOT / "embedding_space_tsne.png")
