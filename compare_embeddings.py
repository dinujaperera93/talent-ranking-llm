"""
Compare embedding techniques for candidate ranking (similarity only).

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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gensim.downloader as gensim_dl
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

from config import DATA_FILE, TARGET_KEYWORDS
from src.data_loader import load_data
from src.preprocessing import preprocess, TARGETS_CLEAN

# ── Ground truth ──────────────────────────────────────────────────────────────
STARRED_IDS = [
    1, 3, 6, 7, 9, 10, 14, 15, 17, 19, 21, 24, 25, 27, 28, 29, 30,
    31, 33, 36, 37, 39, 40, 44, 46, 49, 50, 52, 53, 57, 58, 60, 62,
    66, 72, 73, 75, 76, 79, 82, 97, 99, 100,
]

# ── Shared helper ─────────────────────────────────────────────────────────────

def _avg_word_vecs(texts: list[str], model, vector_size: int) -> np.ndarray:
    """Average non-OOV word vectors; return zero vector for all-OOV texts."""
    result = []
    for text in texts:
        vecs = [model.wv[tok] for tok in text.split() if tok in model.wv]
        result.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(result)


def _avg_keyed_vecs(texts: list[str], kv, vector_size: int) -> np.ndarray:
    """Same as above but for a bare KeyedVectors object (e.g. GloVe)."""
    result = []
    for text in texts:
        vecs = [kv[tok] for tok in text.split() if tok in kv]
        result.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(result)


# ── Ranking methods (return per-candidate similarity scores) ──────────────────
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

def embed_tfidf(df: pd.DataFrame) -> np.ndarray:
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    vec = TfidfVectorizer()
    vec.fit(all_texts)
    cand_embs = vec.transform(df["job_title_clean"].tolist()).toarray()
    tgt_embs  = vec.transform(TARGETS_CLEAN).toarray()
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


def embed_word2vec(df: pd.DataFrame) -> np.ndarray:
    DIM = 100
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    corpus = [t.split() for t in all_texts]
    model = Word2Vec(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    cand_embs = _avg_word_vecs(df["job_title_clean"].tolist(), model, DIM)
    tgt_embs  = _avg_word_vecs(TARGETS_CLEAN, model, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


def embed_fasttext(df: pd.DataFrame) -> np.ndarray:
    DIM = 100
    all_texts = df["job_title_clean"].tolist() + TARGETS_CLEAN
    corpus = [t.split() for t in all_texts]
    model = FastText(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    cand_embs = _avg_word_vecs(df["job_title_clean"].tolist(), model, DIM)
    tgt_embs  = _avg_word_vecs(TARGETS_CLEAN, model, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


def embed_glove(df: pd.DataFrame) -> np.ndarray:
    DIM = 100
    kv = gensim_dl.load("glove-wiki-gigaword-100")
    cand_embs = _avg_keyed_vecs(df["job_title_clean"].tolist(), kv, DIM)
    tgt_embs  = _avg_keyed_vecs(TARGETS_CLEAN, kv, DIM)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


def embed_bert(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cand_embs = model.encode(df["job_title_clean"].tolist(), show_progress_bar=False)
    tgt_embs  = model.encode(TARGET_KEYWORDS, show_progress_bar=False)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


def embed_e5(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer("intfloat/e5-small-v2")
    cand_texts = ["passage: " + t for t in df["job_title_clean"].tolist()]
    tgt_texts  = ["query: "   + t for t in TARGET_KEYWORDS]
    cand_embs = model.encode(cand_texts, show_progress_bar=False)
    tgt_embs  = model.encode(tgt_texts,  show_progress_bar=False)
    return cosine_similarity(cand_embs, tgt_embs).max(axis=1)


METHODS = {
    "TF-IDF":   embed_tfidf,
    "Word2Vec": embed_word2vec,
    "FastText": embed_fasttext,
    "GloVe":    embed_glove,
    "BERT":     embed_bert,
    "E5-small": embed_e5,
}


# ── Full-embedding methods (candidates + targets, for visualization) ───────────

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
    texts = ["passage: " + t for t in cands] + ["query: " + t for t in targets]
    return normalize(model.encode(texts, show_progress_bar=False))


_VIZ_METHODS = {
    "TF-IDF":   _get_tfidf,
    "Word2Vec": _get_word2vec,
    "FastText": _get_fasttext,
    "GloVe":    _get_glove,
    "BERT":     _get_bert,
    "E5-small": _get_e5,
}


# ── Plot helpers ──────────────────────────────────────────────────────────────

_COLORS = {
    "relevant":     "#2ca02c",
    "non-relevant": "#aec7e8",
    "target":       "#d62728",
}


def _scatter(ax, coords, ids, n_targets, title):
    n_cands = len(ids)
    for i, cid in enumerate(ids):
        color = _COLORS["relevant"] if cid in STARRED_IDS else _COLORS["non-relevant"]
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=30, alpha=0.8, zorder=2)
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
    return [
        mpatches.Patch(color=_COLORS["relevant"],     label="Relevant candidate"),
        mpatches.Patch(color=_COLORS["non-relevant"], label="Non-relevant candidate"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=_COLORS["target"], markersize=12, label="Target keyword"),
    ]


def _plot_grid(all_coords_2d, ids, n_targets, reducer_name, out_path):
    n = len(all_coords_2d)
    ncols = 3
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"Embedding Vector Space — {reducer_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    for ax, (name, coords) in zip(axes.flat, all_coords_2d.items()):
        _scatter(ax, coords, ids, n_targets, name)
    for ax in axes.flat[n:]:  # hide unused subplots if n % ncols != 0
        ax.set_visible(False)
    fig.legend(handles=_make_legend(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ── Public visualization entry point ─────────────────────────────────────────

def visualize(df: pd.DataFrame) -> None:
    """Produce PCA and t-SNE plots for all embedding methods."""
    cands = df["job_title_clean"].tolist()
    ids   = df["id"].tolist()
    n_targets = len(TARGET_KEYWORDS)

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
    perp = min(30, len(cands) // 3)
    tsne_coords = {
        name: TSNE(n_components=2, perplexity=perp, random_state=42,
                   max_iter=1000, verbose=0).fit_transform(embs)
        for name, embs in high_dim.items()
    }
    _plot_grid(tsne_coords, ids, n_targets, "t-SNE (2D)", ROOT / "embedding_space_tsne.png")
