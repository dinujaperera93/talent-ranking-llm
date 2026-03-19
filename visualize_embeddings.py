"""
Visualize the vector space for each embedding technique.

For each method (TF-IDF, Word2Vec, FastText, BERT):
  - Embed all candidates + target keywords
  - Reduce to 2D via PCA and t-SNE
  - Plot: relevant candidates (green), non-relevant (gray), targets (red star)

Saves: embedding_space_pca.png  and  embedding_space_tsne.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATA_FILE, TARGET_KEYWORDS
from src.data_loader import load_data
from src.preprocessing import preprocess

# ── Ground truth ──────────────────────────────────────────────────────────────
STARRED_IDS = [
    1, 3, 6, 7, 9, 10, 14, 15, 17, 19, 21, 24, 25, 27, 28, 29, 30,
    31, 33, 36, 37, 39, 40, 44, 46, 49, 50, 52, 53, 57, 58, 60, 62,
    66, 72, 73, 75, 76, 79, 82, 97, 99, 100,
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _avg_word_vecs(texts, model, dim):
    result = []
    for text in texts:
        vecs = [model.wv[t] for t in text.split() if t in model.wv]
        result.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))
    return np.array(result)


# ── Embedding methods (return matrix: candidates + targets at the end) ─────────

def get_tfidf(cands, targets):
    vec = TfidfVectorizer()
    vec.fit(cands + targets)
    embs = vec.transform(cands + targets).toarray()
    return normalize(embs)


def get_word2vec(cands, targets):
    DIM = 100
    corpus = [t.split() for t in cands + targets]
    model = Word2Vec(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    embs = _avg_word_vecs(cands + targets, model, DIM)
    return normalize(embs)


def get_fasttext(cands, targets):
    DIM = 100
    corpus = [t.split() for t in cands + targets]
    model = FastText(corpus, vector_size=DIM, window=5, min_count=1,
                     workers=4, epochs=100, seed=42)
    embs = _avg_word_vecs(cands + targets, model, DIM)
    return normalize(embs)


def get_bert(cands, targets):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(cands + targets, show_progress_bar=False)
    return normalize(embs)


METHODS = {
    "TF-IDF":   get_tfidf,
    "Word2Vec": get_word2vec,
    "FastText": get_fasttext,
    "BERT":     get_bert,
}

# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = {
    "relevant":     "#2ca02c",   # green
    "non-relevant": "#aec7e8",   # light blue-gray
    "target":       "#d62728",   # red
}


def _scatter(ax, coords, ids, n_targets, title):
    n_cands = len(ids)

    for i, cid in enumerate(ids):
        color = COLORS["relevant"] if cid in STARRED_IDS else COLORS["non-relevant"]
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=30, alpha=0.8, zorder=2)

    # Target keywords (last n_targets rows)
    for j in range(n_targets):
        idx = n_cands + j
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=COLORS["target"], s=120, marker="*", zorder=3,
                   label=f'Target {j+1}' if j == 0 else "")
        ax.annotate(f"T{j+1}", (coords[idx, 0], coords[idx, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color=COLORS["target"], fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _make_legend():
    return [
        mpatches.Patch(color=COLORS["relevant"],     label="Relevant candidate"),
        mpatches.Patch(color=COLORS["non-relevant"], label="Non-relevant candidate"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=COLORS["target"], markersize=12, label="Target keyword"),
    ]


def plot_grid(all_coords_2d, ids, n_targets, reducer_name, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Embedding Vector Space — {reducer_name}", fontsize=14, fontweight="bold", y=1.01)

    for ax, (name, coords) in zip(axes.flat, all_coords_2d.items()):
        _scatter(ax, coords, ids, n_targets, name)

    fig.legend(handles=_make_legend(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)

    cands = df["job_title_clean"].tolist()
    ids   = df["id"].tolist()
    targets = TARGET_KEYWORDS
    n_targets = len(targets)

    # Compute high-dim embeddings for all methods
    print("Computing embeddings...")
    high_dim = {}
    for name, fn in METHODS.items():
        print(f"  {name}...", end=" ", flush=True)
        high_dim[name] = fn(cands, targets)
        print("done")

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("\nReducing with PCA...")
    pca_coords = {}
    for name, embs in high_dim.items():
        pca = PCA(n_components=2, random_state=42)
        pca_coords[name] = pca.fit_transform(embs)

    plot_grid(pca_coords, ids, n_targets, "PCA (2D)",
              ROOT / "embedding_space_pca.png")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("Reducing with t-SNE (slower)...")
    tsne_coords = {}
    for name, embs in high_dim.items():
        perp = min(30, len(cands) // 3)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000, verbose=0)
        tsne_coords[name] = tsne.fit_transform(embs)

    plot_grid(tsne_coords, ids, n_targets, "t-SNE (2D)",
              ROOT / "embedding_space_tsne.png")

    print("\nDone. Two plots saved in the project root.")
