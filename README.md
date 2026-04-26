# Talent Spotting & Candidate Ranking System

> An end-to-end NLP pipeline that benchmarks six embedding strategies — from classical bag-of-words to fine-tuned transformers — augmented with three large language models and a human-in-the-loop feedback mechanism, to automatically surface the most relevant HR candidates from a raw applicant pool.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Pipeline: Step by Step](#3-project-pipeline-step-by-step)
   - [Step 1: Data Loading](#step-1-data-loading)
   - [Step 2: Data Cleaning and Preprocessing](#step-2-data-cleaning-and-preprocessing)
   - [Step 3: Embedding and Similarity Scoring](#step-3-embedding-and-similarity-scoring)
   - [Step 4: Candidate Ranking](#step-4-candidate-ranking)
   - [Step 5: Re-ranking with Recruiter Feedback](#step-5-re-ranking-with-recruiter-feedback)
   - [Step 6: LLM-based Listwise Ranking](#step-6-llm-based-listwise-ranking)
   - [Step 7: Evaluation](#step-7-evaluation)
   - [Step 8: Candidate Comparison Table](#step-8-candidate-comparison-table)
4. [Embedding Methods Compared](#4-embedding-methods-compared)
5. [LLM Models Compared](#5-llm-models-compared)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Project Structure](#7-project-structure)
8. [How to Run](#8-how-to-run)
9. [Results](#9-results)
10. [Concluding Remarks for Hiring Managers and Recruiters](#10-concluding-remarks-for-hiring-managers-and-recruiters)

---

## 1. Project Overview

This project builds an **intelligent candidate ranking system** for a real recruiting use case. Given a dataset of 104 candidates with free-text job titles and LinkedIn connection counts, the system automatically ranks them by how well they fit the profile of someone *aspiring to or seeking a Human Resources role*.

The core contribution is a **systematic benchmark of six embedding strategies** (TF-IDF to fine-tuned transformers) combined with **two LLMs across four prompting techniques**, all evaluated against recruiter-provided ground truth. A feedback-driven re-ranking loop improves results as recruiters interact with the system.

Every hyperparameter in this pipeline was validated by grid search. Every design decision is justified in code comments and documentation.

---

## 2. Problem Statement

A recruiter has a pool of candidates and two target search phrases:

- `"aspiring human resources"`
- `"seeking human resources"`

**Goal:** Automatically rank every candidate by fit for an HR role, placing the most relevant candidates at the top of the list.

**Challenges addressed:**

| Challenge | How it is handled |
|---|---|
| Varied phrasing across job titles | Semantic similarity via transformer embeddings |
| Synonym mismatch (`"HR"` vs `"Human Resources"`) | Dense vector spaces that encode meaning, not just tokens |
| Long titles diluting keyword signals | MAX cosine similarity across targets instead of average |
| Connection count as a noisy secondary signal | Normalized and blended at a grid-search-validated weight |
| Static rankings that do not improve over time | Human-in-the-loop re-ranking from recruiter-starred feedback |

---

## 3. Project Pipeline: Step by Step

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│  Data Loading   │  data_loader.py
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Cleaning & Preprocessing   │  preprocessing.py
│  - Clean job titles         │
│  - Parse connection counts  │
│  - Normalize connections    │
└────────────┬────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Embedding & Similarity Scoring        │  compare_embeddings.py
│  (6 methods benchmarked)               │
│  TF-IDF → Word2Vec → FastText →        │
│  GloVe-FT → BERT-FT → E5-small-FT     │
└────────────┬───────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Candidate Ranking                   │  ranking.py
│  fit = 0.7 × title_sim               │
│       + 0.3 × connections_norm       │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Re-ranking with Recruiter Feedback  │  reranking.py
│  Blend similarity-to-starred into    │
│  the fit score (α = 0.9)             │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│  LLM Listwise Ranking                         │  llm_ranking.py
│  Qwen2.5-1.5B    ×  4 prompting techniques   │
│  Gemma-4-E2B-it  ×  4 prompting techniques   │
│  Llama-3.1-8B   ×  4 prompting techniques   │
│  Zero-shot | Few-shot | Chat | CoT            │
└────────────┬─────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Candidate Comparison Table          │  visualisation.py
│  18 columns × 10 rows                │
│  (6 embedding + 12 LLM results)      │
│  Saved to outputs/candidate_table.csv│
└──────────────────────────────────────┘
```

---

### Step 1: Data Loading

**File:** `src/data_loader.py`

The raw dataset (`data/PotentialTalents.csv`) contains 104 candidate records with:

| Column | Description |
|---|---|
| `id` | Unique candidate identifier |
| `job_title` | Free-text LinkedIn-style job title |
| `connection` | LinkedIn connection count (may include `"500+"`) |
| `fit` | Target column; initially empty, filled by the pipeline |

---

### Step 2: Data Cleaning and Preprocessing

**File:** `src/preprocessing.py`

| Operation | Implementation detail |
|---|---|
| Job title cleaning | Preserved as-is; BERT and E5 handle casing and punctuation internally |
| Connection parsing | `"500+"` is capped at 500; invalid or missing values default to 0 |
| Connection normalization | Linearly scaled to `[0, 1]` by dividing by 500 |

---

### Step 3: Embedding and Similarity Scoring

**File:** `src/compare_embeddings.py`

Each of the six methods follows an identical contract:

1. Embed all candidate job titles → matrix `(104 × dim)`
2. Embed the two target keywords → matrix `(2 × dim)`
3. Compute cosine similarity between every candidate and every target
4. Score = **MAX** similarity across both targets

Using MAX ensures a candidate with one highly relevant phrase is not penalized for surrounding context.

---

### Step 4: Candidate Ranking

**File:** `src/ranking.py`

```
fit = 0.7 × title_similarity + 0.3 × connections_norm
```

The `0.7 / 0.3` split was confirmed optimal by grid search. Title relevance dominates; connections act as a principled tie-breaker.

---

### Step 5: Re-ranking with Recruiter Feedback

**File:** `src/reranking.py`

When a recruiter stars candidates they approve of, those selections become a live feedback signal:

1. Encode all titles and all starred titles using `all-MiniLM-L6-v2`
2. Compute each candidate's mean cosine similarity to the starred set
3. Blend into the existing fit score:

```
fit_new = (1 - α) × fit_old + α × starred_similarity     [α = 0.9]
```

`α = 0.9` was validated by grid search.

---

### Step 6: LLM-based Listwise Ranking

**File:** `src/llm_ranking.py`

Instead of scoring candidates one-by-one, the full candidate list is given to the model and it is asked to directly select the top 10 HR-relevant candidates. Three models are compared across four prompting techniques:

**Models:**

| Model | Parameters | Access |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Local HF model (~3 GB bfloat16) |
| `google/gemma-4-E2B-it` | ~2.3B effective / 5.1B total | Local HF model (~15 GB bfloat16) |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Cerebras free API |

**Prompting techniques:**

| Technique | Description |
|---|---|
| Zero-shot | Plain instruction, no examples |
| Few-shot | Small worked example shown before the question |
| Chat | Framed as a user ↔ assistant conversation |
| CoT | Model reasons step-by-step before answering |

Models are loaded and unloaded sequentially to stay within CPU RAM. Output is parsed with regex; responses with fewer than 10 valid numbers are padded with `None`.

---

### Step 7: Evaluation

**File:** `src/evaluation.py`

Rankings are evaluated with **NDCG@k** (Normalized Discounted Cumulative Gain):

```
DCG@k  = Σ(i=1 to k)  relevance_i / log₂(i + 2)
IDCG@k = DCG of the perfect ranking (all 20 starred candidates in positions 1–20)
NDCG@k = DCG@k / IDCG@k          (range: 0.0 to 1.0)
```

---

### Step 8: Candidate Comparison Table

**File:** `src/visualisation.py`

Produces an **18-column × 10-row DataFrame** showing the top-10 candidate IDs selected by each method:

| Columns | Source |
|---|---|
| TF-IDF, Word2Vec, FastText, GloVe-FT, BERT-FT, E5-small-FT | Embedding similarity + connections blend |
| Qwen-Zero-shot, Qwen-Few-shot, Qwen-Chat, Qwen-CoT | Qwen2.5-1.5B-Instruct |
| Gemma-Zero-shot, Gemma-Few-shot, Gemma-Chat, Gemma-CoT | Gemma-4-E2B-it |
| Llama-Zero-shot, Llama-Few-shot, Llama-Chat, Llama-CoT | Llama-3.1-8B (Cerebras) |

Saved to `outputs/candidate_table.csv`.

---

## 4. Embedding Methods Compared

| Method | Corpus | Vector dim | Key characteristic |
|---|---|---|---|
| **TF-IDF** | Our data only | ~312 | Bag-of-words; cannot handle synonyms |
| **Word2Vec** | Our data only | 100 | Dense word semantics; small corpus limits quality |
| **FastText** | Our data only | 100 | Subword n-grams; still bottlenecked by ~100-title corpus |
| **GloVe + Mittens** | Wikipedia + fine-tuned | 100 | Pretrained co-occurrences, nudged toward job-title vocabulary |
| **BERT + SimCSE** | Pretrained + fine-tuned | 384 | Full sentence context; domain-adapted with SimCSE |
| **E5-small + SimCSE** | Pretrained + fine-tuned | 384 | Retrieval-optimized; asymmetric query/passage encoding |

---

## 5. LLM Models Compared

| Model | Size | Prompting | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Zero-shot, Few-shot, Chat, CoT | Lightweight local model; loaded and unloaded from CPU RAM |
| `google/gemma-4-E2B-it` | 2.3B eff / 5.1B total | Zero-shot, Few-shot, Chat, CoT | Stronger instruction following; fits in ~15 GB bfloat16 on CPU |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Zero-shot, Few-shot, Chat, CoT | Served via Cerebras free API; no local memory needed |

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| SimCSE fine-tuning for BERT and E5 | Domain-adapts pretrained transformers without labelled data |
| Mittens fine-tuning for GloVe | Bridges the Wikipedia-to-job-title vocabulary gap at low compute cost |
| MAX cosine similarity (not average) | Prevents long titles from diluting a single highly relevant phrase |
| `w = 0.7` for ranking blend | Grid-search validated; title relevance dominates over connections |
| `α = 0.9` for re-ranking blend | Grid-search validated; starred-candidate similarity is the dominant signal |
| Listwise LLM ranking | More natural than pointwise scoring; models the full ranking task directly |
| Sequential model loading | Qwen unloaded before Gemma loads; both unloaded before Cerebras API call, to stay within CPU RAM budget |
| `_parse_ids` padding to 10 | Ensures DataFrame consistency when a model returns fewer than 10 valid IDs |
| Greedy decoding | Ensures deterministic, reproducible rankings across runs |

---

## 7. Project Structure

```
.
├── main.py                     # Entry point: runs the full pipeline
├── requirements.toml           # All dependencies with version pins
├── data/
│   └── PotentialTalents.csv    # Raw candidate dataset (104 records)
├── models/
│   ├── qwen2.5-1.5b-instruct-ranking/
│   │   └── generation_config.json
│   └── gemma-4-e2b-it-ranking/
│       └── generation_config.json
├── outputs/
│   ├── embedding_space_pca.png
│   ├── embedding_space_tsne.png
│   └── candidate_table.csv     # 18-column comparison table
├── src/
│   ├── config.py               # Paths, model IDs, hyperparameters, ground truth
│   ├── data_loader.py          # CSV ingestion
│   ├── preprocessing.py        # Title cleaning, connection parsing and normalization
│   ├── compare_embeddings.py   # All 6 embedding methods + PCA/t-SNE visualizations
│   ├── ranking.py              # Blended fit scoring (title similarity + connections)
│   ├── reranking.py            # Human-in-the-loop re-ranking via recruiter feedback
│   ├── llm_ranking.py          # LLM listwise ranking (Qwen + Gemma + Llama-3.1, 4 prompt techniques)
│   ├── visualisation.py        # Candidate comparison table (18 columns × 10 rows)
│   ├── evaluation.py           # NDCG@k evaluation metric
│   └── analysis.py             # Jupytext notebook source
└── notebooks/
    └── analysis.ipynb          # Interactive exploration and result inspection
```

---

## 8. How to Run

**1. Install dependencies**

```bash
pip install -r requirements.toml
```

**2. Set API keys** (required for model access)

```bash
echo "HUGGING_FACE_API_KEY=your_hf_token_here" > .env
echo "CEREBRAS_API_KEY=your_cerebras_key_here" >> .env
```

**3. Run the full pipeline**

```bash
python main.py
```

This will:
- Load and preprocess the 104-candidate dataset
- Run all six embedding methods
- Load Qwen2.5-1.5B → run 4 prompting techniques → unload
- Load Gemma-4-E2B-it → run 4 prompting techniques → unload
- Call Cerebras API (Llama-3.1-8B) → run 4 prompting techniques
- Print and save the 18-column candidate comparison table to `outputs/candidate_table.csv`

**4. Open the notebook**

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 9. Results

### Embedding Space: PCA
![PCA](outputs/embedding_space_pca.png)

### Embedding Space: t-SNE
![t-SNE](outputs/embedding_space_tsne.png)

### Candidate Comparison Table

The table (`outputs/candidate_table.csv`) shows the top-10 candidate IDs selected by each of the 14 methods side by side, enabling direct comparison of which candidates each approach surfaces and where they agree or diverge.

---

## 10. Concluding Remarks for Hiring Managers and Recruiters

This is an **original end-to-end NLP engineering investigation**, covering the full machine learning development lifecycle with deliberate, documented decisions at every stage.

**Comparative analysis across eighteen methods.** Six embedding methods and three LLMs (each tested with four prompting techniques) are benchmarked systematically, not just implemented. Each method's failure modes are explained in terms of the specific characteristics of this dataset.

**Principled, data-validated engineering.** Every hyperparameter (`w = 0.7`, `α = 0.9`) was confirmed by grid search. Generation configs are serialized for reproducibility. Code is modular with single-responsibility source files.

**Human-in-the-loop architecture.** The re-ranking module reflects an understanding that production ML systems are not static. Rankings improve as recruiters star candidates, turning the system into a feedback loop.

**Practical LLM integration.** Demonstrates hands-on experience beyond API calls: chat template formatting, four prompt strategies (zero-shot, few-shot, chat, chain-of-thought), sequential memory-safe model loading/unloading, and robust output parsing.

---

### Skills demonstrated

`Python` · `NLP` · `Information Retrieval` · `Sentence Transformers` · `Hugging Face Transformers` · `scikit-learn` · `gensim` · `TF-IDF` · `Word2Vec` · `FastText` · `GloVe` · `Mittens` · `BERT` · `E5` · `SimCSE` · `LLM Prompting` · `Qwen2.5` · `Gemma 4` · `Llama 3.3` · `Cerebras` · `NDCG` · `PCA` · `t-SNE` · `Matplotlib` · `Pandas` · `NumPy` · `Reproducible ML`

---

*Built as a deep-dive into NLP-based candidate ranking, exploring the practical trade-offs between classical embedding methods, fine-tuned transformers, and large language model listwise ranking in a real recruiting context.*
