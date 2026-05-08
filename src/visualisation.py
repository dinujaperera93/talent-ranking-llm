"""
Candidate comparison table — 22 columns × 10 rows.

Embedding (6 cols): top-10 IDs ranked by similarity score blended with
                    connections_norm via rank_candidates (w=0.7).
  TF-IDF | Word2Vec | FastText | GloVe-FT | BERT-FT | E5-small-FT

LLM (16 cols): top-10 IDs from each of 4 prompting techniques × 4 models.
  Qwen-Zero-shot   | Qwen-Few-shot   | Qwen-Chat   | Qwen-CoT
  Gemma-Zero-shot  | Gemma-Few-shot  | Gemma-Chat  | Gemma-CoT
  Llama-Zero-shot  | Llama-Few-shot  | Llama-Chat  | Llama-CoT
  GLM-Zero-shot    | GLM-Few-shot    | GLM-Chat    | GLM-CoT
"""

import gc

import pandas as pd
import torch

from .compare_embeddings import METHODS
from .config import CEREBRAS_MODEL, GEMMA_MODEL, NVIDIA_MODEL, QWEN_MODEL
from .llm_ranking import (
    _generate_cerebras,
    _generate_local,
    _generate_nvidia,
    cerebras_client,
    load_model,
    nvidia_client,
    rank_with_llm,
)
from .ranking import rank_candidates

_LOCAL_MODELS = {
    "Qwen":  QWEN_MODEL,
    "Gemma": GEMMA_MODEL,
}


def build_table(df) -> pd.DataFrame:
    table = {}

    # --- Embedding methods ---
    for name, fn in METHODS.items():
        scored = df.copy()
        scored["fit"] = fn(df)
        ranked = rank_candidates(scored)
        table[name] = ranked["id"].head(10).tolist()

    # --- Local HF models: Qwen (1.5B) and Gemma (2B) ---
    for label, model_id in _LOCAL_MODELS.items():
        print(f"\nLoading {label} ({model_id}) locally …")
        model, tokenizer = load_model(model_id)
        results = rank_with_llm(
            df,
            generate_fn=lambda msgs, max_tok, m=model, t=tokenizer: _generate_local(m, t, msgs, max_tok),
        )
        for technique, ids in results.items():
            table[f"{label}-{technique}"] = ids
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {label} unloaded.")

    # --- Llama 3.1 8B: Cerebras free API ---
    print(f"\nCalling Cerebras API ({CEREBRAS_MODEL}) …")
    _cerebras = cerebras_client()
    results = rank_with_llm(
        df,
        generate_fn=lambda msgs, max_tok: _generate_cerebras(_cerebras, msgs, max_tok),
    )
    for technique, ids in results.items():
        table[f"Llama-{technique}"] = ids

    # --- GLM-4-9B: NVIDIA NIM API ---
    print(f"\nCalling NVIDIA NIM API ({NVIDIA_MODEL}) …")
    _nvidia = nvidia_client()
    results = rank_with_llm(
        df,
        generate_fn=lambda msgs, max_tok: _generate_nvidia(_nvidia, msgs, max_tok),
    )
    for technique, ids in results.items():
        table[f"GLM-{technique}"] = ids

    tbl = pd.DataFrame(table, index=range(1, 11))
    tbl.index.name = "Rank"
    return tbl
