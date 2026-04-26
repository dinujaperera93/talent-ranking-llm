"""
Candidate comparison table — 18 columns × 10 rows.

Embedding (6 cols): top-10 IDs ranked by similarity score blended with
                    connections_norm via rank_candidates (w=0.7).
  TF-IDF | Word2Vec | FastText | GloVe-FT | BERT-FT | E5-small-FT

LLM (12 cols): top-10 IDs from each of 4 prompting techniques × 3 models.
  Qwen-Zero-shot   | Qwen-Few-shot   | Qwen-Chat   | Qwen-CoT
  Gemma-Zero-shot  | Gemma-Few-shot  | Gemma-Chat  | Gemma-CoT
  Llama-Zero-shot  | Llama-Few-shot  | Llama-Chat  | Llama-CoT
"""

import gc

import pandas as pd
import torch

from .compare_embeddings import METHODS
from .config import CEREBRAS_MODEL, GEMMA_MODEL, QWEN_MODEL
from .llm_ranking import (
    _generate_cerebras,
    _generate_local,
    cerebras_client,
    load_model,
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

    # --- Llama 3.3 70B: Cerebras free API ---
    print(f"\nCalling Cerebras API ({CEREBRAS_MODEL}) …")
    _cerebras = cerebras_client()
    results = rank_with_llm(
        df,
        generate_fn=lambda msgs, max_tok: _generate_cerebras(_cerebras, msgs, max_tok),
    )
    for technique, ids in results.items():
        table[f"Llama-{technique}"] = ids

    tbl = pd.DataFrame(table, index=range(1, 11))
    tbl.index.name = "Rank"
    return tbl
