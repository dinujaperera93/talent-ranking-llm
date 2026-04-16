"""
Candidate comparison table — 14 columns × 10 rows.

Embedding (6 cols): top-10 IDs ranked by similarity score blended with
                    connections_norm via rank_candidates (w=0.7).
  TF-IDF | Word2Vec | FastText | GloVe-FT | BERT-FT | E5-small-FT

LLM (8 cols): top-10 IDs from each of 4 prompting techniques × 2 models.
  Qwen-Zero-shot | Qwen-Few-shot | Qwen-Chat | Qwen-CoT
  Gemma-Zero-shot | Gemma-Few-shot | Gemma-Chat | Gemma-CoT
"""

import gc
import pandas as pd
import torch

from .compare_embeddings import METHODS
from .config import QWEN_MODEL, GEMMA_MODEL
from .ranking import rank_candidates
from .llm_ranking import load_model, rank_with_llm

LLM_MODELS = {
    "Qwen":  QWEN_MODEL,
    "Gemma": GEMMA_MODEL,
}


def build_table(df) -> pd.DataFrame:
    table = {}

    # --- Embedding methods (similarity score + connections blend) ---
    for name, fn in METHODS.items():
        scored = df.copy()
        scored["fit"] = fn(df)
        ranked = rank_candidates(scored)
        table[name] = ranked["id"].head(10).tolist()

    # --- LLM methods: all 4 prompts per model, load/unload sequentially ---
    for model_label, model_id in LLM_MODELS.items():
        print(f"\nLoading {model_label} ({model_id}) …")
        model, tokenizer = load_model(model_id=model_id)

        results = rank_with_llm(df, model=model, tokenizer=tokenizer)
        for technique, ids in results.items():
            table[f"{model_label}-{technique}"] = ids

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {model_label} unloaded.")

    tbl = pd.DataFrame(table, index=range(1, 11))
    tbl.index.name = "Rank"
    return tbl
