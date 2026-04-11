import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE, DISPLAY_COLS
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.compare_embeddings import METHODS, visualize, plot_all_scores
from src.llm_ranking import load_model, rank_with_llm, plot_scores

if __name__ == "__main__":
    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)

    # Embedding methods
    all_scores = {}
    for name, fnc in METHODS.items():
        sep = "*" * 10
        print(f"\n{sep}\n  {name}\n{sep}")
        scores = fnc(df)
        all_scores[name] = scores
        ranked = df.copy()
        ranked["fit"] = scores
        ranked = ranked.sort_values("fit", ascending=False).reset_index(drop=True)
        print("Top 10 candidates:")
        print(ranked[DISPLAY_COLS].head(10).to_string(index=False))
    visualize(df)
    plot_all_scores(df, all_scores)

    # LLM listwise ranking — all 4 prompting techniques
    print(f"\n{sep}\n  LLM (Qwen2.5-0.5B-Instruct)\n{sep}")
    model, tokenizer = load_model()

    rankings = rank_with_llm(df, model=model, tokenizer=tokenizer)
    plot_scores(rankings, ROOT / "outputs" / "prompt_comparison.png")
