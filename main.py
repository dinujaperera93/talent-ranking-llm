import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE, DISPLAY_COLS
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.compare_embeddings import METHODS, visualize, plot_all_scores
from src.llm_ranking import rank_with_llm

if __name__ == "__main__":
    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)

    all_scores = {}
    for name, fnc in METHODS.items():
        sep = "=" * 55
        print(f"\n{sep}\n  {name}\n{sep}")
        scores = fnc(df)
        all_scores[name] = scores
        ranked = df.copy()
        ranked["fit"] = scores
        ranked = ranked.sort_values("fit", ascending=False).reset_index(drop=True)

        print("Top 10 candidates:")
        print(ranked[DISPLAY_COLS].head(10).to_string(index=False))

    visualize(df)

    sep = "=" * 55
    print(f"\n{sep}\n  LLM (Qwen2.5-0.5B-Instruct)\n{sep}")
    llm_ranked = rank_with_llm(df)
    print(llm_ranked[["id", "job_title", "fit_llm"]].head(20).to_string(index=False))
    all_scores["LLM"] = llm_ranked.set_index("id").reindex(df["id"])["fit_llm"].values

    plot_all_scores(df, all_scores)
