from pathlib import Path
from config import DATA_FILE
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import add_features
from src.ranking import rank_candidates
from src.reranking import rerank
from src.evaluation import ndcg_at_k

STARRED_IDS = [1, 3, 6]
COLS = ["id", "job_title", "location", "connections_raw", "fit"]

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent

    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)
    df = add_features(df)
    df = rank_candidates(df)

    print("Initial ranking:")
    print(df[COLS].head(10).to_string(index=False))

    df_reranked = rerank(df, STARRED_IDS)

    print("\nAfter re-ranking:")
    print(df_reranked[COLS].head(10).to_string(index=False))

    print(f"\nNDCG@10 before: {ndcg_at_k(df, STARRED_IDS):.4f}")
    print(f"NDCG@10 after : {ndcg_at_k(df_reranked, STARRED_IDS):.4f}")
