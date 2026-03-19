import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE, DISPLAY_COLS
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.compare_embeddings import METHODS, visualize

if __name__ == "__main__":
    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)

    for name, fnc in METHODS.items():
        sep = "=" * 55
        print(f"\n{sep}\n  {name}\n{sep}")
        ranked = df.copy()
        ranked["fit"] = fnc(df)
        ranked = ranked.sort_values("fit", ascending=False).reset_index(drop=True)

        print("Top 10 candidates:")
        print(ranked[DISPLAY_COLS].head(10).to_string(index=False))

    visualize(df)
