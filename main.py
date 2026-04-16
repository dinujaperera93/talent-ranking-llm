import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.visualisation import build_table

if __name__ == "__main__":
    df = load_data(ROOT / DATA_FILE)
    df = preprocess(df)

    # Candidate comparison table (all embedding methods + LLMs)
    sep = "***************"
    print(f"\n{sep}\n  Candidate Comparison Table\n{sep}")
    tbl = build_table(df)
    print("\n" + tbl.to_string())
