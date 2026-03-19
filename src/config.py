DATA_FILE = "data/PotentialTalents.csv"
TARGET_KEYWORDS = ["aspiring human resources", "seeking human resources"]
CONNECTIONS_MAX = 500
DISPLAY_COLS = ["id", "job_title", "connections_norm", "fit"]

# Ground truth: candidates starred by the recruiter
STARRED_IDS = [
    1, 3, 6, 7, 9, 10, 14, 15, 17, 19, 21, 24, 25, 27, 28, 29, 30,
    31, 33, 36, 37, 39, 40, 44, 46, 49, 50, 52, 53, 57, 58, 60, 62,
    66, 72, 73, 75, 76, 79, 82, 97, 99, 100,
]

# Embedding models
BERT_MODEL = "all-MiniLM-L6-v2"
E5_MODEL = "intfloat/e5-small-v2"
GLOVE_MODEL = "glove-wiki-gigaword-100"

# Hyperparameters (confirmed optimal by grid search)
RANK_WEIGHT = 0.7    # title similarity weight in rank_candidates
RERANK_ALPHA = 0.9   # starred-similarity blend weight in rerank
