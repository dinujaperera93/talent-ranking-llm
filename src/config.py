DATA_FILE = "data/PotentialTalents.csv"
TARGET_KEYWORDS = ["aspiring human resources", "seeking human resources"]
CONNECTIONS_MAX = 500

# Ground truth: candidates starred by the recruiter
STARRED_IDS = [27, 28, 29, 30, 40, 10, 62, 53, 75, 97, 73, 58, 33, 17, 3,
    21, 46, 99, 100, 24]

# Embedding models
BERT_MODEL = "all-MiniLM-L6-v2"
E5_MODEL = "intfloat/e5-small-v2"
GLOVE_MODEL = "glove-wiki-gigaword-100"

# LLM models
QWEN_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"  # loaded locally
GEMMA_MODEL  = "google/gemma-4-E2B-it"         # loaded locally
CEREBRAS_MODEL = "llama3.1-8b"                 # Cerebras free API

# Hyperparameters (confirmed optimal by grid search)
RANK_WEIGHT  = 0.7   # title similarity weight in rank_candidates
RERANK_ALPHA = 0.9   # starred-similarity blend weight in rerank
