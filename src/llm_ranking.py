"""
LLM-based candidate ranking using a small instruct model.

The model scores each candidate's job title (0-10) based on HR fit.
GenerationConfig is saved to models/qwen-ranking/generation_config.json
so the same settings can be reloaded without re-specifying them.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ── env & paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GEN_CONFIG_DIR = ROOT / "models" / "qwen-ranking"

# prompt options

# --- Option 1: FEW-SHOT ---
# Show the model input→output examples before the real question.
# The model learns the expected format and score range from the examples.
#
# PROMPT_TEMPLATE = """Score a job title for Human Resources fit (0.0 to 1.0).
#
# Job title: Aspiring Human Resources Professional
# Score: 1.0
#
# Job title: Software Engineer at Google
# Score: 0.0
#
# Job title: Seeking Human Resources Opportunities
# Score: 0.9
#
# Job title: {title}
# Score:"""


# --- Option 2: CHAT FORMAT ---
# Pass few-shot examples as alternating user/assistant turns.
# The model sees its own prior "replies", which strongly anchors the format.
# Use with: tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#
# def build_chat_messages(title: str) -> list:
#     return [
#         {"role": "user",      "content": "Score for HR fit (0.0 to 1.0).\nJob title: Aspiring Human Resources Professional"},
#         {"role": "assistant", "content": "1.0"},
#         {"role": "user",      "content": "Score for HR fit (0.0 to 1.0).\nJob title: Software Engineer at Google"},
#         {"role": "assistant", "content": "0.0"},
#         {"role": "user",      "content": "Score for HR fit (0.0 to 1.0).\nJob title: Seeking Human Resources Opportunities"},
#         {"role": "assistant", "content": "0.9"},
#         {"role": "user",      "content": f"Score for HR fit (0.0 to 1.0).\nJob title: {title}"},
#     ]


# --- Option 3: CHAIN OF THOUGHT ---
# Ask the model to reason step-by-step before giving the final score.
# Note: set max_new_tokens=150 to give room for the reasoning.
#
# PROMPT_TEMPLATE = """Score this job title for Human Resources fit (0.0 to 1.0).
# Think step by step, then give the final score.
#
# Job title: {title}
#
# Step 1 - Does the title mention HR keywords (human resources, HR, talent, recruiting)?
# Step 2 - Does it suggest the person is aspiring/seeking (not already senior)?
# Step 3 - Based on steps 1-2, what decimal score (0.0 to 1.0) is appropriate?
# Score:"""


# --- ACTIVE PROMPT (simple baseline) ---
# Plain instruction with output lead. Fast, no extra tokens.
PROMPT_TEMPLATE = """Score from 0.0 to 1.0 how well this job title matches someone aspiring and seeking a human resources position.
Reply with ONLY a decimal number (e.g. 0.85), nothing else.

Job title: {title}
Score:"""


# generation config
def _save_generation_config() -> GenerationConfig:
    config = GenerationConfig(
        max_new_tokens=10,  # enough for a decimal like "0.85"
        do_sample=False,    # greedy — deterministic, reproducible scores
    )
    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(GEN_CONFIG_DIR))
    print(f"GenerationConfig saved → {GEN_CONFIG_DIR / 'generation_config.json'}")
    return config


def _load_generation_config() -> GenerationConfig:
    """Load from JSON if it exists, otherwise create and save it."""
    if (GEN_CONFIG_DIR / "generation_config.json").exists():
        print(f"GenerationConfig loaded from {GEN_CONFIG_DIR}")
        return GenerationConfig.from_pretrained(str(GEN_CONFIG_DIR))
    return _save_generation_config()


# scoring
def _score_title(model, tokenizer, gen_config, title: str) -> float:
    prompt = PROMPT_TEMPLATE.format(title=title)
    # apply Qwen chat template so instruct formatting is correct
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, generation_config=gen_config)

    # decode only the newly generated tokens (skip the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    match = re.search(r"\d+\.?\d*", generated)
    score = float(match.group()) if match else 0.0
    return min(max(score, 0.0), 1.0)


# main ranking function
def rank_with_llm(df):
    """Return df sorted by LLM fit score (column `fit_llm`)."""
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN, device_map="auto"
    )
    gen_config = _load_generation_config()

    titles = df["job_title_clean"].tolist()
    scores = []
    for i, title in enumerate(titles, 1):
        scores.append(_score_title(model, tokenizer, gen_config, title))
        if i % 10 == 0:
            print(f"  scored {i}/{len(titles)}")

    df = df.copy()
    df["fit_llm"] = scores
    return df.sort_values("fit_llm", ascending=False).reset_index(drop=True)


def plot_scores(df, out_path=None):
    """Horizontal bar chart of LLM fit scores, colored by recruiter-starred status.

    Same color scheme as the PCA/t-SNE plots:
      green = recruiter-starred (relevant)
      light blue = non-relevant
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from .config import STARRED_IDS

    ranked = df.sort_values("fit_llm", ascending=True)  # ascending so top is at chart top
    colors = ["#2ca02c" if cid in STARRED_IDS else "#aec7e8" for cid in ranked["id"]]

    _, ax = plt.subplots(figsize=(8, max(6, len(ranked) * 0.18)))
    ax.barh(range(len(ranked)), ranked["fit_llm"], color=colors)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(
        [t[:45] + "…" if len(t) > 45 else t for t in ranked["job_title_clean"]],
        fontsize=7,
    )
    ax.set_xlabel("LLM fit score (0.0 – 1.0)")
    ax.set_title(f"LLM Ranking — {MODEL_ID}", fontweight="bold")
    ax.set_xlim(0, 1)

    legend = [
        mpatches.Patch(color="#2ca02c", label="Relevant candidate (starred)"),
        mpatches.Patch(color="#aec7e8", label="Non-relevant candidate"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    if out_path is None:
        out_path = ROOT / "llm_scores.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()