"""
LLM-based candidate ranking.

Instead of scoring candidates one-by-one, each prompting technique receives
the full candidate list and is asked to directly rank the top 10.

Techniques compared: Zero-shot, Few-shot, Chat format, Chain of Thought.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GEN_CONFIG_DIR = ROOT / "models" / "qwen-ranking"

from .config import STARRED_IDS


# 1. Prompt builders (listwise)

def _numbered_list(titles: list) -> str:
    """Format titles as a numbered list for the prompt."""
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))


def build_zero_shot(titles: list) -> str:
    return (
        f"Below are {len(titles)} job titles. Identify the top 10 candidates "
        "most likely aspiring to a human resources position.\n"
        "Reply with ONLY the candidate numbers in ranked order, comma-separated "
        "(e.g. 3, 15, 7, ...).\n\n"
        f"{_numbered_list(titles)}\n\n"
        "Top 10 (best to worst):"
    )


def build_few_shot(titles: list) -> str:
    example = (
        "Example — given this short list:\n"
        "1. Aspiring Human Resources Professional\n"
        "2. Software Engineer at Google\n"
        "3. Seeking HR Opportunities\n"
        "Answer: 1, 3\n\n"
    )
    return (
        "Rank the top 10 candidates most suitable for a human resources role.\n"
        "Reply with ONLY the candidate numbers in ranked order, comma-separated.\n\n"
        f"{example}"
        f"Now rank this list:\n{_numbered_list(titles)}\n\n"
        "Top 10 (best to worst):"
    )


def build_chat_messages(titles: list) -> list:
    return [
        {"role": "user",      "content": "I will give you a list of job titles. Your job is to pick the top 10 most relevant for a human resources position and return their numbers in order."},
        {"role": "assistant", "content": "Understood. Please share the list and I will return the top 10 candidate numbers in ranked order, comma-separated."},
        {"role": "user",      "content": f"Here is the list:\n{_numbered_list(titles)}\n\nTop 10 (best to worst):"},
    ]


def build_cot(titles: list) -> str:
    return (
        f"Here are {len(titles)} job titles.\n\n"
        f"{_numbered_list(titles)}\n\n"
        "Step 1 — What keywords indicate an HR-aspiring candidate "
        "(e.g. 'human resources', 'HR', 'aspiring', 'seeking')?\n"
        "Step 2 — Scan the list and identify candidates with those keywords.\n"
        "Step 3 — Rank the top 10 from best to worst match.\n\n"
        "Final answer — top 10 candidate numbers, comma-separated:"
    )


# 2. Model loading

def load_model():
    """Load model & tokenizer, save GenerationConfig for later reuse."""
    print(f"Loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, device_map="auto")

    gen_config = GenerationConfig(do_sample=False, max_new_tokens=100)
    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    gen_config.save_pretrained(str(GEN_CONFIG_DIR))
    print(f"GenerationConfig saved: {GEN_CONFIG_DIR}")

    model.eval()
    return model, tokenizer


# 3. Inference

def _generate(model, tokenizer, messages, max_new_tokens=100):
    """Send messages to the model and return the generated text."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens)
    outputs = model.generate(**inputs, generation_config=gen_config)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# 4. Run all techniques

def rank_with_llm(df, model=None, tokenizer=None):
    """Run each prompting technique once and return top-10 candidate IDs per technique."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    titles = df["job_title_clean"].tolist()
    ids    = df["id"].tolist()
    results = {}

    # Zero-shot
    print("\n── Zero-shot ──")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_zero_shot(titles)}])
    print(f"  {raw[:120]}")
    results["Zero-shot"] = [ids[int(i)-1] for i in re.findall(r"\b(\d+)\b", raw) if 1 <= int(i) <= len(titles)][:10]

    # Few-shot
    print("\n── Few-shot ──")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_few_shot(titles)}])
    print(f"  {raw[:120]}")
    results["Few-shot"] = [ids[int(i)-1] for i in re.findall(r"\b(\d+)\b", raw) if 1 <= int(i) <= len(titles)][:10]

    # Chat
    print("\n── Chat ──")
    raw = _generate(model, tokenizer, build_chat_messages(titles))
    print(f"  {raw[:120]}")
    results["Chat"] = [ids[int(i)-1] for i in re.findall(r"\b(\d+)\b", raw) if 1 <= int(i) <= len(titles)][:10]

    # Chain of Thought
    print("\n── CoT ──")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_cot(titles)}], max_new_tokens=300)
    print(f"  {raw[:120]}")
    results["CoT"] = [ids[int(i)-1] for i in re.findall(r"\b(\d+)\b", raw) if 1 <= int(i) <= len(titles)][:10]

    return results


# 5. Comparison chart

def plot_scores(rankings: dict, out_path=None):
    """
    Bar chart: how many recruiter-starred candidates each technique
    found in its top 10 (higher = better).
    """
    starred = set(STARRED_IDS)
    techniques = list(rankings.keys())
    hits = [sum(1 for cid in rankings[t] if cid in starred) for t in techniques]

    _, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(techniques, hits, color=["#4878cf", "#e8604c", "#6acc65", "#ce7e45"], alpha=0.85)

    for bar, h in zip(bars, hits):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, str(h),
                ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 10)
    ax.axhline(10, color="grey", linestyle="--", linewidth=0.8, label="Max possible (10)")
    ax.set_ylabel("Starred candidates found in top 10")
    ax.set_title(f"Listwise Ranking — {MODEL_ID}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if out_path is None:
        out_path = ROOT / "outputs" / "prompt_comparison.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()
