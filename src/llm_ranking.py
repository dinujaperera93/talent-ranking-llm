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
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")
from .config import GEMMA_MODEL

MODEL_ID = GEMMA_MODEL    # default model; override via load_model(model_id=...)


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

def load_model(model_id: str = None):
    """
    Load model & tokenizer.
    model_id defaults to MODEL_ID (Gemma-4-E2B-it).
    Pass a different model_id to load any other HF model (e.g. Qwen).
    """
    if model_id is None:
        model_id = MODEL_ID
    print(f"Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    slug = model_id.split("/")[-1].lower()
    config_dir = ROOT / "models" / f"{slug}-ranking"
    gen_config = GenerationConfig(do_sample=False, max_new_tokens=100)
    config_dir.mkdir(parents=True, exist_ok=True)
    gen_config.save_pretrained(str(config_dir))
    print(f"  GenerationConfig saved: {config_dir}")

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

def _parse_ids(raw: str, ids: list, n_titles: int) -> list:
    """
    Extract up to 10 valid candidate IDs from a model response.
    Pads with None if fewer than 10 valid numbers found, so every
    column in the comparison table always has exactly 10 rows.
    """
    found = [
        ids[int(i) - 1]
        for i in re.findall(r"\b(\d+)\b", raw)
        if 1 <= int(i) <= n_titles
    ][:10]
    return found + [None] * (10 - len(found))


def rank_with_llm(df, model=None, tokenizer=None):
    """Run each prompting technique once and return top-10 candidate IDs per technique."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    titles = df["job_title_clean"].tolist()
    ids    = df["id"].tolist()
    results = {}

    print("\n─ Zero-shot ─")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_zero_shot(titles)}])
    print(f"  {raw[:120]}")
    results["Zero-shot"] = _parse_ids(raw, ids, len(titles))

    print("\n─ Few-shot ─")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_few_shot(titles)}])
    print(f"  {raw[:120]}")
    results["Few-shot"] = _parse_ids(raw, ids, len(titles))

    print("\n─ Chat ─")
    raw = _generate(model, tokenizer, build_chat_messages(titles))
    print(f"  {raw[:120]}")
    results["Chat"] = _parse_ids(raw, ids, len(titles))

    print("\n─ CoT ─")
    raw = _generate(model, tokenizer, [{"role": "user", "content": build_cot(titles)}], max_new_tokens=300)
    print(f"  {raw[:120]}")
    results["CoT"] = _parse_ids(raw, ids, len(titles))

    return results