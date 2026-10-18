"""
Dataset preparation for Shiba Coder 27B Max.

Loads three public Hugging Face datasets, normalizes them to a unified
``conversations`` schema (user / assistant / tool), applies the
``qwen3-thinking`` chat template, filters by token length, and writes
train/eval artifacts that ``src/train.py`` can consume.

Usage:
    uv run python data/prepare_dataset.py

Inputs (all public, no authentication required):
    * r0b0tlab/qwen3.8-max-distillation-50k
    * guell00/qwen-3.8-code
    * TIGER-Lab/SWE-Next-SFT-Trajectories

Outputs:
    data/processed/train/   (HF dataset saved with ``save_to_disk``)
    data/processed/eval/
"""

import multiprocessing as mp
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template

# ---------------------------------------------------------------------------
# Configuration (mirrors configs/lora_config.yaml)
# ---------------------------------------------------------------------------
RANDOM_SEED = 12181531
MAX_CONTEXT_WINDOW = 8192
TEST_SIZE = 0.02
BASE_MODEL = "prism-ml/Bonsai-27B-unpacked"
CHAT_TEMPLATE = "qwen3-thinking"

NUM_EXAMPLES = {
    "qwen_38_data_set": 5000,
    "qwen_38_code": 10000,
    "swe_next": 2000,
}

ALLOWED_DOMAINS = {"code", "reasoning"}
OUT_DIR = Path(__file__).resolve().parent / "processed"

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_strip(data):
    """Trim whitespace; treat ``None`` as an empty string."""
    return (data or "").strip()


def normalize_assistant_to_think_solution(text: str) -> str:
    """Force every assistant turn into the ``qwen3-thinking`` shape.

    * empty content                  -> ``<think></think>\\n``
    * already has ``<think>…</think>`` -> keep block, append the rest
    * otherwise                      -> prepend an empty ``<think></think>``
    """
    text = _clean_strip(text)

    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end() :].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"

    return f"<think></think>\n{text}".rstrip()


# ---------------------------------------------------------------------------
# Per-dataset normalizers
# ---------------------------------------------------------------------------
def format_qwen(examples):
    """Normalize Qwen-style ``messages`` -> ``conversations``."""
    out = []
    for messages in examples["messages"]:
        cleaned = []
        for message in messages:
            role = message.get("role")
            content = _clean_strip(message.get("content", ""))

            if role == "system":
                continue
            if role == "assistant":
                content = normalize_assistant_to_think_solution(content)
            if role in ("user", "assistant"):
                cleaned.append({"role": role, "content": content})

        if len(cleaned) >= 2:
            out.append(cleaned)

    return {"conversations": out}


def format_swe(examples):
    """Normalize SWE-Next style ``messages`` -> ``conversations`` (keeps tool turns)."""
    out = []
    for messages in examples["messages"]:
        cleaned = []
        for message in messages:
            role = message.get("role")
            content = _clean_strip(message.get("content", ""))

            if role == "system":
                continue
            if role in ("user", "assistant", "tool"):
                cleaned.append({"role": role, "content": content})

        if len(cleaned) >= 2:
            out.append(cleaned)

    return {"conversations": out}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_sample(name, sample_count=None, domains=None, split="train", subset=None):
    data_set = (
        load_dataset(name, subset, split=split)
        if subset
        else load_dataset(name, split=split)
    )

    if domains is not None:
        data_set = data_set.filter(lambda ex: ex["domain"] in domains)

    if sample_count is not None:
        sample_count = min(sample_count, len(data_set))
        data_set = data_set.shuffle(seed=RANDOM_SEED).select(range(sample_count))

    return data_set


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = get_chat_template(
        AutoTokenizer.from_pretrained(BASE_MODEL),
        chat_template=CHAT_TEMPLATE,
    )
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    # 1. General reasoning + code (Qwen3.8-Max distillation)
    qwen_38_data_set = load_sample(
        "r0b0tlab/qwen3.8-max-distillation-50k",
        NUM_EXAMPLES["qwen_38_data_set"],
        domains=ALLOWED_DOMAINS,
    )

    # 2. Specialized code dataset
    qwen_38_code = load_sample(
        "guell00/qwen-3.8-code",
        NUM_EXAMPLES["qwen_38_code"],
    )

    # 3. Software engineering / agent trajectories
    swe_next = load_sample(
        "TIGER-Lab/SWE-Next-SFT-Trajectories",
        NUM_EXAMPLES["swe_next"],
    )

    # Normalize each source to the unified `conversations` schema
    qwen_38_data_set = qwen_38_data_set.map(
        format_qwen, batched=True, remove_columns=qwen_38_data_set.column_names
    )
    qwen_38_code = qwen_38_code.map(
        format_qwen, batched=True, remove_columns=qwen_38_code.column_names
    )
    swe_next = swe_next.map(
        format_swe, batched=True, remove_columns=swe_next.column_names
    )

    # Drop empty conversations
    qwen_38_data_set = qwen_38_data_set.filter(
        lambda x: x["conversations"] is not None and len(x["conversations"]) > 0
    )
    qwen_38_code = qwen_38_code.filter(
        lambda x: x["conversations"] is not None and len(x["conversations"]) > 0
    )
    swe_next = swe_next.filter(
        lambda x: x["conversations"] is not None and len(x["conversations"]) > 0
    )

    # Mix all three sources with a single seed for reproducibility
    combined = concatenate_datasets([qwen_38_data_set, qwen_38_code, swe_next]).shuffle(
        seed=RANDOM_SEED
    )

    # Render each conversation through the chat template -> "text" column
    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = combined.map(formatting_prompts_func, batched=True)

    num_proc = mp.cpu_count()

    # Length filter: drop anything exceeding the model's context window
    def filter_long_data(examples):
        tokenized = text_tok(
            examples["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        return [len(toks) <= MAX_CONTEXT_WINDOW for toks in tokenized]

    dataset = dataset.filter(filter_long_data, batched=True, num_proc=num_proc)

    # Sanity check: assistant turns must carry a <think> block or a tool call
    def check_assistant_format(examples):
        ok = []
        for convo in examples["conversations"]:
            good = True
            for m in convo:
                if m["role"] != "assistant":
                    continue
                c = m.get("content", "")

                # SWE tool call -> valid
                if "<function=" in c:
                    continue

                if "<think>" not in c or "</think>" not in c:
                    good = False
                    break
                if not re.search(r"</think>\n", c):
                    good = False
                    break
            ok.append(good)
        return {"_ok": ok}

    check = dataset.map(
        check_assistant_format,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    bad = len(check) - sum(check["_ok"])
    if bad > 0:
        dataset = dataset.filter(
            lambda x: all(
                (m["role"] != "assistant")
                or ("<function=" in m.get("content", ""))
                or (
                    ("<think>" in m.get("content", ""))
                    and ("</think>\n" in m.get("content", ""))
                )
                for m in x["conversations"]
            )
        )

    # 98 / 2 split
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    print("Final dataset:", len(dataset))
    print("Train:", len(train_dataset))
    print("Eval:", len(eval_dataset))

    train_dataset.save_to_disk(OUT_DIR / "train")
    eval_dataset.save_to_disk(OUT_DIR / "eval")
    print(f"Wrote {OUT_DIR}/train and {OUT_DIR}/eval")


if __name__ == "__main__":
    main()
