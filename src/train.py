"""
Shiba Coder 27B Max — LoRA fine-tuning entrypoint.

Trains a PEFT/LoRA adapter on top of ``prism-ml/Bonsai-27B-unpacked`` using the
exact hyperparameters recorded in ``configs/lora_config.yaml``.

Usage:
    uv run python src/train.py

The script assumes the dataset has already been prepared (see
``data/prepare_dataset.py``). Output checkpoints are written to the directory
given by the ``BONSAI_CKPT`` environment variable.
"""

import os

# Line-buffer stdout so long-running logs are readable immediately.
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "prism-ml/Bonsai-27B-unpacked"
MAX_SEQ_LENGTH = 8192
CHAT_TEMPLATE = "qwen3-thinking"
OUTPUT_DIR = os.environ.get("BONSAI_CKPT", "./checkpoints")

LORA_CONFIG = {
    "r": 32,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_alpha": 64,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
}

SFT_ARGS = {
    "dataset_text_field": "text",
    "max_length": MAX_SEQ_LENGTH,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.03,
    "optim": "adamw_8bit",
    "weight_decay": 0.001,
    "lr_scheduler_type": "linear",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 1,
    "fp16": False,
    "bf16": True,
    "seed": 12181531,
    "report_to": "wandb",
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """Load the base model in 4-bit and attach the LoRA adapter."""
    model, tokenizer_model = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    trainable = sum(
        p.numel() for n, p in model.named_parameters() if "lora_" in n.lower()
    )
    print(
        f"LoRA {trainable / 1e6:.2f}M  VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    return model, tokenizer_model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prepared_dataset(tokenizer):
    """Load the parquet files produced by ``data/prepare_dataset.py``."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    eval_dataset = load_from_disk(os.path.join(data_dir, "eval"))
    print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")
    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """Assemble the TRL SFTTrainer with response-only loss masking."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**SFT_ARGS, output_dir=output_dir),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    return trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer_model = build_model()
    tokenizer = get_chat_template(tokenizer_model, chat_template=CHAT_TEMPLATE)

    train_dataset, eval_dataset = load_prepared_dataset(tokenizer)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, OUTPUT_DIR)

    print(
        "trainer ready, VRAM", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB"
    )

    start = time.time()
    trainer.train()
    print(f"TRAIN DONE {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
