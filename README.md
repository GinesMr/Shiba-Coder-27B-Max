---
base_model: prism-ml/Bonsai-27B-unpacked
library_name: peft
pipeline_tag: text-generation
tags:
  - code
  - reasoning
  - lora
  - sft
  - peft
  - unsloth
  - trl
  - qwen3.5
  - qwen3-thinking
datasets:
  - r0b0tlab/qwen3.8-max-distillation-50k
  - guell00/qwen-3.8-code
  - TIGER-Lab/SWE-Next-SFT-Trajectories
metrics:
  - loss
---

# Shiba Coder 27B Max

**LoRA adapter for [`prism-ml/Bonsai-27B-unpacked`](https://huggingface.co/prism-ml/Bonsai-27B-unpacked) — code generation + reasoning.**

Trained on a mix of reasoning, code, and real software-engineering trajectories.
The adapter is **not merged** — it is loaded on top of the base model at
inference time.

| | |
|---|---|
| **Base model** | [`prism-ml/Bonsai-27B-unpacked`](https://huggingface.co/prism-ml/Bonsai-27B-unpacked) |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (27 B) |
| **Method** | LoRA (SFT) via Unsloth + TRL `SFTTrainer` |
| **Chat template** | `qwen3-thinking` |
| **Context window** | 8 192 tokens |
| **Trainable params** | 159,383,552 / 27,516,112,112 (0.58 %) |
| **Weights** | [`Gmartinezr/Shiba-Coder-27B-Max`](https://huggingface.co/Gmartinezr/Shiba-Coder-27B-Max) |
| **License** | Apache 2.0 |

---

## Quick start

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "prism-ml/Bonsai-27B-unpacked", torch_dtype="auto"
)
tok = AutoTokenizer.from_pretrained("prism-ml/Bonsai-27B-unpacked")
model = PeftModel.from_pretrained(base, "Gmartinezr/Shiba-Coder-27B-Max")

messages = [{"role": "user", "content": "Write a FastAPI endpoint for user registration."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**inputs, max_new_tokens=512)[0], skip_special_tokens=True))
```

### llama.cpp (no merge)

```bash
llama-cli -m Bonsai-27B-Q1_0.gguf \
  --lora Shiba-Coder-LoRA-F16.gguf \
  -ngl 99 --seed 42 --temp 0.2 -n 1024 \
  -p "your prompt"
```

> `convert_lora_to_gguf.py` needs a `transformers` build that understands
> `qwen3_5` (≥ 5.x from source at the time of writing):
> `pip install git+https://github.com/huggingface/transformers.git`

---

## Training at a glance

| | |
|---|---|
| **Duration** | 27.30 h |
| **Steps** | 930 (1 epoch) |
| **Effective batch** | 16 (1 × 16 grad-accum) |
| **Optimizer** | `adamw_8bit` (LR 1e-4, linear, warmup 3 %) |
| **Precision** | bf16 over a 4-bit base |
| **Final train loss** | 0.2722 |
| **Best eval loss** | 0.2888 (step 800) |
| **GPU** | single node |

See [`results/train_summary.md`](./results/train_summary.md) for the full
hyperparameter table and loss progression.

---

## Dataset

| Source | Examples | Purpose |
|---|---|---|
| [`r0b0tlab/qwen3.8-max-distillation-50k`](https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-distillation-50k) | 5 000 | Qwen3.8-Max distillation — reasoning + code trajectories |
| [`guell00/qwen-3.8-code`](https://huggingface.co/datasets/guell00/qwen-3.8-code) | 10 000 | Code-focused SFT |
| [`TIGER-Lab/SWE-Next-SFT-Trajectories`](https://huggingface.co/datasets/TIGER-Lab/SWE-Next-SFT-Trajectories) | 2 000 | Real SWE bugfix / feature trajectories (tool calls) |

≈ 17 000 examples total, filtered to ≤ 8 192 tokens, split 98 / 2 train / eval.
See [`data/DATASETS.md`](./data/DATASETS.md) for the selection rationale.

---

## Repository layout

```
.
├── README.md                    this file
├── LICENSE                      Apache 2.0
├── requirements.txt
├── configs/
│   └── lora_config.yaml         exact hyperparameters (YAML)
├── data/
│   ├── DATASETS.md              selection & filtering rationale
│   └── prepare_dataset.py       build train / eval artifacts
├── src/
│   └── train.py                 training entrypoint
├── evaluation/
│   ├── test_prompts.md          10 A/B eval prompts
│   ├── comparison.md            A/B results
│   └── run_ab.sh                reproducible A/B runner
├── results/
│   ├── train_summary.md         hyperparameters & loss curve
│   └── train_final.log          raw training log
├── notebooks/
│   ├── bonsai_max.ipynb         exploratory notebook
│   └── bonsai_max_final.ipynb   notebook used for the run
└── .github/workflows/ci.yml     lint + secret scan
```

---

## Reproducing

```bash
uv venv && uv pip install -r requirements.txt

uv run python data/prepare_dataset.py
BONSAI_CKPT=./checkpoints uv run python src/train.py
```

Exact hyperparameters live in [`configs/lora_config.yaml`](./configs/lora_config.yaml).

---

## Evaluation

Hand-written prompts covering code generation, debugging, refactoring,
feature implementation, and algorithmic reasoning
([`evaluation/test_prompts.md`](./evaluation/test_prompts.md)).
The A/B comparison report is at [`evaluation/comparison.md`](./evaluation/comparison.md).

---

## Limitations

- Specialized for **code + reasoning**; general chat quality is inherited
  from the base model and may differ.
- LoRA r=32 / α=64 is intentionally low-rank — it adapts style and behaviour,
  not the full knowledge of the base.
- No public benchmark numbers (HumanEval / MBPP / SWE-bench) yet.
- Inherits the biases and failure modes of the base model and the training data.

---

## License

Apache 2.0 — see [`LICENSE`](./LICENSE). Training datasets and the base model
carry their own licenses; verify compatibility before commercial use.
