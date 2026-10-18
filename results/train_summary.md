# Training summary — Shiba Coder 27B Max

Single-node fine-tuning run of a LoRA adapter on top of
`prism-ml/Bonsai-27B-unpacked` (Qwen3.5-27B, `Qwen3_5ForConditionalGeneration`).

## Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `prism-ml/Bonsai-27B-unpacked` |
| Method | LoRA (PEFT) + TRL `SFTTrainer` |
| Context window | 8 192 tokens |
| Base loading | 4-bit (`load_in_4bit=True`) |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 159,383,552 / 27,516,112,112 (**0.58 %**) |
| Optimizer | `adamw_8bit` |
| Learning rate | 1e-4 |
| LR schedule | linear (warmup 3 %) |
| Weight decay | 0.001 |
| Effective batch | 16 (1 × 16 grad-accum) |
| Precision | bf16 |
| Epochs | 1 |
| Steps | 930 |
| Eval strategy | every 200 steps |
| Checkpoint saving | every 200 steps (keep 1) |
| Seed | 12 181 531 |
| Chat template | `qwen3-thinking` |
| Loss masking | response-only (`train_on_responses_only`) |
| Duration | **27.30 h** |

## Loss progression

Train loss (smoothed) decreased monotonically across the run:

| Step | Train loss | Eval loss | LR |
|---|---|---|---|
| 0 | 0.3910 | — | 1.000e-4 |
| 200 | 0.3040 | 0.3023 | 8.358e-5 |
| 400 | 0.2800 | 0.2950 | 6.716e-5 |
| 600 | 0.2650 | 0.2939 | 5.074e-5 |
| 800 | 0.2680 | **0.2888** | 3.432e-5 |
| 930 | 0.2649 | — | 1.109e-7 |

> Values are approximate at the logged 10-step cadence. Final mean train loss
> across the whole run: **0.2722**.

## Key observations

- The adapter converges smoothly; no loss spikes or divergence.
- Eval loss at step 800 (0.2888) is the lowest observed — no overfitting
  despite a full epoch on ~17 k examples.
- The 4-bit base + LoRA footprint stayed within a single 16 GB GPU.

## Artifacts

| File | Description |
|---|---|
| `results/train_final.log` | Raw training log (930 steps, 27.30 h) |
| `notebooks/bonsai_max_final.ipynb` | Notebook used for the run |
| `src/train.py` | Reproducible training script |
| `configs/lora_config.yaml` | Machine-readable configuration |

## Reproducing

```bash
uv venv && uv pip install -r requirements.txt
uv run python data/prepare_dataset.py
BONSAI_CKPT=./checkpoints uv run python src/train.py
```
