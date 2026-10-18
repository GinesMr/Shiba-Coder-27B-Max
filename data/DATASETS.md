# Dataset selection — Shiba Coder 27B Max

The adapter is specialized for **code generation** and **reasoning**, with real-world
software-engineering trajectories mixed in to keep tool-call / multi-turn behaviour
intact. Three public Hugging Face datasets are mixed:

| # | Dataset | Examples used | Split | Why it was selected |
|---|---------|---------------|-------|---------------------|
| 1 | [`r0b0tlab/qwen3.8-max-distillation-50k`](https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-distillation-50k) | 5 000 | filtered by `domain ∈ {code, reasoning}` | High-quality reasoning + code trajectories distilled from Qwen3.8-Max. Gives the adapter a consistent "think → solution" pattern. |
| 2 | [`guell00/qwen-3.8-code`](https://huggingface.co/datasets/guell00/qwen-3.8-code) | 10 000 | all | Pure code SFT data. Anchors the model on real code generation tasks. |
| 3 | [`TIGER-Lab/SWE-Next-SFT-Trajectories`](https://huggingface.co/datasets/TIGER-Lab/SWE-Next-SFT-Trajectories) | 2 000 | all | Real bugfix / feature trajectories with tool calls. Keeps multi-turn + tool-use ability from degrading. |

**Total** ≈ 17 000 examples.

## Why these three?

- **Coverage**: reasoning (1) + pure code (2) + agentic SWE (3) covers the target skill
  set without bloat.
- **Complementary formats**: Qwen-style multi-turn messages for 1 & 2, SWE-style
  tool-call trajectories for 3. Both are normalized to a single `conversations`
  schema in `data/prepare_dataset.py`.
- **Reproducibility**: all three are public HF datasets; shuffling uses a fixed
  seed (`12181531`) so the split is deterministic.

## Filtering & normalisation

1. **Drop `system` messages** — the chat template (`qwen3-thinking`) already encodes
   the role we want; a second system prompt would just confuse the model.
2. **Normalize assistant outputs** to the `qwen3-thinking` shape:
   - if the response already contains `<think>…</think>`, keep it;
   - if a `reasoning` field exists but no `<think>` block, wrap it;
   - otherwise emit an empty `<think></think>` block followed by the answer.
   This guarantees every assistant turn matches the template the adapter was
   trained on.
3. **SWE tool calls** (`<function=…`) are kept verbatim — they are valid content
   under this template and stripping them would damage agentic behaviour.
4. **Length filter**: any example whose tokenized length exceeds
   `MAX_CONTEXT_WINDOW = 8192` is dropped.
5. **Format check**: assistant turns must contain a well-formed `<think>…</think>\n`
   block (or a `<function=` tool call). Examples that fail are removed.

## Train / eval split

`train_test_split(test_size=0.02, seed=12181531)` → **~98% train / ~2% eval**.
The eval split is only used for loss tracking during training; the A/B eval in
`evaluation/` uses hand-written prompts (see `evaluation/test_prompts.md`).

## Reproducing the dataset

```bash
uv run python data/prepare_dataset.py
```

This downloads all three datasets, applies the filters above, and writes
`data/processed/train.parquet` and `data/processed/eval.parquet`.
