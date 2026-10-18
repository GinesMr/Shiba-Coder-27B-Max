# A/B comparison — Bonsai Q1 vs. Shiba Coder 27B Max

> Status: **pending** — the A/B run has not been executed yet.
> See [`evaluation/run_ab.sh`](./run_ab.sh) for the reproducible runner.

## Objective

Determine whether the `Shiba-Coder-27B-Max` LoRA adapter changes the behaviour
of the base model **without merging** — i.e. by loading the adapter at
inference time on top of the quantized base GGUF.

## Test setup

| | Prueba A | Prueba B |
|---|---|---|
| Base model | `Bonsai-27B-Q1_0.gguf` | `Bonsai-27B-Q1_0.gguf` |
| LoRA | — | `Shiba-Coder-LoRA-F16.gguf` |
| Quantization | Q1_0 | Q1_0 + F16 LoRA |
| Inference | `llama-cli -ngl 99 --seed 42 --temp 0.2 -n 1024` | same |
| Prompts | 10 (see [`test_prompts.md`](./test_prompts.md)) | same |

### Environment

- **GPU**: AMD Radeon RX 7800 XT (16 GB, gfx1101)
- **RAM**: 32 GB
- **Runtime**: Docker `ghcr.io/ggml-org/llama.cpp:full-rocm`
- **Backend**: ROCm (fallback: Vulkan → CPU)

## Results

| Metric | Prueba A (base) | Prueba B (base + LoRA) |
|---|---|---|
| LoRA loaded? | n/a | _pending_ |
| Tokens / s | _pending_ | _pending_ |
| VRAM used | _pending_ | _pending_ |
| Coherent outputs (10) | _pending_ | _pending_ |
| Effect of the LoRA | n/a | _pending_ |

## Per-prompt verdicts

| # | Category | Effect observed | Notes |
|---|---|---|---|
| 1 | API endpoint | | |
| 2 | Bug diagnosis | | |
| 3 | Refactor / SOLID | | |
| 4 | Algorithm | | |
| 5 | SQL | | |
| 6 | Type-safe data layer | | |
| 7 | Debugging concurrency | | |
| 8 | Design / architecture | | |
| 9 | React + TypeScript | | |
| 10 | Test-driven utility | | |

## Conclusion

_Pending — fill in after the A/B run._

## Reproducing

```bash
# 1. Convert the LoRA to F16 GGUF (no merge)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$HOME/shiba-q1-test:/work" --entrypoint bash \
  ghcr.io/ggml-org/llama.cpp:full-rocm \
  -c "python3 /app/convert_lora_to_gguf.py /work/lora/checkpoint-930 \
       --base-model-id prism-ml/Bonsai-27B-unpacked \
       --outfile /work/output/Shiba-Coder-LoRA-F16.gguf --outtype f16"

# 2. Run the A/B
bash evaluation/run_ab.sh
```
