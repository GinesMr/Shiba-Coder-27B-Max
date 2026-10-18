#!/usr/bin/env bash
# =============================================================================
# A/B evaluation runner — Shiba Coder 27B Max
#
# Runs the same 10 prompts against:
#   (A) the base model  — Bonsai-27B-Q1_0.gguf
#   (B) the base model + LoRA adapter — Shiba-Coder-LoRA-F16.gguf
#
# Requirements:
#   * llama.cpp binaries in $LLAMA_BIN (default: /app/llama-cli)
#   * $WORK_DIR pointing at a directory with:
#       models/Bonsai-27B-Q1_0.gguf
#       output/Shiba-Coder-LoRA-F16.gguf
#
# Output is written to logs/bonsai-q1-base.txt and logs/shiba-q1-plus-lora.txt.
# =============================================================================
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/app/llama-cli}"
WORK_DIR="${WORK_DIR:-/work}"
MODEL_BASE="${WORK_DIR}/models/Bonsai-27B-Q1_0.gguf"
MODEL_LORA="${WORK_DIR}/output/Shiba-Coder-LoRA-F16.gguf"
LOG_DIR="${WORK_DIR}/logs"

SEED=42
TEMP=0.2
N_PREDICT=1024
NGL=99

mkdir -p "$LOG_DIR"

# --- Prompts ---------------------------------------------------------------
PROMPTS=(
  "Create a Python FastAPI endpoint for user registration. Use Pydantic validation, proper typing, useful HTTP errors, and separate validation logic from persistence logic."

  "This Python function is supposed to return the second-largest element in a list but returns the wrong value for some inputs. Find the bug, explain it, and provide a fixed version with tests.\n\ndef second_largest(xs):\n    xs = list(set(xs))\n    xs.sort()\n    return xs[-2]"

  "Refactor the following class to follow the Single Responsibility Principle. Split it into at least two classes and explain the trade-off.\n\nclass OrderService:\n    def __init__(self, db): self.db = db\n    def create_order(self, data):\n        if not data.get('items'): raise ValueError('empty')\n        self.db.save(data)\n        self.send_email(data['customer'], 'Order created')\n    def send_email(self, to, body): ..."

  "Implement a function that returns the length of the longest increasing subsequence of an integer list. Include a complexity analysis and at least three test cases."

  "Given tables users(id, email) and orders(id, user_id, amount, created_at), write a SQL query that returns, for each user, their total spend in the last 30 days. Then explain how you would index it for performance."

  "Write a small typed Python module that wraps a SQLite table 'articles' with CRUD operations. Use dataclasses or Pydantic, type hints everywhere, and raise specific exceptions (e.g. ArticleNotFound)."

  "The following code sometimes deadlocks. Explain why and rewrite it so it cannot deadlock.\n\nimport threading\nlock_a = threading.Lock()\nlock_b = threading.Lock()\n\ndef t1():\n    lock_a.acquire(); lock_b.acquire()\n    ... ; lock_b.release(); lock_a.release()\n\ndef t2():\n    lock_b.acquire(); lock_a.acquire()\n    ... ; lock_a.release(); lock_b.release()"

  "Design a rate limiter as a small Python library. Support at least two algorithms (token bucket, sliding window), be thread-safe, and document the public API. No external services."

  "Write a React + TypeScript component SearchList that fetches a list of items from /api/items?q=, debounces the input by 300 ms, and shows loading / error / empty states. Use modern hooks and handle race conditions."

  "Write a Python function parse_duration('1h30m') -> timedelta that accepts combinations of d, h, m, s. Include a table-driven test suite that covers at least six cases, including invalid input."
)

# --- Runner ----------------------------------------------------------------
run_one() {
  local label="$1"; shift
  local out_file="$1"; shift
  local extra_args=("$@")

  echo "=== $label ===" | tee -a "$out_file"
  for i in "${!PROMPTS[@]}"; do
    local n=$((i + 1))
    echo "" >> "$out_file"
    echo "--- prompt $n ---" >> "$out_file"
    echo "${PROMPTS[$i]}" >> "$out_file"
    echo "" >> "$out_file"

    "$LLAMA_BIN" \
      -m "$MODEL_BASE" \
      --seed "$SEED" --temp "$TEMP" -n "$N_PREDICT" -ngl "$NGL" \
      "${extra_args[@]}" \
      -p "${PROMPTS[$i]}" 2>&1 | tee -a "$out_file"
  done
}

# Prueba A — base model only
run_one "Prueba A: Bonsai Q1 base" "$LOG_DIR/bonsai-q1-base.txt"

# Prueba B — base model + LoRA
run_one "Prueba B: Bonsai Q1 + LoRA" "$LOG_DIR/shiba-q1-plus-lora.txt" --lora "$MODEL_LORA"

echo ""
echo "Done. Logs in $LOG_DIR/"
