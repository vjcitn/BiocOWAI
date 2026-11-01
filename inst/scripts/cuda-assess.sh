#!/usr/bin/env bash
# cuda-assess.sh — T4 analog of metal-assess.sh (ADR 0001, phases 4/5).
#
# Deliberately does NOT reimplement any CUDA/apt/model-download logic --
# t4-coder.sh already does that correctly (preflight, nvcc/CUDA-toolkit
# handling, build for sm_75, Qwen3.6-35B-A3B UD-Q4_K_XL download, harness
# install). Duplicating that here would just risk re-making mistakes
# already found and fixed elsewhere in this project (e.g. bash's
# unbraced-$VAR-before-unicode-ellipsis parsing quirk, which is still
# latent but unfixed in t4-coder.sh itself -- see ADR 0001). This script
# only adds what t4-coder.sh doesn't have: a single-shot benchmark +
# report in the same format metal-assess.sh produced on the M4, so the
# two backends can be compared side by side.
#
# Run this ON THE T4 HOST, not on the Mac -- it drives t4-coder.sh, which
# is Ubuntu/CUDA/sudo-specific.
#
# Usage:
#   ./cuda-assess.sh assess     # t4-coder.sh install + start, then bench + report
#   ./cuda-assess.sh stop       # t4-coder.sh stop
#
# Requires t4-coder.sh to be reachable (default: two directories up from
# this script, i.e. the BiocOWAI repo root -- override with T4_CODER).
#
# Env overrides: anything t4-coder.sh itself accepts (ALIAS, N_CPU_MOE,
# PORT, CTX, HARNESS, ...) -- see its own header -- plus:
#   T4_CODER=../../t4-coder.sh   path to t4-coder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T4_CODER="${T4_CODER:-$SCRIPT_DIR/../../t4-coder.sh}"
PORT="${PORT:-8181}"

# Pick up ALIAS/N_CPU_MOE/etc. from t4-coder.sh's saved config, same as t4-coder.sh
# itself does -- otherwise the report below shows "unset" whenever this script is
# invoked without re-passing every override already saved from a prior install/start.
LLM_HOME="${LLM_HOME:-$HOME/llm}"
CONFIG="$LLM_HOME/config.env"
# shellcheck source=/dev/null
[[ -f "$CONFIG" ]] && source "$CONFIG"

log()  { printf '\033[1;36m[cuda-assess]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[cuda-assess] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "$T4_CODER" ]] || die "t4-coder.sh not found/executable at '$T4_CODER' (set T4_CODER=/path/to/t4-coder.sh)."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- this script expects to run on the T4 host itself."

# --------------------------------------------------------------- bench
bench() {
  log "Running one benchmark completion against 127.0.0.1:${PORT}…"
  local result pp_ms pp_n tg_ms tg_n pp_tps tg_tps

  result=$(curl -fsS "http://127.0.0.1:$PORT/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Explain what a MIG slice is on an NVIDIA GPU, in three sentences.","n_predict":200}')

  pp_ms=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['timings']['prompt_ms'])" "$result")
  pp_n=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['timings']['prompt_n'])" "$result")
  tg_ms=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['timings']['predicted_ms'])" "$result")
  tg_n=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['timings']['predicted_n'])" "$result")
  pp_tps=$(python3 -c "print(round($pp_n / ($pp_ms/1000), 1))")
  tg_tps=$(python3 -c "print(round($tg_n / ($tg_ms/1000), 1))")

  echo
  log "=== CUDA assessment report ==="
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  log "Model alias: ${ALIAS:-unset}, n-cpu-moe=${N_CPU_MOE:-unset}"
  log "Prompt processing: ${pp_tps} tok/s (${pp_n} tokens in ${pp_ms} ms)"
  log "Token generation:  ${tg_tps} tok/s (${tg_n} tokens in ${tg_ms} ms)"
  echo
  log "Compare against the other backends' numbers in ADR 0001 -- but check the model and"
  log "n-cpu-moe (CPU-offloaded MoE experts, 0 = fully GPU-resident) logged above first:"
  log "different models/offload settings across runs make this a per-backend sanity check"
  log "that the harness-serving pipeline works end to end here, not an apples-to-apples"
  log "comparison unless both sides used the same model and offload setting."
}

usage() { die "Usage: $0 {assess|stop}"; }

main() {
  local cmd="${1:-}"
  case "$cmd" in
    assess)
      "$T4_CODER" install
      "$T4_CODER" start
      bench
      ;;
    stop) "$T4_CODER" stop ;;
    *) usage ;;
  esac
}

main "$@"
