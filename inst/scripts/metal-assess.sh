#!/usr/bin/env bash
# metal-assess.sh — ADR 0001, pre-work for phases 4/5 on Apple Silicon.
#
# NOT the generalized Metal serving script yet. This is a hands-on
# assessment tool: build llama.cpp with the Metal backend, serve a small
# model, run one benchmark pass, and report what unified memory / Metal
# actually look like on this machine -- so phase 4 (model selection) and
# phase 5 (tuning) are designed from real M-series behavior instead of by
# analogy with the CUDA/T4 path (which has no unified memory, no
# `--n-cpu-moe`-style offload concept, and different unit CUDA-arch flags).
#
# Mirrors t4-coder.sh's shape (preflight -> build -> download -> serve ->
# bench -> stop) but only the parts that matter for a first look; it does
# not attempt t4-coder.sh's full tune() sweep.
#
# Usage:
#   ./metal-assess.sh assess     # build + download + serve + bench + report
#   ./metal-assess.sh stop       # stop a server left running by `assess`
#
# Env overrides:
#   LLM_HOME=$HOME/llm             working dir (llama.cpp checkout, model, logs)
#   PORT=8182                      llama-server port (8182, not t4-coder.sh's
#                                   8181, so both can coexist on one machine)
#   MODEL_REPO=ggml-org/gpt-oss-20b-GGUF
#   MODEL_FILE=gpt-oss-20b-MXFP4.gguf
#     gpt-oss-20b in its native MXFP4 quant (~12.1 GB, single file). This
#     is the actual target model for this machine (32 GB unified memory):
#     it matches the "ollama/gpt-oss:20b" alias already referenced by the
#     compbiobench-runner fork's PiProvider (see ADR 0001's phase 2
#     finding), unlike the first assessment pass's placeholder 7B Qwen
#     model, which was only a quick look at Metal build/serve mechanics.
#   CTX=8192                       context size for the assessment server
#   SKIP_BUILD=0 / SKIP_DOWNLOAD=0 set to 1 to reuse a prior build/model
#   CMAKE_BIN (required, no default)   full path to the cmake binary this
#     script should use to configure/build llama.cpp. This script never
#     installs or falls back to a package-manager cmake (e.g. Homebrew) --
#     a bioinformatics dev machine commonly already has a cmake managed
#     for R/Bioconductor source builds (e.g.
#     ~/Library/Caches/org.R-project.R/R/biocmake/<ver>/CMake.app/Contents/bin/cmake)
#     and a second cmake on PATH from a package manager risks shadowing or
#     conflicting with that. Point CMAKE_BIN at whichever cmake you want
#     this build to use; the script fails immediately if it's unset.

set -euo pipefail

LLM_HOME="${LLM_HOME:-$HOME/llm}"
PORT="${PORT:-8182}"
MODEL_REPO="${MODEL_REPO:-ggml-org/gpt-oss-20b-GGUF}"
MODEL_FILE="${MODEL_FILE:-gpt-oss-20b-MXFP4.gguf}"
CTX="${CTX:-8192}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
CMAKE_BIN="${CMAKE_BIN:-}"

LLAMA_DIR="$LLM_HOME/llama.cpp"
MODEL_PATH="$LLM_HOME/models/$MODEL_FILE"
PID_FILE="$LLM_HOME/metal-assess.pid"
LOG_FILE="$LLM_HOME/metal-assess.log"

log()  { printf '\033[1;35m[metal-assess]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[metal-assess] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[metal-assess] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- preflight
preflight() {
  [[ "$(uname -s)" == "Darwin" ]] || die "This script is Apple Silicon / Metal only (host is $(uname -s))."
  [[ "$(uname -m)" == "arm64" ]] || die "This script targets Apple Silicon (arm64); host reports $(uname -m)."
  local chip
  chip=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip/{print $2; exit}')
  [[ "$chip" == Apple* ]] || die "Could not confirm an Apple-silicon chip (got: '${chip:-unknown}')."
  log "Host: $chip, $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB unified memory"
  mkdir -p "$LLM_HOME" "$LLM_HOME/models"
}

# ------------------------------------------------------------- build deps
ensure_build_deps() {
  xcode-select -p >/dev/null 2>&1 || die "Xcode command line tools not installed. Run: xcode-select --install"
  [[ -n "$CMAKE_BIN" ]] || die "CMAKE_BIN is not set. Point it at the cmake you want this build to use (e.g. a Bioconductor-managed one under ~/Library/Caches/org.R-project.R/R/biocmake/<ver>/CMake.app/Contents/bin/cmake, or any other) -- this script will not install or assume one."
  [[ -x "$CMAKE_BIN" ]] || die "CMAKE_BIN='$CMAKE_BIN' is not an executable file."
  log "cmake: $("$CMAKE_BIN" --version | head -1) ($CMAKE_BIN)"
}

# ------------------------------------------------------- build llama.cpp
build_llama_cpp() {
  if [[ "$SKIP_BUILD" == "1" ]]; then
    log "SKIP_BUILD=1, reusing existing build at $LLAMA_DIR"
    [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || die "No existing llama-server binary at $LLAMA_DIR/build/bin/llama-server"
    return
  fi

  if [[ -d "$LLAMA_DIR/.git" ]]; then
    log "Updating existing llama.cpp checkout…"
    git -C "$LLAMA_DIR" pull --ff-only
  else
    log "Cloning llama.cpp into ${LLAMA_DIR}…"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi

  log "Configuring with -DGGML_METAL=ON (Release)…"
  "$CMAKE_BIN" -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON

  log "Building (this is the slow step; using $(sysctl -n hw.ncpu) cores)…"
  "$CMAKE_BIN" --build "$LLAMA_DIR/build" -j "$(sysctl -n hw.ncpu)" --target llama-server

  [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || die "Build finished but llama-server binary not found."
  log "Built: $LLAMA_DIR/build/bin/llama-server"
}

# ----------------------------------------------------------- model
download_model() {
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    log "SKIP_DOWNLOAD=1, expecting model already at $MODEL_PATH"
    [[ -f "$MODEL_PATH" ]] || die "No model file at $MODEL_PATH"
    return
  fi
  if [[ -f "$MODEL_PATH" ]]; then
    log "Model already present: $MODEL_PATH"
    return
  fi
  command -v python3 >/dev/null 2>&1 || die "python3 required to fetch the model via huggingface_hub."
  python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --quiet "huggingface_hub[hf_xet]"

  log "Downloading $MODEL_REPO/$MODEL_FILE (~12.1 GB) into ${LLM_HOME}/models…"
  python3 - "$MODEL_REPO" "$MODEL_FILE" "$LLM_HOME/models" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
print(f"Downloaded to: {path}")
PY
}

# ----------------------------------------------------------- serve
is_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

start_server() {
  if is_running; then
    log "Server already running (pid $(cat "$PID_FILE"))."
    return
  fi
  log "Starting llama-server on 127.0.0.1:$PORT (ctx=$CTX, all layers on GPU via Metal)…"
  nohup "$LLAMA_DIR/build/bin/llama-server" \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --n-gpu-layers 999 \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  log "Waiting for health check…"
  local tries=0
  until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 2
    tries=$(( tries + 1 ))
    if (( tries > 60 )); then
      die "Server did not become healthy within 2 minutes; see $LOG_FILE"
    fi
  done
  log "Server healthy (pid $(cat "$PID_FILE"))."
}

stop_server() {
  if ! is_running; then
    log "No running server (or pid file stale)."
    rm -f "$PID_FILE"
    return
  fi
  local pid
  pid=$(cat "$PID_FILE")
  log "Stopping server (pid $pid)…"
  kill "$pid"
  rm -f "$PID_FILE"
}

# ----------------------------------------------------------- bench
bench() {
  log "Running one benchmark completion (watch Activity Monitor / asitop for GPU + memory while this runs)…"
  local mem_before mem_after result
  mem_before=$(vm_stat | awk '/Pages active/{print $3}' | tr -d '.')

  result=$(curl -fsS "http://127.0.0.1:$PORT/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Explain what a MIG slice is on an NVIDIA GPU, in three sentences.","n_predict":200}')

  mem_after=$(vm_stat | awk '/Pages active/{print $3}' | tr -d '.')

  local pp_ms pp_n tg_ms tg_n pp_tps tg_tps
  pp_ms=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['prompt_ms'])")
  pp_n=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['prompt_n'])")
  tg_ms=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['predicted_ms'])")
  tg_n=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['predicted_n'])")
  pp_tps=$(python3 -c "print(round($pp_n / ($pp_ms/1000), 1))")
  tg_tps=$(python3 -c "print(round($tg_n / ($tg_ms/1000), 1))")

  echo
  log "=== Metal assessment report ==="
  log "Model: $MODEL_REPO/$MODEL_FILE, ctx=$CTX, n-gpu-layers=999 (all)"
  log "Prompt processing: ${pp_tps} tok/s (${pp_n} tokens in ${pp_ms} ms)"
  log "Token generation:  ${tg_tps} tok/s (${tg_n} tokens in ${tg_ms} ms)"
  log "Active pages before/after (4KB pages, rough proxy for RSS growth): $mem_before -> $mem_after"
  log "Full server log: $LOG_FILE"
  echo
  log "Open questions this should inform (ADR 0001 phase 4/5):"
  log "  - Does gpt-oss-20b (this machine's actual target model, matching the fork's"
  log "    PiProvider ollama/gpt-oss:20b alias) deliver usable interactive throughput at"
  log "    full GPU offload, or does it need context/offload tuning to feel responsive?"
  log "  - Is --n-gpu-layers 999 (all layers on GPU) actually optimal here, or does holding"
  log "    some layers back change throughput -- i.e. is there a Metal-side tuning knob"
  log "    worth a phase-5 sweep, analogous to --n-cpu-moe on CUDA?"
}

usage() { die "Usage: $0 {assess|stop}"; }

main() {
  local cmd="${1:-}"
  case "$cmd" in
    assess)
      preflight
      ensure_build_deps
      build_llama_cpp
      download_model
      start_server
      bench
      ;;
    stop) stop_server ;;
    *) usage ;;
  esac
}

main "$@"
