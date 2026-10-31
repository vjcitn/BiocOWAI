#!/usr/bin/env bash
# harness-install.sh — ADR 0001 phase 3: install and configure the coding
# harness (Pi, and/or OpenCode), pointed at a local OpenAI-compatible model
# endpoint.
#
# Generalizes t4-coder.sh's install_pi()/install_opencode()/ensure_node() so
# harness setup stands on its own, independent of which backend is serving
# the model. Pi and OpenCode only need a base URL, a provider key name, and
# a model alias -- the same inputs whether the server behind them is
# llama.cpp CUDA on a T4/A100/MIG slice, llama.cpp Metal on an M4, or
# Ollama. Phase 1's GPU profile and phase 4/5's model/tuning choices decide
# BASE_URL/ALIAS; this script doesn't need to know which backend produced
# them, which is why it has no GPU-detection logic of its own.
#
# Usage:
#   ./harness-install.sh            # installs $HARNESS (default: pi)
#
# Env overrides:
#   HARNESS=pi|opencode|both     which harness(es) to install (default: pi)
#   PROVIDER_KEY=llama-cpp        provider key written into the harness's
#                                 config (models.json / opencode.json); use
#                                 e.g. "ollama" if BASE_URL points at Ollama
#                                 instead of llama.cpp's llama-server
#   PORT=8181                     used to build the default BASE_URL
#   BASE_URL=http://127.0.0.1:$PORT/v1   OpenAI-compatible endpoint to point at
#   ALIAS=qwen3.6-35b-a3b         model id harness-side (matches the
#                                 backend's --alias, or whatever it calls
#                                 the model it's serving)
#   CTX=65536                     context length advertised to OpenCode
#                                 only (Pi doesn't need this)
#   NODE_DIR=$HOME/.local/node    where a private Node 22 LTS gets installed
#
# This only configures the harness; it does not start or manage a model
# server (that's the backend-specific profile's job, e.g. t4-coder.sh's
# cmd_start/cmd_stop) and does not touch GPU detection (gpu-detect.sh,
# phase 1). Matches t4-coder.sh's model string convention so the
# compbiobench-runner fork's PiProvider (uses_conda_env = False) can drive
# it unchanged, e.g. "llama-cpp/qwen3.6-35b-a3b".

set -euo pipefail

HARNESS="${HARNESS:-pi}"
PROVIDER_KEY="${PROVIDER_KEY:-llama-cpp}"
PORT="${PORT:-8181}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT/v1}"
ALIAS="${ALIAS:-qwen3.6-35b-a3b}"
CTX="${CTX:-65536}"
NODE_DIR="${NODE_DIR:-$HOME/.local/node}"

log()  { printf '\033[1;34m[harness-install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[harness-install] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[harness-install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- node
# Private Node 22 LTS, no sudo. Node's release assets differ by platform:
# Darwin ships .tar.gz only in the naming scheme we match on; Linux ships
# both .tar.gz and .tar.xz. We always ask for the matching .tar.gz so one
# extraction path (tar -xz) works everywhere, rather than branching on
# platform to pick -xJ vs -xz.
ensure_node() {
  if [[ -x "$NODE_DIR/bin/node" ]]; then
    export PATH="$NODE_DIR/bin:$PATH"
    return
  fi

  local platform arch
  case "$(uname -s)" in
    Darwin) platform="darwin" ;;
    Linux)  platform="linux" ;;
    *) die "Unsupported platform for Node install: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64)  arch="x64" ;;
    arm64|aarch64) arch="arm64" ;;
    *) die "Unsupported arch for Node install: $(uname -m)" ;;
  esac

  log "Installing Node 22 LTS ($platform-$arch) into ${NODE_DIR}…"
  local shasums tarball
  shasums=$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt)
  tarball=$(awk -v f="${platform}-${arch}.tar.gz" '$0 ~ f {print $2; exit}' <<<"$shasums")
  [[ -n "$tarball" ]] || die "Could not find a Node 22 LTS .tar.gz for $platform-$arch."

  mkdir -p "$NODE_DIR"
  curl -fsSL "https://nodejs.org/dist/latest-v22.x/$tarball" \
    | tar -xz -C "$NODE_DIR" --strip-components=1

  export PATH="$NODE_DIR/bin:$PATH"
  local shell_rc="$HOME/.bashrc"
  [[ -n "${ZSH_VERSION:-}" ]] && shell_rc="$HOME/.zshrc"
  if ! grep -qs "$NODE_DIR/bin" "$shell_rc" 2>/dev/null; then
    echo "export PATH=\"$NODE_DIR/bin:\$PATH\"" >> "$shell_rc"
    log "Added $NODE_DIR/bin to PATH in $shell_rc (open a new shell to pick it up)."
  fi
}

# ------------------------------------------------------------------ pi
install_pi() {
  ensure_node
  if ! command -v pi >/dev/null 2>&1; then
    log "Installing Pi coding agent (@earendil-works/pi-coding-agent)…"
    npm install -g @earendil-works/pi-coding-agent
  fi

  log "Writing ~/.pi/agent/models.json: provider='$PROVIDER_KEY', baseUrl='$BASE_URL', model id='${ALIAS}'…"
  mkdir -p "$HOME/.pi/agent"
  PI_PROVIDER_KEY="$PROVIDER_KEY" PI_BASE_URL="$BASE_URL" PI_ALIAS="$ALIAS" python3 <<'PY'
import json, os, shutil

provider_key = os.environ["PI_PROVIDER_KEY"]
base_url = os.environ["PI_BASE_URL"]
alias = os.environ["PI_ALIAS"]

path = os.path.expanduser("~/.pi/agent/models.json")
cfg = {}
if os.path.exists(path):
    shutil.copy(path, path + ".bak")
    with open(path) as f:
        cfg = json.load(f)

cfg.setdefault("providers", {})[provider_key] = {
    "baseUrl": base_url,
    "api": "openai-completions",
    "apiKey": "none",
    "models": [{"id": alias}],
}

with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
  log "Pi ready. Drive it with model string '$PROVIDER_KEY/$ALIAS', e.g.:"
  log "  pi --model $PROVIDER_KEY/$ALIAS"
}

# ------------------------------------------------------------ opencode
install_opencode() {
  if ! command -v opencode >/dev/null 2>&1 && [[ ! -x "$HOME/.opencode/bin/opencode" ]]; then
    log "Installing OpenCode…"
    curl -fsSL https://opencode.ai/install | bash
  fi

  log "Writing ~/.config/opencode/opencode.json: provider='$PROVIDER_KEY', baseURL='${BASE_URL}'…"
  mkdir -p "$HOME/.config/opencode"
  OC_PROVIDER_KEY="$PROVIDER_KEY" OC_BASE_URL="$BASE_URL" OC_ALIAS="$ALIAS" OC_CTX="$CTX" python3 <<'PY'
import json, os, shutil

provider_key = os.environ["OC_PROVIDER_KEY"]
base_url = os.environ["OC_BASE_URL"]
alias = os.environ["OC_ALIAS"]
ctx = int(os.environ["OC_CTX"])

path = os.path.expanduser("~/.config/opencode/opencode.json")
cfg = {"$schema": "https://opencode.ai/config.json"}
if os.path.exists(path):
    shutil.copy(path, path + ".bak")
    with open(path) as f:
        cfg = json.load(f)

cfg.setdefault("provider", {})[provider_key] = {
    "npm": "@ai-sdk/openai-compatible",
    "name": f"{provider_key} (local)",
    "options": {"baseURL": base_url},
    "models": {alias: {"name": alias, "limit": {"context": ctx, "output": 16384}}},
}
cfg.setdefault("model", f"{provider_key}/{alias}")

with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
  log "OpenCode ready. Model string '$PROVIDER_KEY/$ALIAS' is now its default model."
}

main() {
  case "$HARNESS" in
    pi)       install_pi ;;
    opencode) install_opencode ;;
    both)     install_pi; install_opencode ;;
    *)        die "HARNESS must be pi, opencode, or both (got '$HARNESS')" ;;
  esac
  log "Done."
}

main "$@"
