#!/usr/bin/env bash
# compbiobench-check.sh — ADR 0001 phase 2: check compatibility between
# compbiobench-runner's actual requirements and this host's configuration.
#
# Grounded in real source, not assumption -- and in two different repos,
# which this script distinguishes rather than conflating:
#
#   - Genentech/compbiobench-runner (upstream): a conda-based orchestrator,
#     one fresh isolated conda env per question (CPU bioinformatics tools --
#     samtools/bwa/blast/minimap2/etc, python=3.11) driving a system-
#     installed LLM CLI. It ships only three LLMProvider subclasses: claude,
#     codex, gemini. No "pi" or "opencode" provider exists there.
#   - vjcitn/compbiobench-runner (fork, this project's actual target,
#     https://github.com/vjcitn/compbiobench-runner, PR #1 "Add PiProvider"
#     merged to main): adds a PiProvider wired to Pi
#     (`@earendil-works/pi-coding-agent`) pointed at a locally-served model
#     -- `ollama/gpt-oss:20b` or `llama-cpp/qwen3.6-35b-a3b` (the alias
#     t4-coder.sh's llama-server + ~/.pi/agent/models.json produce). Unlike
#     the conda-isolated providers, PiProvider sets `uses_conda_env = False`:
#     Pi is a per-user Node.js CLI install, not cloned into a conda env per
#     question, so conda/disk sizing for per-question envs does not apply to
#     Pi-backed runs at all.
#
# compbiobench-runner itself (either repo) has NO GPU/CUDA dependency --
# GPU only enters indirectly, through whether the self-hosted model behind
# a conda-free provider like Pi answers within the per-question timeout
# (`-t/--timeout`, default 120 min). This script treats the phase-1 GPU
# profile (if present) as informational context for that question, not a
# pass/fail input.
#
# Usage:
#   ./compbiobench-check.sh
#
# Env overrides:
#   HARNESS=pi|opencode|claude|codex|gemini    # which harness you intend to drive (default: pi)
#   PARALLEL=5                                  # intended -n/--parallel workers (default: 5)
#   COMPBIOBENCH_REMOTE=fork|upstream           # which fallback provider list to assume
#     when COMPBIOBENCH_HOME isn't set (default: fork, since that's what
#     this project actually targets; upstream has no pi provider at all)
#   COMPBIOBENCH_HOME=/path/to/compbiobench-runner  # a local checkout, if you
#     have one; when set, providers AND each one's uses_conda_env flag are
#     read from its run_benchmark.py instead of the hardcoded fallback
#     knowledge below, so this stays correct as either repo evolves.
#
# Exit status is always 0; this reports, it does not gate anything.

set -euo pipefail

LLM_HOME="${LLM_HOME:-$HOME/llm}"
HARNESS="${HARNESS:-pi}"
PARALLEL="${PARALLEL:-5}"
COMPBIOBENCH_HOME="${COMPBIOBENCH_HOME:-}"
COMPBIOBENCH_REMOTE="${COMPBIOBENCH_REMOTE:-fork}"

# Fallback provider knowledge, used only when COMPBIOBENCH_HOME isn't a real
# checkout to introspect. Captured 2026-10-31: fork adds "pi" over upstream's
# claude/codex/gemini (see header comment).
FALLBACK_PROVIDERS_FORK="claude codex gemini pi"
FALLBACK_PROVIDERS_UPSTREAM="claude codex gemini"
# Providers known (fallback only) to skip conda env cloning (uses_conda_env = False).
FALLBACK_NO_CONDA_PROVIDERS="pi"

# Rough per-env disk cost for conda-isolated providers: environment.yml pulls
# blast/minimap2/bwa/samtools/bcftools/htslib/etc on top of a python=3.11
# base. ~6 GB/env is a conservative floor from similar bioconda stacks;
# PARALLEL envs can coexist when not using --keep-envs cleanup lag, so size
# for that worst case. Does not apply to providers with uses_conda_env=False
# (e.g. Pi in the fork), which clone no conda env at all.
GB_PER_ENV=6

log()  { printf '\033[1;34m[compbiobench-check]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[compbiobench-check] WARNING:\033[0m %s\n' "$*"; }
note() { printf '  * %s\n' "$*"; }

notes=()
add_note() { notes+=("$1"); }

capitalize() { awk '{print toupper(substr($0,1,1)) substr($0,2)}' <<<"$1"; }

# ------------------------------------------------------------ providers
# Providers compbiobench-runner actually supports: introspected from a local
# checkout if COMPBIOBENCH_HOME is set and looks right, else the fallback
# list for whichever repo (fork vs upstream) COMPBIOBENCH_REMOTE names.
list_providers() {
  if [[ -n "$COMPBIOBENCH_HOME" && -f "$COMPBIOBENCH_HOME/run_benchmark.py" ]]; then
    local found
    found=$(grep -oE 'class [A-Za-z]+Provider\(LLMProvider\)' \
              "$COMPBIOBENCH_HOME/run_benchmark.py" 2>/dev/null \
            | sed -E 's/class ([A-Za-z]+)Provider.*/\1/' | tr 'A-Z' 'a-z')
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return
    fi
  fi
  if [[ "$COMPBIOBENCH_REMOTE" == "upstream" ]]; then
    printf '%s\n' "$FALLBACK_PROVIDERS_UPSTREAM" | tr ' ' '\n'
  else
    printf '%s\n' "$FALLBACK_PROVIDERS_FORK" | tr ' ' '\n'
  fi
}

# Whether a given provider clones a fresh conda env per question. Introspects
# COMPBIOBENCH_HOME's run_benchmark.py when available (scans that provider's
# class body for `uses_conda_env = False`); otherwise uses fallback knowledge.
provider_uses_conda() {
  local provider="$1" cap
  cap=$(capitalize "$provider")
  if [[ -n "$COMPBIOBENCH_HOME" && -f "$COMPBIOBENCH_HOME/run_benchmark.py" ]]; then
    awk -v cls="${cap}Provider" '
      $0 ~ ("^class " cls "\\(LLMProvider\\)") { in_block=1; next }
      in_block && /^class [A-Za-z]+Provider\(LLMProvider\)/ { in_block=0 }
      in_block && /uses_conda_env[ ]*=[ ]*False/ { found_false=1 }
      END { print (found_false ? "false" : "true") }
    ' "$COMPBIOBENCH_HOME/run_benchmark.py"
    return
  fi
  if grep -qx "$provider" <<<"$FALLBACK_NO_CONDA_PROVIDERS"; then
    printf 'false\n'
  else
    printf 'true\n'
  fi
}

# ---------------------------------------------------------------- conda
check_conda() {
  local uses_conda; uses_conda=$(provider_uses_conda "$HARNESS")
  if [[ "$uses_conda" == "false" ]]; then
    log "Harness '$HARNESS' has uses_conda_env=False; it clones no per-question conda env, so conda availability doesn't gate it."
    return
  fi
  log "Checking conda/mamba (harness '$HARNESS' clones one isolated conda env per question)"
  local tool=""
  if command -v mamba >/dev/null 2>&1; then
    tool="mamba"
  elif command -v conda >/dev/null 2>&1; then
    tool="conda"
  fi

  if [[ -z "$tool" ]]; then
    warn "Neither conda nor mamba found on PATH."
    add_note "Install Miniconda/Mambaforge before running compbiobench-runner with '$HARNESS'; it builds a fresh env per question from environment.yml and has no fallback path without conda."
    return
  fi
  log "Found $tool: $("$tool" --version 2>/dev/null | head -1)"
}

# ---------------------------------------------------------------- disk
check_disk() {
  local uses_conda; uses_conda=$(provider_uses_conda "$HARNESS")
  if [[ "$uses_conda" == "false" ]]; then
    log "Harness '$HARNESS' clones no conda env, so the per-env disk estimate below doesn't apply to it."
    return
  fi
  log "Checking disk headroom for $PARALLEL parallel per-question conda envs"
  local disk_gb needed
  disk_gb=$(df -k "$HOME" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}')
  needed=$(( GB_PER_ENV * PARALLEL ))
  log "Free: ${disk_gb:-0} GB under \$HOME; rough need for $PARALLEL concurrent envs: ~${needed} GB (${GB_PER_ENV} GB/env estimate)"
  if [[ -n "${disk_gb:-}" ]] && (( disk_gb < needed )); then
    warn "Free disk (${disk_gb} GB) is under the rough estimate (${needed} GB) for $PARALLEL concurrent envs."
    add_note "Lower PARALLEL (compbiobench-runner's -n/--parallel), or free disk, or run with fewer concurrent envs than the default of 5."
  fi
}

# ------------------------------------------------------------ harness check
check_harness_provider() {
  log "Checking whether compbiobench-runner can drive harness '$HARNESS' natively"
  local providers source_desc
  if [[ -n "$COMPBIOBENCH_HOME" && -f "$COMPBIOBENCH_HOME/run_benchmark.py" ]]; then
    source_desc="introspected from $COMPBIOBENCH_HOME/run_benchmark.py"
  else
    source_desc="fallback list for COMPBIOBENCH_REMOTE=$COMPBIOBENCH_REMOTE, captured 2026-10-31 (set COMPBIOBENCH_HOME to introspect a real checkout instead)"
  fi
  providers=$(list_providers)
  log "compbiobench-runner's providers ($source_desc): $(tr '\n' ' ' <<<"$providers")"

  if grep -qx "$HARNESS" <<<"$providers"; then
    log "'$HARNESS' is a built-in provider; compbiobench-runner can drive it directly via --llm $HARNESS."
    if [[ "$HARNESS" == "pi" ]]; then
      note "Pi's model string selects the backend via ~/.pi/agent/models.json -- known priced aliases in the fork's PiProvider: ollama/gpt-oss:20b, llama-cpp/qwen3.6-35b-a3b (the latter matches t4-coder.sh's ALIAS default). Make sure -m/--model matches whatever alias your setup actually serves."
    fi
  else
    warn "'$HARNESS' is NOT a provider in this list."
    if [[ "$HARNESS" == "pi" && "$COMPBIOBENCH_REMOTE" == "upstream" ]]; then
      add_note "Genentech/compbiobench-runner (upstream) has no PiProvider. Use the fork instead (COMPBIOBENCH_REMOTE=fork, or clone https://github.com/vjcitn/compbiobench-runner), which already has one (PR #1, merged to main)."
    else
      add_note "No LLMProvider found for '$HARNESS' in the checked source. To add it, implement an LLMProvider subclass (build_command/parse_output/extract_answer/build_test_command) and register it in LLM_PROVIDERS -- see the fork's PiProvider for a worked example of a conda-free, locally-served provider."
    fi
  fi
}

# ------------------------------------------------------------- GPU context
check_gpu_context() {
  log "Reading phase-1 GPU profile for informational context (compbiobench-runner itself has no GPU dependency)"
  local profile="$LLM_HOME/gpu-profile.json"
  if [[ ! -f "$profile" ]]; then
    note "No GPU profile at $profile; run gpu-detect.sh --save first if you want this noted."
    return
  fi
  log "Profile found at $profile:"
  sed 's/^/  /' "$profile"
  note "GPU profile is context only: it matters for whether the self-hosted model behind '$HARNESS' answers within compbiobench-runner's per-question timeout (-t/--timeout, default 120 min), not for compbiobench-runner's own requirements."
}

main() {
  check_conda
  check_disk
  check_harness_provider
  check_gpu_context

  if (( ${#notes[@]} )); then
    echo
    log "Action items:"
    for n in "${notes[@]}"; do note "$n"; done
  else
    echo
    log "No blocking items found."
  fi
}

main "$@"
