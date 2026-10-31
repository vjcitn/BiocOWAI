#!/usr/bin/env bash
# compbiobench-check.sh — ADR 0001 phase 2: check compatibility between
# compbiobench-runner's actual requirements and this host's configuration.
#
# Grounded in the real upstream source (Genentech/compbiobench-runner,
# `environment.yml` + `run_benchmark.py` as of 2026-10-31), not assumption:
#
#   - compbiobench-runner itself has NO GPU/CUDA dependency. It is a
#     conda-based orchestrator: one fresh, isolated conda environment per
#     question (CPU bioinformatics tools — samtools/bwa/blast/minimap2/etc,
#     python=3.11), driving a system-installed, authenticated LLM CLI.
#   - It ships exactly three built-in LLMProvider subclasses: claude, codex,
#     gemini (registered in LLM_PROVIDERS). There is no "pi" or "opencode"
#     provider upstream, so BiocOWAI's Pi-pointed-at-a-local-model harness is
#     NOT natively drivable through compbiobench-runner's own `run_benchmark.py`
#     without writing and registering a custom LLMProvider subclass. CBBBioc's
#     cbb_run_question() sidesteps this today by driving Pi directly in R
#     rather than going through compbiobench-runner's runner.
#   - GPU only enters indirectly: whatever model backs the harness has to
#     answer within the runner's per-question timeout (`-t/--timeout`,
#     default 120 minutes), so this script treats the phase-1 GPU profile
#     (if present) as informational context, not a pass/fail input.
#
# This script therefore checks compbiobench-runner's real prerequisites
# (conda, disk headroom for parallel per-question envs) and reports, rather
# than hides, the harness-provider gap -- it does not invent a GPU
# compatibility matrix the upstream project does not have.
#
# Usage:
#   ./compbiobench-check.sh
#
# Env overrides:
#   HARNESS=pi|opencode|claude|codex|gemini   # which harness you intend to drive (default: pi)
#   PARALLEL=5                                 # intended -n/--parallel workers (default: 5)
#   COMPBIOBENCH_HOME=/path/to/compbiobench-runner  # a local checkout, if you
#     have one; when set, providers are read from its run_benchmark.py
#     instead of the hardcoded fallback list below, so this stays correct if
#     upstream adds providers.
#
# Exit status is always 0; this reports, it does not gate anything.

set -euo pipefail

LLM_HOME="${LLM_HOME:-$HOME/llm}"
HARNESS="${HARNESS:-pi}"
PARALLEL="${PARALLEL:-5}"
COMPBIOBENCH_HOME="${COMPBIOBENCH_HOME:-}"

# Known upstream providers as of 2026-10-31 (LLM_PROVIDERS in run_benchmark.py);
# used only when no local checkout is available to introspect directly.
FALLBACK_PROVIDERS="claude codex gemini"

# Rough per-env disk cost: environment.yml pulls blast/minimap2/bwa/samtools/
# bcftools/htslib/etc on top of a python=3.11 base. ~6 GB/env is a conservative
# floor from similar bioconda stacks; PARALLEL envs can coexist when not using
# --keep-envs cleanup lag, so size for that worst case.
GB_PER_ENV=6

log()  { printf '\033[1;34m[compbiobench-check]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[compbiobench-check] WARNING:\033[0m %s\n' "$*"; }
note() { printf '  * %s\n' "$*"; }

notes=()
add_note() { notes+=("$1"); }

# ---------------------------------------------------------------- conda
check_conda() {
  log "Checking conda/mamba (compbiobench-runner creates one isolated env per question)"
  local tool=""
  if command -v mamba >/dev/null 2>&1; then
    tool="mamba"
  elif command -v conda >/dev/null 2>&1; then
    tool="conda"
  fi

  if [[ -z "$tool" ]]; then
    warn "Neither conda nor mamba found on PATH."
    add_note "Install Miniconda/Mambaforge before running compbiobench-runner; it builds a fresh env per question from environment.yml and has no fallback path without conda."
    return
  fi
  log "Found $tool: $("$tool" --version 2>/dev/null | head -1)"
}

# ---------------------------------------------------------------- disk
check_disk() {
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

# ------------------------------------------------------------ providers
# Print the providers compbiobench-runner actually supports: from a local
# checkout if COMPBIOBENCH_HOME is set and looks right, else the fallback
# list captured from upstream source on 2026-10-31.
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
  printf '%s\n' "$FALLBACK_PROVIDERS" | tr ' ' '\n'
}

check_harness_provider() {
  log "Checking whether compbiobench-runner can drive harness '$HARNESS' natively"
  local providers source_desc
  if [[ -n "$COMPBIOBENCH_HOME" && -f "$COMPBIOBENCH_HOME/run_benchmark.py" ]]; then
    source_desc="introspected from $COMPBIOBENCH_HOME/run_benchmark.py"
  else
    source_desc="fallback list captured from upstream source on 2026-10-31 (set COMPBIOBENCH_HOME to introspect a real checkout instead)"
  fi
  providers=$(list_providers)
  log "compbiobench-runner's built-in providers ($source_desc): $(tr '\n' ' ' <<<"$providers")"

  if grep -qx "$HARNESS" <<<"$providers"; then
    log "'$HARNESS' is a built-in provider; compbiobench-runner can drive it directly via --llm $HARNESS."
  else
    warn "'$HARNESS' is NOT a built-in compbiobench-runner provider."
    add_note "compbiobench-runner's run_benchmark.py has no LLMProvider for '$HARNESS'. To use it through compbiobench-runner's own runner, implement an LLMProvider subclass (build_command/parse_output/extract_answer/build_test_command) and register it in LLM_PROVIDERS."
    add_note "CBBBioc's cbb_run_question() avoids this gap today by driving Pi directly in R, bypassing compbiobench-runner's runner entirely -- that remains the lower-friction path for Pi/OpenCode-backed questions."
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
