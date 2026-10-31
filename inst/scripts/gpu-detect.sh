#!/usr/bin/env bash
# gpu-detect.sh — ADR 0001 phase 1: identify GPU/driver requirements on the
# host system, so later phases (model selection, tuning, harness setup) key
# off a profile instead of hardcoding assumptions about one card, the way
# t4-coder.sh currently does for the T4.
#
# Backends detected: cuda (NVIDIA, including MIG slices such as a Jetstream2
# g3.large), metal (Apple Silicon unified memory). Anything else reports
# backend "none" with a warning rather than failing, so callers can still
# choose CPU-only operation deliberately instead of the script deciding for
# them.
#
# This step is always host-native, even when later phases run in a
# container: a container cannot reliably see the real driver/MIG state (see
# ADR 0001), so detection has to happen before any containerization decision.
#
# Usage:
#   ./gpu-detect.sh            # profile to stdout (JSON)
#   ./gpu-detect.sh --save     # also write to $LLM_HOME/gpu-profile.json
#
# Exit status is always 0; a missing/unsupported GPU is a valid detection
# result (backend "none"), not a script failure.

set -euo pipefail

LLM_HOME="${LLM_HOME:-$HOME/llm}"

log()  { printf '\033[1;34m[gpu-detect]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[gpu-detect] WARNING:\033[0m %s\n' "$*" >&2; }

json_str() { # escape a string for embedding in a JSON string literal
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

host_info() {
  local ram_gb disk_gb
  if [[ "$(uname -s)" == Darwin ]]; then
    ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  else
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  fi
  disk_gb=$(df -k "$HOME" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}')
  printf '%s %s' "${ram_gb:-0}" "${disk_gb:-0}"
}

# ---------------------------------------------------------------- CUDA
detect_cuda() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  local line name vram driver cc
  line=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader,nounits 2>/dev/null | head -1) || return 1
  [[ -n "$line" ]] || return 1

  name=$(cut -d',' -f1 <<<"$line" | sed 's/^ *//;s/ *$//')
  vram=$(cut -d',' -f2 <<<"$line" | tr -dc '0-9')
  driver=$(cut -d',' -f3 <<<"$line" | sed 's/^ *//;s/ *$//')
  cc=$(cut -d',' -f4 <<<"$line" | sed 's/^ *//;s/ *$//')

  # mig.mode.current describes the physical card; it may be unqueryable from
  # inside a slice depending on driver version, hence the 2>/dev/null guard.
  local mig_mode
  mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader \
               2>/dev/null | head -1 | sed 's/^ *//;s/ *$//')
  [[ -z "$mig_mode" ]] && mig_mode="N/A"

  # A MIG slice shows up as its own device in `nvidia-smi -L`, e.g.
  # "  MIG 3g.20gb Device 0: (UUID: MIG-...)". Detect it from that listing
  # rather than trusting mig.mode.current alone, since that field reflects
  # the parent card and is not a reliable "am I running inside a slice?"
  # signal by itself.
  local is_mig="false" mig_slice="null" mig_line
  mig_line=$(nvidia-smi -L 2>/dev/null | grep -i "MIG" | head -1) || true
  if [[ -n "$mig_line" ]]; then
    is_mig="true"
    local slice_name
    slice_name=$(sed -n 's/.*MIG \([0-9]*g\.[0-9]*gb\).*/\1/p' <<<"$mig_line")
    [[ -z "$slice_name" ]] && slice_name="unknown"
    mig_slice="\"$(json_str "$slice_name")\""
  fi

  local arch
  case "$cc" in
    7.5) arch="sm_75" ;;   # T4
    8.0) arch="sm_80" ;;   # A100
    8.6) arch="sm_86" ;;   # e.g. A40/A6000
    8.9) arch="sm_89" ;;   # e.g. L4/L40/RTX 40xx
    9.0) arch="sm_90" ;;   # H100
    "") arch="unknown (no compute_cap reported)" ;;
    *)  arch="unknown (compute_cap=$cc)" ;;
  esac

  CUDA_NAME="$name"; CUDA_VRAM_MB="$vram"; CUDA_DRIVER="$driver"
  CUDA_CC="$cc"; CUDA_ARCH="$arch"
  CUDA_MIG_MODE="$mig_mode"; CUDA_IS_MIG="$is_mig"; CUDA_MIG_SLICE="$mig_slice"
  return 0
}

# ---------------------------------------------------------------- Metal
detect_metal() {
  [[ "$(uname -s)" == Darwin ]] || return 1
  command -v system_profiler >/dev/null 2>&1 || return 1

  local chip
  chip=$(system_profiler SPHardwareDataType 2>/dev/null \
           | awk -F': ' '/Chip/ {print $2; exit}')
  [[ -n "$chip" ]] || chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
  # Intel Macs report a brand string, not "Apple Mx"; there is no unified-
  # memory GPU story to detect there, so treat that as "no Metal backend".
  [[ "$chip" == Apple* ]] || return 1

  local mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))

  METAL_CHIP="$chip"; METAL_UNIFIED_MEM_GB="$mem_gb"
  return 0
}

# ---------------------------------------------------------------- output
emit_profile() {
  local backend="$1" ram_gb disk_gb
  read -r ram_gb disk_gb < <(host_info)

  printf '{\n'
  printf '  "backend": "%s",\n' "$backend"
  case "$backend" in
    cuda)
      printf '  "gpu_name": "%s",\n' "$(json_str "$CUDA_NAME")"
      printf '  "vram_mb": %s,\n' "$CUDA_VRAM_MB"
      printf '  "driver_version": "%s",\n' "$(json_str "$CUDA_DRIVER")"
      printf '  "compute_capability": "%s",\n' "$CUDA_CC"
      printf '  "cuda_arch": "%s",\n' "$(json_str "$CUDA_ARCH")"
      printf '  "mig": {"enabled": %s, "mode": "%s", "slice": %s},\n' \
        "$CUDA_IS_MIG" "$(json_str "$CUDA_MIG_MODE")" "$CUDA_MIG_SLICE"
      ;;
    metal)
      printf '  "chip": "%s",\n' "$(json_str "$METAL_CHIP")"
      printf '  "unified_memory_gb": %s,\n' "$METAL_UNIFIED_MEM_GB"
      ;;
    none)
      printf '  "note": "no supported GPU backend detected; CPU-only",\n'
      ;;
  esac
  printf '  "host": {"ram_gb": %s, "disk_free_gb": %s}\n' "$ram_gb" "$disk_gb"
  printf '}\n'
}

main() {
  local save=0
  [[ "${1:-}" == "--save" ]] && save=1

  local backend="none"
  if detect_cuda; then
    backend="cuda"
    log "CUDA backend: $CUDA_NAME, ${CUDA_VRAM_MB} MiB, cc $CUDA_CC ($CUDA_ARCH), MIG=$CUDA_IS_MIG"
  elif detect_metal; then
    backend="metal"
    log "Metal backend: $METAL_CHIP, ${METAL_UNIFIED_MEM_GB} GB unified memory"
  else
    warn "No CUDA (nvidia-smi) or Apple Silicon Metal backend detected; profile will say backend=none."
  fi

  local profile; profile=$(emit_profile "$backend")
  printf '%s\n' "$profile"

  if (( save )); then
    mkdir -p "$LLM_HOME"
    printf '%s\n' "$profile" > "$LLM_HOME/gpu-profile.json"
    log "Saved to $LLM_HOME/gpu-profile.json"
  fi
}

main "$@"
