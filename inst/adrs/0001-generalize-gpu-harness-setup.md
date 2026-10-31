# ADR 0001: Generalize GPU/harness setup beyond the T4

- Status: Proposed
- Date: 2026-10-31

## Context

`t4-coder.sh` is a working, single-file script that installs and tunes a local
agentic-coding stack (llama.cpp + an open-weight MoE model + Pi/OpenCode) on a
single NVIDIA Tesla T4. It hard-codes T4 assumptions throughout: CUDA
architecture `sm_75`, a model/quant sized to fit a 16 GB card with CPU-expert
offload, and a `tune` step that searches `--n-cpu-moe` values for that one
memory budget.

BiocOWAI needs to run the same kind of stack on hardware that is not a T4:

- **Jetstream2 `g3.large`**: a MIG slice of an A100, ~20 GB VRAM. MIG slices
  report differently from a physical card under `nvidia-smi`, disable some
  CUDA features (e.g. peer access), and are not "a smaller A100" in every
  respect that matters for sizing a model.
- **A full A100 (40 GB)**, e.g. the Channing host:

  ```
  NVIDIA-SMI 610.57.04   KMD Version: 610.57.04   CUDA UMD Version: 13.3
  GPU 0  NVIDIA A100-PCIE-40GB   40960MiB total, 0% util, MIG Disabled
  ```

  Enough VRAM headroom that CPU-expert offload and aggressive quantization
  are no longer the binding constraint; the sizing question changes shape.
- **Apple silicon (M4) via Metal**: not CUDA at all. Different build
  (llama.cpp Metal backend), different memory model (unified memory, not a
  VRAM/system-RAM split), and no `--n-cpu-moe`-style offload concept in the
  same form.

At the same time, `CBBBioc` (`cbb_run_question()`) drives Pi against a fixed
question set and does not want to know anything about GPUs — it just needs a
running harness pointed at a model. Any redesign of BiocOWAI's setup script
should keep that boundary clean: CBBBioc talks to a harness endpoint and a
model alias, never to GPU detection or tuning logic directly.

The work splits into six phases, in order of dependency:

1. Identify GPU and driver requirements on the host system.
2. Check compatibility between `compbiobench-runner`'s requirements and the
   detected GPU configuration.
3. Install and configure the harness (currently Pi; OpenCode is a secondary
   target `t4-coder.sh` already supports).
4. Help select an appropriate open-weight model for the detected hardware.
5. Support tuning, where tuning is a meaningful concept for the hardware
   (as in `t4-coder.sh`'s `--n-cpu-moe` search).
6. Document interactive use and scripted/driven use of the harness.

## Decision

**Replace the single T4-specific script with a small GPU-detection layer plus
per-backend profiles, keeping the existing verb-based CLI
(`install`/`tune`/`start`/`status`/`stop`) as the stable interface.**

Concretely:

- **Phase 1 (detection) is host-native and backend-agnostic.** A detection
  step runs outside any container (it must see the real host) and emits a
  small profile describing what it found: backend (`cuda` | `metal`),
  VRAM or unified-memory size, whether the device is a MIG slice (and if so,
  which features that disables), compute capability / CUDA arch, and driver
  version. This replaces the single `preflight()` check in `t4-coder.sh` that
  assumes `nvidia-smi` and warns-but-continues if the card isn't a T4.
- **Everything downstream keys off that profile, not off hardcoded
  assumptions.** Model/quant choice (phase 4), build flags (CUDA arch or
  Metal), and whether `tune` (phase 5) does anything at all are all profile
  lookups, not script branches sprinkled through install logic. A MIG 20 GB
  slice and a dedicated 20 GB card are *not* assumed equivalent — MIG gets
  its own detection branch, not a fallback onto "generic 20 GB CUDA."
- **Metal is a separate profile module, not a variant of the CUDA path.**
  Unified memory and the absence of an offload-split concept mean the
  tuning and sizing logic for M4 is genuinely different code, not a
  parameterization of the CUDA offload search.
- **Containerize phases 3 onward, per backend family; leave phase 1 and
  Metal out of containers.** A CUDA container (building on the existing
  `docker_support/terra-rstudio-anvil` pattern) can carry the harness +
  llama.cpp CUDA build once a profile has already been chosen on the host.
  Metal has no meaningful container story on macOS, so M4 stays host-native
  end to end. GPU detection itself always runs on the host regardless of
  whether later phases are containerized, since a container cannot see the
  real driver/MIG state reliably.
- **CBBBioc integration stays a stable, backend-agnostic contract.** Phase 6
  documents a harness endpoint (URL) and model alias as the only things
  CBBBioc-side code needs; it never inspects GPU profile or tuning state.
  This preserves the existing boundary where `cbb_run_question()` only knows
  about Pi's HTTP interface.
- **Keep `t4-coder.sh`'s operational shape** (config file under `$LLM_HOME`,
  env-var overrides winning over saved config, `install` skipping finished
  steps, background server + `status`/`stop`) as the pattern the
  generalized tool follows, rather than inventing a new CLI convention.

## Consequences

- `t4-coder.sh` becomes the T4 profile implementation under the new
  structure, not a separate script to maintain in parallel; existing T4
  users see the same CLI.
- Adding a new target (e.g. a different MIG size, a future GPU generation)
  means adding a profile, not editing branch logic throughout an
  install script.
- MIG-specific handling and Metal-specific handling are both net-new work,
  not reuse of the CUDA/T4 offload-tuning code — this ADR accepts that cost
  in exchange for not forcing three hardware families through one code path.
- Phase 2 (compbiobench-runner compatibility) has no prior art in this repo
  to generalize from; it will need to be scoped from
  `compbiobench-runner`'s actual requirements, not inferred from `t4-coder.sh`.
- Containerization is partial by design: useful for the CUDA backends,
  not applicable to Metal, and never covering phase 1. Anyone expecting a
  single image to cover all three targets will be disappointed; that is a
  deliberate scope limit, not an oversight.

## Open questions (not decided here)

- Exact schema for the GPU profile (what fields phase 1 emits).
- Where model-selection guidance (phase 4) draws its catalog from — static
  table vs. a live lookup.
- Whether `compbiobench-runner`'s GPU requirements (phase 2) are documented
  anywhere authoritative yet, or need to be reverse-engineered from its code.
