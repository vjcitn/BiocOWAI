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
2. Check compatibility between `compbiobench-runner`'s requirements and this
   host's configuration (see "Phase 2 finding" below — this turned out not
   to be primarily a GPU question).
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

### Phase 2 finding (resolved 2026-10-31, corrected same day)

Phase 2 was scoped from real source, not inferred from `t4-coder.sh` — but
the first pass only checked upstream `Genentech/compbiobench-runner`
(`environment.yml` + `run_benchmark.py`) and missed that this project
actually has its own fork with further commits and a merged PR. Checking
both gives the accurate picture:

- **`compbiobench-runner` has no GPU/CUDA dependency at all**, in either
  repo. It is a conda-based orchestrator — one fresh, isolated conda
  environment per question (CPU bioinformatics tools: samtools, bwa, blast,
  minimap2, etc., `python=3.11`) driving a system-installed, authenticated
  LLM CLI, via an extensible `LLMProvider` ABC
  (`build_command`/`parse_output`/`extract_answer`/`build_test_command`,
  registered in `LLM_PROVIDERS`).
- **Upstream `Genentech/compbiobench-runner`** ships only three providers:
  `claude`, `codex`, `gemini`. No `pi` or `opencode` provider exists there.
- **`vjcitn/compbiobench-runner` (the fork this project actually
  targets)** already has a `PiProvider`, merged to `main` via PR #1 ("Add
  PiProvider: Pi coding agent CLI support for locally-served models").
  It wires Pi (`@earendil-works/pi-coding-agent`) to a locally-served model
  selected purely by the model string + Pi's own `~/.pi/agent/models.json`
  — `ollama/gpt-oss:20b` initially, then `llama-cpp/qwen3.6-35b-a3b`
  matching `t4-coder.sh`'s llama-server + `ALIAS` convention. Crucially,
  `PiProvider` sets `uses_conda_env = False`: Pi is a per-user Node.js CLI
  install, not cloned into a conda env per question, so the conda/disk
  sizing that applies to `claude`/`codex`/`gemini` simply doesn't apply to
  Pi-backed runs.
- So the harness-provider gap identified in the first pass — "Pi isn't
  natively drivable through `compbiobench-runner`'s own runner" — is true
  of upstream but **already solved in the fork**. `CBBBioc`'s
  `cbb_run_question()` driving Pi directly in R remains a separate, valid
  path (useful when you don't want `compbiobench-runner`'s runner at all),
  but it is not the only option, and BiocOWAI's actual target is the fork's
  `PiProvider`, not a provider that still needs to be written.

GPU still enters only indirectly: whatever model backs the harness has to
answer within the runner's per-question timeout (`-t/--timeout`, default
120 minutes). `inst/scripts/compbiobench-check.sh` checks conda, disk
headroom (skipped for providers with `uses_conda_env = False`, such as
`pi` in the fork), and harness-provider support — introspecting a real
`COMPBIOBENCH_HOME` checkout when available (both which providers exist
*and* each one's `uses_conda_env` flag), falling back to a
`COMPBIOBENCH_REMOTE=fork`-vs-`upstream` distinction otherwise — and
reports the phase-1 GPU profile as informational context for the timeout
question rather than as a pass/fail input.

**Lesson applied going forward:** when checking compatibility against an
external dependency this project has forked, check the fork and its
branches first, not just upstream — upstream's limitations don't
necessarily apply once local changes are in play.

### Phase 3: harness install/config (resolved 2026-10-31)

`inst/scripts/harness-install.sh` lifts `t4-coder.sh`'s
`ensure_node()`/`install_pi()`/`install_opencode()` out into a standalone
script and generalizes them. The key realization driving the design: Pi
and OpenCode don't need to know anything about GPU backend at all — they
only need a base URL, a provider key, and a model alias, which is the same
shape whether the server behind that URL is llama.cpp CUDA on a T4/A100/MIG
slice, llama.cpp Metal on an M4, or Ollama. So phase 3 takes no dependency
on phase 1's GPU profile; it is backend-agnostic by construction, not by
branching on the profile. `BASE_URL`/`PROVIDER_KEY`/`ALIAS` are the only
inputs, and phases 4/5 (model selection, tuning) are what will eventually
supply them per backend.

Preserves exact compatibility with the compbiobench-runner fork's
`PiProvider`: the default `PROVIDER_KEY=llama-cpp` and `ALIAS=qwen3.6-35b-a3b`
reproduce `t4-coder.sh`'s existing `~/.pi/agent/models.json` entry, so
`llama-cpp/qwen3.6-35b-a3b` keeps working as a `PiProvider` model string
unchanged.

One portability bug found and fixed during testing: Node.js's release
feed names Darwin and Linux assets differently per-architecture but both
publish a `.tar.gz`; the script matches on that instead of mixing `-xJ`
(tar.xz) and `-xz` (tar.gz) by platform. Verified locally end-to-end
(Node download/extraction + `~/.pi/agent/models.json` write + existing-Pi-
binary detection) on this machine's Apple M4; the OpenCode branch's
`curl | bash` install step is unexercised live (sandbox correctly declines
to run arbitrary third-party install scripts) but its config-writing logic
is the same proven pattern as Pi's.

### Early Metal assessment (2026-10-31)

Before designing phases 4/5 for Apple Silicon by analogy with the CUDA/T4
path, `inst/scripts/metal-assess.sh` was written and run end-to-end on this
project's actual M4 (32 GB unified memory) to get real numbers instead of
assumptions:

- `llama.cpp` built cleanly with `-DGGML_METAL=ON` using Xcode's AppleClang
  + cmake; no CUDA-path dependencies apply (no `nvcc`, no CUDA-toolkit
  pinning). The first pass installed `cmake` via Homebrew when none was on
  `PATH` — this was corrected (see "cmake policy" below) once it became
  clear that was a real conflict risk, not a hypothetical one.
- A modest single-file quant (`Qwen2.5-7B-Instruct`, Q3_K_M, ~3.8 GB) was
  used deliberately instead of the 35B MoE class `t4-coder.sh` targets, to
  keep this first pass fast; most of that HF repo's larger quants
  (Q4_K_M/Q4_0/Q5/Q6/Q8) are shipped as multi-file shards, which is itself
  a phase-4 consideration (multi-shard download is more involved than
  `t4-coder.sh`'s single-file `hf_hub_download`).
- With all layers offloaded to the GPU (`--n-gpu-layers 999`): ~48 tok/s
  prompt processing, ~18 tok/s token generation, negligible resident-memory
  growth (expected — a 3.8 GB model is a rounding error against 32 GB
  unified memory).
- Two bugs found and fixed during this exercise, both now fixed in
  `metal-assess.sh` and `harness-install.sh`: (1) an unbraced `$VAR`
  immediately followed by the unicode ellipsis in log strings was
  misparsed as part of the variable name under this shell/locale
  (`${VAR}…` fixed it; the same latent bug still exists in `t4-coder.sh`,
  noted but not fixed here since that file is out of this round's scope);
  (2) the assumed GGUF filename convention didn't match the real
  Hugging Face repo (hyphens vs underscores, and quant availability as
  single-file vs multi-shard varies by repo and has to be checked, not
  assumed).

### cmake policy: no package-manager fallback (2026-11-01)

The first assessment pass's `ensure_build_deps()` installed `cmake` via
Homebrew when none was found on `PATH`. The user flagged this as a real
conflict risk, not a hypothetical one: this is a bioinformatics
development machine that already carries an R/Bioconductor-managed cmake
(e.g. `~/Library/Caches/org.R-project.R/R/biocmake/<ver>/CMake.app/Contents/bin/cmake`),
and a second cmake landing on `PATH` from a package manager can shadow or
conflict with source builds that expect that specific one.

**Fixed:** `metal-assess.sh` no longer touches Homebrew (or any package
manager) at all. It requires `CMAKE_BIN` to be set to an explicit,
executable cmake path and fails immediately with a clear message if it
isn't — no default, no installation, no PATH search. The Homebrew `cmake`
formula installed during the first pass was uninstalled; verified the
script subsequently fails correctly with `CMAKE_BIN` unset, and succeeds
using the user's actual Bioconductor-managed cmake (4.4.2) pointed at
explicitly, with consistent benchmark results.

**Lesson applied going forward:** a "just install what's missing" default
is wrong on a machine with other toolchains that care which binary of a
given name is in play (compilers, cmake, python, etc.) — default to
requiring an explicit path from the user rather than reaching for a
package manager, on this project.

### Target model for M4: gpt-oss-20b (2026-11-01)

The Qwen2.5-7B placeholder used for the first Metal pass was replaced with
this machine's actual target model: `gpt-oss-20b`, native MXFP4 quant
(`ggml-org/gpt-oss-20b-GGUF`, single file, ~12.1 GB) — matching the
`ollama/gpt-oss:20b` alias the compbiobench-runner fork's `PiProvider`
already references (see the Phase 2 finding above). With `--n-gpu-layers
999` (full GPU offload) and `--ctx-size 8192`: ~48-69 tok/s prompt
processing, ~35 tok/s token generation, consistent across repeated runs
and across the Homebrew-cmake and Bioconductor-cmake builds. 35 tok/s
generation is comfortably interactive for a coding-agent harness. This is
still a single-request measurement, not a sweep (see open questions
above) — a real phase-5 tuning pass would still check whether holding
some layers off-GPU changes throughput, but the "does the target model
even run well here" question for M4 is now answered: yes.

This answers the open questions the script posed going in: at 32 GB
unified memory, a small-to-mid single-file model is comfortable headroom,
but the 35B MoE class remains untested here and multi-shard download
support would be needed to reach the larger quants on this same repo —
both real inputs for phase 4's model-selection guidance, not resolved by
this pass. Whether `--n-gpu-layers 999` is actually optimal (vs. a Metal-
side tuning sweep analogous to `--n-cpu-moe`) is also still open; this
pass used the naive "offload everything" setting and didn't sweep it.

### T4/CUDA companion script: cuda-assess.sh (2026-11-01, not yet run)

Added `inst/scripts/cuda-assess.sh` as the T4 analog of `metal-assess.sh`,
but it deliberately does not duplicate `t4-coder.sh`'s build/download
logic: it drives `t4-coder.sh install` and `t4-coder.sh start` (already
T4-validated: CUDA build at `sm_75`, CUDA toolkit pinned to 12.8,
`unsloth/Qwen3.6-35B-A3B-GGUF` at `UD-Q4_K_XL`, ~22 GB, matching the
fork's `llama-cpp/qwen3.6-35b-a3b` `PiProvider` alias), then adds the one
piece `t4-coder.sh` doesn't have: a single-shot `/completion` benchmark
and report in the same format `metal-assess.sh` used, so M4 and T4 numbers
are directly comparable.

This was a deliberate reuse decision, not an oversight: `t4-coder.sh`'s
CUDA/apt/model-download logic is already working and already exercised by
real users, so reimplementing it here would only risk reintroducing a bug
already found and fixed elsewhere in this pass — the unbraced-`$VAR`-
before-ellipsis bash parsing quirk, which is fixed in `metal-assess.sh`
and `harness-install.sh` but **still present, unfixed, in `t4-coder.sh`
itself at lines 148 and 163** (noted when first found, carried forward as
a known, currently out-of-scope issue).

The cmake-policy question (no package-manager fallback, require an
explicit path) was deliberately **not** carried over to the T4/Linux side
in this pass. The rationale the user gave for the Mac policy was specific
to this machine's dual toolchain (Homebrew vs. Bioconductor-managed
cmake); `t4-coder.sh`'s existing `sudo apt-get`/CUDA-toolkit install path
predates that policy and runs on disposable cloud GPU instances rather
than a personal dev workstation with a competing R/Bioconductor
toolchain, so the conflict risk that motivated the Mac change doesn't
obviously apply. Left as an open question below rather than assumed
either way, since changing it would mean editing already-working,
already-used code without being able to test the result locally.

**Run on a real T4 (2026-11-01)**, via `cuda-assess.sh assess` on the
user's T4 container (commit 7cc8962, before the `SKIP_APT`/
`SKIP_CUDA_TOOLKIT` addition below). Full pipeline succeeded end to end:
apt deps, CUDA toolkit, llama.cpp CUDA build for sm_75, model already
cached from a prior run, Pi harness install, server start at
`N_CPU_MOE=22`/`ctx=65536`, benchmark. Result: **35.3 tok/s prompt
processing, 31.2 tok/s token generation** (VRAM 12431/15360 MiB). That's
comparable to the M4's ~35 tok/s generation on gpt-oss-20b, despite
Qwen3.6-35B-A3B being a much larger model running with its MoE experts
partly offloaded to CPU RAM rather than fully GPU-resident -- a good sign
that `--n-cpu-moe` offload is doing its job. Confirms the "does the
harness-serving pipeline work end to end on T4" question: yes. (This run
used the default `N_CPU_MOE=22`, not a `tune` sweep, so whether a
different split pushes throughput higher is still open, same as the M4
Metal-layer-offload question above.)

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
- Phase 2 (compbiobench-runner compatibility) turned out to be a
  harness-provider-support problem, not a GPU-compatibility problem — see
  "Phase 2 finding" above. That problem is already solved for Pi in the
  fork (`vjcitn/compbiobench-runner`'s `PiProvider`, `uses_conda_env =
  False`); it is only unsolved for OpenCode, and only if running through
  `compbiobench-runner`'s own runner rather than CBBBioc's direct-driving
  path.
- Containerization is partial by design: useful for the CUDA backends,
  not applicable to Metal, and never covering phase 1. Anyone expecting a
  single image to cover all three targets will be disappointed; that is a
  deliberate scope limit, not an oversight.

## Open questions (not decided here)

- Exact schema for the GPU profile (what fields phase 1 emits).
- Where model-selection guidance (phase 4) draws its catalog from — static
  table vs. a live lookup.
- Whether an OpenCode `LLMProvider` is worth writing for the fork (Pi's is
  already done), or whether OpenCode-backed questions should just go
  through CBBBioc's direct-driving path indefinitely.
- Whether `vjcitn/compbiobench-runner` stays a long-lived fork or its
  `PiProvider` work gets proposed upstream to Genentech at some point.
- ~~Whether the "no package-manager fallback, require an explicit path"
  cmake policy (M4/Homebrew, above) should extend to `t4-coder.sh`'s `sudo
  apt-get`/CUDA-toolkit install steps on the T4/Linux side~~ — partially
  addressed 2026-11-01: a live T4/container run surfaced that `sudo` isn't
  guaranteed on every T4 host (the user asked directly: "is this assuming
  sudo privileges on cuda machines?" — yes, unconditionally, before this
  fix). Added `SKIP_APT=1` / `SKIP_CUDA_TOOLKIT=1` env vars to
  `t4-coder.sh`: each skips its `sudo apt-get`/`sudo dpkg -i` step and
  instead fails fast with a clear message listing what's missing (cmake,
  git, curl, jq, python3, gcc/g++, libcurl for apt; nvcc on PATH or under
  `/usr/local/cuda*/bin` for the toolkit) rather than hanging on a sudo
  password prompt or silently assuming privilege. This is opt-in, not the
  default -- `t4-coder.sh`'s existing sudo-apt behavior is unchanged when
  the vars are unset, so existing users see no difference. Verified both
  the fail-fast and pass-through paths in isolation (extracted function
  bodies run standalone, since this machine has no CUDA/apt to test
  against directly). Not yet exercised on an actual no-sudo T4 host.
- Whether a `tune`-style `--n-cpu-moe` sweep on T4 beats the default 22
  used in the 2026-11-01 run above (31.2 tok/s generation) -- not yet
  swept, same open question as the M4's untried `--n-gpu-layers` sweep.
