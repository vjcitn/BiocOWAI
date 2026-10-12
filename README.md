# BiocOWAI - open weight open harness AI with rstudio targeting AnVIL T4 GPU

The main resources in this repo are 

- Dockerfiles that defines a preconfigured environment with llama.cpp to serve LLMs and pi (or OpenCode) as a coding harness
- a script, t4-coder.sh, authored Claude Opus 5.5, that sets up and tunes the server to the available GPU/CPU resources


t4-coder runs an open-weight coding model entirely inside a Terra environment: 

- llama.cpp’s llama-server serves Qwen3.6-35B-A3B through an OpenAI-compatible API on localhost, and a lightweight agent harness (Pi or OpenCode) uses it to read code, edit files and run commands
- The model is a mixture-of-experts with 35B total but only ~3B active parameters per token, so it can be split across hardware.  Attention and shared weights stay on the T4’s 16 GB of GPU memory, while the experts for some layers live in system RAM and are computed on the CPU
- The `tune` process for t4-coder finds the best GPU/CPU split for your terra configuration

```
# t4-coder.sh — local agentic coding on a single Tesla T4 (Ubuntu, CUDA 12.x, sudo)
#
#   Model:   Qwen3.6-35B-A3B, Unsloth dynamic GGUF UD-Q4_K_XL (~22 GB)
#   Server:  llama.cpp llama-server built for sm_75, MoE experts partly in CPU RAM
#   Harness: Pi (default) and/or OpenCode, pointed at http://127.0.0.1:8181/v1
#
# Usage:
#   ./t4-coder.sh install      # deps, build llama.cpp, download model, install harness
#   ./t4-coder.sh tune         # try several --n-cpu-moe values, save the fastest that fits
#   ./t4-coder.sh start        # launch llama-server in the background
#   ./t4-coder.sh status       # health, VRAM, last log lines
#   ./t4-coder.sh stop
#
# Overrides go in the environment and win over the saved config, e.g.
#   N_CPU_MOE=28 ./t4-coder.sh start      THINK=0 ./t4-coder.sh start
#   HARNESS=both ./t4-coder.sh install    TUNE_VALUES="20 22 24 26" ./t4-coder.sh tune
#
# Everything lives under $LLM_HOME (default ~/llm) so it sits on the Terra persistent
# disk. apt packages may need reinstalling if the cloud environment is recreated;
```
