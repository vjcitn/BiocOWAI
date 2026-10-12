# BiocOWAI - open weight open harness AI with rstudio targeting AnVIL T4 GPU

## Overview

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

## Note on tuning and configuration

On the 8 CPU, 52 GB RAM terra cloud env, we have
```
rstudio@6adc68012135:~$ time t4-coder tune
n-cpu-moe  prefill t/s  decode t/s   VRAM used     
16         failed to load (likely OOM; see /home/rstudio/llm/logs/tune-16.log)
20         289          31.1         13337 MiB     
24         250          28.7         11483 MiB     
28         214          26.9         9625 MiB      
32         189          24           7771 MiB      
[t4-coder] Fastest was n-cpu-moe=20 (31.1 t/s). Saved N_CPU_MOE=22 (+2 for context headroom).
[t4-coder] Next: ./t4-coder.sh start

real    2m0.712s
user    2m39.004s
sys     0m15.214s
```
Note the location of the configuration file:
```
rstudio@6adc68012135:~$ ls ~/llm
config.env  logs  models
rstudio@6adc68012135:~$ cat ~/llm/config.env
: "${HF_REPO:=unsloth/Qwen3.6-35B-A3B-GGUF}"
: "${QUANT:=UD-Q4_K_XL}"
: "${ALIAS:=qwen3.6-35b-a3b}"
: "${PORT:=8181}"
: "${CTX:=65536}"
: "${N_CPU_MOE:=22}"
: "${HARNESS:=pi}"
: "${THINK:=1}"
```
