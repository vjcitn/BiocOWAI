# BiocOWAI - open weight open harness AI with rstudio targeting AnVIL T4 GPU

The main resources in this repo are 

- a Dockerfile that defines a preconfigured environment with llama.cpp to serve LLMs and pi (or OpenCode) as a coding harness
- a script, t4-coder.sh, authored Claude Opus 5.5, that sets up and tunes the server to the available GPU/CPU resources


t4-coder runs an open-weight coding model entirely inside a Terra environment: 

- llama.cpp’s llama-server serves Qwen3.6-35B-A3B through an OpenAI-compatible API on localhost, and a lightweight agent harness (Pi or OpenCode) uses it to read code, edit files and run commands
- The model is a mixture-of-experts with 35B total but only ~3B active parameters per token, so it can be split across hardware.  Attention and shared weights stay on the T4’s 16 GB of GPU memory, while the experts for some layers live in system RAM and are computed on the CPU
- The `tune` process for t4-coder finds the best GPU/CPU split for your terra configuration
