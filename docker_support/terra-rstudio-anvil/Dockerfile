# check=skip=SecretsUsedInArgOrEnv

#######################################################################
# Stage 1: build llama.cpp for t4-coder (local agentic coding)
#   CUDA 12.8 to match Terra's T4 driver; Ubuntu 22.04 to match terra-base.
#   Only the binaries plus three CUDA runtime libs are carried into the
#   final image, not the multi-GB CUDA toolkit.
#######################################################################
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS llama-build

# Pin a llama.cpp release tag for reproducible images
ARG LLAMA_CPP_REF=b11435
# 75-real = native T4 kernels; 80-virtual = PTX fallback for newer GPUs
ARG CUDA_ARCHS="75-real;80-virtual"

RUN apt-get update \
 && apt-get install -y --no-install-recommends git cmake build-essential libssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp /src

# Same recipe as llama.cpp's own Linux CUDA release: CPU backend variants are
# selected at runtime (the build machine's CPU flags don't leak in), and
# libraries are found next to the binary via an $ORIGIN rpath.
RUN cmake -S /src -B /src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 && cmake --build /src/build --config Release -j"$(nproc)"

RUN mkdir -p /opt/llama/bin \
 && cp -a /src/build/bin/. /opt/llama/bin/ \
 && cp -L /usr/local/cuda/lib64/libcudart.so.12 \
          /usr/local/cuda/lib64/libcublas.so.12 \
          /usr/local/cuda/lib64/libcublasLt.so.12 /opt/llama/bin/

#######################################################################
# Stage 2: RStudio + Bioconductor on terra-base
#######################################################################
FROM us.gcr.io/broad-dsp-gcr-public/terra-base:1.0.1

USER root

ENV R_VERSION="4.6.1"
ENV R_HOME="/usr/local/lib/R"
ENV TZ="Etc/UTC"

COPY scripts/install_R_source.sh /rocker_scripts/install_R_source.sh
RUN /rocker_scripts/install_R_source.sh

ENV CRAN="https://cloud.r-project.org"
ENV LANG=en_US.UTF-8

COPY scripts/bin/ /rocker_scripts/bin/
COPY scripts/setup_R.sh /rocker_scripts/setup_R.sh
RUN <<EOF
if grep -q "1000" /etc/passwd; then
    userdel --remove "$(id -un 1000)";
fi
/rocker_scripts/setup_R.sh
EOF

ENV S6_VERSION="v2.1.0.2"
ENV RSTUDIO_VERSION="2026.09.0+174"
ENV DEFAULT_USER="rstudio"
ENV RSTUDIO_PORT=8001
ENV RSTUDIO_HOME=/etc/rstudio
ENV RSTUDIO_USERSETTING=/home/rstudio/.config/rstudio/rstudio-prefs.json

COPY scripts/install_rstudio.sh /rocker_scripts/install_rstudio.sh
COPY scripts/install_s6init.sh /rocker_scripts/install_s6init.sh
COPY scripts/default_user.sh /rocker_scripts/default_user.sh
COPY scripts/init_set_env.sh /rocker_scripts/init_set_env.sh
COPY scripts/init_userconf.sh /rocker_scripts/init_userconf.sh

RUN apt update && apt install -y --no-install-recommends zlib1g-dev perl xz-utils
RUN Rscript -e "install.packages(c('tinytex', 'BiocManager'), repos = 'https://cloud.r-project.org')"

RUN apt-get update \
 && apt-get install -y --no-install-recommends perl xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "tinytex::install_tinytex(dir = '/opt/TinyTeX', add_path = FALSE)" \
 && /opt/TinyTeX/bin/x86_64-linux/tlmgr option sys_bin /usr/local/bin \
 && /opt/TinyTeX/bin/x86_64-linux/tlmgr path add \
 && chmod -R a+rwX /opt/TinyTeX

RUN Rscript -e "tinytex::tlmgr_install(c('fancyhdr', 'titlesec', 'enumitem'))"

RUN apt-get update && apt install -y libuv1-dev
RUN Rscript -e "BiocManager::install(c('GenomicRanges', 'Seqinfo', 'limma', 'DESeq2', 'scater', 'scran', \
     'SpatialExperiment', 'SpatialFeatureExperiment'))"

RUN /rocker_scripts/install_rstudio.sh

## System libraries for common R/Bioconductor source builds
##   libcurl/bz2: curl, Rhtslib; magick++: magick; udunits/gdal/geos/proj/sqlite: units, sf, terra;
##   fftw3: fftwtools/EBImage. Plus vim and xz-utils for convenience.
## NVIDIA user-space driver (libcuda.so.1 + nvidia-smi): Terra did not provide these in this
## RStudio container. The 580 series must match the host's kernel driver.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      xz-utils vim libcurl4-openssl-dev libbz2-dev libmagick++-dev \
      libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite3-dev libfftw3-dev \
      libnvidia-compute-580 nvidia-utils-580 \
 && rm -rf /var/lib/apt/lists/*

## Local agentic coding: prebuilt llama.cpp, Node 22, Pi/OpenCode, hf CLI, t4-coder
COPY --from=llama-build /opt/llama /opt/llama
COPY scripts/t4-coder.sh /usr/local/bin/t4-coder
COPY scripts/install_llm_tools.sh /rocker_scripts/install_llm_tools.sh
RUN /rocker_scripts/install_llm_tools.sh

ENV PASSWORD="bioc"

ENV RSTUDIO_HOME=/home/rstudio
EXPOSE 8001
ENTRYPOINT ["/init"]

