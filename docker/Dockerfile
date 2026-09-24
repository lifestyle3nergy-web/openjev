# All three OpenJev images, from one file. They share every layer up to and
# including torch (about 8.7 GB on disk), so on one server that part is pulled and stored once:
#   docker build -f docker/Dockerfile                                              -t razorback16/openjev .
#   docker build -f docker/Dockerfile --target encoder --build-arg BACKEND=laya    -t razorback16/openjev-laya .
#   docker build -f docker/Dockerfile --target encoder --build-arg BACKEND=verdict -t razorback16/openjev-verdict .
# Nothing above the torch stage may depend on a build arg, or the images stop sharing it.

# ---- torch: CUDA runtime, Python 3.12 and PyTorch, common to every image ----
# The CUDA base image carries only the driver shims; torch's wheel brings its own
# CUDA 13.0 libraries (kernels include sm_120, Blackwell) as nvidia-* packages.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04 AS torch

ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/opt/python PYTHONUNBUFFERED=1
# gcc: Triton compiles a small C launcher for each kernel on first use
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
# The version vLLM pins; the encoder models accept it too
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache torch==2.13.0

# ---- encoder: Laya or Verdict, loaded in process with PyTorch. No vLLM. ----
# Both models are small (under 1 GB of weights), so they share a GPU with vLLM.
FROM torch AS encoder

ARG BACKEND
ENV OPENJEV_BACKEND=${BACKEND} OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=8080
WORKDIR /app
COPY pyproject.toml README.md ./
COPY openjev ./openjev
RUN test -n "$BACKEND" && uv pip install --python /opt/venv/bin/python --no-cache ".[${BACKEND}]"

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=300s \
    CMD curl -sf http://127.0.0.1:8080/health || exit 1
ENTRYPOINT []
CMD ["python", "-m", "openjev"]

# ---- openjev (default): vLLM with DiffusionGemma structured reads, plus the API ----
# PR #57250 merged on 2026-09-22, carrying the two sampler fixes we ran on a fork
# (eager fallback past torch.compile's recompile limit; logprob stashes of
# different widths), so the pin is upstream's own merge commit. Its image-input
# fix came in separately, as #57589. The commit has a published wheel, so the
# compiled kernels are downloaded rather than built: no CUDA build here.
FROM torch AS openjev

ARG VLLM_REPO=https://github.com/vllm-project/vllm
ARG VLLM_COMMIT=1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8
ARG VLLM_BASE_COMMIT=1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8

# FlashInfer compiles its NVFP4 MoE kernels with nvcc on first start. This is the
# subset of the CUDA toolkit upstream vLLM's own image installs for that, rather
# than the whole devel image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 cuda-nvrtc-dev-13-0 \
        cuda-cuobjdump-13-0 libcurand-dev-13-0 libcublas-dev-13-0 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

RUN git clone --filter=blob:none "$VLLM_REPO" /opt/vllm && git -C /opt/vllm checkout "$VLLM_COMMIT"
# A choice may have 255 options, and one read asks for every option's exact
# logprob. Upstream's cap of 128 ids per request is the only thing in the way;
# it also sizes a per-request buffer in the logprob kernel (512 ids is 2 KiB a
# request). The grep fails the build if the line ever moves or changes.
RUN grep -q '^MAX_LOGPROB_TOKEN_IDS = 128$' /opt/vllm/vllm/sampling_params.py \
    && sed -i 's/^MAX_LOGPROB_TOKEN_IDS = 128$/MAX_LOGPROB_TOKEN_IDS = 512/' /opt/vllm/vllm/sampling_params.py
# Image tokens attend bidirectionally inside their own span, as this checkpoint
# was trained to expect. The patch fails the build if its anchors moved.
COPY docker/patches/vision_prefix_lm.py /tmp/
RUN python /tmp/vision_prefix_lm.py /opt/vllm/vllm/model_executor/models/diffusion_gemma.py \
    && cd /opt/vllm && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_BASE_COMMIT" \
       uv pip install --python /opt/venv/bin/python . \
    && rm -rf /root/.cache/uv

# The API server goes last so a code change only rebuilds a few small layers.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY openjev ./openjev
RUN uv pip install --python /opt/venv/bin/python --no-cache .
COPY docker/entrypoint.sh /usr/local/bin/openjev-entrypoint

ENV OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=900s \
    CMD curl -sf http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["openjev-entrypoint"]
