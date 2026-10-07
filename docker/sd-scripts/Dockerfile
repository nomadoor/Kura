ARG PYTORCH_BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

# The CUDA version the image runs; Kura records it with the pinned digest.
ARG KURA_CUDA=12.8
LABEL io.kura.cuda="${KURA_CUDA}"

ARG SD_SCRIPTS_REF=37a1cbbc5725ed2a3575506e7bd2001c9908ac92
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV SD_SCRIPTS_ROOT=/opt/sd-scripts

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/kohya-ss/sd-scripts.git "$SD_SCRIPTS_ROOT" \
    && cd "$SD_SCRIPTS_ROOT" \
    && git checkout "$SD_SCRIPTS_REF" \
    && test "$(git rev-parse HEAD)" = "$SD_SCRIPTS_REF" \
    && pip install --no-cache-dir -r requirements.txt

COPY docker/sd-scripts/patches/0001-preserve-safetensors-symlink-name.patch /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch

RUN cd "$SD_SCRIPTS_ROOT" \
    && git apply --check /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch \
    && git apply /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch

LABEL org.opencontainers.image.source="https://github.com/kohya-ss/sd-scripts" \
      org.opencontainers.image.revision="37a1cbbc5725ed2a3575506e7bd2001c9908ac92" \
      io.kura.patch.symlink-safetensors="preserve-input-filename-v2" \
      io.kura.backend="sd-scripts"

# Installing the trainer must not replace the base image's torch build.
RUN python -c "import sys, torch; cuda = torch.version.cuda or ''; sys.exit(0 if cuda.startswith('${KURA_CUDA}') else 'torch ' + torch.__version__ + ' uses CUDA ' + cuda + ', expected ${KURA_CUDA}')"

WORKDIR /workspace
CMD ["sleep", "infinity"]
