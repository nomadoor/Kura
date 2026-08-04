ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

ARG SD_SCRIPTS_REF=6721028c79ee85a78b3a06dfd8954dae310a1cce
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
      org.opencontainers.image.revision="6721028c79ee85a78b3a06dfd8954dae310a1cce" \
      io.kura.patch.symlink-safetensors="preserve-input-filename-v2" \
      io.kura.backend="sd-scripts"

WORKDIR /workspace
CMD ["sleep", "infinity"]
