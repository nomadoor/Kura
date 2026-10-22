ARG PYTORCH_BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

# The CUDA version the image runs; Kura records it with the pinned digest.
ARG KURA_CUDA=12.8
LABEL io.kura.cuda="${KURA_CUDA}"

ARG SD_SCRIPTS_REF=690ea7f96c23182352ec63def76d431c6120bd2f
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV SD_SCRIPTS_ROOT=/opt/sd-scripts

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The base image still trusts the eMudhra intermediate CAs whose keys were compromised on
# 2026-10-19. Raise every trust store in the image to the first release that drops them; the
# build fails if one stays below it. It runs before anything else is fetched, and its own
# downloads check against the Ubuntu store it has just raised. Remove this step once the
# base image ships them.
RUN apt-get update \
    && apt-get install -y --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge 20260816ubuntu0.22.04.1 \
    && CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt conda install -y -n base -c conda-forge "ca-certificates>=2026.10.19" "certifi>=2026.10.19" "pip>=26.2.2" \
    && conda clean -afy

RUN git clone https://github.com/kohya-ss/sd-scripts.git "$SD_SCRIPTS_ROOT" \
    && cd "$SD_SCRIPTS_ROOT" \
    && git checkout "$SD_SCRIPTS_REF" \
    && test "$(git rev-parse HEAD)" = "$SD_SCRIPTS_REF" \
    && pip install --no-cache-dir -r requirements.txt

COPY docker/sd-scripts/patches/0001-preserve-safetensors-symlink-name.patch /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch
# Upstream train_network.py (v0.12.0, lines 1512-1516) sets global_step to the remainder
# within the epoch after a Resume skips whole epochs; 0002 keeps it at the logical step,
# which Kura's sd-scripts training-state contract reads. See the patch header.
COPY docker/sd-scripts/patches/0002-resume-continues-the-logical-step.patch /tmp/kura-sd-scripts-patches/0002-resume-continues-the-logical-step.patch
# Upstream library/flux_utils.py (v0.12.0, lines 136-141 and 161-166) stops stripping the
# model.diffusion_model. prefix at the first key without it; with safetensors 0.8 key order an
# all-in-one FLUX.1 file then leaves the DiT on the meta device. 0003 strips it from every key.
COPY docker/sd-scripts/patches/0003-strip-the-flux-prefix-from-every-key.patch /tmp/kura-sd-scripts-patches/0003-strip-the-flux-prefix-from-every-key.patch

RUN cd "$SD_SCRIPTS_ROOT" \
    && git apply --check /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch \
    && git apply /tmp/kura-sd-scripts-patches/0001-preserve-safetensors-symlink-name.patch \
    && git apply --check /tmp/kura-sd-scripts-patches/0002-resume-continues-the-logical-step.patch \
    && git apply /tmp/kura-sd-scripts-patches/0002-resume-continues-the-logical-step.patch \
    && git apply --check /tmp/kura-sd-scripts-patches/0003-strip-the-flux-prefix-from-every-key.patch \
    && git apply /tmp/kura-sd-scripts-patches/0003-strip-the-flux-prefix-from-every-key.patch

LABEL org.opencontainers.image.source="https://github.com/kohya-ss/sd-scripts" \
      org.opencontainers.image.revision="${SD_SCRIPTS_REF}" \
      io.kura.patch.symlink-safetensors="preserve-input-filename-v2" \
      io.kura.patch.resume-step="logical-global-step-v1" \
      io.kura.patch.flux-prefix="every-key-v1" \
      io.kura.backend="sd-scripts"

# Installing the trainer must not replace the base image's torch build.
RUN python -c "import sys, torch; cuda = torch.version.cuda or ''; sys.exit(0 if cuda.startswith('${KURA_CUDA}') else 'torch ' + torch.__version__ + ' uses CUDA ' + cuda + ', expected ${KURA_CUDA}')"

WORKDIR /workspace
CMD ["sleep", "infinity"]
