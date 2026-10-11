ARG PYTORCH_BASE=pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_BASE}

# The CUDA version the image runs; Kura records it with the pinned digest.
ARG KURA_CUDA=13.0
LABEL io.kura.cuda="${KURA_CUDA}"

ARG MUSUBI_TUNER_REF=v0.3.5
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner \
    && cd /opt/musubi-tuner \
    && git checkout "$MUSUBI_TUNER_REF" \
    && pip install --no-cache-dir -e .

# Upstream trainer_base.py (v0.3.5, lines 2050-2054) restarts global_step at zero after
# --resume; 0001 continues the logical step the restored scheduler counted, which Kura's
# Musubi Tuner training-state contract reads. See the patch header.
COPY docker/musubi-tuner/patches/0001-resume-continues-the-logical-step.patch /tmp/kura-musubi-tuner-patches/0001-resume-continues-the-logical-step.patch
RUN cd /opt/musubi-tuner \
    && git apply --check /tmp/kura-musubi-tuner-patches/0001-resume-continues-the-logical-step.patch \
    && git apply /tmp/kura-musubi-tuner-patches/0001-resume-continues-the-logical-step.patch \
    && rm -r /tmp/kura-musubi-tuner-patches

LABEL io.kura.patch.resume-step="logical-global-step-v1"

COPY docker/musubi-tuner/patch_flux2_diffusers_vae.py /tmp/patch_flux2_diffusers_vae.py
RUN python /tmp/patch_flux2_diffusers_vae.py \
    && rm /tmp/patch_flux2_diffusers_vae.py

COPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py
# Installing the trainer must not replace the base image's torch build.
RUN python -c "import sys, torch; cuda = torch.version.cuda or ''; sys.exit(0 if cuda.startswith('${KURA_CUDA}') else 'torch ' + torch.__version__ + ' uses CUDA ' + cuda + ', expected ${KURA_CUDA}')"

WORKDIR /workspace

CMD ["sleep", "infinity"]
