ARG PYTORCH_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_BASE}

ARG MUSUBI_TUNER_REF=main
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

COPY docker/musubi-tuner/patch_flux2_diffusers_vae.py /tmp/patch_flux2_diffusers_vae.py
RUN python /tmp/patch_flux2_diffusers_vae.py \
    && rm /tmp/patch_flux2_diffusers_vae.py

COPY docker/ai-toolkit/kura_runpod_object_job.py /opt/kura_runpod_object_job.py
WORKDIR /workspace

CMD ["sleep", "infinity"]
