# Reproducible CPU environment for the TASCP artifact.
#
#   docker build -f environment/Dockerfile -t tascp-eval .
#   docker run --rm -v "$PWD:/work" -w /work tascp-eval make smoke
#
# For GPU reproduction build with a CUDA base image (e.g.
# nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04) and install the CUDA torch
# wheel; every target except the exp22 Transformer comparison is CPU-feasible.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends make git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt requirements.txt
COPY environment/requirements.lock environment/requirements.lock
# Install the pinned direct requirements (the lock additionally pins the full
# transitive closure; use it if you need bit-for-bit interpreter parity).
RUN pip install -r requirements.txt

COPY . .

# Install the package itself and run the reproducibility smoke checks.
RUN pip install -e . --no-deps
CMD ["make", "smoke"]
