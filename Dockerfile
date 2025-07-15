# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# --- Constants ---
ARG WORKDIR=/workspace/neuralfeels

# --- System setup ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    sudo \
    git \
    git-lfs \
    vim \
    curl \
    wget \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- Micromamba install ---
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# --- Working directory ---
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

USER root

# --- Entrypoint ---
CMD ["bash"]
