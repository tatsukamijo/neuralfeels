# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

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
    libgl1 \
    libglu1-mesa \
    build-essential \
    cmake \
    unzip \
    x11-apps \
    xvfb \
    libgtk2.0-dev \
    libgtk-3-dev \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# --- ROS1 Noetic install ---
RUN apt-get update && apt-get install -y --no-install-recommends curl
RUN curl -sS https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $( . /etc/os-release && echo $UBUNTU_CODENAME ) main" > /etc/apt/sources.list.d/ros1-latest.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true \
    && rosdep update --include-eol-distros

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# --- Micromamba install ---
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# --- Working directory ---
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

USER root

# Source ROS1 Noetic in bash profile
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# --- Entrypoint ---
CMD ["bash"]
