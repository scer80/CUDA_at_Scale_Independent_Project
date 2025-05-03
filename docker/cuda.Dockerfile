# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

ENV TZ=Europe/Budapest
RUN echo "tzdata tzdata/Areas select Europe" > debconf-preseed.txt && \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> debconf-preseed.txt && \
    debconf-set-selections debconf-preseed.txt && \
    apt-get update && \
    TZ=Etc/UTC apt-get install -y --no-install-recommends \
        tzdata

##################
# Ubuntu packages
##################
RUN apt update \
    && apt install -y --no-install-recommends \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    cudnn9-cuda-11 \
    curl \
    ffmpeg \
    git \
    gnupg2 \
    krb5-user \
    libgl1 \
    libffi-dev \
    libgmp-dev \
    libopencv-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    lshw \
    nano \
    openssh-client \
    openssh-server \
    pciutils \
    software-properties-common \
    ssh \
    sudo \
    vim \
    wget \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN cd opt/ \
    && git clone https://github.com/NVIDIA/cudnn-frontend.git \
    && cd cudnn-frontend \
    && mkdir build \
    && cd build \
    && cmake -DCUDNN_PATH=/usr/include/x86_64-linux-gnu -DCUDAToolkit_ROOT=/usr/local/cuda-11.8  ../ \
    && cmake --build . -j16

RUN apt update \
    && apt install -y \
    python3.10 \
    python3-virtualenv \
    python3-distutils \
    python3-pip \
    python3-apt \
    python-is-python3

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    nvidia_cudnn_frontend \
    tqdm

ARG HOME
ARG USER_NAME
ARG USER_UID
ARG USER_GID

RUN echo "Creating group user" && \
    groupadd --gid ${USER_GID} user && \
    echo "Creating user " ${USER_NAME} " with home " ${HOME} && \
    useradd -l -u $USER_UID -s /bin/bash -g user --home-dir ${HOME} -m $USER_NAME  && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER_NAME

WORKDIR $HOME