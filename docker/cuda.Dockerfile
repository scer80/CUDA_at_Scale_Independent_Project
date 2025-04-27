FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt update \
    && apt install -y \
    apt-utils \
    ca-certificates \
    cmake \
    cudnn9-cuda-12 \
    curl \
    gdb \
    git \
    gnupg2 \
    krb5-user \
    libgl1 \
    libgl1-mesa-glx \
    libopencv-dev \
    mesa-utils \
    nano \
    openssh-server \
    sudo \
    vim \
    wget \
    tzdata \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

    # nvidia-cuda-toolkit \
    # nvidia-cudnn \
    # 


RUN apt update \
    && apt install -y \
    python3.10 \
    python3-virtualenv \
    python3-distutils \
    python3-pip \
    python3-apt \
    python-is-python3

RUN pip install \
    nvidia_cudnn_frontend    

    RUN cd /tmp/ \
    && git clone https://github.com/catchorg/Catch2.git \
    && cd Catch2 \
    && cmake -B build -S . -DBUILD_TESTING=OFF \
    && cmake --build build \
    && sudo cmake --install build

RUN cd opt/ \
    && git clone https://github.com/NVIDIA/cudnn-frontend.git \
    && cd cudnn-frontend \
    && mkdir build \
    && cd build \
    && cmake -DCUDNN_PATH=/usr/include/x86_64-linux-gnu -DCUDAToolkit_ROOT=/usr/local/cuda-12.2  ../ \
    && cmake --build . -j16

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
