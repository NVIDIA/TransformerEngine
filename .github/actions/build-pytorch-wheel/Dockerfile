# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION=12.9.1
ARG CUDNN_MAJOR_VERSION=9
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ARG AARCH=x86_64

# Install Python
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python$PYTHON_VERSION-dev python$PYTHON_VERSION-venv python3-pip && \
    python$PYTHON_VERSION -m venv /opt/venv


# Install cuda-toolkit
RUN CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | awk -F \. {'print $1'}) && \
    CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | awk -F \. {'print $2'}) && \
    rm /etc/apt/sources.list.d/cuda*.list || true && \
    rm /etc/apt/sources.list.d/nvidia-cuda.list || true && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${AARCH}/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} cudnn-cuda-$CUDA_MAJOR_VERSION libcudnn$CUDNN_MAJOR_VERSION-cuda-$CUDA_MAJOR_VERSION libnccl2 libnccl-dev cmake

# Install PyTorch
RUN export MATRIX_CUDA_VERSION=$(echo $CUDA_VERSION | awk -F \. {'print $1 $2'}) && \
    export MATRIX_TORCH_VERSION=$(echo $TORCH_VERSION | awk -F \. {'print $1 "." $2'}) && \
    export TORCH_CUDA_VERSION=$(python -c "from os import environ as env; \ 
    minv = {'2.5': 118, '2.6': 118, '2.7': 118, '2.8': 126, '2.9': 126}[env['MATRIX_TORCH_VERSION']]; \
    maxv = {'2.5': 124, '2.6': 126, '2.7': 128, '2.8': 129, '2.9': 130}[env['MATRIX_TORCH_VERSION']]; \
    print(minv if int(env['MATRIX_CUDA_VERSION']) < 120 else maxv)" \
    ) && \
    pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}