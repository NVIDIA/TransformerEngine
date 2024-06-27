#!/bin/bash

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

env MPICC=$(which mpicc) python -m pip install mpi4py

mpiexec \
--oversubscribe \
-np ${NUM_GPU} \
-x CUDA_DEVICE_MAX_CONNECTIONS=1 \
-x MASTER_ADDR=$(hostname) \
-x MASTER_PORT=33558 \
-x PATH \
python ln_mlp_with_overlap.py --num-replicas 2
