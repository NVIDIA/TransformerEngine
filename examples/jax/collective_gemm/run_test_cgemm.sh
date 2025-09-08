# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=$(nvidia-smi -L | wc -l)

mpirun --map-by core -n $NUM_GPUS pytest -vs examples/jax/collective_gemm/test_gemm.py
wait

mpirun --map-by core -n $NUM_GPUS pytest -vs examples/jax/collective_gemm/test_dense_grad.py
wait

mpirun --map-by core -n $NUM_GPUS pytest -vs examples/jax/collective_gemm/test_layernorm_mlp_grad.py
