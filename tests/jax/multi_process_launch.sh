# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

#!/bin/bash

SCRIPT_NAME="${SCRIPT_NAME:-test.py}"


XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_command_buffer=''"

export XLA_FLAGS="${XLA_BASE_FLAGS}"

NUM_RUNS=$(nvidia-smi -L | wc -l)
for ((i=1; i<NUM_RUNS; i++))
do
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME 127.0.0.1:12345 $i $NUM_RUNS > /dev/null 2>&1 &
done

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_NAME 127.0.0.1:12345 0 $NUM_RUNS

wait
