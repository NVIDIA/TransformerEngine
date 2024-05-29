#!/bin/bash

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

ARGS=()
NPROCS=4
while [[ $# -gt 0 ]]; do
      case "$1" in
        --nproc-per-node)
            shift;
            NPROCS="$1";
            shift;
            ;;
        --nproc-per-node=*)
            NPROCS=$(echo "$1" | rev | cut -d'=' -f1 | rev);
            shift;
            ;;
        *)
            ARGS+=("$1");
            shift;
            ;;
    esac;
done;

UB_SKIPMC=1 torchrun --nproc-per-node=${NPROCS} \
    ../../../examples/pytorch/comm_gemm_overlap/test_gemm.py "${ARGS[@]}";
