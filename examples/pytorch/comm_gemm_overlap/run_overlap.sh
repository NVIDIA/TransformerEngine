#!/bin/bash

NVTE_BIAS_GELU_NVFUSION=0 \
UB_SKIPMC=1 \
torchrun --nproc-per-node=4 overlapped_ln_mlp.py "$@"
