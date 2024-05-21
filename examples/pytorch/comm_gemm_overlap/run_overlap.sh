#!/bin/bash

UB_SKIPMC=1 torchrun --nproc-per-node=4 overlapped_ln_mlp.py "$@"
