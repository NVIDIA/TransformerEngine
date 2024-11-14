# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

## 4 GPU Tests

# DP 2 TP 2
bash test-pax.sh --data-parallel 2 --tensor-parallel 2 --enable-te --multiprocess

# PP 2 FSDP 2
bash test-pax.sh --fsdp 2 --pipeline-parallel 2 --enable-te --multiprocess

# PP 2 FSDP 2 + FP 8
ENABLE_FP8=1 bash test-pax.sh --fsdp 2 --pipeline-parallel 2 --enable-te --multiprocess

# PP 2 FSDP 2 + FP 8 + single-process
ENABLE_FP8=1 bash test-pax.sh --fsdp 2 --pipeline-parallel 2 --enable-te
