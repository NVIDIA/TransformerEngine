# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

# WAR(rewang) for the "Check failed: reduction_kind.has_value()"
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_xla_runtime_executable=true"

: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/jax/test_distributed_*

