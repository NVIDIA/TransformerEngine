# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

set -xe

export NVTE_JAX_TEST_TIMING=1

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Use --xla_gpu_enable_triton_gemm=false to ensure the reference JAX implementation we are using is accurate.
XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_triton_gemm=false" NVTE_JAX_UNITTEST_LEVEL="L2" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_*

# NCCL EP multi-process suite. The launcher skips when fewer than 4 GPUs or no NVLink is detected.
export NVTE_JAX_UNITTEST_LEVEL="L2"

# Self-hosted NCCL classes (one process per GPU).
NVTE_TEST_EP_CLASSES="TestEP,TestEPOverflowDrop,TestEpDomainGrouping" \
    TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh

# Borrowed-comm classes, in a fresh process group: self-hosted NCCL
# teardown followed by a borrowed-comm re-init in the *same* process is
# fragile (observed a spurious combine-backward mismatch), so keep them
# process-isolated rather than chasing it -- real deployments don't mix
# the two modes in one process either.
NVTE_TEST_EP_CLASSES="TestEPBorrowedComm,TestEpDomainGrouping" \
    TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh

# Same 4 devices split 2 processes x 2 GPUs each (borrowed-comm-only classes;
# self-hosted NCCL doesn't support >1 local device per process). dp=2,ep=2
# here keeps each EP domain inside one process (JAX groups devices by
# process), so this alone doesn't exercise EP traffic crossing a process.
NVTE_TEST_EP_DEVICES_PER_PROC=2 NVTE_TEST_EP_CLASSES="TestEPBorrowedComm,TestEpDomainGrouping" \
    TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh

# Same 2x2 split, but ep=4 (dp=1): the single EP domain spans devices
# [0,1,2,3], forcing NCCL EP traffic across the process boundary.
NVTE_TEST_EP_MESH=1x4 NVTE_TEST_EP_DEVICES_PER_PROC=2 \
    NVTE_TEST_EP_CLASSES="TestEPBorrowedComm,TestEpDomainGrouping" \
    TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh

# All 4 devices in a single process (DEVICES_PER_PROC=4 -> NUM_RUNS=1): the
# single-controller, multi-domain scenario -- one process bootstraps and
# drives two independent EP domains (dp=2, ep=2) across its 4 local GPUs.
NVTE_TEST_EP_DEVICES_PER_PROC=4 \
    NVTE_TEST_EP_CLASSES="TestEpSingleProcessMultiDomain,TestEpDomainGrouping" \
    TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh
