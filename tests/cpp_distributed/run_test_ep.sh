#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Run TE EP distributed unit tests via mpirun. Each MPI rank pins to one GPU
# (rank % device_count) and exchanges ncclUniqueId through MPI_Bcast.
#
# Usage:
#   bash run_test_ep.sh [num_gpus] [build_dir]
#
# Defaults:
#   num_gpus  = number of GPUs visible to nvidia-smi
#   build_dir = <script_dir>/build
#
# Environment variables:
#   GTEST_FILTER     - forwarded to all processes (e.g., "EPPipelineTest.*")
#   GTEST_XML_PREFIX - if set, each rank writes JUnit XML to
#                      ${GTEST_XML_PREFIX}.rank<N>.xml
#   MPIRUN           - override the mpirun binary (default: mpirun)
#   MPIRUN_EXTRA     - extra flags forwarded to mpirun

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${2:-${SCRIPT_DIR}/build}"
NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MPIRUN="${MPIRUN:-mpirun}"

# Skip cleanly on pre-Hopper: NCCL EP requires SM>=90.
MIN_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | awk -F. 'NR==1 || ($1*10+$2)<min { min=$1*10+$2 } END { print min+0 }')
if (( MIN_SM > 0 && MIN_SM < 90 )); then
    echo "NCCL EP requires SM>=90 (lowest visible GPU is SM${MIN_SM}); SKIPPING."
    exit 0
fi

# NCCL EP requires active NVLink P2P among ranks on the node.
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
    echo "NVLink not detected on this platform; SKIPPING."
    exit 0
fi

TEST_BIN="${BUILD_DIR}/test_ep"
if [[ ! -x "${TEST_BIN}" ]]; then
    echo "ERROR: binary not found: ${TEST_BIN}"
    echo "Build:  cd ${SCRIPT_DIR} && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

if (( NUM_GPUS < 2 )); then
    echo "EP Tests: requires at least 2 GPUs, found ${NUM_GPUS}. Skipping."
    exit 0
fi

GTEST_ARGS="${GTEST_FILTER:+--gtest_filter=${GTEST_FILTER}}"

echo "=== EP Tests ==="
echo "  GPUs: ${NUM_GPUS}   Binary: ${TEST_BIN}"
echo

if [[ -n "${GTEST_XML_PREFIX:-}" ]]; then
    # bash -c so OMPI_COMM_WORLD_RANK expands per-rank, avoiding a write race
    # on a single shared output path.
    "${MPIRUN}" --allow-run-as-root --oversubscribe -n "${NUM_GPUS}" ${MPIRUN_EXTRA:-} bash -c \
        "exec '${TEST_BIN}' ${GTEST_ARGS} --gtest_output=xml:${GTEST_XML_PREFIX}.rank\${OMPI_COMM_WORLD_RANK}.xml"
else
    "${MPIRUN}" --allow-run-as-root --oversubscribe -n "${NUM_GPUS}" ${MPIRUN_EXTRA:-} "${TEST_BIN}" ${GTEST_ARGS}
fi
