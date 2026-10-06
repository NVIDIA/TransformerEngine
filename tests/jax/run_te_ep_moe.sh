#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Multiprocess (one-GPU-per-process) launcher for the TE-EP MoE custom_vjp
# test suite. Forks one pytest invocation per visible GPU, passing each
# its own --num-process=N --process-id=i, and waits for all of them. Each
# child calls jax.distributed.initialize(..., local_device_ids=process_id)
# so each Python process only sees its one GPU as a local device and the
# participating processes form a global (ep, fsdp) mesh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_FILE="${TEST_FILE:-$TE_ROOT/tests/jax/test_te_ep_moe.py}"
PYTEST_INI="$TE_ROOT/tests/jax/pytest.ini"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "[run_te_ep_moe.sh] need >=4 GPUs (got $NUM_GPUS); SKIPPING."
    exit 0
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export TE_EP_MOE_COORDINATOR_ADDRESS="${TE_EP_MOE_COORDINATOR_ADDRESS:-127.0.0.1:13457}"

echo "============================================================"
echo "TE-EP MoE MULTIPROCESS test (one process per GPU, ${NUM_GPUS} GPUs)"
echo "  test file          : $TEST_FILE"
echo "  coordinator        : $TE_EP_MOE_COORDINATOR_ADDRESS"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
if [ "$#" -gt 0 ]; then
    echo "  extra pytest args  : $*"
fi
echo "============================================================"

if [ -n "${TE_EP_MOE_MP_LOG_DIR:-}" ]; then
    LOG_DIR="$TE_EP_MOE_MP_LOG_DIR"
    mkdir -p "$LOG_DIR"
else
    LOG_DIR=$(mktemp -d -t te_ep_moe_mp_XXXXXX)
fi
echo "Per-process logs: $LOG_DIR"

PIDS=()
EXITS=()
PHASE_FAILED=0

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

run_phase() {
    local phase_name="$1"
    local use_cudnn_fusion="$2"
    shift 2
    local -a phase_args=("$@")
    local phase_log_dir="$LOG_DIR/$phase_name"
    mkdir -p "$phase_log_dir"
    PIDS=()
    EXITS=()

    echo
    echo "============================================================"
    echo "Phase: $phase_name"
    echo "  use_cudnn_fusion=$use_cudnn_fusion"
    echo "  phase pytest args : ${phase_args[*]:-<none>}"
    echo "  logs              : $phase_log_dir"
    echo "============================================================"

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local log_file="$phase_log_dir/proc_${i}.log"
        local -a pytest_cmd=(
            python3 -m pytest -c "$PYTEST_INI"
            "$TEST_FILE"
            -p no:typeguard
            -v -s
            --num-process="$NUM_GPUS"
            --process-id="$i"
            --use-cudnn-fusion="$use_cudnn_fusion"
            "${phase_args[@]}"
        )
        if [ "$i" -eq 0 ]; then
            echo "=== Live output from process 0 ($phase_name) ==="
            "${pytest_cmd[@]}" 2>&1 | tee "$log_file" &
        else
            "${pytest_cmd[@]}" > "$log_file" 2>&1 &
        fi
        PIDS+=("$!")
    done

    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            EXITS+=("0")
        else
            EXITS+=("$?")
        fi
    done

    echo
    echo "Per-process exit codes for $phase_name:"
    for i in "${!EXITS[@]}"; do
        echo "  proc $i -> ${EXITS[$i]}"
    done

    local failed=0
    for e in "${EXITS[@]}"; do
        if [ "$e" != "0" ] && [ "$e" != "5" ]; then
            failed=1
            break
        fi
    done
    if [ "$failed" -ne 0 ]; then
        PHASE_FAILED=1
        echo "[run_te_ep_moe.sh] phase $phase_name FAILED"
        echo "  process 0 tail:"
        tail -20 "$phase_log_dir/proc_0.log" 2>/dev/null || true
    else
        echo "[run_te_ep_moe.sh] phase $phase_name PASSED"
    fi
}

# Keep ordinary CUDA C++ and cuDNN JAX coverage in separate process groups.
# TE EP/NCCL caches alignment process-wide; these use 128 and 256 respectively.
run_phase "ordinary" "0" -k "not TestTeEpMoeCudnnCutedslFusion" "$@"
run_phase "cutedsl" "1" -k "TestTeEpMoeCudnnCutedslFusion" "$@"

echo
if [ "$PHASE_FAILED" -eq 0 ]; then
    echo "[run_te_ep_moe.sh] all phases PASSED"
    if [ -z "${TE_EP_MOE_MP_LOG_DIR:-}" ]; then
        rm -rf "$LOG_DIR"
    fi
    exit 0
fi

echo "[run_te_ep_moe.sh] at least one phase FAILED"
echo "  retaining logs at $LOG_DIR for diagnosis"
exit 1
