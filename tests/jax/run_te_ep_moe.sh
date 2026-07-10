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
TEST_FILE="$TE_ROOT/tests/jax/test_te_ep_moe.py"
PYTEST_INI="$TE_ROOT/tests/jax/pytest.ini"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "[run_te_ep_moe.sh] need >=4 GPUs (got $NUM_GPUS); aborting" >&2
    exit 1
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
echo "============================================================"

if [ -n "${TE_EP_MOE_MP_LOG_DIR:-}" ]; then
    LOG_DIR="$TE_EP_MOE_MP_LOG_DIR"
    mkdir -p "$LOG_DIR"
else
    LOG_DIR=$(mktemp -d -t te_ep_moe_mp_XXXXXX)
fi
echo "Per-process logs: $LOG_DIR"

PIDS=()

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

for i in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="$LOG_DIR/proc_${i}.log"
    PYTEST_CMD=(
        python3 -m pytest -c "$PYTEST_INI"
        "$TEST_FILE"
        -p no:typeguard
        -v -s
        --num-process="$NUM_GPUS"
        --process-id="$i"
    )
    if [ "$i" -eq 0 ]; then
        echo "=== Live output from process 0 ==="
        "${PYTEST_CMD[@]}" 2>&1 | tee "$LOG_FILE" &
    else
        "${PYTEST_CMD[@]}" > "$LOG_FILE" 2>&1 &
    fi
    PIDS+=("$!")
done

EXITS=()
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        EXITS+=("0")
    else
        EXITS+=("$?")
    fi
done

echo
echo "============================================================"
echo "Per-process exit codes:"
for i in "${!EXITS[@]}"; do
    echo "  proc $i -> ${EXITS[$i]}"
done

# Treat exit 0 (pass) and exit 5 (pytest "no tests collected", which the
# file emits via pytest.skip(allow_module_level=True) on pre-Blackwell
# GPUs) as success.
FAILED=0
for e in "${EXITS[@]}"; do
    if [ "$e" != "0" ] && [ "$e" != "5" ]; then
        FAILED=1
        break
    fi
done

echo
if [ "$FAILED" -eq 0 ]; then
    echo "[run_te_ep_moe.sh] all processes PASSED"
    if [ -z "${TE_EP_MOE_MP_LOG_DIR:-}" ]; then
        rm -rf "$LOG_DIR"
    fi
    exit 0
fi

echo "[run_te_ep_moe.sh] at least one process FAILED"
echo "  retaining logs at $LOG_DIR for diagnosis"
echo "  process 0 tail:"
tail -20 "$LOG_DIR/proc_0.log" 2>/dev/null || true
exit 1
