#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Multiprocess (one-GPU-per-process) launcher for the unified MoE VJP
# test suite. Forks one pytest invocation per visible GPU, passing each
# its own --num-process=N --process-id=i, and waits for all of them.
# Each child calls jax.distributed.initialize(..., local_device_ids=
# process_id) so each Python process only sees its one GPU as a local
# device and the participating processes form a global mesh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_FILE="$TE_ROOT/tests/jax/test_multiprocess_moe_vjp.py"
PYTEST_INI="$TE_ROOT/tests/jax/pytest.ini"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "[run_multiprocess_moe_vjp.sh] need >=4 GPUs (got $NUM_GPUS); aborting" >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export MOE_VJP_COORDINATOR_ADDRESS="${MOE_VJP_COORDINATOR_ADDRESS:-127.0.0.1:13456}"

echo "============================================================"
echo "MoE VJP MULTIPROCESS test (one process per GPU, ${NUM_GPUS} GPUs)"
echo "  test file          : $TEST_FILE"
echo "  coordinator        : $MOE_VJP_COORDINATOR_ADDRESS"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "============================================================"

# Per-process logs. MOE_VJP_MP_LOG_DIR can be set to a host-mounted dir
# (e.g. when running inside a container that throws away /tmp on exit)
# so logs survive for postmortem inspection. Defaults to a fresh /tmp.
if [ -n "${MOE_VJP_MP_LOG_DIR:-}" ]; then
    LOG_DIR="$MOE_VJP_MP_LOG_DIR"
    mkdir -p "$LOG_DIR"
else
    LOG_DIR=$(mktemp -d -t moe_vjp_mp_XXXXXX)
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

# Launch one pytest per GPU. Process 0 streams to stdout; others log
# only to file so the live output isn't a mosaic.
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

# Wait for all and collect exit codes.
EXITS=()
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        EXITS+=("0")
    else
        EXITS+=("$?")
    fi
done

# Summary.
echo
echo "============================================================"
echo "Per-process exit codes:"
for i in "${!EXITS[@]}"; do
    echo "  proc $i -> ${EXITS[$i]}"
done

# Final pass/fail. Any non-zero in any process fails the suite, but
# we tolerate non-zero on the non-zero processes only if proc 0
# reports PASS (this matches the encoder launcher's logic). Simplest
# strict rule: any non-zero is a failure.
FAILED=0
for e in "${EXITS[@]}"; do
    if [ "$e" != "0" ]; then
        FAILED=1
        break
    fi
done

echo
if [ "$FAILED" -eq 0 ]; then
    echo "[run_multiprocess_moe_vjp.sh] all processes PASSED"
    if [ -z "${MOE_VJP_MP_LOG_DIR:-}" ]; then
        rm -rf "$LOG_DIR"
    fi
    exit 0
fi

echo "[run_multiprocess_moe_vjp.sh] at least one process FAILED"
echo "  retaining logs at $LOG_DIR for diagnosis"
echo "  process 0 tail:"
tail -20 "$LOG_DIR/proc_0.log" 2>/dev/null || true
exit 1
