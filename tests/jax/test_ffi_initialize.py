# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""The FFI initialize stage must not capture work other threads enqueue on XLA's stream."""
import threading

import jax
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.activation import activation
from transformer_engine.jax.layernorm import layernorm

ROWS, COLS = 8, 4096


@jax.jit
def _write_row(buf, row, value):
    return jax.lax.dynamic_update_slice(buf, row[None, :], (value % ROWS, 0))


def _place_and_check(started, stop, errors):
    """Write row `k % ROWS` with value `k` from host, and read the buffer back every ROWS writes."""
    try:
        buf = jnp.zeros((ROWS, COLS), jnp.float32)
        k = 0
        while not stop.is_set():
            for _ in range(ROWS):
                k += 1
                row = jax.device_put(np.full((COLS,), k, np.float32))
                buf = _write_row(buf, row, jnp.int32(k))
            values = np.arange(k - ROWS + 1, k + 1)
            expected = np.empty((ROWS, COLS), np.float32)
            expected[values % ROWS] = values[:, None]
            if not np.array_equal(np.asarray(buf), expected):
                raise AssertionError(f"writes up to {k} did not land")
            started.set()
    except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(e)


def test_initialize_stage_does_not_capture_other_threads_work():
    """First executions of TE programs while another thread places data on the same device."""
    hidden = 1024
    x = jax.random.normal(jax.random.PRNGKey(0), (512, hidden), jnp.bfloat16)
    gamma = jnp.ones((hidden,), jnp.bfloat16)
    beta = jnp.zeros((hidden,), jnp.bfloat16)

    def program(scale):
        def fn(x, gamma, beta):
            # layernorm and activation both register an FFI initialize stage.
            h = layernorm(x, gamma, beta, "layernorm")
            return activation(h[:, None, :], ("gelu",)) * scale

        # Compiled ahead, so each call below is the executable's first execution and nothing else.
        return jax.jit(fn).lower(x, gamma, beta).compile()

    programs = [program(1.0 + i) for i in range(8)]
    started, stop, errors = threading.Event(), threading.Event(), []
    placer = threading.Thread(target=_place_and_check, args=(started, stop, errors), daemon=True)
    placer.start()
    try:
        assert started.wait(timeout=60), errors
        for fn in programs:
            jax.block_until_ready(fn(x, gamma, beta))
    finally:
        stop.set()
        placer.join(timeout=60)
    assert not placer.is_alive(), "the placing thread hung"
    assert not errors, errors
