# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import jax.numpy as jnp
import time
import math

from typing import Callable, Any, Dict, Optional, Tuple
from flax import linen as nn
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.flax.transformer import DotProductAttention as TEDotProductAttention


def speedometer(
    model_apply_fn: Callable,
    variables: Any,
    input: jnp.ndarray,
    output_grad: jnp.ndarray,
    dropout_key: jax.random.PRNGKey,
    model_init_fn: Callable = None,
    forward_kwargs: dict = {},
    autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> None:
    """Measure average runtime for a JAX module
    Perform forward and backward passes .
    """
    if autocast_kwargs is None:
        autocast_kwargs = {"enabled": False}
        model_init_fn = None

    train_step_fn = create_train_step_fn(model_apply_fn, autocast_kwargs, forward_kwargs)

    # Warm up runs
    key = dropout_key
    for _ in range(warmup_iters):
        key, step_key = jax.random.split(key)
        loss, (param_grads, other_grads) = train_step_fn(variables, input, output_grad, step_key)

    # Timing runs
    start = time.time()
    for _ in range(timing_iters):
        key, step_key = jax.random.split(key)
        loss, (param_grads, other_grads) = train_step_fn(variables, input, output_grad, step_key)
    end = time.time()

    print(f"Mean time: {(end - start) * 1000 / timing_iters} ms")


def create_train_step_fn(
    model_apply_fn: Callable,
    autocast_kwargs: Dict[str, Any],
    forward_kwargs: Dict[str, Any] = None,
) -> Callable:
    """
    Creates a JIT-compiled function that performs one forward/backward pass.
    """

    if forward_kwargs is None:
        forward_kwargs = {}

    def loss_fn(variables: Any, inp: jnp.ndarray, grad_target: jnp.ndarray, dropout_key):
        rngs = {"dropout": dropout_key}
        with te.autocast(**autocast_kwargs):
            # Forward Pass: Apply the model using current parameters and variables
            call_kwargs = {**forward_kwargs, "rngs": rngs}
            out = model_apply_fn(variables, inp, **call_kwargs)

        # grad_target = derivative of L (loss fn) over y (output) = signma(L)/sigma(y)
        # where grad_w(L) = gradient of loss over params = sigma(L)/sigma(y) * sigma(y)/sigma(w) --> chain rule
        #  sigma(y)/sigma(w) = J_model(w)
        return jnp.vdot(out, grad_target)

    def fwd_bwd_fn(*args, **kwargs):
        return jax.value_and_grad(loss_fn, argnums=(0, 1))(*args, **kwargs)

    # Use jax.value_and_grad to get the loss value and gradients simultaneously. (forward + backward pass)
    # ∇_params[output^T · grad_target] = grad_target^T · J_output(params) = VJP
    # fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    # JIT-compile the fwd_bwd_fn
    return jax.jit(fwd_bwd_fn)
