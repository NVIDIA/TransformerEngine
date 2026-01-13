# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import jax.numpy as jnp
import time

from typing import Callable, Any, Dict, Optional, Tuple
import transformer_engine.jax as te


def speedometer(
    model_apply_fn: Callable,
    variables: Any,
    input: jnp.ndarray,
    output_grad: jnp.ndarray,
    model_init_fn: Callable = None,
    forward_kwargs: dict = {},
    autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
    rngs: Dict[str, jax.random.PRNGKey] = None,
) -> None:
    """Measure average runtime for a JAX module
    Perform forward and backward passes .
    """
    if autocast_kwargs is None:
        autocast_kwargs = {"enabled": False}
        model_init_fn = None

    if rngs is None:
        rngs = {}

    train_step_fn = create_train_step_fn(model_apply_fn, autocast_kwargs, forward_kwargs)

    # Warm up runs
    for _ in range(warmup_iters):
        rngs, step_rngs = _split_step_rngs(rngs)
        loss, (param_grads, other_grads) = train_step_fn(variables, input, output_grad, step_rngs)

    # Timing runs
    start = time.time()
    for _ in range(timing_iters):
        rngs, step_rngs = _split_step_rngs(rngs)
        loss, (param_grads, other_grads) = train_step_fn(variables, input, output_grad, step_rngs)
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

    def loss_fn(
        variables: Any,
        inp: jnp.ndarray,
        grad_target: jnp.ndarray,
        rngs: Dict[str, jax.random.PRNGKey],
    ):
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


def _split_step_rngs(
    rngs: Dict[str, jax.random.PRNGKey],
) -> Tuple[Dict[str, jax.random.PRNGKey], Dict[str, jax.random.PRNGKey]]:
    """Splits each RNG in the rngs dictionary for a new step."""
    step_rngs = {}
    new_rngs = {}
    for name, key in rngs.items():
        new_key, step_key = jax.random.split(key)
        new_rngs[name] = new_key
        step_rngs[name] = step_key
    return new_rngs, step_rngs
