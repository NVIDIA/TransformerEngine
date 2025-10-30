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


def speedometer(
    model_apply_fn: Callable,
    variables: Any,
    input: jnp.ndarray,
    output_grad: jnp.ndarray,
    dropout_key: jax.random.PRNGKey,
    model_init_fn: Callable = None,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> None:
    """Measure average runtime for a JAX module
    Perform forward and backward passes .
    """
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}
        model_init_fn = None

    train_step_fn = create_train_step_fn(model_apply_fn, fp8_autocast_kwargs, forward_kwargs)

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
    fp8_autocast_kwargs: Dict[str, Any],
    forward_kwargs: Dict[str, Any] = None,
) -> Callable:
    """
    Creates a JIT-compiled function that performs one forward/backward pass.
    """

    if forward_kwargs is None:
        forward_kwargs = {}

    def loss_fn(variables: Any, inp: jnp.ndarray, grad_target: jnp.ndarray, dropout_key):
        rngs = {"dropout": dropout_key}
        with te.fp8_autocast(**fp8_autocast_kwargs):
            # Forward Pass: Apply the model using current parameters and variables
            call_kwargs = {**forward_kwargs, "rngs": rngs}
            out = model_apply_fn(variables, inp, **call_kwargs)

        # grad_target = derivative of L (loss fn) over y (output) = signma(L)/sigma(y)
        # where grad_w(L) = gradient of loss over params = sigma(L)/sigma(y) * sigma(y)/sigma(w) --> chain rule
        #  sigma(y)/sigma(w) = J_model(w)
        return jnp.vdot(out, grad_target)

    # Use jax.value_and_grad to get the loss value and gradients simultaneously. (forward + backward pass)
    # ∇_params[output^T · grad_target] = grad_target^T · J_output(params) = VJP
    fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    # JIT-compile the fwd_bwd_fn
    return jax.jit(fwd_bwd_fn)


def create_train_step_fn_vjp(
    model_apply_fn: Callable,
    fp8_autocast_kwargs: Dict[str, Any],
    forward_kwargs: Dict[str, Any] = None,
) -> Callable:
    """
    Alternative implementation using JAX's vjp directly instead of jnp.vdot + jax.grad.
    This is more explicit about computing the Vector-Jacobian Product.
    """

    if forward_kwargs is None:
        forward_kwargs = {}

    def train_step_fn(variables: Any, inp: jnp.ndarray, grad_target: jnp.ndarray, dropout_key):
        """Compute forward pass and VJP in one step"""

        # Define forward function that closes over dropout_key
        def forward_fn(variables: Any, inp: jnp.ndarray):
            """Pure forward function for VJP computation"""
            rngs = {"dropout": dropout_key}
            with te.fp8_autocast(**fp8_autocast_kwargs):
                call_kwargs = {**forward_kwargs, "rngs": rngs}
                return model_apply_fn(variables, inp, **call_kwargs)

        # Compute forward pass and get VJP function (w.r.t. variables and inp)
        output, vjp_fn = jax.vjp(forward_fn, variables, inp)

        # Compute gradients using VJP - returns gradients w.r.t. variables and inp
        var_grads, inp_grads = vjp_fn(grad_target)

        # Return loss value and gradients
        loss_value = jnp.vdot(output, grad_target)
        return loss_value, (var_grads, inp_grads)

    return jax.jit(train_step_fn)


class DotProductAttention(nn.Module):
    """Attention operation in Transformer layer
    Built with plain JAX/Flax modules.
    """

    num_attention_heads: int
    kv_channels: int
    attention_dropout: float

    def setup(self):
        self.projection_size = self.kv_channels * self.num_attention_heads
        self.hidden_size_per_attention_head = self.kv_channels
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def masked_softmax(self, inp: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        if mask is not None:
            inp = jnp.where(mask, -10000.0, inp)
        return nn.softmax(inp, axis=-1)

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        sq, b, np, hn = query.shape
        sk = key.shape[0]
        # sq: sequence length of Query
        # b : batch size
        # np: num parallel heads
        # hn: hidden size per attention head, also == self.kv_channels
        # sk: sequence length of Key

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query = query.reshape(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.reshape(sk, b * np, -1)

        # Batch matrix multiplication: b = b * np, q = sq, k = sk, n = hn
        # getting QK^T per head/batch / sqrt(d_k). with output shape (batch * num head, query seq length, key seq length)
        bmm1 = jnp.einsum("qbn,kbn->bqk", query, key) / self.norm_factor

        # change view to [b, np, sq, sk]
        # separate num heads and batch
        attention_scores = bmm1.reshape(b, np, sq, sk)

        # softmax (QK^T / sqrt(d_k))
        attention_probs = self.masked_softmax(attention_scores, attention_mask)

        attention_probs = nn.Dropout(rate=self.attention_dropout)(
            attention_probs, deterministic=deterministic
        )

        # change view [sk, b * np, hn]
        value = value.reshape(sk, b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = jnp.einsum("bqk,kbn->bqn", attention_probs, value)

        # change view [b, np, sq, hn]
        context = context.reshape(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = jnp.transpose(context, (2, 0, 1, 3))

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.reshape(sq, b, self.projection_size)

        return context


class BasicMLP(nn.Module):
    """Feed-forward network in Transformer layer
    Built with plain JAX/Flax modules.
    """

    hidden_size: int
    ffn_hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.ffn_hidden_size, use_bias=True)(x)
        x = nn.gelu(x, approximate=True)  # equivalent to tanh approximation
        x = nn.Dense(features=self.hidden_size, use_bias=True)(x)
        return x