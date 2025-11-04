# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import jax.numpy as jnp
import time
import math
from enum import Enum

from typing import Callable, Any, Dict, Optional, Tuple
from flax import linen as nn
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.flax.transformer import DotProductAttention as TEDotProductAttention


class AttentionType(Enum):
    """Enum for selecting attention implementation type."""
    CUSTOM_DOT_PRODUCT = "custom_dot_product"
    FLAX_LINEN_MULTIHEAD = "flax_linen_multihead"
    TE_FLAX_MULTIHEAD = "te_flax_multihead"


class AttentionWrapper(nn.Module):
    """
    Args:
        num_attention_heads: Number of attention heads
        hidden_size: Hidden dimension size
        kv_channels: Dimension per attention head (hidden_size // num_attention_heads)
        attention_dropout: Dropout rate for attention
        attention_type: Type of attention implementation to use (default: TE_FLAX_MULTIHEAD)
        attention_mask_type: Mask type for TE attention (default: 'no_mask')
    """
    
    num_attention_heads: int
    hidden_size: int
    kv_channels: int
    attention_dropout: float = 0.1
    attention_type: AttentionType = AttentionType.TE_FLAX_MULTIHEAD
    attention_mask_type: str = 'no_mask'
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:

        # Create the attention module based on attention_type
        if self.attention_type == AttentionType.CUSTOM_DOT_PRODUCT:
            attention = DotProductAttention(
                num_attention_heads=self.num_attention_heads,
                kv_channels=self.kv_channels,
                attention_dropout=self.attention_dropout,
            )
            x = attention(query, key, value, attention_mask, deterministic=deterministic)
            x = te_flax.DenseGeneral(features=self.hidden_size, use_bias=True)(x)
            x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)
            return x
        
        elif self.attention_type == AttentionType.FLAX_LINEN_MULTIHEAD:
            # Flax Linen expects [batch, seq_len, num_heads, head_dim]
            # Input is [seq_len, batch, num_heads, head_dim]
            sq, b, np, hn = query.shape
            
            # [seq_len, batch, num_heads, head_dim] -> [batch, seq_len, num_heads * head_dim]
            query_reshaped = jnp.transpose(query, (1, 0, 2, 3)).reshape(b, sq, np * hn)
            key_reshaped = jnp.transpose(key, (1, 0, 2, 3)).reshape(b, key.shape[0], np * hn)
            value_reshaped = jnp.transpose(value, (1, 0, 2, 3)).reshape(b, value.shape[0], np * hn)
            
            attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.kv_channels,
                dropout_rate=self.attention_dropout,
            )
            # Output shape: [batch, seq_len, num_heads * head_dim]
            output = attention(query_reshaped, key_reshaped, value_reshaped, mask=attention_mask, deterministic=deterministic)
            
            # Reshape back to [seq_len, batch, num_heads * head_dim]
            output = jnp.transpose(output, (1, 0, 2))
            return output
        
        elif self.attention_type == AttentionType.TE_FLAX_MULTIHEAD:
            # Use DotProductAttention (not MultiHeadAttention which includes QKV projection)
            attention = TEDotProductAttention(
                head_dim=self.kv_channels,
                num_attention_heads=self.num_attention_heads,
                num_gqa_groups=self.num_attention_heads,  # No GQA, set equal to num_attention_heads
                attention_dropout=self.attention_dropout,
                attn_mask_type=self.attention_mask_type,
                transpose_batch_sequence=True,  # Expected format: [seq_len, batch, num_heads, head_dim]
            )
            x = attention(query, key, value, mask=attention_mask, deterministic=deterministic)
            # Reshape from [seq_len, batch, num_heads, head_dim] to [seq_len, batch, hidden_size]
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
            return x
        
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


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

    def fwd_bwd_fn(*args, **kwargs):
        return jax.value_and_grad(loss_fn, argnums=(0, 1))(*args, **kwargs)

    # Use jax.value_and_grad to get the loss value and gradients simultaneously. (forward + backward pass)
    # ∇_params[output^T · grad_target] = grad_target^T · J_output(params) = VJP
    # fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    # JIT-compile the fwd_bwd_fn
    return jax.jit(fwd_bwd_fn)

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