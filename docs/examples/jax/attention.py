# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX: BSHD attention with TransformerEngine.

Companion source for ``attention.rst``. Code blocks between
``# ATTENTION_*_START`` / ``# ATTENTION_*_END`` markers are pulled into the RST
via ``literalinclude``.

Run as a script to exercise the example end-to-end:

    python docs/examples/jax/attention.py
"""

# ATTENTION_IMPORTS_START
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import quickstart_jax_utils as utils

from transformer_engine.jax.attention import SequenceDescriptor
from transformer_engine.jax.flax import DotProductAttention

# ATTENTION_IMPORTS_END


# ATTENTION_INPUTS_START
batch, seq, num_query_heads, num_kv_heads, head_dim = 2, 4096, 128, 8, 128
window_size = (128, 0)
dtype = jnp.bfloat16
timing_iters = 20
warmup_iters = 10


def create_qkv_inputs(
    *,
    seed: int,
    kv_heads: int = num_kv_heads,
    qk_head_dim: int = head_dim,
    v_head_dim: int = head_dim,
):
    """Create separate BSHD query, key, value tensors and an output gradient."""

    q_key, k_key, v_key, dout_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(q_key, (batch, seq, num_query_heads, qk_head_dim)).astype(dtype)
    k = jax.random.normal(k_key, (batch, seq, kv_heads, qk_head_dim)).astype(dtype)
    v = jax.random.normal(v_key, (batch, seq, kv_heads, v_head_dim)).astype(dtype)
    dout = jax.random.normal(dout_key, (batch, seq, num_query_heads, v_head_dim)).astype(dtype)
    return q, k, v, dout


def create_full_sequence_descriptor():
    """Describe a BSHD batch with no padding."""

    seqlens = jnp.full((batch,), seq, dtype=jnp.int32)
    return SequenceDescriptor.from_seqlens(seqlens)


q, k, v, dout = create_qkv_inputs(seed=2026)
qkv = (q, k, v)
sequence_descriptor = create_full_sequence_descriptor()
# ATTENTION_INPUTS_END


# ATTENTION_BASELINE_MODEL_START
def _repeat_kv_for_gqa(x, query_heads):
    """Repeat each KV head across its group of query heads."""

    repeats = query_heads // x.shape[2]
    return jnp.repeat(x, repeats, axis=2)


def _make_causal_swa_mask(q_len, kv_len, window: Optional[Tuple[int, int]]):
    """Create a boolean causal mask, optionally restricted to an SWA window."""

    q_pos = jnp.arange(q_len)[:, None]
    kv_pos = jnp.arange(kv_len)[None, :]

    if window is None:
        return kv_pos <= q_pos

    left, right = window
    allowed = kv_pos <= q_pos + right
    if left >= 0:
        allowed = allowed & (kv_pos >= q_pos - left)
    return allowed


class FlaxNativeGQAAttention(nn.Module):
    """Plain JAX/Flax GQA used as the bf16 baseline."""

    window_size: Optional[Tuple[int, int]] = None

    @nn.compact
    def __call__(self, qkv_tensors):
        query, key, value = qkv_tensors
        key = _repeat_kv_for_gqa(key, query.shape[2])
        value = _repeat_kv_for_gqa(value, query.shape[2])

        scale = query.shape[-1] ** -0.5
        scores = jnp.einsum(
            "bqhd,bkhd->bhqk",
            query.astype(jnp.float32),
            key.astype(jnp.float32),
        )
        scores *= scale

        mask = _make_causal_swa_mask(query.shape[1], key.shape[1], self.window_size)
        scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, value.astype(jnp.float32))
        return out.astype(query.dtype)


baseline = FlaxNativeGQAAttention(window_size=window_size)
baseline_vars = baseline.init(jax.random.PRNGKey(2026), qkv)
# ATTENTION_BASELINE_MODEL_END


# ATTENTION_TE_MODEL_START
class TEDotProductAttention(nn.Module):
    """Thin Flax wrapper around TE's DotProductAttention."""

    num_kv_heads: int
    qk_head_dim: int = head_dim
    attn_mask_type: str = "causal"
    qkv_layout: str = "bshd_bshd_bshd"
    window_size: Optional[Tuple[int, int]] = None

    @nn.compact
    def __call__(
        self,
        qkv_tensors,
        sequence_descriptor: Optional[SequenceDescriptor] = None,
        *,
        deterministic: bool = False,
    ):
        query, key, value = qkv_tensors
        return DotProductAttention(
            head_dim=self.qk_head_dim,
            num_attention_heads=num_query_heads,
            num_gqa_groups=self.num_kv_heads,
            attn_mask_type=self.attn_mask_type,
            qkv_layout=self.qkv_layout,
            attention_dropout=0.0,
            transpose_batch_sequence=False,
            window_size=self.window_size,
        )(
            query,
            key,
            value,
            sequence_descriptor=sequence_descriptor,
            deterministic=deterministic,
        )


te_model = TEDotProductAttention(num_kv_heads=num_kv_heads, window_size=window_size)
te_vars = te_model.init(
    jax.random.PRNGKey(2026),
    qkv,
    sequence_descriptor=sequence_descriptor,
    deterministic=False,
)
# ATTENTION_TE_MODEL_END


def run_forward_backward(model, variables, input_qkv, output_grad, seq_desc=None):
    """Run one compiled forward+backward pass through an attention module."""

    def loss_fn(qkv_arg):
        if seq_desc is None:
            out = model.apply(variables, qkv_arg)
        else:
            out = model.apply(
                variables,
                qkv_arg,
                sequence_descriptor=seq_desc,
                deterministic=False,
            )
        return jnp.vdot(out.astype(jnp.float32), output_grad.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss_fn))(input_qkv)


def compare_te_to_baseline(input_qkv=qkv, output_grad=dout, seq_desc=sequence_descriptor):
    """Compare the TE example to the native baseline."""

    loss_ref, grads_ref = run_forward_backward(
        baseline, baseline_vars, input_qkv, output_grad
    )
    loss_te, grads_te = run_forward_backward(te_model, te_vars, input_qkv, output_grad, seq_desc)
    out_ref = baseline.apply(baseline_vars, input_qkv)
    out_te = te_model.apply(te_vars, input_qkv, sequence_descriptor=seq_desc, deterministic=False)

    jax.block_until_ready((loss_ref, grads_ref, loss_te, grads_te, out_ref, out_te))
    np.testing.assert_allclose(out_te, out_ref, rtol=5e-2, atol=5e-2)
    for got, expected in zip(grads_te, grads_ref):
        np.testing.assert_allclose(got, expected, rtol=8e-2, atol=8e-2)


# ATTENTION_SINGLE_GPU_BENCH_START
def run_single_gpu_bench():
    forward_kwargs = {
        "sequence_descriptor": sequence_descriptor,
        "deterministic": False,
    }

    print("Native JAX bf16 GQA + SWA:")
    utils.speedometer(
        model_apply_fn=baseline.apply,
        variables=baseline_vars,
        input=qkv,
        output_grad=dout,
        timing_iters=timing_iters,
        warmup_iters=warmup_iters,
    )

    print("\nTE DotProductAttention GQA + SWA:")
    utils.speedometer(
        model_apply_fn=te_model.apply,
        variables=te_vars,
        input=qkv,
        output_grad=dout,
        forward_kwargs=forward_kwargs,
        timing_iters=timing_iters,
        warmup_iters=warmup_iters,
    )


# ATTENTION_SINGLE_GPU_BENCH_END


# ATTENTION_MLA_START
mla_head_dim_qk, mla_head_dim_v = 128, 64
mla_q, mla_k, mla_v, mla_dout = create_qkv_inputs(
    seed=2027,
    kv_heads=num_kv_heads,
    qk_head_dim=mla_head_dim_qk,
    v_head_dim=mla_head_dim_v,
)
mla_qkv = (mla_q, mla_k, mla_v)

mla_model = TEDotProductAttention(
    num_kv_heads=num_kv_heads,
    qk_head_dim=mla_head_dim_qk,
    window_size=None,
)
mla_vars = mla_model.init(
    jax.random.PRNGKey(4),
    mla_qkv,
    sequence_descriptor=sequence_descriptor,
    deterministic=False,
)


def run_mla_variant():
    out = mla_model.apply(
        mla_vars,
        mla_qkv,
        sequence_descriptor=sequence_descriptor,
        deterministic=False,
    )
    loss, grads = run_forward_backward(
        mla_model, mla_vars, mla_qkv, mla_dout, sequence_descriptor
    )
    jax.block_until_ready((out, loss, grads))
    print(
        "TE MLA-style BSHD: "
        f"q/k head dim={mla_head_dim_qk}, v head dim={mla_head_dim_v}"
    )
    print(f"Output shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"Grad shapes={[tuple(grad.shape) for grad in grads]}")


# ATTENTION_MLA_END


if __name__ == "__main__":
    print("# SINGLE_GPU_OUTPUT_START")
    run_single_gpu_bench()
    print("# SINGLE_GPU_OUTPUT_END")

    print("\n# MLA_OUTPUT_START")
    run_mla_variant()
    print("# MLA_OUTPUT_END")
