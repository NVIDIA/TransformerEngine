# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Getting Started with Transformer Engine - JAX Example
======================================================

This example shows how to build a Transformer decoder layer using JAX/Flax
and how to optimize it with Transformer Engine.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.sharding import MeshResource
from transformer_engine.common.recipe import Format, DelayedScaling

from getting_started_utils_jax import speedometer


# Configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 8
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = jnp.bfloat16

# Create synthetic data
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (batch_size, sequence_length, hidden_size)).astype(dtype)
mesh_resource = MeshResource()


# =============================================================================
# Baseline: Pure Flax Implementation
# =============================================================================


# BASELINE_MLP_START
class FlaxMLP(nn.Module):
    """Feed-forward network in Transformer layer.
    Built with plain Flax modules.
    """

    hidden_size: int
    ffn_hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.ffn_hidden_size, use_bias=True)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(features=self.hidden_size, use_bias=True)(x)
        return x


# BASELINE_MLP_END


# BASELINE_LAYER_START
class FlaxTransformerLayer(nn.Module):
    """Basic Transformer layer using plain Flax modules."""

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    layernorm_eps: float = 1e-5
    attention_dropout: float = 0.1

    def setup(self):
        self.kv_channels = self.hidden_size // self.num_attention_heads

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nn.make_causal_mask(x[..., 0], dtype=jnp.bool_)

        res = x
        x = nn.LayerNorm(epsilon=self.layernorm_eps)(x)

        # Fused QKV projection
        qkv = nn.Dense(features=3 * self.hidden_size, use_bias=True)(x)
        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = jnp.split(qkv, 3, axis=3)

        dropout_rng = None
        if not deterministic and self.attention_dropout > 0:
            dropout_rng = self.make_rng("dropout")

        x = nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.attention_dropout,
            deterministic=deterministic,
            broadcast_dropout=True,
        )

        x = x.reshape(x.shape[0], x.shape[1], self.hidden_size)
        x = nn.Dense(features=self.hidden_size, use_bias=True)(x)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)
        x = res + x

        res = x
        x = nn.LayerNorm(epsilon=self.layernorm_eps)(x)
        mlp = FlaxMLP(hidden_size=self.hidden_size, ffn_hidden_size=self.ffn_hidden_size)
        x = mlp(x)

        return x + res


# BASELINE_LAYER_END


print("# BENCHMARK_BASELINE_OUTPUT_START")
# BENCHMARK_BASELINE_START
baseline = FlaxTransformerLayer(
    hidden_size=hidden_size,
    ffn_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
)
params = baseline.init(key, x, deterministic=False)

print("Baseline Flax:")
time_baseline = speedometer(
    baseline.apply, params, x, forward_kwargs={"deterministic": True}, label="baseline"
)
# BENCHMARK_BASELINE_END
print("# BENCHMARK_BASELINE_OUTPUT_END\n")


# =============================================================================
# TE Unfused: Basic TE Modules
# =============================================================================


# TE_UNFUSED_MLP_START
class TEUnfusedMLP(nn.Module):
    """MLP using TE modules."""

    hidden_size: int
    ffn_hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = te_flax.DenseGeneral(features=self.ffn_hidden_size, use_bias=True)(x)
        x = x.reshape(*x.shape[:-1], 1, x.shape[-1])
        x = te.activation.activation(x, activation_type=("gelu",))
        x = te_flax.DenseGeneral(features=self.hidden_size, use_bias=True)(x)
        return x


# TE_UNFUSED_MLP_END


# TE_UNFUSED_LAYER_START
class TEUnfusedTransformerLayer(nn.Module):
    """Transformer layer using basic TE modules (without TE attention)."""

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    layernorm_eps: float = 1e-5
    attention_dropout: float = 0.1

    def setup(self):
        self.kv_channels = self.hidden_size // self.num_attention_heads

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nn.make_causal_mask(x[..., 0], dtype=jnp.bool_)

        res = x
        x = te_flax.LayerNorm(epsilon=self.layernorm_eps)(x)

        qkv = te_flax.DenseGeneral(features=3 * self.hidden_size, use_bias=True)(x)
        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = jnp.split(qkv, 3, axis=3)

        dropout_rng = None
        if not deterministic and self.attention_dropout > 0:
            dropout_rng = self.make_rng("dropout")

        x = nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.attention_dropout,
            deterministic=deterministic,
            broadcast_dropout=True,
        )

        x = x.reshape(x.shape[0], x.shape[1], self.hidden_size)
        x = te_flax.DenseGeneral(features=self.hidden_size, use_bias=True)(x)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        x = res + x

        res = x
        x = te_flax.LayerNorm(epsilon=self.layernorm_eps)(x)
        mlp = TEUnfusedMLP(hidden_size=self.hidden_size, ffn_hidden_size=self.ffn_hidden_size)
        x = mlp(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        return x + res


# TE_UNFUSED_LAYER_END


print("# BENCHMARK_TE_UNFUSED_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_START
te_unfused = TEUnfusedTransformerLayer(
    hidden_size=hidden_size,
    ffn_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
)
params = te_unfused.init(key, x, deterministic=False)

print("TE Unfused:")
time_te_unfused = speedometer(
    te_unfused.apply, params, x, forward_kwargs={"deterministic": True}, label="te_unfused"
)
# BENCHMARK_TE_UNFUSED_END
print("# BENCHMARK_TE_UNFUSED_OUTPUT_END\n")


# =============================================================================
# TE Unfused + TE Attention
# =============================================================================


# TE_UNFUSED_ATTN_LAYER_START
class TEUnfusedAttnTransformerLayer(nn.Module):
    """Transformer layer using TE modules including TE DotProductAttention."""

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    layernorm_eps: float = 1e-5
    attention_dropout: float = 0.1

    def setup(self):
        self.kv_channels = self.hidden_size // self.num_attention_heads

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        res = x
        x = te_flax.LayerNorm(epsilon=self.layernorm_eps, dtype=jnp.bfloat16)(x)

        qkv = te_flax.DenseGeneral(
            features=3 * self.hidden_size, use_bias=True, dtype=jnp.bfloat16
        )(x)
        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = jnp.split(qkv, 3, axis=3)

        attention = te_flax.DotProductAttention(
            head_dim=self.kv_channels,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            attn_mask_type="causal",
            transpose_batch_sequence=False,
        )
        x = attention(q, k, v, deterministic=deterministic)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = te_flax.DenseGeneral(features=self.hidden_size, use_bias=True, dtype=jnp.bfloat16)(x)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        x = res + x

        res = x
        x = te_flax.LayerNorm(epsilon=self.layernorm_eps)(x)
        mlp = TEUnfusedMLP(hidden_size=self.hidden_size, ffn_hidden_size=self.ffn_hidden_size)
        x = mlp(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        return x + res


# TE_UNFUSED_ATTN_LAYER_END


print("# BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_ATTN_START
te_unfused_attn = TEUnfusedAttnTransformerLayer(
    hidden_size=hidden_size,
    ffn_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
)

with te.autocast(enabled=False, mesh_resource=mesh_resource):
    params = te_unfused_attn.init(key, x, deterministic=False)

print("TE Unfused + TE Attention:")
time_te_unfused_attn = speedometer(
    te_unfused_attn.apply,
    params,
    x,
    forward_kwargs={"deterministic": True},
    autocast_kwargs={"enabled": False, "mesh_resource": mesh_resource},
    label="te_unfused_attn",
)
# BENCHMARK_TE_UNFUSED_ATTN_END
print("# BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_END\n")


# =============================================================================
# TE Unfused + FP8
# =============================================================================

print("# BENCHMARK_TE_UNFUSED_FP8_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_FP8_START
recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

te_unfused_fp8 = TEUnfusedAttnTransformerLayer(
    hidden_size=hidden_size,
    ffn_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
)

with te.autocast(enabled=True, recipe=recipe, mesh_resource=mesh_resource):
    params = te_unfused_fp8.init(key, x, deterministic=False)

print("TE Unfused + TE Attention + FP8:")
time_te_unfused_fp8 = speedometer(
    te_unfused_fp8.apply,
    params,
    x,
    forward_kwargs={"deterministic": True},
    autocast_kwargs={"enabled": True, "recipe": recipe, "mesh_resource": mesh_resource},
    label="te_unfused_fp8",
)
# BENCHMARK_TE_UNFUSED_FP8_END
print("# BENCHMARK_TE_UNFUSED_FP8_OUTPUT_END\n")


# =============================================================================
# TE Fused + FP8: Optimized Modules with FP8
# =============================================================================


# TE_FUSED_LAYER_START
class TEFusedTransformerLayer(nn.Module):
    """Transformer layer using fused TE modules for better performance."""

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    layernorm_eps: float = 1e-5
    attention_dropout: float = 0.1

    def setup(self):
        self.kv_channels = self.hidden_size // self.num_attention_heads

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        res = x

        # Fused LayerNorm + QKV projection
        qkv, _ = te_flax.LayerNormDenseGeneral(
            features=3 * self.hidden_size,
            epsilon=self.layernorm_eps,
            use_bias=True,
            return_layernorm_output=False,
        )(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_attention_heads, self.kv_channels)
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]

        attention = te_flax.DotProductAttention(
            head_dim=self.kv_channels,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            attn_mask_type="causal",
            qkv_layout="bshd_bshd_bshd",
            transpose_batch_sequence=False,
        )
        x = attention(q, k, v, deterministic=deterministic)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = te_flax.DenseGeneral(features=self.hidden_size, use_bias=True)(x)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        x = res + x

        res = x
        # Fused LayerNorm + MLP
        x, _ = te_flax.LayerNormMLP(
            intermediate_dim=self.ffn_hidden_size,
            epsilon=self.layernorm_eps,
            use_bias=True,
            activations=("gelu",),
            intermediate_dropout_rate=0.0,
            return_layernorm_output=False,
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.attention_dropout)(x, deterministic=deterministic)

        return x + res


# TE_FUSED_LAYER_END


print("# BENCHMARK_TE_FUSED_FP8_OUTPUT_START")
# BENCHMARK_TE_FUSED_FP8_START
te_fused_fp8 = TEFusedTransformerLayer(
    hidden_size=hidden_size,
    ffn_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
)

with te.autocast(enabled=True, recipe=recipe, mesh_resource=mesh_resource):
    params = te_fused_fp8.init(key, x, deterministic=False)

print("TE Fused + TE Attention + FP8:")
time_te_fused_fp8 = speedometer(
    te_fused_fp8.apply,
    params,
    x,
    forward_kwargs={"deterministic": True},
    autocast_kwargs={"enabled": True, "recipe": recipe, "mesh_resource": mesh_resource},
    label="te_fused_fp8",
)
# BENCHMARK_TE_FUSED_FP8_END
print("# BENCHMARK_TE_FUSED_FP8_OUTPUT_END\n")


# =============================================================================
# TE TransformerLayer + FP8: Ready-to-use Module
# =============================================================================

print("# BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_START")
# BENCHMARK_TE_TRANSFORMER_LAYER_START
te_transformer_layer = te_flax.TransformerLayer(
    hidden_size=hidden_size,
    mlp_hidden_size=ffn_hidden_size,
    num_attention_heads=num_attention_heads,
    mlp_activations=("gelu",),
    self_attn_mask_type="causal",
    layernorm_epsilon=1e-5,
    use_bias=True,
    attention_dropout=0.0,
    intermediate_dropout=0.0,
    hidden_dropout=0.0,
    enable_relative_embedding=False,
    self_attn_bias_type="no_bias",
    dtype=jnp.bfloat16,
    transpose_batch_sequence=False,
)

with te.autocast(enabled=True, recipe=recipe, mesh_resource=mesh_resource):
    params = te_transformer_layer.init(key, x, deterministic=False)

print("TE TransformerLayer + FP8:")
time_te_transformer_layer = speedometer(
    te_transformer_layer.apply,
    params,
    x,
    forward_kwargs={"deterministic": True},
    autocast_kwargs={"enabled": True, "recipe": recipe, "mesh_resource": mesh_resource},
    label="te_transformer_layer",
)
# BENCHMARK_TE_TRANSFORMER_LAYER_END
print("# BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_END\n")

# Write summary CSV for RST documentation
with open("getting_started_jax_summary.csv", "w") as f:
    f.write("Implementation,Time (ms),Speedup\n")
    f.write(f"Baseline Flax,{time_baseline:.2f},1.00x\n")
    f.write(f"TE Unfused,{time_te_unfused:.2f},{time_baseline/time_te_unfused:.2f}x\n")
    f.write(
        "TE Unfused + TE"
        f" Attention,{time_te_unfused_attn:.2f},{time_baseline/time_te_unfused_attn:.2f}x\n"
    )
    f.write(
        "TE Unfused + TE Attention +"
        f" FP8,{time_te_unfused_fp8:.2f},{time_baseline/time_te_unfused_fp8:.2f}x\n"
    )
    f.write(
        "TE Fused + TE Attention +"
        f" FP8,{time_te_fused_fp8:.2f},{time_baseline/time_te_fused_fp8:.2f}x\n"
    )
    f.write(
        "TE TransformerLayer +"
        f" FP8,{time_te_transformer_layer:.2f},{time_baseline/time_te_transformer_layer:.2f}x\n"
    )
print("\nSummary written to getting_started_jax_summary.csv")
