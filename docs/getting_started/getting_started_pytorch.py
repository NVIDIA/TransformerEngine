# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Getting Started with Transformer Engine - PyTorch Example
==========================================================

This example shows how to build a Transformer layer using PyTorch
and how to optimize it with Transformer Engine.
"""

from typing import Optional
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from getting_started_utils_pytorch import DotProductAttention, speedometer


# Configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 8
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = torch.bfloat16

# Create synthetic data
x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)


# =============================================================================
# Baseline: Pure PyTorch Implementation
# =============================================================================


# BASELINE_MLP_START
class PyTorchMLP(torch.nn.Module):
    """Feed-forward network in Transformer layer.
    Built with plain PyTorch modules.
    """

    hidden_size: int
    ffn_hidden_size: int

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.linear1 = torch.nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


# BASELINE_MLP_END


# BASELINE_LAYER_START
class PyTorchTransformerLayer(torch.nn.Module):
    """Basic Transformer layer using plain PyTorch modules."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = PyTorchMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x

        # Second residual connection
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


# BASELINE_LAYER_END


print("# BENCHMARK_BASELINE_OUTPUT_START")
# BENCHMARK_BASELINE_START
baseline = (
    PyTorchTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("Baseline PyTorch:")
time_baseline = speedometer(baseline, x, forward_kwargs={"attention_mask": None}, label="baseline")
# BENCHMARK_BASELINE_END
print("# BENCHMARK_BASELINE_OUTPUT_END\n")


# =============================================================================
# TE Unfused: Basic TE Modules
# =============================================================================


# TE_UNFUSED_MLP_START
class TEUnfusedMLP(torch.nn.Module):
    """MLP using TE modules."""

    hidden_size: int
    ffn_hidden_size: int

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


# TE_UNFUSED_MLP_END


# TE_UNFUSED_LAYER_START
class TEUnfusedTransformerLayer(torch.nn.Module):
    """Transformer layer using basic TE modules."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = torch.nn.Dropout(hidden_dropout)
        self.ln2 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = TEUnfusedMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        self.dropout2 = torch.nn.Dropout(hidden_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout1(x)
        x = res + x

        # Second residual connection
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout2(x)

        return x + res


# TE_UNFUSED_LAYER_END


print("# BENCHMARK_TE_UNFUSED_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_START
te_unfused = (
    TEUnfusedTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("TE Unfused:")
time_te_unfused = speedometer(
    te_unfused, x, forward_kwargs={"attention_mask": None}, label="te_unfused"
)
# BENCHMARK_TE_UNFUSED_END
print("# BENCHMARK_TE_UNFUSED_OUTPUT_END\n")


# =============================================================================
# TE Unfused + TE Attention
# =============================================================================


# TE_UNFUSED_ATTN_LAYER_START
class TEUnfusedAttnTransformerLayer(torch.nn.Module):
    """Transformer layer using TE modules including TE DotProductAttention."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = te.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
            attn_mask_type="causal",
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = torch.nn.Dropout(hidden_dropout)
        self.ln2 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = TEUnfusedMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        self.dropout2 = torch.nn.Dropout(hidden_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout1(x)
        x = res + x

        # Second residual connection
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout2(x)

        return x + res


# TE_UNFUSED_ATTN_LAYER_END


print("# BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_ATTN_START
te_unfused_attn = (
    TEUnfusedAttnTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("TE Unfused + TE Attention:")
time_te_unfused_attn = speedometer(
    te_unfused_attn, x, forward_kwargs={"attention_mask": None}, label="te_unfused_attn"
)
# BENCHMARK_TE_UNFUSED_ATTN_END
print("# BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_END\n")


# =============================================================================
# TE Unfused + FP8
# =============================================================================

print("# BENCHMARK_TE_UNFUSED_FP8_OUTPUT_START")
# BENCHMARK_TE_UNFUSED_FP8_START
recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

te_unfused_fp8 = (
    TEUnfusedAttnTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("TE Unfused + TE Attention + FP8:")
time_te_unfused_fp8 = speedometer(
    te_unfused_fp8,
    x,
    forward_kwargs={"attention_mask": None},
    autocast_kwargs={"enabled": True, "recipe": recipe},
    label="te_unfused_fp8",
)
# BENCHMARK_TE_UNFUSED_FP8_END
print("# BENCHMARK_TE_UNFUSED_FP8_OUTPUT_END\n")


# =============================================================================
# TE Fused + FP8: Optimized Modules with FP8
# =============================================================================


# TE_FUSED_LAYER_START
class TEFusedTransformerLayer(torch.nn.Module):
    """Transformer layer using fused TE modules for better performance."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads

        # Fused LayerNorm + QKV projection
        self.ln_qkv = te.LayerNormLinear(hidden_size, 3 * hidden_size, eps=layernorm_eps, bias=True)
        self.attention = te.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
            attn_mask_type="causal",
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = torch.nn.Dropout(hidden_dropout)

        # Fused LayerNorm + MLP
        self.ln_mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size, eps=layernorm_eps, bias=True)
        self.dropout2 = torch.nn.Dropout(hidden_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x

        # Fused LayerNorm + QKV projection
        qkv = self.ln_qkv(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout1(x)
        x = res + x

        # Fused LayerNorm + MLP
        res = x
        x = self.ln_mlp(x)
        x = self.dropout2(x)

        return x + res


# TE_FUSED_LAYER_END


print("# BENCHMARK_TE_FUSED_FP8_OUTPUT_START")
# BENCHMARK_TE_FUSED_FP8_START
te_fused_fp8 = (
    TEFusedTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("TE Fused + TE Attention + FP8:")
time_te_fused_fp8 = speedometer(
    te_fused_fp8,
    x,
    forward_kwargs={"attention_mask": None},
    autocast_kwargs={"enabled": True, "recipe": recipe},
    label="te_fused_fp8",
)
# BENCHMARK_TE_FUSED_FP8_END
print("# BENCHMARK_TE_FUSED_FP8_OUTPUT_END\n")


# =============================================================================
# TE TransformerLayer + FP8: Ready-to-use Module
# =============================================================================

print("# BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_START")
# BENCHMARK_TE_TRANSFORMER_LAYER_START
te_transformer_layer = (
    te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        self_attn_mask_type="causal",
        layernorm_epsilon=1e-5,
        bias=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    .to(dtype=dtype)
    .cuda()
)

print("TE TransformerLayer + FP8:")
time_te_transformer_layer = speedometer(
    te_transformer_layer,
    x,
    forward_kwargs={"attention_mask": None},
    autocast_kwargs={"enabled": True, "recipe": recipe},
    label="te_transformer_layer",
)
# BENCHMARK_TE_TRANSFORMER_LAYER_END
print("# BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_END\n")


# Write summary CSV for RST documentation
with open("getting_started_pytorch_summary.csv", "w") as f:
    f.write("Implementation,Time (ms),Speedup\n")
    f.write(f"Baseline PyTorch,{time_baseline:.2f},1.00x\n")
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
print("\nSummary written to getting_started_pytorch_summary.csv")
