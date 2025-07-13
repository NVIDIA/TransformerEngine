# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import pytest
import torch

import transformer_engine
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert type name to PyTorch dtype"""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name.replace("torch.", "", 1)
    if name.startswith("fp"):
        name = name.replace("fp", "float", 1)
    dtype = dict(
        float32=torch.float32,
        float=torch.float32,
        float64=torch.float64,
        double=torch.float64,
        float16=torch.float16,
        half=torch.float16,
        bfloat16=torch.bfloat16,
        bf16=torch.bfloat16,
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e4m3=torch.float8_e4m3fn,
        float8e4m3=torch.float8_e4m3fn,
        float8=torch.float8_e4m3fn,
        float8_e5m2=torch.float8_e5m2,
        float8e5m2=torch.float8_e5m2,
        uint8=torch.uint8,
        byte=torch.uint8,
        int8=torch.int8,
        char=torch.int8,
        int16=torch.int16,
        short=torch.int16,
        int32=torch.int32,
        int=torch.int32,
        int64=torch.int64,
        long=torch.int64,
        bool=torch.bool,
    )[name]
    return dtype


def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
            tex.DType.kFloat8E5M2: torch.float8_e5m2,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    if dtype == torch.float8_e4m3fn:
        return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
    if dtype == torch.float8_e5m2:
        return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
    raise ValueError(f"Unsupported dtype ({dtype})")


def make_recipe(name: Optional[str]) -> Optional[Recipe]:
    """Make recipe for quantization scheme"""
    if name is None:
        return None
    if name in ("fp8", "fp8_delayed_scaling"):
        return transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "fp8_current_scaling":
        return transformer_engine.common.recipe.Float8CurrentScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "mxfp8":
        return transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "fp8_block_scaling":
        return transformer_engine.common.recipe.Float8BlockScaling()
    raise ValueError(f"Unsupported quantization scheme ({name})")


def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    fp8.FP8GlobalStateManager.reset()


class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_gqa_groups: int,
        head_dim_qk: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
        head_dim_v: int = None,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
        total_requests: int = None,
        max_ctx_len: int = None,
        eps: float = 1e-5,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        if self.head_dim_qk == self.head_dim_v:
            self.kv_channels = self.head_dim_qk
        else:
            self.kv_channels = (self.head_dim_qk, self.head_dim_v)
        self.hidden_size = num_heads * head_dim_qk
        self.hidden_size_kv = num_gqa_groups * self.head_dim_v
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (max_seqlen_q == max_seqlen_kv) else "cross"
        self.num_layers = num_layers
        self.bias_shape = bias_shape
        self.window_size = window_size
        self.total_requests = total_requests
        self.max_ctx_len = max_ctx_len
        self.eps = eps
