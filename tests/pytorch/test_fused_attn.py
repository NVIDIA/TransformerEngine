# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
import functools
from importlib.metadata import version
import os
from typing import Any, Dict, List, Tuple, Union

from pkg_resources import packaging
import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    RotaryPositionEmbedding,
)
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch.cpp_extensions as ext
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import (
    _set_cuda_rng_state,
    CudaRNGStatesTracker,
)
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch.module.base import (
    TransformerEngineBaseModule,
    _prepare_backward,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
)
import transformer_engine_extensions as tex


# Only run FP8 tests on H100
fp8_available, reason_for_no_fp8 = fp8.FP8GlobalStateManager.is_fp8_available()

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

def reset_rng_states() -> None:
    """Revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    _set_cuda_rng_state(_cuda_rng_state)

@functools.cache
def _cudnn_version() -> Tuple[int, int, int]:
    """Current cuDNN version (major, minor, patch)"""
    encoded_version = ext.get_cudnn_version()
    major, encoded_version = divmod(encoded_version, 1000)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)

_dtypes = [torch.float16]
if torch.cuda.is_bf16_supported():
    _dtypes.append(torch.bfloat16)

@functools.cache
def _default_dtype() -> torch.dtype:
    """BF16 if supported, FP16 otherwise"""
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16

@dataclass
class ModelConfig:
    """Configuration for multi-head attention"""
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    attn_mask_type: str = "causal"
    bias_type: str = "no_bias"
    num_layers: int = 1
    dropout_prob: float = 0.0

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

def _is_fused_attention_supported(
    config: ModelConfig,
    dtype: torch.dtype,
    qkv_layout: str = "sbh3d",
) -> bool:
    """Check if cuDNN fused attention supports a model configuration"""
    backend = tex.get_fused_attn_backend(
        TE_DType[dtype],
        TE_DType[dtype],
        QKVLayout[qkv_layout],
        AttnBiasType[config.bias_type],
        AttnMaskType[config.attn_mask_type],
        config.dropout_prob,
        config.seq_len,
        config.seq_len,
        config.head_dim,
    )
    return backend != FusedAttnBackend["No_Backend"]

@functools.cache
def _is_flash_attention_2_available() -> bool:
    """Check if Flash Attention 2.0+ is available"""
    Version = packaging.version.Version
    return Version(version("flash-attn")) > Version("2")

def _is_flash_attention_supported(config: ModelConfig) -> bool:
    """Check if Flash Attention supports a model configuration"""
    if get_device_compute_capability() < (8, 0):
        return False
    if config.bias_type != "no_bias":
        return False
    return True


_test_dot_product_attention_configs = {
    # Baseline cases
    "s128-b4-h16-d64": ModelConfig(128, 4, 16, 64),
    "s1024-b4-h16-d64": ModelConfig(1024, 4, 16, 64),

    # Small case
    "s32-b2-h2-d32": ModelConfig(32, 2, 2, 32),

    # Large case
    "s2048-b32-h16-d64-no_mask-post_scale_bias": ModelConfig(
        2048, 32, 16, 64,
        attn_mask_type="no_mask",
        bias_type="post_scale_bias",
    ),

    # Sequence length
    "s512-b4-h16-d64": ModelConfig(512, 4, 16, 64),
    "s2048-b4-h16-d64": ModelConfig(2048, 4, 16, 64),

    # Batch size
    "s128-b1-h16-d64": ModelConfig(128, 1, 16, 64),
    "s128-b32-h16-d64": ModelConfig(128, 32, 16, 64),
    "s1024-b1-h16-d64": ModelConfig(1024, 1, 16, 64),
    "s1024-b32-h16-d64": ModelConfig(1024, 32, 16, 64),

    # Num heads
    "s128-b4-h24-d64": ModelConfig(128, 4, 24, 64),
    "s1024-b4-h24-d64": ModelConfig(1024, 4, 24, 64),

    # Head dim
    "s128-b4-h16-d128": ModelConfig(128, 4, 16, 128),
    "s1024-b4-h16-d128": ModelConfig(1024, 4, 16, 128),

    # Attention mask type
    "s128-b32-h16-d64-no_mask": ModelConfig(
        128, 32, 16, 64,
        attn_mask_type="no_mask",
    ),
    "s1024-b32-h16-d64-no_mask": ModelConfig(
        1024, 32, 16, 64,
        attn_mask_type="no_mask",
    ),

    # Bias type
    "s128-b4-h16-d64-post_scale_bias": ModelConfig(
        128, 4, 16, 64,
        bias_type="post_scale_bias",
    ),
    "s1024-b4-h16-d64": ModelConfig(
        1024, 4, 16, 64,
        bias_type="post_scale_bias",
    ),
}

@pytest.mark.parametrize("config_name", _test_dot_product_attention_configs.keys())
@pytest.mark.parametrize("dtype", _dtypes)
def test_dot_product_attention(
    config_name: str,
    dtype: torch.dtype,
    checkpoint_attention: bool = False,
) -> None:
    """Test DotProductAttention module"""

    # Get configs
    config = _test_dot_product_attention_configs[config_name]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(config, dtype)
    flash_attn_supported = _is_flash_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    # UnfusedDotProductAttention backend
    unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
        dtype,
        config,
        "UnfusedDotProductAttention",
        checkpoint_attention,
    )

    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "FusedAttention",
            checkpoint_attention,
        )
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "FlashAttention",
            checkpoint_attention,
        )
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)

def _run_dot_product_attention(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    checkpoint_attention: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1"

    inp = torch.randn(
        config.seq_len,
        config.batch_size,
        3,
        config.num_heads,
        config.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = torch.randn(
        config.seq_len,
        config.batch_size,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    if config.bias_type != "no_bias":
        bias = torch.randn(
            1,
            config.num_heads,
            config.seq_len,
            config.seq_len,
            dtype=dtype,
            device="cuda",
        )
    else:
        bias = None

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    block = (
         DotProductAttention(
                config.num_heads,
                config.head_dim,
                attention_dropout=config.dropout_prob,
                sequence_parallel=False,
                tp_size=1,
                get_rng_state_tracker=get_dummy_cuda_rng_tracker,
                tp_group=None,
                layer_number=1,
                attention_type="self",
        ).to(dtype=dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(
        q, k, v,
        qkv_format='sbhd',
        cu_seqlens_q = cu_seqlens,
        cu_seqlens_kv = cu_seqlens,
        attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=checkpoint_attention,
        core_attention_bias_type=config.bias_type,
        core_attention_bias=bias,
    )
    op.backward(op_grad)

    return op, inp.grad


@pytest.mark.parametrize("config_name", ["s128-b4-h16-d64"])
def test_dot_product_attention_checkpoint(
    config_name: str,
    dtype: torch.dtype = _default_dtype(),
) -> None:
    test_dot_product_attention(
        config_name=config_name,
        dtype=dtype,
        checkpoint_attention=True,
    )


_qkv_layouts = [
    'sb3hd', 'sbh3d', 'sbhd_sb2hd', 'sbhd_sbh2d', 'sbhd_sbhd_sbhd',
    'bs3hd', 'bsh3d', 'bshd_bs2hd', 'bshd_bsh2d', 'bshd_bshd_bshd',
    # will add tests for thd layouts later when the support is available in fused attention
    #'t3hd', 'th3d', 'thd_t2hd', 'thd_th2d', 'thd_thd_thd',
]

_test_dpa_qkv_layout_configs = {
    "s128-b4-h16-d64": ModelConfig(128, 4, 16, 64),
    "s1024-b4-h16-d64": ModelConfig(1024, 4, 16, 64),
    "s128-b4-h16-d64_no_mask": ModelConfig(
        128, 4, 16, 64,
        attn_mask_type="no_mask",
    ),
    "s1024-b4-h16-d64_no_mask": ModelConfig(
        1024, 4, 16, 64,
        attn_mask_type="no_mask",
    ),
}

@pytest.mark.skipif(_cudnn_version() < (8,9,5), reason="cuDNN 8.9.5+ is required")
@pytest.mark.parametrize("config_name", _test_dpa_qkv_layout_configs.keys())
@pytest.mark.parametrize("qkv_layout", _qkv_layouts)
@pytest.mark.parametrize("workspace_opt", [True, False])
def test_dpa_qkv_layout(
    config_name: str,
    qkv_layout: str,
    workspace_opt: bool,
    dtype: torch.dtype = _default_dtype(),
) -> None:
    """Test DotProductAttention module with different QKV layouts"""

    # Get configs
    config = _test_dpa_qkv_layout_configs[config_name]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(config, dtype)
    flash_attn_supported = _is_flash_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    # UnfusedDotProductAttention backend
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa_qkv_layout(
        dtype, config, "UnfusedDotProductAttention", qkv_layout, workspace_opt)

    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_dpa_qkv_layout(
            dtype, config, "FusedAttention", qkv_layout, workspace_opt)
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i in range(len(unfused_attn_bwd)):
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_dpa_qkv_layout(
            dtype, config, "FlashAttention", qkv_layout, workspace_opt)
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        for i in range(len(unfused_attn_bwd)):
            torch.testing.assert_close(flash_attn_bwd[i], unfused_attn_bwd[i], **tols)

def _run_dpa_qkv_layout(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    qkv_layout: str,
    workspace_opt: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"

    dim_to_num = {
        'b': config.batch_size,
        's': config.seq_len,
        'h': config.num_heads,
        'd': config.head_dim,
        't': config.batch_size * config.seq_len,
        '3': 3,
        '2': 2,
    }

    inp = []
    for i,layout in enumerate(qkv_layout.split('_')):
        tensor_shape = [dim_to_num[j] for j in layout]
        tensor = 0.1 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout):
             if l.isdigit():
                 tensor_count = int(l)
                 split_dim = dim
                 break
        tensors = torch.split(tensor, 1, dim = split_dim) if split_dim != 0 else [tensor]
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
    for i in range(3):
        inp[i].requires_grad = True

    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
    qkv_format_no_thd = qkv_format if qkv_format != 'thd' else 'bshd'
    op_grad_shape = [dim_to_num[i] for i in qkv_format_no_thd]
    op_grad_shape_new = [*op_grad_shape[:-2], op_grad_shape[-2] * op_grad_shape[-1]]
    op_grad = 0.001 * torch.randint(0, 200, op_grad_shape_new, dtype=dtype, device="cuda")

    block = (
         DotProductAttention(
                config.num_heads,
                config.head_dim,
                attention_dropout = config.dropout_prob,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = None,
                tp_group = None,
                layer_number = 1,
                attention_type = "self"
        ).to(dtype = dtype).cuda()
    )

    if qkv_format != 'thd':
        op = block(inp[0], inp[1], inp[2], qkv_format=qkv_format)
    else:
        cu_seqlens_q = torch.arange(
                0,
                (config.batch_size + 1) * config.seq_len,
                step=config.seq_len,
                dtype=torch.int32,
                device="cuda")
        cu_seqlens_kv = torch.arange(
                0,
                (batch_size + 1) * config.seq_len,
                step=config.seq_len,
                dtype=torch.int32,
                device="cuda")
        op = block(inp[0], inp[1], inp[2],
                qkv_format=qkv_format,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_kv = cu_seqlens_kv)
    op.backward(op_grad)

    return op, (inp[0].grad, inp[1].grad, inp[2].grad)


_test_transformer_layer_configs = {
    "s32-b2-h2-d32": ModelConfig(32, 2, 2, 32),
    "s1024-b2-h2-d32": ModelConfig(1024, 2, 2, 32),
    "s32-b2-h2-d32_no_mask": ModelConfig(
        32, 2, 2, 32,
        attn_mask_type="no_mask",
    ),
    "s32-b2-h2-d32_post_scale_bias": ModelConfig(
        32, 2, 2, 32,
        bias_type="post_scale_bias",
    ),
}

@pytest.mark.parametrize("config_name", _test_transformer_layer_configs.keys())
@pytest.mark.parametrize("fused_qkv_params", [True, False])
def test_transformer_layer(
    config_name: str,
    fused_qkv_params: bool,
    dtype: torch.dtype = _default_dtype(),
    RoPE: bool = False,
) -> None:
    """Test TransformerLayer module when its DotProductAttention is enabled with
    FlashAttention, FusedAttention, or UnfusedDotProductAttention backend"""

    # Get configs
    config = _test_transformer_layer_configs[config_name]
    tols = dict(atol=5e-1, rtol=5e-2)

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(
        config,
        dtype,
        qkv_layout="sbh3d" if fused_qkv_params else "sb3hd",
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    # UnfusedDotProductAttention backend
    unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
        dtype,
        config,
        "UnfusedDotProductAttention",
        fused_qkv_params,
        RoPE,
    )

    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FusedAttention",
            fused_qkv_params,
            RoPE,
        )
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FlashAttention",
            fused_qkv_params,
            RoPE,
        )
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)

def _run_transformer_layer(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    fused_qkv_params: bool,
    RoPE: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.randn(
        config.seq_len,
        config.batch_size,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        device=inp.device,
        dtype=torch.int32,
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [
            rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]
    if config.bias_type != "no_bias":
        bias = torch.randn(
            1,
            config.num_heads,
            config.seq_len,
            config.seq_len,
            dtype=dtype,
            device="cuda",
        )
    else:
        bias = None

    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim)
        rotary_pos_emb = PE(config.seq_len).cuda().to(dtype=dtype)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.0,
            attention_dropout=config.dropout_prob,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            kv_channels=config.head_dim,
            tp_group=None,
            tp_size=1,
            params_dtype=dtype,
            get_rng_state_tracker=None,
            fuse_wgrad_accumulation=False,
            seq_length=config.seq_len,
            micro_batch_size=config.batch_size,
            sequence_parallel=False,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="encoder",
            drop_path_rate=drop_path_rates[layer_number - 1],
            set_parallel_mode=True,
            fuse_qkv_params=fused_qkv_params,
            zero_centered_gamma=False,
            qkv_weight_interleaved=False,
            ub_tp_comm_overlap=False,
            bias=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    op = block(
        inp,
        self_attn_mask_type=config.attn_mask_type,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.bias_type,
        core_attention_bias=bias,
    )
    loss = op.sum()
    loss.backward()

    return op, inp.grad


@pytest.mark.parametrize("config_name", ["s32-b2-h2-d32"])
def test_transformer_layer_rope(
    config_name: str,
    fused_qkv_params: bool = True,
) -> None:
    test_transformer_layer(
        config_name=config_name,
        fused_qkv_params=fused_qkv_params,
        RoPE=True,
    )


@pytest.mark.parametrize("config_name", _test_transformer_layer_configs.keys())
def test_transformer_layer_gqa(
    config_name: str,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test TransformerLayer module when its DotProductAttention is enabled with
    FlashAttention or UnfusedDotProductAttention backend"""

    config = _test_transformer_layer_configs[config_name]
    def find_factors(x):
       f = []
       for i in range(1, x + 1):
           if x % i == 0:
               f.append(i)
       return f

    # Skip if only unfused backend is supported
    if not _is_flash_attention_2_available():
        pytest.skip("FlashAttention 1 does not support GQA")
    if not _is_flash_attention_supported(config):
        pytest.skip("FlashAttention does not support this model config")

    num_querys_per_gqa_group = find_factors(config.num_heads)

    for num_q_per_gqa_group in num_querys_per_gqa_group:
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer_gqa(
            dtype,
            config,
            "FlashAttention",
            num_q_per_gqa_group,
        )
        unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer_gqa(
            dtype,
            config,
            "UnfusedDotProductAttention",
            num_q_per_gqa_group,
        )

        atol, rtol = 5e-1, 5e-2
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, atol=atol, rtol=rtol)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, atol=atol, rtol=rtol)

def _run_transformer_layer_gqa(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    num_querys_per_gqa_group: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.randn(
        config.seq_len,
        config.batch_size,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = torch.randn(
        config.seq_len,
        config.batch_size,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
    )

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [
            rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            num_gqa_groups=config.num_heads / num_querys_per_gqa_group,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.0,
            attention_dropout=config.dropout_prob,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            kv_channels=config.head_dim,
            tp_group=None,
            tp_size= 1,
            params_dtype=dtype,
            get_rng_state_tracker=None,
            fuse_wgrad_accumulation=False,
            seq_length=config.seq_len,
            micro_batch_size=config.batch_size,
            sequence_parallel=False,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="encoder",
            drop_path_rate=drop_path_rates[layer_number - 1],
            set_parallel_mode=True,
            fuse_qkv_params=True,
            zero_centered_gamma=False,
            qkv_weight_interleaved=False,
            ub_tp_comm_overlap=False,
            bias=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    op = block(inp, self_attn_mask_type=config.attn_mask_type)
    op.backward(op_grad)

    return op, inp.grad


_test_dpa_fp8_configs = {
    "s128-b4-h16-d64-no_mask": ModelConfig(
        128, 4, 16, 64,
        attn_mask_type="no_mask",
    ),
    "s1024-b4-h16-d64-no_mask": ModelConfig(
        1024, 4, 16, 64,
        attn_mask_type="no_mask",
    ),
}

@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("config_name", _test_dpa_fp8_configs.keys())
def test_dpa_fp8(
    config_name: str,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test FP8 dot-product attention with different backends

    FusedAttention uses fused_attn_fwd/bwd_qkvpacked from
    cpp_extensions. UnfusedDotProductAttention uses plain PyTorch
    operations.

    """

    config = _test_dpa_fp8_configs[config_name]

    # Skip if not supported
    if not _is_fused_attention_supported(config, dtype):
        pytest.skip("FusedAttention does not support this model config")

    # Run dot-product attention with different backends
    fused_attn_fwd, fused_attn_bwd = _run_dpa_fp8(
        dtype,
        config,
        "FusedAttention"
    )
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa_fp8_ref(
        dtype,
        config,
        "UnfusedDotProductAttention",
    )

    # Check that results match
    tols = dict(atol=2.5e-2, rtol=2.5e-2)
    torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
    torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

def _run_dpa_fp8(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.01 * torch.randn(
        config.batch_size * config.seq_len,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = 0.01 * torch.randn(
        config.batch_size * config.seq_len,
        config.num_heads * config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    torch.save(op_grad, 'op_grad.pt')

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        interval=1,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

    dpa = DPA_FP8(config).to(dtype=torch.float16).cuda()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        op = dpa(inp, cu_seqlens, config.seq_len)
    op.backward(op_grad)

    context = torch.load("ctx.pt")
    dqkv = torch.load('dqkv.pt')
    return (
        context.view(config.batch_size, config.seq_len, -1).transpose(0,1),
        dqkv.view(
            config.batch_size,
            config.seq_len,
            3,
            config.num_heads,
            config.head_dim,
        ).transpose(0,1).contiguous(),
    )

def _run_dpa_fp8_ref(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.load('qkv.pt').cuda()
    inp.requires_grad=True
    seqlens = torch.full(
        [config.batch_size],
        config.seq_len,
        dtype=torch.int32,
        device="cuda",
    ).cuda()
    cu_seqlens = torch.zeros(
        config.batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = torch.load('op_grad.pt').cuda().view(config.batch_size, config.seq_len, -1).transpose(0,1)

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker():
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    block = (
         DotProductAttention(
                config.num_heads,
                config.head_dim,
                attention_dropout=config.dropout_prob,
                sequence_parallel=False,
                tp_size=1,
                get_rng_state_tracker=get_dummy_cuda_rng_tracker,
                tp_group=None,
                layer_number=1,
                attention_type="self"
        ).to(dtype=dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(q, k, v, attn_mask_type=config.attn_mask_type)
    op.backward(op_grad)

    return op, inp.grad

_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = False
_2X_ACC_WGRAD = False

META_QKV  = tex.FP8FwdTensors.GEMM1_OUTPUT
META_O    = tex.FP8FwdTensors.GEMM2_INPUT
META_DO   = tex.FP8BwdTensors.GRAD_INPUT2
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1

META_S    = tex.FP8FwdTensors.GEMM3_WEIGHT
META_DS   = tex.FP8BwdTensors.GRAD_INPUT3

class _dpa_fp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_heads: int,
        p_dropout: float,
        max_s: int,
        fast_zero_fill: bool,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        is_training: bool,
    ) -> torch.Tensor:

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1
        is_nl = False
        if b < 4 and b > 1:
            max_s = 512
            is_nl = True

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        inputmat, inputmat_t = ext.fp8_cast_transpose_fused(
            inp,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )

        qkv_weight_fp8, qkv_weight_t_fp8 = ext.fp8_cast_transpose_fused(
            qkv_weight,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
        )

        M = None
        ZInv = None
        philox_unpacked = None

        qkv_out, _ = ext.fp8_gemm(
            qkv_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            torch.uint8,
            workspace,
            bias=qkv_bias,
            use_bias=True,
            out_index=META_QKV,
            fp8_meta_tensor=fp8_meta["scaling_fwd"],
            use_split_accumulator=_2X_ACC_FPROP,
            D_dtype=fp8_dtype_forward,
        )
        qkv_out = qkv_out.view(-1, 3, h, d)
        qkv_out_fp16 = ext.cast_from_fp8(qkv_out, fp8_meta["scaling_fwd"],
                META_QKV, fp8_dtype_forward,
                tex.DType.kFloat16).view(b, max_s, 3, h, d).transpose(0,1).contiguous()
        torch.save(qkv_out_fp16, 'qkv.pt')

        # FMHA
        context_, aux_ctx_tensors, *rest = fused_attn_fwd(
                is_training,
                max_s,
                max_s,
                cu_seqlens,
                cu_seqlens,
                qkv_out[:,0,:,:],
                qkv_out[:,1,:,:],
                qkv_out[:,2,:,:],
                fp8_dtype_forward,
                FusedAttnBackend["FP8"],
                None,
                fp8_meta["scaling_fwd"].scale_inv[META_QKV],
                fp8_meta["scaling_fwd"].scale[META_S],
                fp8_meta["scaling_fwd"].scale[META_O],
                fp8_meta["scaling_fwd"].amax_history[0][META_S],
                fp8_meta["scaling_fwd"].amax_history[0][META_O],
                attn_scale=None,
                dropout=p_dropout,
                fast_zero_fill=fast_zero_fill,
                qkv_layout="t3hd",
                attn_bias_type="no_bias",
                attn_mask_type="padding",
                rng_gen=None,
                )
        M, ZInv, philox_unpacked = aux_ctx_tensors

        context = context_.view(-1, in_features)
        context_t = tex.fp8_transpose(context, fp8_dtype_forward)

        ctx.save_for_backward(
            inputmat_t, qkv_weight_t_fp8, workspace,
            qkv_out,
            context_, context_t,
            fp8_meta["scaling_fwd"].scale,
            fp8_meta["scaling_fwd"].scale_inv,
        )
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.fast_zero_fill = fast_zero_fill
        ctx.is_nl = is_nl
        ctx.hidden_size = in_features
        ctx.num_heads = num_heads

        context_fp16 = ext.cast_from_fp8(context, fp8_meta["scaling_fwd"],
                META_O, fp8_dtype_forward, tex.DType.kFloat16)
        torch.save(context_fp16, 'ctx.pt')
        return context_fp16


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with _prepare_backward(True, ctx.fp8_meta, None, 1, name="_DPA"):
            (
                inputmat_t,
                qkv_weight_t_fp8,
                workspace,
                qkv_out,
                context, context_t,
                fwd_scales,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            proj_dgrad = ext.cast_to_fp8(
                grad_output, ctx.fp8_meta["scaling_bwd"], META_DO, fp8_dtype_backward
            )

            dq, dk, dv, *rest = fused_attn_bwd(
                    ctx.max_s,
                    ctx.max_s,
                    ctx.cu_seqlens,
                    ctx.cu_seqlens,
                    qkv_out[:,0,:,:],
                    qkv_out[:,1,:,:],
                    qkv_out[:,2,:,:],
                    context,
                    proj_dgrad.view_as(context),
                    fp8_dtype_forward,
                    ctx.aux_ctx_tensors,
                    FusedAttnBackend["FP8"],
                    fwd_scale_inverses[META_QKV], # d_scale_qkv,
                    fwd_scale_inverses[META_S], # d_scale_s,
                    fwd_scale_inverses[META_O], # d_scale_o,
                    ctx.fp8_meta['scaling_bwd'].scale_inv[META_DO], # d_scale_do
                    fwd_scales[META_S], # q_scale_s
                    ctx.fp8_meta['scaling_bwd'].scale[META_DS], # q_scale_ds
                    ctx.fp8_meta['scaling_bwd'].scale[META_DQKV], # q_scale_dqkv
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DS], # amax_ds
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DQKV], # amax_dqkv
                    None,
                    ctx.p_dropout,
                    ctx.fast_zero_fill,
                    "t3hd",
                    "no_bias",
                    "padding",
                    )
            dqkv = torch.cat([dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)], dim=1)

            dqkv_grad_output_c = dqkv.view(-1, 3*ctx.hidden_size)
            dqkv_grad_output_c_fp16 = ext.cast_from_fp8(dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"], META_DQKV,
                fp8_dtype_backward, tex.DType.kFloat16)
            torch.save(dqkv_grad_output_c_fp16, 'dqkv.pt')

            qkv_bgrad, dqkv_grad_output_t = ext.fp8_transpose_bgrad_fused(
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"],
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
            )

            # QKV DGRAD
            qkv_dgrad, _ = ext.fp8_gemm(
                qkv_weight_t_fp8,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            # QKV WGRAD
            qkv_wgrad, _ = ext.fp8_gemm(
                inputmat_t,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                dqkv_grad_output_t,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        return (qkv_dgrad,
            qkv_wgrad,
            qkv_bgrad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None)

class DPA_FP8(TransformerEngineBaseModule):
    def __init__(
        self,
        config,
        params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.dropout_prob
        self.h = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.fast_zero_fill = True

        self.qkv_weight = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.qkv_weight.shape)
        self.qkv_bias = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        with torch.no_grad():
            self.qkv_bias.zero_()
            self.qkv_weight.fill_(1.0)
        self.workspace = torch.empty(
            _CUBLASLT_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )

    def forward(
        self, inp: torch.Tensor,
        cu_seqlens, max_s,
    ) -> torch.Tensor:
        with self.prepare_forward(inp, None, num_gemms=3) as inp:
            out = _dpa_fp8.apply(
                inp,
                self.qkv_weight,
                self.qkv_bias,
                cu_seqlens,
                self.h,
                self.p_dropout,
                max_s,
                self.fast_zero_fill,
                self.fp8_meta,
                self.workspace,
                self.training)
        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Needs override."""
