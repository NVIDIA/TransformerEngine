# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention import DotProductAttention
import os

class ModelConfig:
    def __init__(
        self, num_layers, hidden_size, num_attention_heads, head_dim, seq_len,
        dropout_p, attn_mask_type,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert (hidden_size == num_attention_heads * head_dim
                ), """hidden_size must be = num_heads x head_dim."""
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type

model_configs = {
    "test1": ModelConfig(1, 1024, 16, 64, 128, 0.0, "causal"),
    "test2": ModelConfig(1, 1024, 16, 64, 512, 0.0, "causal"),
    "test3": ModelConfig(1, 1024, 16, 64, 2048, 0.0, "causal"),
    "test4": ModelConfig(1, 2048, 16, 128, 128, 0.0, "causal"),
    "test5": ModelConfig(1, 2048, 16, 128, 512, 0.0, "causal"),
    "test6": ModelConfig(1, 2048, 16, 128, 2048, 0.0, "causal"),
    "test7": ModelConfig(1, 1024, 16, 64, 128, 0.0, "no_mask"),
    "test8": ModelConfig(1, 1024, 16, 64, 512, 0.0, "no_mask"),
}

param_types = [torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2, 32]

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_dot_product_attention(dtype, bs, model):
    """Test DotProductAttention module with three backends,
    FlashAttention, FusedAttention and UnfusedDotProductAttention"""

    config = model_configs[model]

    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FlashAttention")
    fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "UnfusedDotProductAttention")

    atol, rtol = (2.5e-2, 2.5e-2) if dtype == torch.bfloat16 else (2.5e-3, 2.5e-3)
    assert torch.allclose(fused_attn_fwd, flash_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, flash_attn_bwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol = atol, rtol = rtol)

def _run_dot_product_attention(dtype, bs, config, backend):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.1 * torch.randn(
            config.seq_len, bs, 3, config.num_attention_heads, config.head_dim,
            dtype = dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = 0.001 * torch.randint(0, 200, (
        config.seq_len, bs, config.num_attention_heads * config.head_dim
        ), dtype = dtype).cuda()

    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                attention_dropout = config.dropout_p,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = None,
                tp_group = None,
                layer_number = 1,
                attention_type = "self"
        ).to(dtype = dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(q, k, v)
    op.backward(op_grad)

    return op, inp.grad

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_transformer_layer(dtype, bs, model):
    """Test TransformerLayer module when its DotProductAttention is enabled with
    FlashAttention, FusedAttention, or UnfusedDotProductAttention backend"""

    config = model_configs[model]

    flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype, bs, config, "FlashAttention")
    fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
            dtype, bs, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
            dtype, bs, config, "UnfusedDotProductAttention")

    atol, rtol = (5e-1, 5e-1) if dtype == torch.bfloat16 else (5e-1, 5e-1)
    assert torch.allclose(fused_attn_fwd, flash_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, flash_attn_bwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol = atol, rtol = rtol)

def _run_transformer_layer(dtype, bs, config, backend):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.1 * torch.randn(
            config.seq_len, bs, config.num_attention_heads * config.head_dim,
            dtype = dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = 0.001 * torch.randint(0, 200, (
        config.seq_len, bs, config.num_attention_heads * config.head_dim
        ), dtype = dtype).cuda()

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
            config.num_attention_heads,
            layernorm_epsilon = 1e-5,
            hidden_dropout = 0.0,
            attention_dropout = config.dropout_p,
            init_method = init_method,
            output_layer_init_method = output_layer_init_method,
            layer_number = layer_number,
            kv_channels = config.head_dim,
            self_attn_mask_type = config.attn_mask_type,
            tp_group = None,
            tp_size =  1,
            params_dtype = dtype,
            get_rng_state_tracker = None,
            fuse_wgrad_accumulation = False,
            seq_length = config.seq_len,
            micro_batch_size = bs,
            sequence_parallel = False,
            apply_residual_connection_post_layernorm = False,
            output_layernorm = False,
            layer_type = "encoder",
            drop_path_rate = drop_path_rates[layer_number - 1],
            set_parallel_mode = True,
            fuse_qkv_params = True,
            zero_centered_gamma = False,
            qkv_weight_interleaved = False,
            ub_tp_comm_overlap = False,
            bias = True,
        )
        .to(dtype = dtype)
        .cuda()
    )

    op = block(inp)
    op.backward(op_grad)

    return op, inp.grad

model_configs_fp8 = {
    "test1": ModelConfig(1, 1024, 16, 64, 512, 0.0, "no_mask"),
}
batch_sizes_fp8 = [1, 4]
param_types_fp8 = [torch.float16]

@pytest.mark.parametrize("dtype", param_types_fp8)
@pytest.mark.parametrize("bs", batch_sizes_fp8)
@pytest.mark.parametrize("model", model_configs_fp8.keys())
def test_dpa_fp8(dtype, bs, model):
    """Test DotProductAttention module with FP8,
    using cpp_extensions import fused_attn_fwd/bwd_qkvpacked and UnfusedDotProductAttention"""

    config = model_configs_fp8[model]

    fused_attn_fwd, fused_attn_bwd = _run_dpa_fp8(
            dtype, bs, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa_fp8_ref(
            dtype, bs, config, "UnfusedDotProductAttention")

    atol, rtol = (5e-2, 1e-1)
    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol = atol, rtol = rtol)

def _run_dpa_fp8(dtype, bs, config, backend):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"

    inp = 0.01 * torch.randn(
            bs * config.seq_len, config.num_attention_heads * config.head_dim,
            dtype = dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = 0.001 * torch.randint(0, 200, (
        bs * config.seq_len, config.num_attention_heads * config.head_dim
        ), dtype = dtype).cuda()
    torch.save(op_grad, 'op_grad.pt')

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        interval=1,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

    dpa = DPA_FP8(config).to(dtype = torch.float16).cuda()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        op = dpa(inp, cu_seqlens, config.seq_len)
        op.backward(op_grad)

    context = torch.load("ctx.pt")
    dqkv = torch.load('dqkv.pt')
    return (context.view(bs, config.seq_len, -1).transpose(0,1),
        dqkv.view(bs, config.seq_len, 3, config.num_attention_heads, config.head_dim).transpose(0,1).contiguous())

def _run_dpa_fp8_ref(dtype, bs, config, backend):

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.load('qkv.pt').cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = torch.load('op_grad.pt').cuda().view(bs, config.seq_len, -1).transpose(0,1)

    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                attention_dropout = config.dropout_p,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = None,
                tp_group = None,
                layer_number = 1,
                attention_type = "self"
        ).to(dtype = dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(q, k, v)
    op.backward(op_grad)
    torch.save(op,'ctx_ref.pt')
    torch.save(inp.grad,'dqkv_ref.pt')

    return op, inp.grad

from torch.nn.parameter import Parameter
import transformer_engine.pytorch.cpp_extensions as ext
import transformer_engine_extensions as tex
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule, _prepare_backward
from transformer_engine.common import recipe
from typing import Union, Dict, Any, Tuple, List
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    FusedAttnBackend)

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
        num_attention_heads: int,
        p_dropout: float,
        max_s: int,
        fast_zero_fill: bool,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        is_training: bool,
    ) -> torch.Tensor:

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_attention_heads
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

        qkv_out = ext.fp8_gemm(
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
            out_index = META_QKV,
            fp8_meta_tensor = fp8_meta["scaling_fwd"],
            use_split_accumulator=_2X_ACC_FPROP,
            D_dtype=fp8_dtype_forward,
        )
        qkv_out = qkv_out.view(-1, 3, h, d)
        qkv_out_fp16 = ext.cast_from_fp8(qkv_out, fp8_meta["scaling_fwd"],
                META_QKV, fp8_dtype_forward,
                tex.DType.kFloat16).view(b, max_s, 3, h, d).transpose(0,1).contiguous()
        torch.save(qkv_out_fp16, 'qkv.pt')

        # FMHA
        context_, aux_ctx_tensors, *rest = fused_attn_fwd_qkvpacked(
                is_training,
                max_s,
                cu_seqlens,
                qkv_out,
                fp8_dtype_forward,
                FusedAttnBackend["FP8"],
                None,
                fp8_meta["scaling_fwd"].scale_inv[META_QKV],
                fp8_meta["scaling_fwd"].scale[META_S],
                fp8_meta["scaling_fwd"].scale[META_O],
                fp8_meta["scaling_fwd"].amax_history[0][META_S],
                fp8_meta["scaling_fwd"].amax_history[0][META_O],
                attn_scale = None,
                dropout = p_dropout,
                fast_zero_fill = fast_zero_fill,
                qkv_layout = "qkv_interleaved",
                attn_bias_type = "no_bias",
                attn_mask_type = "padding",
                rng_gen = None,
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
        ctx.num_attention_heads = num_attention_heads

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

            dqkv, *rest = fused_attn_bwd_qkvpacked(
                    ctx.max_s,
                    ctx.cu_seqlens,
                    qkv_out,
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
                    "qkv_interleaved",
                    "no_bias",
                    "padding",
                    )

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
            qkv_dgrad = ext.fp8_gemm(
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
            qkv_wgrad = ext.fp8_gemm(
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
        self.p_dropout = config.dropout_p
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.fast_zero_fill = True

        self.qkv_weight = Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.qkv_weight.shape)
        self.qkv_bias = Parameter(
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
