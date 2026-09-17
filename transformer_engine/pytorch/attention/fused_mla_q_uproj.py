# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused MLA Q up-projection + per-head RoPE + MXFP8 quantize.

The MLA RoPE Triton kernels are taken from:
Megatron-LM megatron/core/fusions/fused_mla_yarn_rope_apply.py
"""

from __future__ import annotations
import functools
import os
import weakref
from importlib.metadata import PackageNotFoundError, version as get_pkg_version

import torch
import transformer_engine_torch as tex
from packaging.version import Version as PkgVersion

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ..constants import MXFP8_BLOCK_SCALING_SIZE
from ..distributed import get_distributed_world_size
from ..quantized_tensor import QuantizedTensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from ..utils import get_device_compute_capability

_CUDNN_FRONTEND_MIN_VERSION = "1.27.0"


def _cudnn_frontend_version_supported() -> bool:
    """Check that the installed nvidia-cudnn-frontend meets the minimum version."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion(
            _CUDNN_FRONTEND_MIN_VERSION
        )
    except PackageNotFoundError:
        return False


if triton is not None:

    @triton.jit
    def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
        token_idx = -1
        this_seq_len = 0
        seq_idx = 0
        last_cum_seqlen = tl.load(cu_seqlens) // cp_size
        while seq_idx < seq_num:
            cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1) // cp_size
            if token_idx == -1 and cur_cum_seqlen > pid_m:
                token_idx = pid_m - last_cum_seqlen
                this_seq_len = cur_cum_seqlen - last_cum_seqlen
            last_cum_seqlen = cur_cum_seqlen
            seq_idx += 1
        if cp_size > 1:
            if token_idx < this_seq_len // 2:
                token_idx = token_idx + cp_rank * this_seq_len // 2
            else:
                token_idx = (token_idx - this_seq_len // 2) + (
                    2 * cp_size - cp_rank - 1
                ) * this_seq_len // 2
        return token_idx

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "head_num"],
        restore_value=["DO"],
    )
    @triton.jit
    def rotary_bwd_q_kernel(
        DO,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        """
        Triton kernel of the backward pass for applying YARN RoPE to MLA's query.
        This kernel inplace modifies the input tensor DO.

        Input:
            DO: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
                or [total_seq_len, head_num, qk_head_dim + emb_dim]
            COS/SIN: [max_seq_len, emb_dim]

            batch_size, seq_num, and cu_seqlens_q are the same as in the forward pass
        """
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)

        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

        DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
        mask = x_off < head_num * stride_x_nheads
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)

        x_1 = x_left * cos_left + x_right * sin_right
        x_2 = -x_left * sin_left + x_right * cos_right

        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        tl.store(DO + x_1_off, x_1, mask=mask)
        tl.store(DO + x_2_off, x_2, mask=mask)

else:
    rotary_bwd_q_kernel = None


class FusedMLAQUpProjRopeQuant:
    """Wrapper for the cuDNN fused MLA Q up-proj + per-head RoPE + MXFP8 quantize kernel.

    - If w is already a QuantizedTensor (primary FP8 parameter in MXFP8BlockScaling recipe),
      this performs an MXFP8 GEMM within the fusion (and quantizes the input if necessary)
    - Otherwise (plain BF16 weight), x and w are passed as-is to the BF16 kernel variant.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _kernel(cls):
        # Import directly from the subpackage to avoid depending on cudnn/__init__.py
        # lazy-import registration (which would require overlaying cudnn/__init__.py and
        # could revert atomicrmw fixes present in the container's version).
        try:
            from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100

            return gemm_proj_rope_mxfp8_wrapper_sm100
        except ImportError:
            return None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the cuDNN FE fused gemm rope quant wrapper is available"""
        if int(os.environ.get("NVTE_FUSED_MLA_Q_UPROJ", "1")) <= 0:
            return False
        if not _cudnn_frontend_version_supported():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        if cls._kernel() is None:
            return False
        return True

    @classmethod
    def run(
        cls,
        x,  # MXFP8Tensor when w is MXFP8 (already quantized by the norm), else bf16 Tensor
        w,  # MXFP8Tensor (primary FP8 param) or bf16 torch.Tensor
        cos: torch.Tensor,
        sin: torch.Tensor,
        s: int,
        b: int,
    ) -> "tuple[MXFP8Tensor, torch.Tensor]":
        """Run the fused kernel; return (Q MXFP8Tensor, activation saved for the wgrad backward).

        The kernel precision is selected by the weight precision.  On the FP8 path ``x`` must
        arrive already MXFP8-quantized: the caller's normalization emits MXFP8 straight from
        its FP32 accumulator, exactly as `TELayerNormColumnParallelLinear` does, and quantizing
        again here would round a second time and change the GEMM's input bytes.
        """

        from cuda.bindings import driver as cuda

        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        wrapper = cls._kernel()

        if isinstance(w, QuantizedTensor):
            assert isinstance(w, MXFP8Tensor), (
                "FusedMLAQUpProjRopeQuant expects an MXFP8Tensor weight (MXFP8BlockScaling"
                f" recipe), got {type(w).__name__}. Use the unfused path for other quantization"
                " recipes."
            )
            # ---- FP8 projection: MXFP8 x straight from the norm + w's fp8 codes -> mxfp8in ----
            # x carries both usages: rowwise feeds this GEMM, columnwise the FP8 wgrad in backward.
            x_mxfp8 = x
            assert isinstance(x_mxfp8, QuantizedTensor) and hasattr(x_mxfp8, "_rowwise_data"), (
                "FusedMLAQUpProjRopeQuant needs an MXFP8-quantized input on the FP8 path, got"
                f" {type(x_mxfp8).__name__}; have the normalization emit MXFP8 directly."
            )
            assert not x_mxfp8._with_gemm_swizzled_scales, (
                "x scales must be unswizzled: the cuDNN kernel reads them as a plain"
                " [tokens, K//32] array."
            )
            x_code = x_mxfp8._rowwise_data.view(torch.float8_e4m3fn)  # [tokens, K]
            x_scale = x_mxfp8._rowwise_scale_inv  # [tokens, K//32] uint8

            # Primary FP8 parameter: already quantized; use its rowwise FP8 codes + E8M0 scales.
            w.update_usage(rowwise_usage=True, columnwise_usage=None)
            w_code = w._rowwise_data.view(torch.float8_e4m3fn)  # [N, K]
            w_scale = w._rowwise_scale_inv  # [N, K//32] uint8

            out = wrapper(
                x_code,
                w_code,
                cos,
                sin,
                x_scale=x_scale,
                w_scale=w_scale,
                w_out_in=True,
                stream=stream,
            )

            # Drop rowwise data now.
            # Only columnwise x is needed for the FP8 wgrad in backward.
            x_mxfp8.update_usage(rowwise_usage=False, columnwise_usage=True)
            x_saved = x_mxfp8
        else:
            # ---- 16-bit projection: bf16 GEMM inputs -> bf16in (the projection stays bf16) ----
            out = wrapper(x, w, cos, sin, w_out_in=True, stream=stream)
            x_saved = x

        nh = out["out_fp8_row"].shape[1]
        d = out["out_fp8_row"].shape[2]
        query = cls.wrap_mxfp8(
            out["out_fp8_row"],
            out["out_scales_row"],
            out["out_fp8_col"],
            out["out_scales_col"],
            s,
            b,
            nh,
            d,
        )
        # 2nd return is the activation to save for wgrad: MXFP8 (fp8 path) or bf16 (16-bit path).
        return query, x_saved

    @classmethod
    def backward_linear(
        cls,
        grad_output,
        x_saved,
        w_q,
        act_dtype,
        wgrad_store,
        fuse_wgrad_accumulation,
        tp_group,
        sequence_parallel,
    ):
        """Linear backward for the fused Q up-proj."""
        from ..module.linear import (
            LinearBwdArgs,
            _linear_backward_impl,
            _2X_ACC_DGRAD,
            _2X_ACC_WGRAD,
        )

        tp_size = get_distributed_world_size(tp_group) if tp_group is not None else 1
        fp8 = isinstance(w_q, QuantizedTensor)

        grad_output_quantizer = None
        if fp8:
            grad_output_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )
            grad_output_quantizer.optimize_for_gemm = True

        bwd_args = LinearBwdArgs(
            grad_output=grad_output,
            inputmat=x_saved,
            weight_fp8=w_q,
            saved_weight=w_q,
            grad_output_quantizer=grad_output_quantizer,
            inp_shape=x_saved.shape,
            activation_dtype=act_dtype,
            fp8=fp8,
            # This temporary fused API always computes both projection gradients.
            requires_dgrad=True,
            requires_wgrad=True,
            dgrad_use_split_accumulator=_2X_ACC_DGRAD,
            wgrad_use_split_accumulator=_2X_ACC_WGRAD,
            is_weight_param_quantized=fp8,
            parallel_mode="column",
            tp_group=tp_group,
            tp_size=tp_size,
            tensor_parallel=tp_size > 1,
            sequence_parallel=sequence_parallel,
            is_fsdp2=False,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            wgrad_store=wgrad_store,
            origin_weight_ref=weakref.ref(w_q) if fuse_wgrad_accumulation else None,
            main_grad_func=(lambda: w_q.main_grad) if fuse_wgrad_accumulation else None,
        )

        wgrad, dgrad, grad_bias = _linear_backward_impl(bwd_args)
        return dgrad, wgrad, grad_bias

    @classmethod
    def wrap_mxfp8(
        cls,
        fp8_row: torch.Tensor,
        scales_row: torch.Tensor,
        fp8_col: torch.Tensor,
        scales_col: torch.Tensor,
        s: int,
        b: int,
        nh: int,
        d: int,
    ) -> MXFP8Tensor:
        """Wrap raw data and scale tensors into an MXFP8Tensor"""

        blk = MXFP8_BLOCK_SCALING_SIZE
        # Both rowwise and columnwise Q are required:
        # - Forward QK^T uses rowwise
        # - cuDNN backward (fused_attn_fp8_bwd_impl) requires columnwise for dK gradient
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        return MXFP8Tensor(
            shape=(s, b, nh, d),
            dtype=torch.bfloat16,
            rowwise_data=fp8_row.view(s, b, nh, d),
            rowwise_scale_inv=scales_row.view(s, b, nh, d // blk),
            columnwise_data=fp8_col.view(s, b, nh, d),
            columnwise_scale_inv=scales_col.view(s // blk, b, nh, d),
            quantizer=quantizer,
            requires_grad=False,
            fp8_dtype=tex.DType.kFloat8E4M3,
            with_gemm_swizzled_scales=False,
        )


class FusedMLAQUpProjFunction(torch.autograd.Function):
    """Fused Q up-proj: q -> (layernorm + MXFP8 quant) -> (GEMM + per-head RoPE + MXFP8) -> MXFP8Tensor Q."""

    @staticmethod
    def forward(
        ctx,
        x,  # [s, b, q_lora_rank] pre-norm input
        gamma,  # [q_lora_rank] q_layernorm weight
        w_q,  # [nh*q_head_dim, q_lora_rank] FP8 QuantizedTensor or bf16 (TE out×in layout)
        cos,  # [s, 1, 1, rope_dim]
        sin,  # [s, 1, 1, rope_dim]
        wgrad_store,
        fuse_wgrad_accumulation,
        nh,
        q_head_dim,
        qk_head_dim,
        qk_pos_emb_head_dim,
        s,
        b,
        tp_group,  # tensor-parallel process group
        sequence_parallel,  # True if sequence parallelism is active
        eps,  # norm epsilon
        normalization,  # "RMSNorm"
        zero_centered_gamma,
    ):
        """Run the normalization (quantized output) then the fused gemm + rope + mxfp8"""
        from ..module._common import apply_normalization

        tokens = s * b
        tp_size = get_distributed_world_size(tp_group) if tp_group is not None else 1
        if tp_size > 1:
            raise RuntimeError(
                "FusedMLAQUpProjFunction does not support tensor parallelism (TP>1): "
                "the backward dgrad is reduce-scattered over TP ranks but the caller "
                "passes a pre-gathered full-sequence input. Use TP=1 or the unfused path."
            )
        if normalization != "RMSNorm":
            raise RuntimeError(
                "FusedMLAQUpProjFunction supports RMSNorm only, got "
                f"{normalization}; LayerNorm would also need mu saved in forward."
            )
        x2d = x.detach().reshape(tokens, -1).contiguous()
        fp8 = isinstance(w_q, QuantizedTensor)

        # Matches LayerNormLinear's input quantizer with one deliberate difference:
        # optimize_for_gemm stays off, because the fused GEMM+RoPE+Quant cuDNN kernel reads the rowwise scales as a
        # plain [tokens, K//32] array.
        x_quantizer = None
        if fp8:
            x_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )

        ln_out, _, rsigma = apply_normalization(
            x2d,
            None,
            gamma,
            None,
            eps,
            x_quantizer,
            x2d.dtype,
            normalization,
            int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0")),
            zero_centered_gamma,
        )

        # Reshape [s, 1, 1, rope_dim] -> [s*b, rope_dim] bf16 as required by the KF kernel.
        def _flat(t):
            t = t.reshape(s, -1)
            if b > 1:
                t = t.unsqueeze(1).expand(s, b, t.shape[-1]).reshape(tokens, -1)
            return t.to(torch.bfloat16).contiguous()

        cos, sin = _flat(cos), _flat(sin)
        query, x_saved = FusedMLAQUpProjRopeQuant.run(ln_out, w_q.detach(), cos, sin, s, b)

        ctx.save_for_backward(x2d, rsigma, gamma, x_saved, w_q, cos, sin)
        ctx.wgrad_store = wgrad_store
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        ctx.act_dtype = x.dtype
        ctx.dims = (nh, q_head_dim, qk_head_dim, qk_pos_emb_head_dim, s, b)
        ctx.tp_group = tp_group
        ctx.sequence_parallel = sequence_parallel
        ctx.normalization = normalization
        ctx.zero_centered_gamma = zero_centered_gamma
        return query

    @staticmethod
    def backward(ctx, dq):
        """Backward is unfused and matches the typical backward pass"""
        if rotary_bwd_q_kernel is None:
            raise RuntimeError("Fused MLA Q up-projection backward requires Triton")

        x2d, rsigma, gamma, x_saved, w_q, cos, sin = ctx.saved_tensors
        nh, q_head_dim, qk_head_dim, qk_pos_emb_head_dim, s, b = ctx.dims
        tokens = s * b
        act_dtype = ctx.act_dtype

        # --- RoPE backward (unchanged: bf16, same rotary_bwd_q_kernel as the unfused path) ---
        dq3 = dq.reshape(tokens, nh, q_head_dim).contiguous()

        def grid(META):
            return (tokens, triton.cdiv(nh, META["BLOCK_H"]))

        rotary_bwd_q_kernel[grid](
            dq3,
            cos.contiguous(),
            sin.contiguous(),
            qk_head_dim,
            qk_pos_emb_head_dim,
            nh,
            1,
            None,
            None,
            dq3.stride(0),
            dq3.stride(1),
            0,
            1,
        )
        # grad w.r.t. the (pre-RoPE) up-proj GEMM output; bf16.
        dq2d = dq3.reshape(tokens, nh * q_head_dim).contiguous()

        # Delegate the projection backward to TE's _linear_backward (via backward_linear)
        grad_ln_out, ret_grad_w, _ = FusedMLAQUpProjRopeQuant.backward_linear(
            grad_output=dq2d,
            x_saved=x_saved,
            w_q=w_q,
            act_dtype=act_dtype,
            wgrad_store=ctx.wgrad_store,
            fuse_wgrad_accumulation=ctx.fuse_wgrad_accumulation,
            tp_group=ctx.tp_group,
            sequence_parallel=ctx.sequence_parallel,
        )

        # --- Norm backward, on the rsigma this forward saved ---
        bwd_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        grad_x, dgamma = tex.rmsnorm_bwd(
            grad_ln_out.reshape(x2d.shape),
            x2d,
            rsigma,
            gamma,
            bwd_sm_margin,
            ctx.zero_centered_gamma,
        )
        grad_x = grad_x.reshape(s, b, -1)

        # grads for: x, gamma, w_q, then cos, sin and the 13 non-tensor args
        return (grad_x, dgamma, ret_grad_w) + (None,) * 15
