# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled masked softmax functions"""
import os
from typing import Callable, Tuple, Union, Optional
import torch
from torch import nn
import torch._C._onnx as _C_onnx
from torch.onnx import _type_utils
import transformer_engine_torch as tex
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.te_onnx_extensions import compute_in_fp32


THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128


_default_causal_mask = {}


def _get_default_causal_mask(sq: int, sk: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input"""
    if sq == 1:
        return torch.zeros((1, sk), dtype=torch.bool, device="cuda")

    matrix_shape = (sq, sk)
    if matrix_shape not in _default_causal_mask:
        diagonal_offset = sk - sq + 1
        _default_causal_mask[matrix_shape] = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device="cuda"), diagonal=diagonal_offset
        )
    return _default_causal_mask[matrix_shape]


def _get_onnx_export_causal_mask(
    seq_q: int, seq_k: int, onnx_causal_mask: torch.Tensor
) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input, for ONNX export.

    ONNX does not support dynamic control-flow and requires non-square masks when
    using a KV-cache (seq_k's length len(context)+len(generative) while seq_q's length is 1).

    Argument `onnx_causal_mask` is a square triu (k=1) mask that is sliced to the correct
    shape for GPT context and generation phases.
    In the context phase the derived mask is a square triu of shape (seq_k, seq_k), and in
    the generation phase the mask is rectangular with shape (1, seq_k).
    """
    assert len(onnx_causal_mask.size()) == 2
    assert onnx_causal_mask.size(0) == onnx_causal_mask.size(1)
    assert onnx_causal_mask.size(0) >= (seq_k - seq_q) >= 0
    derived_mask = onnx_causal_mask[seq_k - seq_q : seq_k, :seq_k]
    return derived_mask


def fp32_compute(onnx_symbolic_fn):
    """A decorator that wraps an ONNX symoblic function with FP32 compute operators."""

    def wrapper(g: torch.Graph, inp: torch._C.Value, scale: float, *args, **kwargs):
        return compute_in_fp32(g, inp, onnx_symbolic_fn, scale, *args, **kwargs)

    return wrapper


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledUpperTriangMaskedSoftmax fwd"""
        scale_t = torch.tensor([scale])
        softmax_results = tex.scaled_upper_triang_masked_softmax_forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledUpperTriangMaskedSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = tex.scaled_upper_triang_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None

    @staticmethod
    @fp32_compute
    def symbolic(g: torch.Graph, inputs: torch._C.Value, scale: float) -> torch._C.Value:
        """ScaledUpperTriangMaskedSoftmax symbolic method"""

        def triangular_mask():
            dtype = _type_utils.JitScalarType.INT64
            ones = torch.onnx.symbolic_opset9.ones_like(g, inputs, dtype)
            k = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
            mask = g.op("Trilu", ones, k, upper_i=1)
            mask = g.op("Cast", mask, to_i=_C_onnx.TensorProtoDataType.BOOL)
            return mask

        # Captures the logic of function scaled_upper_triang_masked_softmax_warp_forward
        mask = triangular_mask()
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        inv_mask = g.op("Sub", one, mask)

        neg_tenK = g.op("Constant", value_t=torch.tensor(-10000.0, dtype=torch.float16))
        softmax_mask = g.op("Mul", mask, neg_tenK)

        scale_input = g.op("Constant", value_t=torch.tensor(scale, dtype=torch.float16))
        scaled = g.op("Mul", inputs, scale_input)
        masked_scaled = g.op("Mul", inv_mask, scaled)
        masked = g.op("Add", masked_scaled, softmax_mask)
        out = g.op("Softmax", masked)
        return out


class ScaledAlignedCausalMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply causal mask aligned to the bottom right corner of the input matrix
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledAlignedCausalMaskedSoftmax fwd"""
        scale_t = torch.tensor([scale])
        softmax_results = tex.scaled_aligned_causal_masked_softmax_forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledAlignedCausalMaskedSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = tex.scaled_aligned_causal_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None

    @staticmethod
    @fp32_compute
    def symbolic(g: torch.Graph, inputs: torch._C.Value, scale: float) -> torch._C.Value:
        """ScaledAlignedCausalMaskedSoftmax symbolic method"""

        def triangular_mask():
            dtype = _type_utils.JitScalarType.INT64
            ones = torch.onnx.symbolic_opset9.ones_like(g, inputs, dtype)
            k = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))

            # rectangular causal mask aligned to the bottom right corner of Attention matrix
            rows = inputs.size(dim=-2)
            cols = inputs.size(dim=-1)
            diag_shift = cols - rows + 1

            mask = g.op("Trilu", ones, k, upper_i=diag_shift)
            mask = g.op("Cast", mask, to_i=_C_onnx.TensorProtoDataType.BOOL)
            return mask

        # Captures the logic of function scaled_aligned_masked_softmax_warp_forward
        mask = triangular_mask()
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        inv_mask = g.op("Sub", one, mask)

        neg_tenK = g.op("Constant", value_t=torch.tensor(-10000.0, dtype=torch.float16))
        softmax_mask = g.op("Mul", mask, neg_tenK)

        scale_input = g.op("Constant", value_t=torch.tensor(scale, dtype=torch.float16))
        scaled = g.op("Mul", inputs, scale_input)
        masked_scaled = g.op("Mul", inv_mask, scaled)
        masked = g.op("Add", masked_scaled, softmax_mask)
        out = g.op("Softmax", masked)
        return out


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledMaskedSoftmax fwd"""
        scale_t = torch.tensor([scale])

        softmax_results = tex.scaled_masked_softmax_forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledMaskedSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = tex.scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None

    @staticmethod
    @fp32_compute
    def symbolic(
        g: torch.Graph, inputs: torch._C.Value, mask: torch._C.Value, scale: float
    ) -> torch._C.Value:
        """ScaledMaskedSoftmax symbolic method"""
        # Captures the logic of function scaled_masked_softmax_warp_forward.
        # output = softmax(mask(input*scale)
        # Computed as:
        #   masked_scaled = (1 - mask)*(input*scale)
        #   softmax_mask = mask * -10000
        #   output = softmax(masked_scaled + softmax_mask)
        scale_input = g.op("Constant", value_t=torch.tensor(scale, dtype=torch.float16))
        scaled = g.op("Mul", inputs, scale_input)
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        inv_mask = g.op("Sub", one, mask)
        # Note: type is hard coded because softmax uses FP16 or BF16
        neg_tenK = g.op("Constant", value_t=torch.tensor(-10000.0, dtype=torch.float16))
        softmax_mask = g.op("Mul", mask, neg_tenK)
        masked_scaled = g.op("Mul", inv_mask, scaled)
        masked = g.op("Add", masked_scaled, softmax_mask)
        out = g.op("Softmax", masked)
        return out


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledSoftmax fwd"""
        scale_t = torch.tensor([scale])

        softmax_results = tex.scaled_softmax_forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = tex.scaled_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None

    @staticmethod
    @fp32_compute
    def symbolic(g: torch.Graph, inputs: torch._C.Value, scale: float) -> torch._C.Value:
        """ScaledSoftmax symbolic method"""
        scale_input = g.op("Constant", value_t=torch.tensor(scale, dtype=torch.float16))
        scaled = g.op("Mul", inputs, scale_input)
        out = g.op("Softmax", scaled)
        return out


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
    """

    def __init__(
        self,
        mask_func: Callable,
        softmax_in_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.scaled_masked_softmax_fusion = bool(int(os.getenv("NVTE_MASKED_SOFTMAX_FUSION", "1")))
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32

        # Users exporting to ONNX can optimize the attention mask for GPT text generation.
        self.kvcache_max_seq = int(os.getenv("NVTE_ONNX_KVCACHE_MAX_SEQ_LEN", "-1"))
        if self.kvcache_max_seq > 0:
            self.register_buffer(
                "onnx_causal_mask",
                torch.triu(
                    torch.ones(self.kvcache_max_seq, self.kvcache_max_seq, device="cuda"),
                    diagonal=1,
                ).bool(),
                persistent=False,
            )

    def forward(
        self,
        inp: torch.Tensor,
        mask: torch.Tensor,
        attn_mask_type: str,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """FusedScaleMaskSoftmax fprop"""
        # [b, np, sq, sk]
        assert inp.dim() == 4
        self.input_in_fp16 = inp.dtype == torch.float16
        self.input_in_bf16 = inp.dtype == torch.bfloat16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type

        assert scale is None or self.softmax_in_fp32, "softmax should be in fp32 when scaled"

        if self.is_kernel_available(mask, *inp.size()) and not is_in_onnx_export_mode():
            return self.forward_fused_softmax(inp, mask, scale)
        return self.forward_torch_softmax(inp, mask, scale)

    def is_kernel_available(self, mask: torch.Tensor, b: int, np: int, sq: int, sk: int) -> bool:
        """Check FusedScaleMaskSoftmax kernel availability based on size"""
        attn_batches = b * np

        if not self.scaled_masked_softmax_fusion:
            return False  # user doesn't want to fuse
        if not self.input_in_float16:
            return False  # input must be fp16
        if not 16 < sk < 16384:
            return False  # sk must be 16 ~ 16384
        if sk % 8 != 0:
            return False  # sk must be divisor of 8
        if self.attn_mask_type == "arbitrary":
            return False  # Custom masks not supported

        if self.attn_mask_type == "causal":  # unfused causal softmax kernel
            return True

        if (
            sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
            and self.attn_mask_type != "arbitrary"  # Custom masks not supported
        ):
            batch_per_block = self.get_batch_per_block(int(sk))

            if self.attn_mask_type == "padding":
                if (
                    mask is not None
                    and sq % batch_per_block == 0
                    and mask.shape[-2] == sq
                    and mask.shape[-1] == sk
                ):
                    return True
            else:
                if sq % batch_per_block == 0:
                    return True
        return False

    def forward_fused_softmax(
        self, inp: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        """Fused masked softmax kernel"""
        scale = 1.0 if scale is None else scale

        if self.attn_mask_type == "causal":
            return ScaledAlignedCausalMaskedSoftmax.apply(inp, scale)

        # input is 4D tensor (b, np, sq, sk)
        if mask is not None and self.attn_mask_type != "no_mask":
            return ScaledMaskedSoftmax.apply(inp, mask, scale)
        return ScaledSoftmax.apply(inp, scale)

    def forward_torch_softmax(
        self, inp: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        """Framework softmax"""
        if self.input_in_float16 and self.softmax_in_fp32:
            inp = inp.float()

        if scale is not None:
            inp = inp * scale

        if self.attn_mask_type == "causal":
            seq_len_q, seq_len_k = inp.size(2), inp.size(3)
            if is_in_onnx_export_mode() and self.kvcache_max_seq > 0:
                assert self.kvcache_max_seq >= seq_len_k
                mask = _get_onnx_export_causal_mask(seq_len_q, seq_len_k, self.onnx_causal_mask)
            else:
                mask = _get_default_causal_mask(seq_len_q, seq_len_k)

        mask_output = inp
        if mask is not None and self.attn_mask_type != "no_mask":
            mask_output = self.mask_func(inp, mask)
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(key_seq_len: int) -> int:
        """Softmax utility"""
        pow2 = 1 << (key_seq_len - 1).bit_length()
        warp_size = pow2 if pow2 < THREADS_PER_WARP else THREADS_PER_WARP
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = THREADS_PER_BLOCK // warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block
