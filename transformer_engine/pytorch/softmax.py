# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled masked softmax functions"""
import os
from typing import Callable, Tuple, Union, Optional
import torch
from torch import nn
import transformer_engine_torch as tex


THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128


_default_causal_mask = {}


def _get_default_causal_mask(mask_type: str, sq: int, sk: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input"""
    matrix_identifiers = (mask_type, sq, sk)
    if matrix_identifiers not in _default_causal_mask:
        diagonal_offset = sk - sq + 1 if "bottom_right" in mask_type else 1
        _default_causal_mask[matrix_identifiers] = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device="cuda"), diagonal=diagonal_offset
        )
    return _default_causal_mask[matrix_identifiers]


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

        if self.is_kernel_available(mask, *inp.size()):
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
        if sq == 1:
            return False  # sq must be > 1
        if self.attn_mask_type == "causal" and sq != sk:
            return False  # Fused causal kernel only support causal_bottom_right

        if (
            sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            batch_per_block = self.get_batch_per_block(int(sk))
            if "padding" in self.attn_mask_type or self.attn_mask_type == "arbitrary":
                if (
                    mask is not None
                    and sq % batch_per_block == 0
                    and mask.shape[0] in [1, b]
                    and mask.shape[1:] == (1, sq, sk)
                ):
                    return True
            else:
                if sq % batch_per_block == 0:
                    return True
        return False

    def forward_fused_softmax(
        self, inp: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Fused masked softmax path.
          attn_mask_type                                       | module
        -----------------------------------------------------------------------------------------
          no_mask                                              | ScaledSoftmax
          causal (self-attention), causal_bottom_right         | ScaledAlignedCausalMaskedSoftmax
          padding, padding_causal, padding_causal_bottom_right | ScaledMaskedSoftmax
          arbitrary ([1, 1, sq, sk] or [b, 1, sq, sk])         | ScaledMaskedSoftmax
        """
        scale = 1.0 if scale is None else scale

        # Disable for now until unalignment bug is fixed.
        # if self.attn_mask_type in ["causal", "causal_bottom_right"]:
        #    return ScaledAlignedCausalMaskedSoftmax.apply(inp, scale)

        # input is 4D tensor (1, 1, sq, sk) or (b, 1, sq, sk)
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

        if self.attn_mask_type in ["causal", "causal_bottom_right"]:
            seq_len_q, seq_len_k = inp.size(2), inp.size(3)
            causal_mask = _get_default_causal_mask(self.attn_mask_type, seq_len_q, seq_len_k)
            if mask is None:
                mask = causal_mask
            else:
                mask = torch.logical_or(mask, causal_mask)

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
