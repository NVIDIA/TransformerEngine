# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Fused scaled masked softmax functions"""

import os
from typing import Callable, Tuple, Union, Optional

import paddle

from transformer_engine.paddle.cpp_extensions import (
    scaled_upper_triang_masked_softmax_forward,
    scaled_upper_triang_masked_softmax_backward,
    scaled_masked_softmax_forward,
    scaled_masked_softmax_backward,
    scaled_softmax_forward,
    scaled_softmax_backward,
)

THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128

_default_causal_mask = {}


def _get_default_causal_mask(sq: int) -> paddle.Tensor:
    """Return the causal upper triangular mask for softmax input"""
    if sq not in _default_causal_mask:
        _default_causal_mask[sq] = paddle.triu(paddle.ones((sq, sq)), diagonal=1).cast('bool')
    return _default_causal_mask[sq]


class ScaledUpperTriangMaskedSoftmax(paddle.autograd.PyLayer):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: paddle.Tensor, scale: float) -> paddle.Tensor:
        """ScaledUpperTriangMaskedSoftmax fwd"""
        scale_t = paddle.Tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        """ScaledUpperTriangMaskedSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensor()
        input_grads = scaled_upper_triang_masked_softmax_backward(output_grads, softmax_results,
                                                                  scale_t[0])

        return input_grads, None


class ScaledMaskedSoftmax(paddle.autograd.PyLayer):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: paddle.Tensor, mask: paddle.Tensor, scale: float) -> paddle.Tensor:
        """ScaledMaskedSoftmax fwd"""
        scale_t = paddle.Tensor([scale])

        softmax_results = scaled_masked_softmax_forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        """ScaledMaskedSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensor()

        input_grads = scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class ScaledSoftmax(paddle.autograd.PyLayer):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: paddle.Tensor, scale: float) -> paddle.Tensor:
        """ScaledSoftmax fwd"""
        scale_t = paddle.Tensor([scale])

        softmax_results = scaled_softmax_forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        """ScaledSoftmax bwd"""
        softmax_results, scale_t = ctx.saved_tensor()

        input_grads = scaled_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class FusedScaleMaskSoftmax(paddle.nn.Layer):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
    """

    def __init__(
        self,
        attn_mask_type: str,
        mask_func: Callable,
        softmax_in_fp32: bool = True,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = bool(int(os.getenv("NVTE_MASKED_SOFTMAX_FUSION", "1")))
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.backend = backend

    def forward(
        self,
        inp: paddle.Tensor,
        mask: paddle.Tensor,
        scale: Optional[float] = None,
    ) -> paddle.Tensor:
        """FusedScaleMaskSoftmax fprop"""
        # [b, np, sq, sk]
        assert inp.dim() == 4
        self.input_in_fp16 = inp.dtype == paddle.float16
        self.input_in_bf16 = inp.dtype == paddle.bfloat16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16

        assert (scale is None or self.softmax_in_fp32), "softmax should be in fp32 when scaled"

        if self.backend == 'transformer_engine':
            return self._te_forward(inp, mask, scale)
        if self.backend == 'paddle':
            return self._pd_forward(inp, mask, scale)
        raise AttributeError(f"Backend {self.backend} is not supported.")

    def is_kernel_available(self, b: int, np: int, sq: int, sk: int) -> bool:
        """Check FusedScaleMaskSoftmax kernel availability based on size"""
        attn_batches = b * np

        if (self.scaled_masked_softmax_fusion    # user want to fuse
                and self.input_in_float16    # input must be fp16
                and 16 < sk <= 4096    # sk must be 16 ~ 2048
                and sq % 4 == 0    # sq must be divisor of 4
                and attn_batches % 4 == 0    # np * b must be divisor of 4
           ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(int(sk))

                if self.attn_mask_type == "causal":
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def _te_forward(self,
                    inp: paddle.Tensor,
                    mask: paddle.Tensor,
                    scale: Optional[float] = None) -> paddle.Tensor:
        """Fused masked softmax kernel"""
        b, np, sq, sk = inp.size()
        scale = 1.0 if scale is None else scale

        if self.attn_mask_type == "causal":
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            inp = inp.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(inp, scale)
            return probs.view(b, np, sq, sk)
        # input is 4D tensor (b, np, sq, sk)
        if mask is not None:
            return ScaledMaskedSoftmax.apply(inp, mask, scale)
        return ScaledSoftmax.apply(inp, scale)

    def _pd_forward(self,
                    inp: paddle.Tensor,
                    mask: paddle.Tensor,
                    scale: Optional[float] = None) -> paddle.Tensor:
        """Call Paddle OP"""
        if self.input_in_float16 and self.softmax_in_fp32:
            inp = paddle.cast(inp, 'float32')

        if scale is not None:
            inp = inp * scale

        if self.attn_mask_type == "causal":
            mask = _get_default_causal_mask(inp.shape[2])

        mask_output = self.mask_func(inp, mask) if mask is not None else inp
        probs = paddle.nn.functional.softmax(mask_output, axis=-1)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = paddle.cast(probs, 'float16')
            else:
                probs = paddle.cast(probs, 'bfloat16')

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
