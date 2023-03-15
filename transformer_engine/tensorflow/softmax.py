# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled masked softmax functions"""

from typing import Callable

import os
import transformer_engine_tensorflow as tex
import tensorflow as tf

from .module import get_stream_id

THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128


class FusedScaleMaskSoftmax(tf.keras.Model):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        attn_mask_type: str,
        mask_func: Callable,
        softmax_in_fp32: bool,
        scale: float,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = bool(
            int(os.getenv("NVTE_MASKED_SOFTMAX_FUSION", "1"))
        )
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.stream = get_stream_id()

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def __call__(self, inp: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """FusedScaleMaskSoftmax fprop"""
        # [b, np, sq, sk]
        assert len(inp.shape) == 4
        self.input_in_fp16 = inp.dtype == tf.float16
        self.input_in_bf16 = inp.dtype == tf.bfloat16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16

        if self.is_kernel_available(*inp.shape):
            return self.forward_fused_softmax(inp, mask)
        return self.forward_tf_softmax(inp, mask)

    def is_kernel_available(self, b: int, np: int, sq: int, sk: int) -> bool:
        """Check FusedScaleMaskSoftmax kernel availability based on size"""
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
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

    @tf.custom_gradient
    def scaled_masked_softmax(self, x: tf.Tensor, mask: tf.Tensor,
                              scale: float):
        """Scaled masked softmax."""
        y = tex.scaled_masked_softmax_forward(x, mask, scale, self.stream)

        def grad_fn(upstream):
            dx = tex.scaled_masked_softmax_backward(upstream, y, scale,
                                                    self.stream)
            return dx, None, None

        return y, grad_fn

    @tf.custom_gradient
    def scaled_softmax(self, x: tf.Tensor, scale: float):
        """Scaled softmax."""
        y = tex.scaled_softmax_forward(x, scale, self.stream)

        def grad_fn(upstream):
            dx = tex.scaled_softmax_backward(upstream, y, scale, self.stream)
            return dx, None

        return y, grad_fn

    @tf.custom_gradient
    def scaled_upper_triang_masked_softmax(self, x: tf.Tensor, scale: float):
        """Scaled upper triangular masked softmax."""
        y = tex.scaled_upper_triang_masked_softmax_forward(x, scale,
                                                           self.stream)

        def grad_fn(upstream):
            dx = tex.scaled_upper_triang_masked_softmax_backward(
                upstream, y, scale, self.stream
            )
            return dx, None

        return y, grad_fn

    def forward_fused_softmax(
        self,
        inp: tf.Tensor,
        mask: tf.Tensor,
    ) -> tf.Tensor:
        """Fused masked softmax kernel"""
        sq, sk = inp.shape[2], inp.shape[3]
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == "causal":
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            inp = tf.reshape(inp, (-1, sq, sk))
            probs = self.scaled_upper_triang_masked_softmax(inp, scale)
            return tf.reshape(probs, inp.shape)
        # input is 4D tensor (b, np, sq, sk)
        if mask is not None:
            ndims = len(mask.shape)
            assert ndims <= 4, "mask ndims should be <= 4"
            if len(mask.shape) < 4:
                # Broadcasting the first dims of mask to match the input ndims.
                broadcast_shape = [1] * (4 - ndims) + mask.shape[:]
                mask = tf.reshape(mask, broadcast_shape)
            return self.scaled_masked_softmax(inp, mask, scale)
        return self.scaled_softmax(inp, scale)

    def forward_tf_softmax(self, inp: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Framework softmax"""
        if self.input_in_float16 and self.softmax_in_fp32:
            inp = tf.cast(inp, tf.float32)

        if self.scale is not None:
            inp = inp * self.scale
        mask_output = self.mask_func(inp, mask) if mask is not None else inp
        probs = tf.nn.softmax(mask_output, axis=-1)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = tf.cast(probs, tf.half)
            else:
                probs = tf.cast(probs, tf.bfloat16)

        return probs

    @staticmethod
    def get_batch_per_block(key_seq_len: int) -> int:
        """Softmax utility"""
        pow2 = 1 << (key_seq_len - 1).bit_length()
        warp_size = pow2 if pow2 < THREADS_PER_WARP else THREADS_PER_WARP
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = THREADS_PER_BLOCK / warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block
