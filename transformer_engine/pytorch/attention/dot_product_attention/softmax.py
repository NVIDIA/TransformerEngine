# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled masked softmax functions"""
import os
from typing import Callable, Optional
import torch
from torch import nn
import transformer_engine_torch as tex
from transformer_engine.pytorch.export import is_in_onnx_export_mode


THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128


_default_causal_mask = {}


def _scale_to_tensor(scale: float) -> torch.Tensor:
    """Wrap a Python float in a 0-D tensor expected by the tex kernels."""
    return torch.tensor([scale])[0]


# ----------------------------- ScaledSoftmax -------------------------------


@torch.library.custom_op("te_softmax::scaled_softmax_fwd", mutates_args=())
def scaled_softmax_forward(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward pass for ScaledSoftmax."""
    return tex.scaled_softmax_forward(inputs, _scale_to_tensor(scale))


@scaled_softmax_forward.register_fake
def _scaled_softmax_forward_fake(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    del scale
    return torch.empty_like(inputs)


@torch.library.custom_op("te_softmax::scaled_softmax_bwd", mutates_args=())
def scaled_softmax_backward(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    """Backward pass for ScaledSoftmax."""
    return tex.scaled_softmax_backward(output_grads, softmax_results, _scale_to_tensor(scale))


@scaled_softmax_backward.register_fake
def _scaled_softmax_backward_fake(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    del softmax_results, scale
    return torch.empty_like(output_grads)


def _scaled_softmax_setup_context(ctx, inputs, output):
    _inp, scale = inputs
    ctx.scale = scale
    ctx.save_for_backward(output)


def _scaled_softmax_backward_wrapper(ctx, grad_output):
    (softmax_results,) = ctx.saved_tensors
    grad_inputs = torch.ops.te_softmax.scaled_softmax_bwd(grad_output, softmax_results, ctx.scale)
    return grad_inputs, None


scaled_softmax_forward.register_autograd(
    _scaled_softmax_backward_wrapper,
    setup_context=_scaled_softmax_setup_context,
)


# --------------------------- ScaledMaskedSoftmax ---------------------------


@torch.library.custom_op("te_softmax::scaled_masked_softmax_fwd", mutates_args=())
def scaled_masked_softmax_forward(
    inputs: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    """Forward pass for ScaledMaskedSoftmax."""
    return tex.scaled_masked_softmax_forward(inputs, mask, _scale_to_tensor(scale))


@scaled_masked_softmax_forward.register_fake
def _scaled_masked_softmax_forward_fake(
    inputs: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    del mask, scale
    return torch.empty_like(inputs)


@torch.library.custom_op("te_softmax::scaled_masked_softmax_bwd", mutates_args=())
def scaled_masked_softmax_backward(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    """Backward pass for ScaledMaskedSoftmax."""
    return tex.scaled_masked_softmax_backward(
        output_grads, softmax_results, _scale_to_tensor(scale)
    )


@scaled_masked_softmax_backward.register_fake
def _scaled_masked_softmax_backward_fake(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    del softmax_results, scale
    return torch.empty_like(output_grads)


def _scaled_masked_softmax_setup_context(ctx, inputs, output):
    _inp, _mask, scale = inputs
    ctx.scale = scale
    ctx.save_for_backward(output)


def _scaled_masked_softmax_backward_wrapper(ctx, grad_output):
    (softmax_results,) = ctx.saved_tensors
    grad_inputs = torch.ops.te_softmax.scaled_masked_softmax_bwd(
        grad_output, softmax_results, ctx.scale
    )
    return grad_inputs, None, None


scaled_masked_softmax_forward.register_autograd(
    _scaled_masked_softmax_backward_wrapper,
    setup_context=_scaled_masked_softmax_setup_context,
)


# ---------------------- ScaledUpperTriangMaskedSoftmax ----------------------


@torch.library.custom_op("te_softmax::scaled_upper_triang_masked_softmax_fwd", mutates_args=())
def scaled_upper_triang_masked_softmax_forward(
    inputs: torch.Tensor, scale: float
) -> torch.Tensor:
    """Forward pass for ScaledUpperTriangMaskedSoftmax."""
    return tex.scaled_upper_triang_masked_softmax_forward(inputs, _scale_to_tensor(scale))


@scaled_upper_triang_masked_softmax_forward.register_fake
def _scaled_upper_triang_masked_softmax_forward_fake(
    inputs: torch.Tensor, scale: float
) -> torch.Tensor:
    del scale
    return torch.empty_like(inputs)


@torch.library.custom_op("te_softmax::scaled_upper_triang_masked_softmax_bwd", mutates_args=())
def scaled_upper_triang_masked_softmax_backward(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    """Backward pass for ScaledUpperTriangMaskedSoftmax."""
    return tex.scaled_upper_triang_masked_softmax_backward(
        output_grads, softmax_results, _scale_to_tensor(scale)
    )


@scaled_upper_triang_masked_softmax_backward.register_fake
def _scaled_upper_triang_masked_softmax_backward_fake(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    del softmax_results, scale
    return torch.empty_like(output_grads)


def _scaled_upper_triang_masked_softmax_setup_context(ctx, inputs, output):
    _inp, scale = inputs
    ctx.scale = scale
    ctx.save_for_backward(output)


def _scaled_upper_triang_masked_softmax_backward_wrapper(ctx, grad_output):
    (softmax_results,) = ctx.saved_tensors
    grad_inputs = torch.ops.te_softmax.scaled_upper_triang_masked_softmax_bwd(
        grad_output, softmax_results, ctx.scale
    )
    return grad_inputs, None


scaled_upper_triang_masked_softmax_forward.register_autograd(
    _scaled_upper_triang_masked_softmax_backward_wrapper,
    setup_context=_scaled_upper_triang_masked_softmax_setup_context,
)


# -------------------- ScaledAlignedCausalMaskedSoftmax ---------------------


@torch.library.custom_op("te_softmax::scaled_aligned_causal_masked_softmax_fwd", mutates_args=())
def scaled_aligned_causal_masked_softmax_forward(
    inputs: torch.Tensor, scale: float
) -> torch.Tensor:
    """Forward pass for ScaledAlignedCausalMaskedSoftmax."""
    return tex.scaled_aligned_causal_masked_softmax_forward(inputs, _scale_to_tensor(scale))


@scaled_aligned_causal_masked_softmax_forward.register_fake
def _scaled_aligned_causal_masked_softmax_forward_fake(
    inputs: torch.Tensor, scale: float
) -> torch.Tensor:
    del scale
    return torch.empty_like(inputs)


@torch.library.custom_op("te_softmax::scaled_aligned_causal_masked_softmax_bwd", mutates_args=())
def scaled_aligned_causal_masked_softmax_backward(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    """Backward pass for ScaledAlignedCausalMaskedSoftmax."""
    return tex.scaled_aligned_causal_masked_softmax_backward(
        output_grads, softmax_results, _scale_to_tensor(scale)
    )


@scaled_aligned_causal_masked_softmax_backward.register_fake
def _scaled_aligned_causal_masked_softmax_backward_fake(
    output_grads: torch.Tensor, softmax_results: torch.Tensor, scale: float
) -> torch.Tensor:
    del softmax_results, scale
    return torch.empty_like(output_grads)


def _scaled_aligned_causal_masked_softmax_setup_context(ctx, inputs, output):
    _inp, scale = inputs
    ctx.scale = scale
    ctx.save_for_backward(output)


def _scaled_aligned_causal_masked_softmax_backward_wrapper(ctx, grad_output):
    (softmax_results,) = ctx.saved_tensors
    grad_inputs = torch.ops.te_softmax.scaled_aligned_causal_masked_softmax_bwd(
        grad_output, softmax_results, ctx.scale
    )
    return grad_inputs, None


scaled_aligned_causal_masked_softmax_forward.register_autograd(
    _scaled_aligned_causal_masked_softmax_backward_wrapper,
    setup_context=_scaled_aligned_causal_masked_softmax_setup_context,
)


_default_causal_mask = {}


def _get_default_causal_mask(mask_type: str, sq: int, sk: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input"""

    def _get_mask():
        diagonal_offset = sk - sq + 1 if "bottom_right" in mask_type else 1
        return torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device="cuda"), diagonal=diagonal_offset
        )

    if is_in_onnx_export_mode():
        return _get_mask()
    matrix_identifiers = (mask_type, sq, sk)
    if matrix_identifiers not in _default_causal_mask:
        _default_causal_mask[matrix_identifiers] = _get_mask()
    return _default_causal_mask[matrix_identifiers]


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
        self.scaled_masked_softmax_fusion_type = bool(
            int(os.getenv("NVTE_MASKED_SOFTMAX_FUSION", "1"))
        )
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
        if is_in_onnx_export_mode():
            return self.forward_torch_softmax(inp, mask, scale)

        # We do not want to connect this if with previous if,
        # because we want to avoid calling is_kernel_available() in ONNX mode.
        if self.is_kernel_available(mask, *inp.size()):
            return self.forward_fused_softmax(inp, mask, scale)
        return self.forward_torch_softmax(inp, mask, scale)

    def is_kernel_available(self, mask: torch.Tensor, b: int, np: int, sq: int, sk: int) -> bool:
        """Check FusedScaleMaskSoftmax kernel availability based on size"""
        attn_batches = b * np

        if not self.scaled_masked_softmax_fusion_type:
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
        scale = 1.0 if scale is None else float(scale)

        # Disable for now until unalignment bug is fixed.
        # if self.attn_mask_type in ["causal", "causal_bottom_right"]:
        #    return torch.ops.te_softmax.scaled_aligned_causal_masked_softmax_fwd(inp, scale)

        # input is 4D tensor (1, 1, sq, sk) or (b, 1, sq, sk)
        if mask is not None and self.attn_mask_type != "no_mask":
            return torch.ops.te_softmax.scaled_masked_softmax_fwd(inp, mask, scale)
        return torch.ops.te_softmax.scaled_softmax_fwd(inp, scale)

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
        probs = torch.nn.functional.softmax(mask_output, dim=-1)

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
