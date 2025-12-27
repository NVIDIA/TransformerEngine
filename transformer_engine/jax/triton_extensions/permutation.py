# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX/TE custom ops for permutation in MOE using Triton kernels."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import triton

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive
from transformer_engine.common.triton.permutation import (
    _row_id_map_pass_1_kernel,
    _row_id_map_pass_2_kernel,
    _row_id_map_pass_3_kernel,
    _permute_kernel,
    _unpermute_kernel,
    _unpermute_bwd_with_merging_probs_kernel,
    _make_chunk_sort_map_kernel,
    _sort_chunks_by_map_kernel,
)
from .utils import triton_call_lowering


__all__ = [
    "make_row_id_map",
    "permute_with_mask_map",
    "permute_with_mask_map_and_pad",
    "unpermute_with_mask_map",
    "unpermute_with_mask_map_and_unpad",
    "unpermute_bwd_with_merging_probs",
    "unpermute_bwd_with_merging_probs_and_unpad",
    "make_chunk_sort_map",
    "sort_chunks_by_map",
]

DEFAULT_BLOCK_SIZE = 1024


def _get_min_block_size(kernel, default=128):
    if hasattr(kernel, "configs"):
        return min(config.kwargs.get("BLOCK_SIZE", default) for config in kernel.configs)
    return default


class RowIdMapPass1Primitive(BasePrimitive):
    """
    Pass 1 of row_id_map generation: block cumsum.

    For each expert, compute the cumsum of every block_size tokens.
    """

    name = "te_row_id_map_pass1_triton"
    multiple_results = True
    impl_static_args = (1, 2, 3)  # num_tokens, num_experts, block_size
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(routing_map_aval, *, num_tokens, num_experts, block_size):
        """Shape/dtype inference for pass 1."""
        del block_size  # Only affects grid, not output shape

        assert routing_map_aval.shape == (
            num_tokens,
            num_experts,
        ), f"routing_map shape mismatch: expected ({num_tokens}, {num_experts})"

        row_id_map_shape = (num_tokens, num_experts * 2 + 1)
        workspace_shape = (
            num_experts,
            triton.cdiv(num_tokens, DEFAULT_BLOCK_SIZE),
        )

        return (
            jax.core.ShapedArray(row_id_map_shape, jnp.int32),
            jax.core.ShapedArray(workspace_shape, jnp.int32),
        )

    @staticmethod
    def impl(routing_map, num_tokens, num_experts, block_size):
        """Forward to inner primitive."""
        assert RowIdMapPass1Primitive.inner_primitive is not None
        return RowIdMapPass1Primitive.inner_primitive.bind(
            routing_map,
            num_tokens=num_tokens,
            num_experts=num_experts,
            block_size=block_size,
        )

    @staticmethod
    def lowering(ctx, routing_map, *, num_tokens, num_experts, block_size):
        """MLIR lowering using triton_call_lowering."""
        # Compute strides
        routing_stride_token = num_experts
        routing_stride_expert = 1
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1

        grid = (num_experts, triton.cdiv(num_tokens, block_size))

        # All scalar arguments must be passed as constexprs
        return triton_call_lowering(
            ctx,
            _row_id_map_pass_1_kernel,
            routing_map,  # Only tensor arguments here
            grid=grid,
            constexprs={
                "num_tokens": num_tokens,
                "stride_routing_map_token": routing_stride_token,
                "stride_routing_map_expert": routing_stride_expert,
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(RowIdMapPass1Primitive)


class RowIdMapPass2Primitive(BasePrimitive):
    """
    Pass 2 of row_id_map generation: cumsum all and process the mask.
    """

    name = "te_row_id_map_pass2_triton"
    multiple_results = True
    impl_static_args = (2, 3, 4)  # num_tokens, num_experts, block_size
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(row_id_map_aval, workspace_aval, *, num_tokens, num_experts, block_size):
        """Shape/dtype inference for pass 2 (in-place operation)."""
        del row_id_map_aval, workspace_aval
        del block_size

        row_id_map_shape = (num_tokens, num_experts * 2 + 1)
        workspace_shape = (num_experts, triton.cdiv(num_tokens, DEFAULT_BLOCK_SIZE))

        return (
            jax.core.ShapedArray(row_id_map_shape, jnp.int32),
            jax.core.ShapedArray(workspace_shape, jnp.int32),
        )

    @staticmethod
    def impl(row_id_map, workspace, num_tokens, num_experts, block_size):
        """Forward to inner primitive."""
        assert RowIdMapPass2Primitive.inner_primitive is not None
        return RowIdMapPass2Primitive.inner_primitive.bind(
            row_id_map,
            workspace,
            num_tokens=num_tokens,
            num_experts=num_experts,
            block_size=block_size,
        )

    @staticmethod
    def lowering(ctx, row_id_map, workspace, *, num_tokens, num_experts, block_size):
        """MLIR lowering using triton_call_lowering."""
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1

        grid = (num_experts, triton.cdiv(num_tokens, block_size))
        workspace_load_width = triton.next_power_of_2(
            num_experts * triton.cdiv(num_tokens, block_size)
        )

        return triton_call_lowering(
            ctx,
            _row_id_map_pass_2_kernel,
            row_id_map,
            workspace,
            grid=grid,
            input_output_aliases={0: 0, 1: 1},
            constexprs={
                "num_tokens": num_tokens,
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "WORKSPACE_LOAD_WIDTH": workspace_load_width,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(RowIdMapPass2Primitive)


class RowIdMapPass3Primitive(BasePrimitive):
    """
    Pass 3 of row_id_map generation: make the row_id_map from sparse to dense structure.
    """

    name = "te_row_id_map_pass3_triton"
    multiple_results = False
    impl_static_args = (1, 2)  # num_tokens, num_experts
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(row_id_map_aval, *, num_tokens, num_experts):
        """Shape/dtype inference for pass 3 (in-place operation)."""
        del row_id_map_aval
        row_id_map_shape = (num_tokens, num_experts * 2 + 1)
        return jax.core.ShapedArray(row_id_map_shape, jnp.int32)

    @staticmethod
    def impl(row_id_map, num_tokens, num_experts):
        """Forward to inner primitive."""
        assert RowIdMapPass3Primitive.inner_primitive is not None
        return RowIdMapPass3Primitive.inner_primitive.bind(
            row_id_map,
            num_tokens=num_tokens,
            num_experts=num_experts,
        )

    @staticmethod
    def lowering(ctx, row_id_map, *, num_tokens, num_experts):
        """MLIR lowering using triton_call_lowering."""
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1

        grid = (num_tokens,)
        load_size = triton.next_power_of_2(num_experts)

        return triton_call_lowering(
            ctx,
            _row_id_map_pass_3_kernel,
            row_id_map,
            grid=grid,
            input_output_aliases={0: 0},
            constexprs={
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "num_experts": num_experts,
                "LOAD_SIZE": load_size,
            },
        )


register_primitive(RowIdMapPass3Primitive)


class PermuteWithMaskMapPrimitive(BasePrimitive):
    """
    Permute the input tensor based on the row_id_map, optionally with fused padding.
    """

    name = "te_permute_with_mask_map_triton"
    multiple_results = True
    # scale, permuted_scale are dummy inputs (not used when PERMUTE_SCALE=False)
    # pad_offsets can be shape (0,) when not doing padding, or (num_experts,) when padding
    impl_static_args = (
        6,
        7,
        8,
        9,
        10,
        11,
    )  # num_tokens, num_experts, num_out_tokens, hidden_size, with_probs, with_pad
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        inp_aval,
        row_id_map_aval,
        probs_aval,
        scale_aval,  # dummy, same shape as inp
        permuted_scale_aval,  # dummy, same shape as inp
        pad_offsets_aval,
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
        with_probs,
        with_pad,
    ):
        """Shape/dtype inference for permute."""
        del row_id_map_aval, scale_aval, permuted_scale_aval, pad_offsets_aval
        del num_tokens, num_experts, with_pad

        output_shape = (num_out_tokens, hidden_size)
        output_aval = jax.core.ShapedArray(output_shape, inp_aval.dtype)

        if with_probs:
            permuted_probs_aval = jax.core.ShapedArray((num_out_tokens,), probs_aval.dtype)
        else:
            permuted_probs_aval = jax.core.ShapedArray((0,), inp_aval.dtype)

        return output_aval, permuted_probs_aval

    @staticmethod
    def impl(
        inp,
        row_id_map,
        probs,
        scale,
        permuted_scale,
        pad_offsets,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
        with_probs,
        with_pad,
    ):
        """Forward to inner primitive."""
        assert PermuteWithMaskMapPrimitive.inner_primitive is not None
        return PermuteWithMaskMapPrimitive.inner_primitive.bind(
            inp,
            row_id_map,
            probs,
            scale,
            permuted_scale,
            pad_offsets,
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_out_tokens=num_out_tokens,
            hidden_size=hidden_size,
            with_probs=with_probs,
            with_pad=with_pad,
        )

    @staticmethod
    def lowering(
        ctx,
        inp,
        row_id_map,
        probs,
        scale,
        permuted_scale,
        pad_offsets,
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
        with_probs,
        with_pad,
    ):
        """MLIR lowering using triton_call_lowering."""
        del num_out_tokens
        inp_stride_token = hidden_size
        inp_stride_hidden = 1
        output_stride_token = hidden_size
        output_stride_hidden = 1
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1
        permuted_probs_stride_token = 1

        if with_probs:
            # Check if probs is 2D [num_tokens, num_experts] or 1D [num_tokens]
            probs_aval = ctx.avals_in[2]
            if len(probs_aval.shape) > 1:
                probs_stride_token = num_experts
                probs_stride_expert = 1
            else:
                probs_stride_token = 1
                probs_stride_expert = 1
        else:
            probs_stride_token = 0
            probs_stride_expert = 0

        # Grid function equivalent: (num_tokens, cdiv(hidden_size, BLOCK_SIZE))
        # Use minimum BLOCK_SIZE from autotune configs to ensure grid covers all elements
        block_size = _get_min_block_size(_permute_kernel)
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))

        return triton_call_lowering(
            ctx,
            _permute_kernel,
            inp,
            row_id_map,
            probs,
            scale,
            permuted_scale,
            pad_offsets,
            grid=grid,
            constexprs={
                "scale_hidden_dim": 0,
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "stride_input_token": inp_stride_token,
                "stride_input_hidden": inp_stride_hidden,
                "stride_output_token": output_stride_token,
                "stride_output_hidden": output_stride_hidden,
                "stride_probs_token": probs_stride_token,
                "stride_probs_expert": probs_stride_expert,
                "stride_scale_token": hidden_size,
                "stride_scale_hidden": 1,
                "stride_permuted_probs_token": permuted_probs_stride_token,
                "stride_permuted_scale_token": hidden_size,
                "stride_permuted_scale_hidden": 1,
                "num_experts": num_experts,
                "hidden_size": hidden_size,
                "PERMUTE_PROBS": with_probs,
                "PERMUTE_SCALE": False,
                "FUSION_PAD": with_pad,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(PermuteWithMaskMapPrimitive)


class UnpermuteWithMaskMapPrimitive(BasePrimitive):
    """
    Unpermute the input tensor based on the row_id_map.
    """

    name = "te_unpermute_with_mask_map_triton"
    multiple_results = True
    impl_static_args = (
        5,
        6,
        7,
        8,
        9,
    )  # num_tokens, num_experts, hidden_size, with_merging_probs, with_probs
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        inp_aval,
        row_id_map_aval,
        merging_probs_aval,
        permuted_probs_aval,
        pad_offsets_aval,  # dummy, not used when FUSION_UNPAD=False
        *,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """Shape/dtype inference for unpermute."""
        del row_id_map_aval, merging_probs_aval, with_merging_probs, pad_offsets_aval

        output_shape = (num_tokens, hidden_size)
        output_aval = jax.core.ShapedArray(output_shape, inp_aval.dtype)

        if with_probs:
            unpermuted_probs_shape = (num_tokens, num_experts)
            unpermuted_probs_aval = jax.core.ShapedArray(
                unpermuted_probs_shape, permuted_probs_aval.dtype
            )
        else:
            unpermuted_probs_aval = jax.core.ShapedArray((0,), inp_aval.dtype)

        return output_aval, unpermuted_probs_aval

    @staticmethod
    def impl(
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """Forward to inner primitive."""
        assert UnpermuteWithMaskMapPrimitive.inner_primitive is not None
        return UnpermuteWithMaskMapPrimitive.inner_primitive.bind(
            inp,
            row_id_map,
            merging_probs,
            permuted_probs,
            pad_offsets,
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            with_merging_probs=with_merging_probs,
            with_probs=with_probs,
        )

    @staticmethod
    def lowering(
        ctx,
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        *,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """MLIR lowering using triton_call_lowering."""
        # Compute strides
        inp_stride_token = hidden_size
        inp_stride_hidden = 1
        output_stride_token = hidden_size
        output_stride_hidden = 1
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1

        if with_merging_probs:
            merging_probs_stride_token = num_experts
            merging_probs_stride_expert = 1
        else:
            merging_probs_stride_token = 0
            merging_probs_stride_expert = 0

        permuted_probs_stride_token = 1
        unpermuted_probs_stride_token = num_experts
        unpermuted_probs_stride_expert = 1

        # Grid - use minimum BLOCK_SIZE from autotune configs
        block_size = _get_min_block_size(_unpermute_kernel)
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))

        # Pass all 5 inputs including pad_offsets (even though FUSION_UNPAD=False)
        return triton_call_lowering(
            ctx,
            _unpermute_kernel,
            inp,
            row_id_map,
            merging_probs,
            permuted_probs,
            pad_offsets,
            grid=grid,
            constexprs={
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "stride_input_token": inp_stride_token,
                "stride_input_hidden": inp_stride_hidden,
                "stride_output_token": output_stride_token,
                "stride_output_hidden": output_stride_hidden,
                "stride_merging_probs_token": merging_probs_stride_token,
                "stride_merging_probs_expert": merging_probs_stride_expert,
                "stride_permuted_probs_token": permuted_probs_stride_token,
                "stride_unpermuted_probs_token": unpermuted_probs_stride_token,
                "stride_unpermuted_probs_expert": unpermuted_probs_stride_expert,
                "num_experts": num_experts,
                "hidden_size": hidden_size,
                "PROBS_LOAD_WIDTH": triton.next_power_of_2(num_experts),
                "WITH_MERGING_PROBS": with_merging_probs,
                "PERMUTE_PROBS": with_probs,
                "FUSION_UNPAD": False,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(UnpermuteWithMaskMapPrimitive)


class UnpermuteWithMaskMapAndUnpadPrimitive(BasePrimitive):
    """
    Unpermute the input tensor based on the row_id_map with fused unpadding.
    """

    name = "te_unpermute_with_mask_map_and_unpad_triton"
    multiple_results = True
    impl_static_args = (
        5,
        6,
        7,
        8,
        9,
    )  # num_tokens, num_experts, hidden_size, with_merging_probs, with_probs
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        inp_aval,
        row_id_map_aval,
        merging_probs_aval,
        permuted_probs_aval,
        pad_offsets_aval,
        *,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """Shape/dtype inference for unpermute with unpadding."""
        del row_id_map_aval, merging_probs_aval, with_merging_probs, pad_offsets_aval

        output_shape = (num_tokens, hidden_size)
        output_aval = jax.core.ShapedArray(output_shape, inp_aval.dtype)

        if with_probs:
            unpermuted_probs_shape = (num_tokens, num_experts)
            unpermuted_probs_aval = jax.core.ShapedArray(
                unpermuted_probs_shape, permuted_probs_aval.dtype
            )
        else:
            unpermuted_probs_aval = jax.core.ShapedArray((0,), inp_aval.dtype)

        return output_aval, unpermuted_probs_aval

    @staticmethod
    def impl(
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """Forward to inner primitive."""
        assert UnpermuteWithMaskMapAndUnpadPrimitive.inner_primitive is not None
        return UnpermuteWithMaskMapAndUnpadPrimitive.inner_primitive.bind(
            inp,
            row_id_map,
            merging_probs,
            permuted_probs,
            pad_offsets,
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            with_merging_probs=with_merging_probs,
            with_probs=with_probs,
        )

    @staticmethod
    def lowering(
        ctx,
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        *,
        num_tokens,
        num_experts,
        hidden_size,
        with_merging_probs,
        with_probs,
    ):
        """MLIR lowering using triton_call_lowering."""
        # Compute strides
        inp_stride_token = hidden_size
        inp_stride_hidden = 1
        output_stride_token = hidden_size
        output_stride_hidden = 1
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1

        if with_merging_probs:
            merging_probs_stride_token = num_experts
            merging_probs_stride_expert = 1
        else:
            merging_probs_stride_token = 0
            merging_probs_stride_expert = 0

        permuted_probs_stride_token = 1
        unpermuted_probs_stride_token = num_experts
        unpermuted_probs_stride_expert = 1

        # Grid - use minimum BLOCK_SIZE from autotune configs
        block_size = _get_min_block_size(_unpermute_kernel)
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))

        return triton_call_lowering(
            ctx,
            _unpermute_kernel,
            inp,
            row_id_map,
            merging_probs,
            permuted_probs,
            pad_offsets,
            grid=grid,
            constexprs={
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "stride_input_token": inp_stride_token,
                "stride_input_hidden": inp_stride_hidden,
                "stride_output_token": output_stride_token,
                "stride_output_hidden": output_stride_hidden,
                "stride_merging_probs_token": merging_probs_stride_token,
                "stride_merging_probs_expert": merging_probs_stride_expert,
                "stride_permuted_probs_token": permuted_probs_stride_token,
                "stride_unpermuted_probs_token": unpermuted_probs_stride_token,
                "stride_unpermuted_probs_expert": unpermuted_probs_stride_expert,
                "num_experts": num_experts,
                "hidden_size": hidden_size,
                "PROBS_LOAD_WIDTH": triton.next_power_of_2(num_experts),
                "WITH_MERGING_PROBS": with_merging_probs,
                "PERMUTE_PROBS": with_probs,
                "FUSION_UNPAD": True,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(UnpermuteWithMaskMapAndUnpadPrimitive)


class UnpermuteBwdWithMergingProbsPrimitive(BasePrimitive):
    """
    Backward pass for unpermute with merging probabilities.

    This kernel computes gradients for both the input and merging_probs.
    """

    name = "te_unpermute_bwd_with_merging_probs_triton"
    multiple_results = True
    impl_static_args = (5, 6, 7, 8)  # num_tokens, num_experts, num_out_tokens, hidden_size
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        fwd_output_grad_aval,
        fwd_input_aval,
        merging_probs_aval,
        row_id_map_aval,
        pad_offsets_aval,  # dummy, not used when FUSION_UNPAD=False
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """Shape/dtype inference for unpermute backward with merging probs."""
        del fwd_input_aval, row_id_map_aval, pad_offsets_aval

        # fwd_input_grad has same shape as fwd_input
        fwd_input_grad_shape = (num_out_tokens, hidden_size)
        fwd_input_grad_aval = jax.core.ShapedArray(fwd_input_grad_shape, fwd_output_grad_aval.dtype)

        # merging_probs_grad has same shape as merging_probs
        merging_probs_grad_shape = (num_tokens, num_experts)
        merging_probs_grad_aval = jax.core.ShapedArray(
            merging_probs_grad_shape, merging_probs_aval.dtype
        )

        return fwd_input_grad_aval, merging_probs_grad_aval

    @staticmethod
    def impl(
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        pad_offsets,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """Forward to inner primitive."""
        assert UnpermuteBwdWithMergingProbsPrimitive.inner_primitive is not None
        return UnpermuteBwdWithMergingProbsPrimitive.inner_primitive.bind(
            fwd_output_grad,
            fwd_input,
            merging_probs,
            row_id_map,
            pad_offsets,
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_out_tokens=num_out_tokens,
            hidden_size=hidden_size,
        )

    @staticmethod
    def lowering(
        ctx,
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        pad_offsets,
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """MLIR lowering using triton_call_lowering."""
        del num_out_tokens

        # Compute strides
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1
        fwd_output_grad_stride_token = hidden_size
        fwd_output_grad_stride_hidden = 1
        fwd_input_grad_stride_token = hidden_size
        fwd_input_grad_stride_hidden = 1
        fwd_input_stride_token = hidden_size
        fwd_input_stride_hidden = 1
        merging_probs_stride_token = num_experts
        merging_probs_stride_expert = 1
        merging_probs_grad_stride_token = num_experts
        merging_probs_grad_stride_expert = 1

        # Grid - one program per token
        grid = (num_tokens,)

        # Get min block size from autotune configs for consistency
        block_size = _get_min_block_size(_unpermute_bwd_with_merging_probs_kernel)

        # Pass all 5 inputs including pad_offsets (even though FUSION_UNPAD=False)
        return triton_call_lowering(
            ctx,
            _unpermute_bwd_with_merging_probs_kernel,
            fwd_output_grad,
            fwd_input,
            merging_probs,
            row_id_map,
            pad_offsets,
            grid=grid,
            constexprs={
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "stride_fwd_output_grad_token": fwd_output_grad_stride_token,
                "stride_fwd_output_grad_hidden": fwd_output_grad_stride_hidden,
                "stride_fwd_input_grad_token": fwd_input_grad_stride_token,
                "stride_fwd_input_grad_hidden": fwd_input_grad_stride_hidden,
                "stride_fwd_input_token": fwd_input_stride_token,
                "stride_fwd_input_hidden": fwd_input_stride_hidden,
                "stride_merging_probs_token": merging_probs_stride_token,
                "stride_merging_probs_expert": merging_probs_stride_expert,
                "stride_merging_probs_grad_token": merging_probs_grad_stride_token,
                "stride_merging_probs_grad_expert": merging_probs_grad_stride_expert,
                "num_experts": num_experts,
                "hidden_size": hidden_size,
                "PROBS_LOAD_WIDTH": triton.next_power_of_2(num_experts),
                "FUSION_UNPAD": False,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(UnpermuteBwdWithMergingProbsPrimitive)


class UnpermuteBwdWithMergingProbsAndUnpadPrimitive(BasePrimitive):
    """
    Backward pass for unpermute with merging probabilities and fused unpadding.

    This kernel computes gradients for both the input and merging_probs,
    while handling padded outputs.
    """

    name = "te_unpermute_bwd_with_merging_probs_and_unpad_triton"
    multiple_results = True
    impl_static_args = (5, 6, 7, 8)  # num_tokens, num_experts, num_out_tokens, hidden_size
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        fwd_output_grad_aval,
        fwd_input_aval,
        merging_probs_aval,
        row_id_map_aval,
        pad_offsets_aval,
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """Shape/dtype inference for unpermute backward with merging probs and unpadding."""
        del fwd_input_aval, row_id_map_aval, pad_offsets_aval

        # fwd_input_grad has same shape as fwd_input
        fwd_input_grad_shape = (num_out_tokens, hidden_size)
        fwd_input_grad_aval = jax.core.ShapedArray(fwd_input_grad_shape, fwd_output_grad_aval.dtype)

        # merging_probs_grad has same shape as merging_probs
        merging_probs_grad_shape = (num_tokens, num_experts)
        merging_probs_grad_aval = jax.core.ShapedArray(
            merging_probs_grad_shape, merging_probs_aval.dtype
        )

        return fwd_input_grad_aval, merging_probs_grad_aval

    @staticmethod
    def impl(
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        pad_offsets,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """Forward to inner primitive."""
        assert UnpermuteBwdWithMergingProbsAndUnpadPrimitive.inner_primitive is not None
        return UnpermuteBwdWithMergingProbsAndUnpadPrimitive.inner_primitive.bind(
            fwd_output_grad,
            fwd_input,
            merging_probs,
            row_id_map,
            pad_offsets,
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_out_tokens=num_out_tokens,
            hidden_size=hidden_size,
        )

    @staticmethod
    def lowering(
        ctx,
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        pad_offsets,
        *,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    ):
        """MLIR lowering using triton_call_lowering."""
        del num_out_tokens

        # Compute strides
        row_id_stride_token = num_experts * 2 + 1
        row_id_stride_expert = 1
        fwd_output_grad_stride_token = hidden_size
        fwd_output_grad_stride_hidden = 1
        fwd_input_grad_stride_token = hidden_size
        fwd_input_grad_stride_hidden = 1
        fwd_input_stride_token = hidden_size
        fwd_input_stride_hidden = 1
        merging_probs_stride_token = num_experts
        merging_probs_stride_expert = 1
        merging_probs_grad_stride_token = num_experts
        merging_probs_grad_stride_expert = 1

        # Grid - one program per token
        grid = (num_tokens,)

        # Get min block size from autotune configs for consistency
        block_size = _get_min_block_size(_unpermute_bwd_with_merging_probs_kernel)

        return triton_call_lowering(
            ctx,
            _unpermute_bwd_with_merging_probs_kernel,
            fwd_output_grad,
            fwd_input,
            merging_probs,
            row_id_map,
            pad_offsets,
            grid=grid,
            constexprs={
                "stride_row_id_map_token": row_id_stride_token,
                "stride_row_id_map_expert": row_id_stride_expert,
                "stride_fwd_output_grad_token": fwd_output_grad_stride_token,
                "stride_fwd_output_grad_hidden": fwd_output_grad_stride_hidden,
                "stride_fwd_input_grad_token": fwd_input_grad_stride_token,
                "stride_fwd_input_grad_hidden": fwd_input_grad_stride_hidden,
                "stride_fwd_input_token": fwd_input_stride_token,
                "stride_fwd_input_hidden": fwd_input_stride_hidden,
                "stride_merging_probs_token": merging_probs_stride_token,
                "stride_merging_probs_expert": merging_probs_stride_expert,
                "stride_merging_probs_grad_token": merging_probs_grad_stride_token,
                "stride_merging_probs_grad_expert": merging_probs_grad_stride_expert,
                "num_experts": num_experts,
                "hidden_size": hidden_size,
                "PROBS_LOAD_WIDTH": triton.next_power_of_2(num_experts),
                "FUSION_UNPAD": True,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(UnpermuteBwdWithMergingProbsAndUnpadPrimitive)


def unpermute_bwd_with_merging_probs(
    fwd_output_grad: jnp.ndarray,
    row_id_map: jnp.ndarray,
    fwd_input: jnp.ndarray,
    merging_probs: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Backward pass for unpermute with merging probabilities.

    This computes gradients for both the input tensor and merging_probs.

    Parameters
    ----------
    fwd_output_grad : jnp.ndarray
        Gradient of the forward output of shape `[num_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    fwd_input : jnp.ndarray
        The input tensor from the forward pass of shape `[num_out_tokens, hidden_size]`.
    merging_probs : jnp.ndarray
        The merging probabilities of shape `[num_tokens, num_experts]`.
    num_tokens : int
        Number of tokens in the unpermuted tensor.
    num_experts : int
        Number of experts.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size.

    Returns
    -------
    fwd_input_grad : jnp.ndarray
        Gradient w.r.t. the input tensor of shape `[num_out_tokens, hidden_size]`.
    merging_probs_grad : jnp.ndarray
        Gradient w.r.t. merging_probs of shape `[num_tokens, num_experts]`.
    """
    # Create dummy pad_offsets (not used when FUSION_UNPAD=False, but required by kernel signature)
    dummy_pad_offsets = jnp.zeros((0,), dtype=jnp.int32)
    # Pass arguments in kernel order: fwd_output_grad, fwd_input, merging_probs, row_id_map, pad_offsets
    return UnpermuteBwdWithMergingProbsPrimitive.outer_primitive.bind(
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        dummy_pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        hidden_size=hidden_size,
    )


def unpermute_bwd_with_merging_probs_and_unpad(
    fwd_output_grad: jnp.ndarray,
    row_id_map: jnp.ndarray,
    fwd_input: jnp.ndarray,
    merging_probs: jnp.ndarray,
    pad_offsets: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Backward pass for unpermute with merging probabilities and fused unpadding.

    This computes gradients for both the input tensor and merging_probs,
    while handling padded outputs.

    Parameters
    ----------
    fwd_output_grad : jnp.ndarray
        Gradient of the forward output of shape `[num_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    fwd_input : jnp.ndarray
        The input tensor from the forward pass of shape `[num_out_tokens, hidden_size]`.
    merging_probs : jnp.ndarray
        The merging probabilities of shape `[num_tokens, num_experts]`.
    pad_offsets : jnp.ndarray
        Per-expert cumulative padding offsets of shape `[num_experts]`.
    num_tokens : int
        Number of tokens in the unpermuted tensor.
    num_experts : int
        Number of experts.
    num_out_tokens : int
        Number of tokens in the permuted tensor (including padding).
    hidden_size : int
        Hidden size.

    Returns
    -------
    fwd_input_grad : jnp.ndarray
        Gradient w.r.t. the input tensor of shape `[num_out_tokens, hidden_size]`.
    merging_probs_grad : jnp.ndarray
        Gradient w.r.t. merging_probs of shape `[num_tokens, num_experts]`.
    """
    return UnpermuteBwdWithMergingProbsAndUnpadPrimitive.outer_primitive.bind(
        fwd_output_grad,
        fwd_input,
        merging_probs,
        row_id_map,
        pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        hidden_size=hidden_size,
    )


class MakeChunkSortMapPrimitive(BasePrimitive):
    """
    Make a row_id_map for chunk sort.
    """

    name = "te_make_chunk_sort_map_triton"
    multiple_results = False
    impl_static_args = (2, 3)  # num_tokens, num_splits
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(split_sizes_aval, sorted_indices_aval, *, num_tokens, num_splits):
        """Shape/dtype inference."""
        del sorted_indices_aval
        assert split_sizes_aval.shape == (num_splits,)
        return jax.core.ShapedArray((num_tokens,), jnp.int32)

    @staticmethod
    def impl(split_sizes, sorted_indices, num_tokens, num_splits):
        """Forward to inner primitive."""
        assert MakeChunkSortMapPrimitive.inner_primitive is not None
        return MakeChunkSortMapPrimitive.inner_primitive.bind(
            split_sizes,
            sorted_indices,
            num_tokens=num_tokens,
            num_splits=num_splits,
        )

    @staticmethod
    def lowering(ctx, split_sizes, sorted_indices, *, num_tokens, num_splits):
        """MLIR lowering using triton_call_lowering."""
        grid = (num_tokens,)

        return triton_call_lowering(
            ctx,
            _make_chunk_sort_map_kernel,
            split_sizes,
            sorted_indices,
            grid=grid,
            constexprs={
                "num_splits": num_splits,
                "IDX_LOAD_WIDTH": triton.next_power_of_2(num_splits),
            },
        )


register_primitive(MakeChunkSortMapPrimitive)


class SortChunksByMapPrimitive(BasePrimitive):
    """
    Sort chunks with row_id_map.
    """

    name = "te_sort_chunks_by_map_triton"
    multiple_results = True
    impl_static_args = (3, 4, 5, 6)  # num_tokens, hidden_size, is_forward, with_probs
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        inp_aval, row_id_map_aval, probs_aval, *, num_tokens, hidden_size, is_forward, with_probs
    ):
        """Shape/dtype inference."""
        del row_id_map_aval, is_forward

        output_aval = jax.core.ShapedArray((num_tokens, hidden_size), inp_aval.dtype)

        if with_probs:
            permuted_probs_aval = jax.core.ShapedArray((num_tokens,), probs_aval.dtype)
        else:
            permuted_probs_aval = jax.core.ShapedArray((0,), inp_aval.dtype)

        return output_aval, permuted_probs_aval

    @staticmethod
    def impl(inp, row_id_map, probs, num_tokens, hidden_size, is_forward, with_probs):
        """Forward to inner primitive."""
        assert SortChunksByMapPrimitive.inner_primitive is not None
        return SortChunksByMapPrimitive.inner_primitive.bind(
            inp,
            row_id_map,
            probs,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            is_forward=is_forward,
            with_probs=with_probs,
        )

    @staticmethod
    def lowering(ctx, inp, row_id_map, probs, *, num_tokens, hidden_size, is_forward, with_probs):
        """MLIR lowering using triton_call_lowering."""
        # Compute strides
        inp_stride_token = hidden_size
        inp_stride_hidden = 1
        output_stride_token = hidden_size
        output_stride_hidden = 1
        probs_stride_token = 1
        permuted_probs_stride_token = 1

        # Grid - use minimum BLOCK_SIZE from autotune configs
        block_size = _get_min_block_size(_sort_chunks_by_map_kernel)
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))

        return triton_call_lowering(
            ctx,
            _sort_chunks_by_map_kernel,
            inp,
            row_id_map,
            probs,
            grid=grid,
            constexprs={
                "stride_input_token": inp_stride_token,
                "stride_input_hidden": inp_stride_hidden,
                "stride_output_token": output_stride_token,
                "stride_output_hidden": output_stride_hidden,
                "stride_probs_token": probs_stride_token,
                "stride_permuted_probs_token": permuted_probs_stride_token,
                "hidden_size": hidden_size,
                "PERMUTE_PROBS": with_probs,
                "FORWARD": is_forward,
                "BLOCK_SIZE": block_size,
            },
        )


register_primitive(SortChunksByMapPrimitive)


def make_row_id_map(
    routing_map: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
) -> jnp.ndarray:
    """
    Prepare the row_id_map for the permutation.

    This function chains 3 Triton kernel passes together.

    Parameters
    ----------
    routing_map : jnp.ndarray
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map : jnp.ndarray
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    block_size = DEFAULT_BLOCK_SIZE

    # Pass 1: Block cumsum
    row_id_map_pass1, workspace_tensor = RowIdMapPass1Primitive.outer_primitive.bind(
        routing_map,
        num_tokens=num_tokens,
        num_experts=num_experts,
        block_size=block_size,
    )

    # Pass 2: Cumsum all and process the mask
    row_id_map_pass2, _ = RowIdMapPass2Primitive.outer_primitive.bind(
        row_id_map_pass1,
        workspace_tensor,
        num_tokens=num_tokens,
        num_experts=num_experts,
        block_size=block_size,
    )

    # Initialize columns [num_experts:] to -1 since Pass 1/2 only wrote to [0:num_experts]
    # Reference implementation expects -1 for invalid entries
    row_id_map = row_id_map_pass2.at[:, num_experts:].set(-1)

    # Pass 3: Make the row_id_map from sparse to dense structure
    row_id_map = RowIdMapPass3Primitive.outer_primitive.bind(
        row_id_map,
        num_tokens=num_tokens,
        num_experts=num_experts,
    )

    return row_id_map


def permute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Permute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size of the input tensor.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape `[num_out_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities if probs was provided, None otherwise.
    """
    with_probs = probs is not None

    # Handle None probs by creating dummy tensor
    if not with_probs:
        probs = jnp.zeros((0,), dtype=inp.dtype)

    # Create dummy scale tensors (not used when PERMUTE_SCALE=False, but required by kernel signature)
    dummy_scale = inp
    dummy_permuted_scale = inp
    # Create dummy pad_offsets (not used when FUSION_PAD=False, but required by kernel signature)
    dummy_pad_offsets = jnp.zeros((0,), dtype=jnp.int32)

    output, permuted_probs = PermuteWithMaskMapPrimitive.outer_primitive.bind(
        inp,
        row_id_map,
        probs,
        dummy_scale,
        dummy_permuted_scale,
        dummy_pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        hidden_size=hidden_size,
        with_probs=with_probs,
        with_pad=False,
    )

    if not with_probs:
        permuted_probs = None

    return output, permuted_probs


def permute_with_mask_map_and_pad(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    pad_offsets: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Permute the input tensor based on the row_id_map with fused padding.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    pad_offsets : jnp.ndarray
        Per-expert cumulative padding offsets of shape `[num_experts]`.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor (including padding).
    hidden_size : int
        Hidden size of the input tensor.

    Returns
    -------
    output : jnp.ndarray
        Permuted and padded output tensor of shape `[num_out_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities if probs was provided, None otherwise.
    """
    with_probs = probs is not None

    # Handle None probs by creating dummy tensor
    if not with_probs:
        probs = jnp.zeros((0,), dtype=inp.dtype)

    # Create dummy scale tensors (not used when PERMUTE_SCALE=False, but required by kernel signature)
    dummy_scale = inp
    dummy_permuted_scale = inp

    output, permuted_probs = PermuteWithMaskMapPrimitive.outer_primitive.bind(
        inp,
        row_id_map,
        probs,
        dummy_scale,
        dummy_permuted_scale,
        pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        hidden_size=hidden_size,
        with_probs=with_probs,
        with_pad=True,
    )

    if not with_probs:
        permuted_probs = None

    return output, permuted_probs


def unpermute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    permuted_probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Unpermute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs : Optional[jnp.ndarray]
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs : Optional[jnp.ndarray]
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    num_tokens : int
        Number of tokens in the permuted tensor.
    num_experts : int
        Number of experts in the permuted tensor.
    hidden_size : int
        Hidden size of the permuted tensor.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape `[num_tokens, hidden_size]`.
    unpermuted_probs : Optional[jnp.ndarray]
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    with_merging_probs = merging_probs is not None
    with_probs = permuted_probs is not None

    # Handle None inputs by creating dummy tensors
    if not with_merging_probs:
        merging_probs = jnp.zeros((0,), dtype=inp.dtype)
    if not with_probs:
        permuted_probs = jnp.zeros((0,), dtype=inp.dtype)
    # Create dummy pad_offsets (not used when FUSION_UNPAD=False, but required by kernel signature)
    dummy_pad_offsets = jnp.zeros((0,), dtype=jnp.int32)

    output, unpermuted_probs = UnpermuteWithMaskMapPrimitive.outer_primitive.bind(
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        dummy_pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        with_merging_probs=with_merging_probs,
        with_probs=with_probs,
    )

    if not with_probs:
        unpermuted_probs = None

    return output, unpermuted_probs


def unpermute_with_mask_map_and_unpad(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    permuted_probs: Optional[jnp.ndarray],
    pad_offsets: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Unpermute the input tensor based on the row_id_map with fused unpadding.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_out_tokens, hidden_size]` (including padding).
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs : Optional[jnp.ndarray]
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs : Optional[jnp.ndarray]
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    pad_offsets : jnp.ndarray
        Per-expert cumulative padding offsets of shape `[num_experts]`.
    num_tokens : int
        Number of tokens in the unpermuted tensor.
    num_experts : int
        Number of experts.
    hidden_size : int
        Hidden size of the tensor.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape `[num_tokens, hidden_size]`.
    unpermuted_probs : Optional[jnp.ndarray]
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    with_merging_probs = merging_probs is not None
    with_probs = permuted_probs is not None

    # Handle None inputs by creating dummy tensors
    if not with_merging_probs:
        merging_probs = jnp.zeros((0,), dtype=inp.dtype)
    if not with_probs:
        permuted_probs = jnp.zeros((0,), dtype=inp.dtype)

    output, unpermuted_probs = UnpermuteWithMaskMapAndUnpadPrimitive.outer_primitive.bind(
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        with_merging_probs=with_merging_probs,
        with_probs=with_probs,
    )

    if not with_probs:
        unpermuted_probs = None

    return output, unpermuted_probs


def make_chunk_sort_map(
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    num_splits: int,
) -> jnp.ndarray:
    """
    Make a row_id_map for chunk sort.

    Parameters
    ----------
    split_sizes : jnp.ndarray
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices : jnp.ndarray
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens : int
        Number of tokens in the input tensor.
    num_splits : int
        Number of splits of split_sizes and sorted_indices.

    Returns
    -------
    row_id_map : jnp.ndarray
        Row ID map for chunk sorting of shape `[num_tokens,]`.
    """
    return MakeChunkSortMapPrimitive.outer_primitive.bind(
        split_sizes,
        sorted_indices,
        num_tokens=num_tokens,
        num_splits=num_splits,
    )


def sort_chunks_by_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Sort chunks with row_id_map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    hidden_size : int
        Hidden size of the input tensor.
    is_forward : bool
        Whether the sort is for forward or backward.

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape `[num_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Sorted probabilities if probs was provided, None otherwise.
    """
    with_probs = probs is not None

    # Handle None probs by creating dummy tensor
    if not with_probs:
        probs = jnp.zeros((0,), dtype=inp.dtype)

    output, permuted_probs = SortChunksByMapPrimitive.outer_primitive.bind(
        inp,
        row_id_map,
        probs,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        is_forward=is_forward,
        with_probs=with_probs,
    )

    if not with_probs:
        permuted_probs = None

    return output, permuted_probs
