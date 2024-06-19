# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for softmax"""
from abc import abstractmethod
from functools import partial, reduce
import operator
import warnings

import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax

from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import get_padded_spec, check_valid_batch_dims, jax_dtype_to_te_dtype
from ..softmax import SoftmaxType


__all__ = [
    "scaled_softmax_fwd",
    "scaled_softmax_bwd",
    "scaled_masked_softmax_fwd",
    "scaled_masked_softmax_bwd",
    "scaled_upper_triang_masked_softmax_fwd",
    "scaled_upper_triang_masked_softmax_bwd",
    "is_softmax_kernel_available",
]


def is_softmax_kernel_available(
    softmax_type: SoftmaxType,
    batch: int,
    heads: int,
    q_seqlen: int,
    k_seqlen: int,
    dtype: jnp.dtype,
):
    """check softmax available"""
    if softmax_type is SoftmaxType.SCALED:
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )
    if softmax_type is SoftmaxType.SCALED_MASKED:
        return ScaledMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )
    if softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )

    raise NotImplementedError


class SoftmaxPrimitive(BasePrimitive):
    """
    Softmax Primitive
    """

    max_k_seqlen_supported = 16384
    name = "te_softmax_internal_placeholder"

    @staticmethod
    @abstractmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        raise NotImplementedError

    @staticmethod
    def get_batch_per_block(k_seqlen: int) -> int:
        """Get batch per CTA in Softmax kernels"""
        threads_per_warp = 32
        threads_per_block = 128  # Depends on the kernel implmentation

        pow2 = 1 << (k_seqlen - 1).bit_length()
        warp_size = pow2 if pow2 < threads_per_warp else threads_per_warp
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = threads_per_block // warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block

    @staticmethod
    def forward_abstract(logits_aval, scale_factor):
        """
        softmax_forward abstract
        """
        del scale_factor
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = logits_aval.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        out_aval = core.raise_to_shaped(logits_aval)
        return out_aval

    @staticmethod
    def forward_lowering(name, ctx, logits, *, scale_factor):
        """
        softmax_forward lowering rules
        """
        (i_aval,) = ctx.avals_in
        i_type = ir.RankedTensorType(logits.type)
        i_shape = i_type.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        pad_batch = batch
        heads = i_shape[-3]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [logits]
        operand_shapes = [i_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch,
            pad_batch,
            heads,
            q_seqlen,
            k_seqlen,
            jax_dtype_to_te_dtype(i_aval.dtype),
            scale_factor,
        )

        out = custom_caller(name, args, opaque, False)

        return [out]

    @staticmethod
    def forward_impl(primitive, logits, scale_factor):
        """
        softmax_forward implementation
        """
        assert primitive is not None
        output = primitive.bind(logits, scale_factor=scale_factor)
        return output

    @staticmethod
    def forward_batcher(primitive, batched_args, batch_dims, *, scale_factor):
        """
        softmax_forward batcher
        """
        assert primitive is not None
        (logits,) = batched_args
        (logits_bdim,) = batch_dims

        out_bdims = logits_bdim
        return primitive.bind(logits, scale_factor=scale_factor), out_bdims

    @classmethod
    def forward_infer_sharding_from_operands(cls, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_forward infer_sharding_from_operands
        """
        del scale_factor, result_infos  # Unused.
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the hidden dimension is not supported in {cls.name}! "
                "Forcing XLA to not shard the hidden dim, which might introduce extra "
                "collective ops and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        return out_sharding

    @classmethod
    def forward_partition(cls, impl, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_forward partitioning
        """
        del result_infos
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the hidden dimension is not supported in {cls.name}! "
                "Forcing XLA to not shard the hidden dim, which might introduce extra "
                "collective ops and hurt performance."
            )
        out_shardings = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        arg_shardings = (out_shardings,)
        impl = partial(impl, scale_factor=scale_factor)
        return mesh, impl, out_shardings, arg_shardings

    @staticmethod
    def backward_abstract(
        dz_aval, softmax_out_aval, scale_factor=None
    ):  # pylint: disable=unused-argument
        """
        softmax_backward abstract
        """
        dz_dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        softmax_out_dtype = dtypes.canonicalize_dtype(softmax_out_aval.dtype)
        assert dz_dtype == softmax_out_dtype
        assert dz_dtype in [jnp.float16, jnp.bfloat16]
        assert softmax_out_dtype in [jnp.float16, jnp.bfloat16]

        assert dz_aval.shape == softmax_out_aval.shape

        dx_aval = core.raise_to_shaped(dz_aval)
        return dx_aval

    @staticmethod
    def backward_lowering(name, ctx, dz, softmax_out, *, scale_factor):
        """
        softmax_backward lowering rules
        """
        dz_aval, _ = ctx.avals_in

        dz_type = ir.RankedTensorType(dz.type)
        dz_shape = dz_type.shape

        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, dz_shape[:-3])
        pad_batch = batch  # unused
        heads = dz_shape[-3]
        q_seqlen = dz_shape[-2]
        k_seqlen = dz_shape[-1]

        softmax_out_type = ir.RankedTensorType(softmax_out.type)
        softmax_out_shape = softmax_out_type.shape

        out_types = [ir.RankedTensorType.get(dz_shape, dz_type.element_type)]
        operands = [dz, softmax_out]
        operand_shapes = [dz_shape, softmax_out_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch,
            pad_batch,
            heads,
            q_seqlen,
            k_seqlen,
            jax_dtype_to_te_dtype(dz_aval.dtype),
            scale_factor,
        )

        out = custom_caller(name, args, opaque, False)

        return [out]

    @staticmethod
    def backward_impl(primitive, dz, softmax_out, scale_factor):
        """
        softmax_backward implementation
        """
        assert primitive is not None
        dx = primitive.bind(dz, softmax_out, scale_factor=scale_factor)
        return dx

    @staticmethod
    def backward_batcher(primitive, batched_args, batch_dims, *, scale_factor):
        """
        softmax_backward batcher
        """
        assert primitive is not None
        dz, softmax_out = batched_args
        _, softmax_out_bdim = batch_dims

        out_bdims = softmax_out_bdim
        return primitive.bind(dz, softmax_out, scale_factor=scale_factor), out_bdims

    @classmethod
    def backward_infer_sharding_from_operands(cls, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_backward infer_sharding_from_operands
        """
        del scale_factor, result_infos  # Unused.
        dz_spec = get_padded_spec(arg_infos[0])
        if dz_spec[-1] is not None:
            warnings.warn(
                f"Sharding the hidden dimension is not supported in {cls.name}! "
                "Forcing XLA to not shard the hidden dim, which might introduce extra "
                "collective ops and hurt performance."
            )
        dx_sharding = NamedSharding(mesh, PartitionSpec(*dz_spec[:-1], None))
        return dx_sharding

    @classmethod
    def backward_partition(cls, impl, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_backward partition
        """
        del result_infos

        dz_spec = get_padded_spec(arg_infos[0])
        softmax_out_spec = get_padded_spec(arg_infos[1])
        if dz_spec[-1] is not None or softmax_out_spec[-1] is not None:
            warnings.warn(
                f"Sharding the hidden dimension is not supported in {cls.name}! "
                "Forcing XLA to not shard the hidden dim, which might introduce extra "
                "collective ops and hurt performance."
            )

        dz_sharding = NamedSharding(mesh, PartitionSpec(*dz_spec[:-1], None))
        softmax_out_sharding = NamedSharding(mesh, PartitionSpec(*softmax_out_spec[:-1], None))
        dx_sharding = dz_sharding
        arg_shardings = (dz_sharding, softmax_out_sharding)
        out_shardings = dx_sharding

        impl = partial(impl, scale_factor=scale_factor)
        return mesh, impl, out_shardings, arg_shardings


class ScaledSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Fwd Primitive
    """

    name = "te_scaled_softmax_forward"
    multiple_results = False
    impl_static_args = (1,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        dtype = dtypes.canonicalize_dtype(dtype)
        if (
            dtype in [jnp.float16, jnp.bfloat16]
            and 16 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
            and q_seqlen % 4 == 0  # q_seqlen must be divisor of 4
            and attn_batches % 4 == 0  # batch * heads must be divisor of 4
        ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, scale_factor):  # pylint: disable=unused-argument
        """
        te_scaled_softmax_forward abstract
        """
        return SoftmaxPrimitive.forward_abstract(logits_aval, scale_factor)

    @staticmethod
    def lowering(ctx, logits, *, scale_factor):
        """
        te_scaled_softmax_forward lowering rules
        """
        return SoftmaxPrimitive.forward_lowering(
            ScaledSoftmaxFwdPrimitive.name, ctx, logits, scale_factor=scale_factor
        )

    @staticmethod
    def impl(logits, scale_factor):
        return SoftmaxPrimitive.forward_impl(
            ScaledSoftmaxFwdPrimitive.inner_primitive, logits, scale_factor
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        return SoftmaxPrimitive.forward_batcher(
            ScaledSoftmaxFwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledSoftmaxFwdPrimitive.forward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledSoftmaxFwdPrimitive.forward_partition(
            ScaledSoftmaxFwdPrimitive.impl, scale_factor, mesh, arg_infos, result_infos
        )


register_primitive(ScaledSoftmaxFwdPrimitive)


def scaled_softmax_fwd(logits: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledSoftmaxFwdPrimitive.outer_primitive.bind(logits, scale_factor=scale_factor)


class ScaledSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Bwd Primitive
    """

    name = "te_scaled_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, scale_factor):
        """
        te_scaled_softmax_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_softmax_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(
            ScaledSoftmaxBwdPrimitive.name, ctx, dz, softmax_out, scale_factor=scale_factor
        )

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(
            ScaledSoftmaxBwdPrimitive.inner_primitive, dz, softmax_out, scale_factor=scale_factor
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        return SoftmaxPrimitive.backward_batcher(
            ScaledSoftmaxBwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledSoftmaxBwdPrimitive.backward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledSoftmaxBwdPrimitive.backward_partition(
            ScaledSoftmaxBwdPrimitive.impl, scale_factor, mesh, arg_infos, result_infos
        )


register_primitive(ScaledSoftmaxBwdPrimitive)


def scaled_softmax_bwd(
    dz: jnp.ndarray, softmax_out: jnp.ndarray, scale_factor: float
) -> jnp.ndarray:
    """
    scaled_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledSoftmaxBwdPrimitive.outer_primitive.bind(
        dz, softmax_out, scale_factor=scale_factor
    )


class ScaledMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Fwd Primitive
    """

    name = "te_scaled_masked_softmax_forward"
    multiple_results = False
    impl_static_args = (2,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        dtype = dtypes.canonicalize_dtype(dtype)
        if (
            dtype in [jnp.float16, jnp.bfloat16]
            and 16 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
            and q_seqlen % 4 == 0  # q_seqlen must be divisor of 4
            and attn_batches % 4 == 0  # batch * heads must be divisor of 4
        ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, mask_aval, scale_factor):  # pylint: disable=unused-argument
        """
        te_scaled_masked_softmax_forward abstract
        """

        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = logits_aval.shape

        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        mask_dtype = dtypes.canonicalize_dtype(mask_aval.dtype)
        assert mask_dtype in [
            jnp.uint8,
        ]
        mask_shape = mask_aval.shape
        pad_batch = batch = reduce(operator.mul, mask_shape[:-3])
        assert pad_batch in (1, batch)  # 1 means broadcast
        assert mask_shape[-3] == 1  # 1 means broadcast
        assert mask_shape[-2] == q_seqlen
        assert mask_shape[-1] == k_seqlen

        out_aval = core.raise_to_shaped(logits_aval)
        return out_aval

    @staticmethod
    def lowering(ctx, logits, mask, *, scale_factor):
        """
        te_scaled_masked_softmax_forward lowering rules
        """

        logits_aval, _ = ctx.avals_in
        i_type = ir.RankedTensorType(logits.type)
        i_shape = i_type.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        heads = i_shape[-3]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]

        mask_type = ir.RankedTensorType(mask.type)
        mask_shape = mask_type.shape
        pad_batch = reduce(operator.mul, mask_shape[:-3])

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [logits, mask]
        operand_shapes = [i_shape, mask_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch,
            pad_batch,
            heads,
            q_seqlen,
            k_seqlen,
            jax_dtype_to_te_dtype(logits_aval.dtype),
            scale_factor,
        )

        out = custom_caller(ScaledMaskedSoftmaxFwdPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(logits, mask, scale_factor):
        assert ScaledMaskedSoftmaxFwdPrimitive.inner_primitive is not None
        output = ScaledMaskedSoftmaxFwdPrimitive.inner_primitive.bind(
            logits, mask, scale_factor=scale_factor
        )
        return output

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        assert ScaledMaskedSoftmaxFwdPrimitive.outer_primitive is not None
        logits, mask = batched_args
        logits_bdim, _ = batch_dims

        out_bdims = logits_bdim
        return (
            ScaledMaskedSoftmaxFwdPrimitive.outer_primitive.bind(
                logits, mask, scale_factor=scale_factor
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledMaskedSoftmaxFwdPrimitive.forward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledMaskedSoftmaxFwdPrimitive.backward_partition(
            ScaledMaskedSoftmaxFwdPrimitive.impl, scale_factor, mesh, arg_infos, result_infos
        )


register_primitive(ScaledMaskedSoftmaxFwdPrimitive)


def scaled_masked_softmax_fwd(
    logits: jnp.ndarray, mask: jnp.ndarray, scale_factor: float
) -> jnp.ndarray:
    """
    scaled_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledMaskedSoftmaxFwdPrimitive.outer_primitive.bind(
        logits, mask, scale_factor=scale_factor
    )


class ScaledMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Bwd Primitive
    """

    name = "te_scaled_masked_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(
            ScaledMaskedSoftmaxBwdPrimitive.name, ctx, dz, softmax_out, scale_factor=scale_factor
        )

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(
            ScaledMaskedSoftmaxBwdPrimitive.inner_primitive,
            dz,
            softmax_out,
            scale_factor=scale_factor,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        return SoftmaxPrimitive.backward_batcher(
            ScaledMaskedSoftmaxBwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledMaskedSoftmaxBwdPrimitive.backward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledMaskedSoftmaxBwdPrimitive.backward_partition(
            ScaledMaskedSoftmaxBwdPrimitive.impl, scale_factor, mesh, arg_infos, result_infos
        )


register_primitive(ScaledMaskedSoftmaxBwdPrimitive)


def scaled_masked_softmax_bwd(
    dz: jnp.ndarray, softmax_out: jnp.ndarray, scale_factor: float
) -> jnp.ndarray:
    """
    scaled_masked_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledMaskedSoftmaxBwdPrimitive.outer_primitive.bind(
        dz, softmax_out, scale_factor=scale_factor
    )


class ScaledUpperTriangMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Fwd Primitive
    """

    name = "te_scaled_upper_triang_masked_softmax_forward"
    multiple_results = False
    impl_static_args = (1,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        dtype = dtypes.canonicalize_dtype(dtype)
        if (
            dtype in [jnp.float16, jnp.bfloat16]
            and 16 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
            and q_seqlen % 4 == 0  # q_seqlen must be divisor of 4
            and attn_batches % 4 == 0  # batch * heads must be divisor of 4
            and k_seqlen == q_seqlen
        ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return attn_batches % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, scale_factor):  # pylint: disable=unused-argument
        """
        te_scaled_upper_triang_masked_softmax_forward abstract
        """
        q_seqlen = logits_aval.shape[-2]
        k_seqlen = logits_aval.shape[-1]
        assert q_seqlen == k_seqlen
        return SoftmaxPrimitive.forward_abstract(logits_aval, scale_factor)

    @staticmethod
    def lowering(ctx, logits, *, scale_factor):
        """
        te_scaled_upper_triang_masked_softmax_forward lowering rules
        """
        return SoftmaxPrimitive.forward_lowering(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.name, ctx, logits, scale_factor=scale_factor
        )

    @staticmethod
    def impl(logits, scale_factor):
        return SoftmaxPrimitive.forward_impl(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.inner_primitive, logits, scale_factor
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        return SoftmaxPrimitive.forward_batcher(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.forward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.forward_partition(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.impl,
            scale_factor,
            mesh,
            arg_infos,
            result_infos,
        )


register_primitive(ScaledUpperTriangMaskedSoftmaxFwdPrimitive)


def scaled_upper_triang_masked_softmax_fwd(logits: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.outer_primitive.bind(
        logits, scale_factor=scale_factor
    )


class ScaledUpperTriangMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Bwd Primitive
    """

    name = "te_scaled_upper_triang_masked_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)  # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(
        batch: int, heads: int, q_seqlen: int, k_seqlen: int, dtype: jnp.dtype
    ) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype
        )

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.name,
            ctx,
            dz,
            softmax_out,
            scale_factor=scale_factor,
        )

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.inner_primitive,
            dz,
            softmax_out,
            scale_factor=scale_factor,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        check_valid_batch_dims(batch_dims)
        return SoftmaxPrimitive.backward_batcher(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor,
        )

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return ScaledUpperTriangMaskedSoftmaxBwdPrimitive.backward_infer_sharding_from_operands(
            scale_factor, mesh, arg_infos, result_infos
        )

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return ScaledUpperTriangMaskedSoftmaxBwdPrimitive.backward_partition(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.impl,
            scale_factor,
            mesh,
            arg_infos,
            result_infos,
        )


register_primitive(ScaledUpperTriangMaskedSoftmaxBwdPrimitive)


def scaled_upper_triang_masked_softmax_bwd(
    dz: jnp.ndarray, softmax_out: jnp.ndarray, scale_factor: float
) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledUpperTriangMaskedSoftmaxBwdPrimitive.outer_primitive.bind(
        dz, softmax_out, scale_factor=scale_factor
    )
