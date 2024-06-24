# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for attention"""
from dataclasses import dataclass
from functools import partial, reduce
import operator
import warnings

import jax.numpy as jnp
from jax import dtypes
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax
from transformer_engine.transformer_engine_jax import (
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_QKV_Layout,
    NVTE_Fused_Attn_Backend,
)
from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    te_dtype_to_jax_dtype,
    get_padded_spec,
)
from ..sharding import (
    all_reduce_sum_along_dp_fsdp,
    get_all_mesh_axes,
    num_of_devices,
)


__all__ = [
    "FusedAttnHelper",
    "fused_attn_fwd_qkvpacked",
    "fused_attn_bwd_qkvpacked",
    "fused_attn_fwd_kvpacked",
    "fused_attn_bwd_kvpacked",
    "fused_attn_fwd",
    "fused_attn_bwd",
]


@dataclass(frozen=True)
class FusedAttnHelper:
    """
    Helper for the fused attention backend
    """

    q_dtype: jnp.dtype
    kv_dtype: jnp.dtype
    qkv_layout: NVTE_QKV_Layout
    attn_bias_type: NVTE_Bias_Type
    attn_mask_type: NVTE_Mask_Type
    dropout_probability: float
    q_num_heads: int
    kv_num_heads: int
    q_max_seqlen: int
    kv_max_seqlen: int
    head_dim: int

    def is_fused_attn_kernel_available(self):
        """Check if there is available fused attention kernel"""
        return self.get_fused_attn_backend() != NVTE_Fused_Attn_Backend.NVTE_No_Backend

    def get_fused_attn_backend(self):
        """Get the fused attention kernel backend"""
        return transformer_engine_jax.get_fused_attn_backend(
            jax_dtype_to_te_dtype(self.q_dtype),
            jax_dtype_to_te_dtype(self.kv_dtype),
            self.qkv_layout,
            self.attn_bias_type,
            self.attn_mask_type,
            self.dropout_probability,
            self.q_num_heads,
            self.kv_num_heads,
            self.q_max_seqlen,
            self.kv_max_seqlen,
            self.head_dim,
        )

    @staticmethod
    def parse_qkv_aval(q_aval, k_aval, v_aval, qkv_layout):
        """Parse qkv aval"""
        match qkv_layout:
            case NVTE_QKV_Layout.NVTE_BS3HD:
                *q_batch_shape, q_max_seqlen, nqkv, attn_heads, q_head_dim = q_aval.shape
                kv_batch_shape = q_batch_shape
                kv_max_seqlen = q_max_seqlen
                num_gqa_groups = attn_heads
                kv_head_dim = q_head_dim
                assert nqkv == 3
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                *q_batch_shape, q_max_seqlen, attn_heads, q_head_dim = q_aval.shape
                *kv_batch_shape, kv_max_seqlen, nkv, num_gqa_groups, kv_head_dim = k_aval.shape
                assert nkv == 2
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                *q_batch_shape, q_max_seqlen, attn_heads, q_head_dim = q_aval.shape
                *kv_batch_shape, kv_max_seqlen, num_gqa_groups, kv_head_dim = k_aval.shape
                assert k_aval.shape == v_aval.shape
            case _:
                raise ValueError(f"Unexpected {qkv_layout=}")
        assert q_batch_shape == kv_batch_shape
        assert q_head_dim == kv_head_dim
        assert q_aval.dtype == k_aval.dtype == v_aval.dtype

        return (q_batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, q_head_dim)


@dataclass(frozen=True)
class _FusedAttnRNGStateChecker:
    """
    Checker for guarding the fused attention rng state.
    The fused attention backend requires a 64 bits seed and a 64 bits offset.
    However, JAX doesn't enable 64 bits by default,
    so we have to emulate seed as two 32 bits array.
    The offset calculation is maintained in the backend.
    """

    rng_state_dtype: jnp.dtype = jnp.uint32
    # (seed,) with internal dtype int64
    seed_size: int = 2
    # (seed, offset) with internal dtype int64
    rng_state_size: int = 2 * 2

    def check_seed(self, seed, dropout_probability, is_training):
        """
        Check the seed and convert the data type of seed if possible.
        """
        # Jax can't bind None, create a dummy tensor for None
        if seed is None:
            dropout_enabled = dropout_probability > 0 and is_training
            assert not dropout_enabled, "seed is not allowed to be None when dropout is enabled."
            seed = jnp.zeros(2, dtype=self.rng_state_dtype)
            seed = jnp.repeat(seed, num_of_devices())

        if seed.dtype != self.rng_state_dtype:
            warnings.warn(
                f"Requested {seed.dtype=} is not available, and will be "
                f"casted to dtype {self.rng_state_dtype}. "
                "Please use threefry/rbg/unsafe_rbg PRNG implementations to remove this warning."
            )
            seed = seed.astype(self.rng_state_dtype)

        assert seed.dtype == self.rng_state_dtype
        # Backend takes an int64_t seed, so only the first two u32 elements are taken
        assert seed.size >= self.seed_size

        return seed


def generate_cu_seqlen(actual_seqlen):
    """
    Generating cumsum seqlen for a batch
    """
    cu_seqlen = jnp.cumsum(actual_seqlen)
    cu_seqlen = jnp.hstack((0, cu_seqlen))
    return cu_seqlen


class FusedAttnFwdPrimitive(BasePrimitive):
    """
    Fused Attention Forward Primitive
    """

    name = "te_fused_attn_forward"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        q_aval,
        k_aval,
        v_aval,
        bias_aval,
        q_seqlen_or_cu_seqlen_aval,
        kv_seqlen_or_cu_seqlen_aval,
        seed_aval,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        """
        Fused attention fwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q_aval.dtype)
        k_dtype = dtypes.canonicalize_dtype(k_aval.dtype)
        v_dtype = dtypes.canonicalize_dtype(v_aval.dtype)
        bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        assert q_dtype == k_dtype == v_dtype == bias_dtype
        assert q_seqlen_or_cu_seqlen_aval.dtype == kv_seqlen_or_cu_seqlen_aval.dtype

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, qkv_layout)
        )

        output_shape = (*batch_shape, q_max_seqlen, attn_heads, head_dim)
        out_aval = q_aval.update(shape=output_shape, dtype=q_dtype)

        # backend determines the softmax buffer shape/dtype
        backend = FusedAttnHelper(
            q_dtype,
            k_dtype,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
            dropout_probability,
            attn_heads,
            num_gqa_groups,
            q_max_seqlen,
            kv_max_seqlen,
            head_dim,
        ).get_fused_attn_backend()

        if backend == NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen:
            softmax_shape = (*batch_shape, attn_heads, q_max_seqlen, kv_max_seqlen)
            softmax_dtype = q_dtype
        elif backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            softmax_shape = (*batch_shape, attn_heads, q_max_seqlen, 1)
            softmax_dtype = dtypes.canonicalize_dtype(jnp.float32)
        else:
            raise ValueError(f"Unsupported {backend=}")
        softmax_aux_aval = q_aval.update(shape=softmax_shape, dtype=softmax_dtype)

        # JAX does not enable 64-bit int by default so we get XLA to allocate x8 memory with
        # 32-bit unsigned int to get the buffer size we need in the C++ kernel
        checker = _FusedAttnRNGStateChecker()
        seed_dtype = dtypes.canonicalize_dtype(seed_aval.dtype)
        assert seed_dtype == checker.rng_state_dtype
        rng_state_shape = (seed_aval.shape[0], checker.rng_state_size)
        rng_state_aval = seed_aval.update(shape=rng_state_shape, dtype=checker.rng_state_dtype)

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        # do a dummy kernel call here to get workspace buffer shapes/dtypes that XLA needs to
        # prepare for the active fused-attn backend
        input_batch = reduce(operator.mul, batch_shape)
        wkspace_info = transformer_engine_jax.get_fused_attn_fwd_workspace_sizes(
            input_batch,
            bias_batch,
            q_max_seqlen,
            kv_max_seqlen,
            attn_heads,
            num_gqa_groups,
            bias_heads,
            head_dim,
            scaling_factor,
            dropout_probability,
            attn_bias_type,
            attn_mask_type,
            qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            is_training,
        )
        wkspace_aval = q_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )

        return out_aval, softmax_aux_aval, rng_state_aval, wkspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Fused attention fwd outer primitive abstract
        """
        out_aval, softmax_aux_aval, rng_state_aval, _ = FusedAttnFwdPrimitive.abstract(
            *args, **kwargs
        )
        return out_aval, softmax_aux_aval, rng_state_aval

    @staticmethod
    def lowering(
        ctx,
        q,
        k,
        v,
        bias,
        q_cu_seqlen,
        kv_cu_seqlen,
        seed,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        """
        Fused attention fwd lowering rules
        """
        operands = [q, k, v, bias, q_cu_seqlen, kv_cu_seqlen, seed]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        q_aval, k_aval, v_aval, bias_aval, *_ = ctx.avals_in

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, qkv_layout)
        )

        input_batch = reduce(operator.mul, batch_shape)

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        wkspace_aval = ctx.avals_out[-1]

        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            input_batch,
            bias_batch,
            q_max_seqlen,
            kv_max_seqlen,
            attn_heads,
            num_gqa_groups,
            bias_heads,
            head_dim,
            wkspace_aval.size,
            scaling_factor,
            dropout_probability,
            attn_bias_type,
            attn_mask_type,
            qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            is_training,
        )

        out = custom_caller(FusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(
        q,
        k,
        v,
        bias,
        q_seqlen,
        kv_seqlen,
        seed,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        assert FusedAttnFwdPrimitive.inner_primitive is not None

        q_cu_seqlen = generate_cu_seqlen(q_seqlen)
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen)

        output, softmax_aux, rng_state, _ = FusedAttnFwdPrimitive.inner_primitive.bind(
            q,
            k,
            v,
            bias,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
        )
        return output, softmax_aux, rng_state

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        check_valid_batch_dims(batch_dims)
        assert FusedAttnFwdPrimitive.outer_primitive is not None
        q_bdim, *_, seed_bdim = batch_dims

        out_bdims = q_bdim, q_bdim, seed_bdim
        return (
            FusedAttnFwdPrimitive.outer_primitive.bind(
                *batched_args,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                qkv_layout=qkv_layout,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_probability,
                is_training=is_training,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        mesh,
        arg_infos,
        result_infos,
    ):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, result_infos
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        match qkv_layout:
            case NVTE_QKV_Layout.NVTE_BS3HD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec[:-3], *q_spec[-2:]))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-4], q_spec[-2], q_spec[-4], None)
                )
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                # k_spec = (...batch, kv_seqlen, 2, num_gqa_groups, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], k_spec[-4])
                )
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                # k_spec = (...batch, kv_seqlen, num_gqa_groups, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], k_spec[-3])
                )
            case _:
                raise ValueError(f"Unsupported {qkv_layout=}")
        rng_state_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        return (out_sharding, softmax_aux_sharding, rng_state_sharding)

    @staticmethod
    def partition(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        mesh,
        arg_infos,
        result_infos,
    ):
        out_sharding = result_infos[0].sharding
        softmax_aux_sharding = result_infos[1].sharding
        rng_state_sharding = seed_sharding = NamedSharding(
            mesh, PartitionSpec(get_all_mesh_axes(), None)
        )
        arg_shardings = tuple([arg_i.sharding for arg_i in arg_infos[:-1]] + [seed_sharding])
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)
        impl = partial(
            FusedAttnFwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
        )
        return mesh, impl, out_shardings, arg_shardings


register_primitive(FusedAttnFwdPrimitive)


class FusedAttnBwdPrimitive(BasePrimitive):
    """
    Fused Attention Backward Primitive
    """

    name = "te_fused_attn_backward"
    multiple_results = True
    impl_static_args = (10, 11, 12, 13, 14, 15)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        q_aval,
        k_aval,
        v_aval,
        bias_aval,
        softmax_aux_aval,
        rng_state_aval,
        output_aval,
        doutput_aval,
        q_cu_seqlen_aval,
        kv_cu_seqlen_aval,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        """
        Fused attention bwd abstract
        """
        del softmax_aux_aval, rng_state_aval, output_aval

        q_dtype = dtypes.canonicalize_dtype(q_aval.dtype)
        k_dtype = dtypes.canonicalize_dtype(k_aval.dtype)
        v_dtype = dtypes.canonicalize_dtype(v_aval.dtype)
        bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        doutput_dtype = dtypes.canonicalize_dtype(doutput_aval.dtype)
        assert q_dtype == k_dtype == v_dtype == bias_dtype == doutput_dtype
        assert q_cu_seqlen_aval.dtype == kv_cu_seqlen_aval.dtype

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, qkv_layout)
        )

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        input_batch = reduce(operator.mul, batch_shape)
        wkspace_shape, wkspace_dtype = transformer_engine_jax.get_fused_attn_bwd_workspace_sizes(
            input_batch,
            bias_batch,
            q_max_seqlen,
            kv_max_seqlen,
            attn_heads,
            num_gqa_groups,
            bias_heads,
            head_dim,
            scaling_factor,
            dropout_probability,
            attn_bias_type,
            attn_mask_type,
            qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            is_training,
        )

        dq_aval = q_aval.update(shape=q_aval.shape, dtype=q_dtype)
        dk_aval = k_aval.update(shape=k_aval.shape, dtype=k_dtype)
        dv_aval = v_aval.update(shape=v_aval.shape, dtype=v_dtype)
        dbias_aval = bias_aval.update(shape=bias_aval.shape, dtype=bias_dtype)
        wkspace_aval = q_aval.update(
            shape=wkspace_shape, dtype=te_dtype_to_jax_dtype(wkspace_dtype)
        )

        return dq_aval, dk_aval, dv_aval, dbias_aval, wkspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Fused attention fwd outer primitive abstract
        """
        dq_aval, dk_aval, dv_aval, dbias_aval, _ = FusedAttnBwdPrimitive.abstract(*args, **kwargs)
        return dq_aval, dk_aval, dv_aval, dbias_aval

    @staticmethod
    def lowering(
        ctx,
        q,
        k,
        v,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_cu_seqlen,
        kv_cu_seqlen,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        """
        Fused attention bwd lowering rules
        """
        operands = [
            q,
            k,
            v,
            bias,
            softmax_aux,
            rng_state,
            output,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
        ]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        q_aval, k_aval, v_aval, bias_aval, *_ = ctx.avals_in

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, qkv_layout)
        )

        input_batch = reduce(operator.mul, batch_shape)

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        wkspace_aval = ctx.avals_out[-1]

        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            input_batch,
            bias_batch,
            q_max_seqlen,
            kv_max_seqlen,
            attn_heads,
            num_gqa_groups,
            bias_heads,
            head_dim,
            wkspace_aval.size,
            scaling_factor,
            dropout_probability,
            attn_bias_type,
            attn_mask_type,
            qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            is_training,
        )

        out = custom_caller(FusedAttnBwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(
        q,
        k,
        v,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        assert FusedAttnBwdPrimitive.inner_primitive is not None

        q_cu_seqlen = generate_cu_seqlen(q_seqlen)
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen)

        dq, dk, dv, dbias, _ = FusedAttnBwdPrimitive.inner_primitive.bind(
            q,
            k,
            v,
            bias,
            softmax_aux,
            rng_state,
            output,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
        )
        return dq, dk, dv, dbias

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
    ):
        check_valid_batch_dims(batch_dims)
        assert FusedAttnBwdPrimitive.outer_primitive is not None
        q_bdim, k_bdim, v_bdim, *_ = batch_dims

        out_bdims = q_bdim, k_bdim, v_bdim, q_bdim
        return (
            FusedAttnBwdPrimitive.outer_primitive.bind(
                *batched_args,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                qkv_layout=qkv_layout,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_probability,
                is_training=is_training,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        mesh,
        arg_infos,
        result_infos,
    ):
        del attn_bias_type, attn_mask_type, qkv_layout, scaling_factor
        del dropout_probability, is_training, result_infos
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        v_spec = get_padded_spec(arg_infos[2])
        bias_spec = get_padded_spec(arg_infos[3])
        dq_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        dk_sharding = NamedSharding(mesh, PartitionSpec(*k_spec))
        dv_sharding = NamedSharding(mesh, PartitionSpec(*v_spec))
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))
        return (dq_sharding, dk_sharding, dv_sharding, dbias_sharding)

    @staticmethod
    def partition(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        v_spec = get_padded_spec(arg_infos[2])
        bias_spec = get_padded_spec(arg_infos[3])
        dq_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        dk_sharding = NamedSharding(mesh, PartitionSpec(*k_spec))
        dv_sharding = NamedSharding(mesh, PartitionSpec(*v_spec))
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (dq_sharding, dk_sharding, dv_sharding, dbias_sharding)

        def sharded_impl(
            q, k, v, bias, softmax_aux, rng_state, output, doutput, q_cu_seqlen, kv_cu_seqlen
        ):
            local_dq, local_dk, local_dv, local_dbias = FusedAttnBwdPrimitive.impl(
                q,
                k,
                v,
                bias,
                softmax_aux,
                rng_state,
                output,
                doutput,
                q_cu_seqlen,
                kv_cu_seqlen,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                qkv_layout=qkv_layout,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_probability,
                is_training=is_training,
            )
            global_dbias = local_dbias
            if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias)
            return local_dq, local_dk, local_dv, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(FusedAttnBwdPrimitive)


def fused_attn_fwd_qkvpacked(
    qkv: jnp.ndarray,
    bias: jnp.ndarray,
    seqlen: jnp.ndarray,
    seed: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE self fused attention fwd
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv.dtype)

    _not_used = jnp.zeros(0, qkv.dtype)
    return FusedAttnFwdPrimitive.outer_primitive.bind(
        qkv,
        _not_used,
        _not_used,
        bias,
        seqlen,
        seqlen,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BS3HD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )


def fused_attn_bwd_qkvpacked(
    qkv: jnp.ndarray,
    bias: jnp.ndarray,
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    seqlen: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE self fused attention bwd
    Return the gradients of self fused attention with packed qkv input
    """
    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv.dtype)
    dummy_input = jnp.zeros(0, dtype=qkv.dtype)
    dqkv, *_, dbias = FusedAttnBwdPrimitive.outer_primitive.bind(
        qkv,
        dummy_input,
        dummy_input,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        seqlen,
        seqlen,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BS3HD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )
    return dqkv, dbias


def fused_attn_fwd_kvpacked(
    q: jnp.ndarray,
    kv: jnp.ndarray,
    bias: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    seed: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE fused attention fwd with kvpacked inputs
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=q.dtype)

    return FusedAttnFwdPrimitive.outer_primitive.bind(
        q,
        kv,
        jnp.zeros(0, q.dtype),
        bias,
        q_seqlen,
        kv_seqlen,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BS2HD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )


def fused_attn_bwd_kvpacked(
    q: jnp.ndarray,
    kv: jnp.ndarray,
    bias: jnp.ndarray,
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE fused attention bwd with kvpacked inputs
    Return the gradients of fused attention with packed kv input
    """
    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=q.dtype)
    dummy_input = jnp.zeros(0, q.dtype)
    dq, dkv, _, dbias = FusedAttnBwdPrimitive.outer_primitive.bind(
        q,
        kv,
        dummy_input,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BS2HD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )
    return dq, dkv, dbias


def fused_attn_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    seed: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE fused attention fwd, where query, key, value are seperated tensors
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=q.dtype)

    return FusedAttnFwdPrimitive.outer_primitive.bind(
        q,
        k,
        v,
        bias,
        q_seqlen,
        kv_seqlen,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )


def fused_attn_bwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
):
    """
    Wrapper for TE fused attention bwd
    Return the gradients of fused attention with seperated query, key, value tensors
    """
    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=q.dtype)
    return FusedAttnBwdPrimitive.outer_primitive.bind(
        q,
        k,
        v,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
    )
