# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for attention"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, reduce
import operator
from typing import Optional, Tuple
import warnings

import jax.numpy as jnp
from jax import dtypes, lax, tree_util
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax
from transformer_engine.transformer_engine_jax import (
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_QKV_Layout,
    NVTE_QKV_Format,
    NVTE_Fused_Attn_Backend,
    nvte_get_qkv_format,
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
    get_mesh_axis_size,
    global_mesh_resource,
    num_of_devices,
    lax_paral_op,
)


__all__ = [
    "FusedAttnHelper",
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
            case NVTE_QKV_Layout.NVTE_BS3HD | NVTE_QKV_Layout.NVTE_T3HD:
                *q_batch_shape, q_max_seqlen, nqkv, attn_heads, q_head_dim = q_aval.shape
                kv_batch_shape = q_batch_shape
                kv_max_seqlen = q_max_seqlen
                num_gqa_groups = attn_heads
                kv_head_dim = q_head_dim
                assert nqkv == 3
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD | NVTE_QKV_Layout.NVTE_THD_T2HD:
                *q_batch_shape, q_max_seqlen, attn_heads, q_head_dim = q_aval.shape
                *kv_batch_shape, kv_max_seqlen, nkv, num_gqa_groups, kv_head_dim = k_aval.shape
                assert nkv == 2
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD | NVTE_QKV_Layout.NVTE_THD_THD_THD:
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
    cu_seqlen = jnp.cumsum(actual_seqlen, axis=-1)
    cu_seqlen = jnp.where(actual_seqlen < 0, -1, cu_seqlen)
    cu_seqlen = jnp.insert(cu_seqlen, 0, values=0, axis=-1)
    return cu_seqlen


class FusedAttnFwdPrimitive(BasePrimitive):
    """
    Fused Attention Forward Primitive
    """

    name = "te_fused_attn_forward"
    multiple_results = True
    impl_static_args = (9, 10, 11, 12, 13, 14, 15)
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
        _q_seq_offsets,
        _k_seq_offsets,
        seed_aval,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
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
            softmax_shape = (*batch_shape, attn_heads, q_max_seqlen, max_segments_per_seq)
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
            max_segments_per_seq,
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
        q_seq_offsets,
        k_seq_offsets,
        seed,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
    ):
        """
        Fused attention fwd lowering rules
        """
        operands = [
            q,
            k,
            v,
            bias,
            q_cu_seqlen,
            kv_cu_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            seed,
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
            max_segments_per_seq,
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
        q_seq_offsets,
        k_seq_offsets,
        seed,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
    ):
        assert FusedAttnFwdPrimitive.inner_primitive is not None

        if nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format.NVTE_THD:

            def _fix_len_take(x, condition):
                x_shape = x.shape
                x = x.flatten()
                size = x.size
                indices = jnp.nonzero(condition.flatten(), size=size, fill_value=size)[0]
                y = jnp.take(x, indices, fill_value=-1)
                return jnp.reshape(y, x_shape)

            def convert_to_2d(offsets, batch, max_seqlen):
                offsets_2d = jnp.where(
                    offsets >= 0,
                    offsets + (jnp.arange(batch) * max_seqlen)[..., jnp.newaxis],
                    offsets,
                )
                return offsets_2d

            match qkv_layout:
                case NVTE_QKV_Layout.NVTE_T3HD:
                    kv_max_seqlen = q_max_seqlen = q.shape[-4]
                    kv_batch = q_batch = reduce(operator.mul, q.shape[:-4])
                case NVTE_QKV_Layout.NVTE_THD_T2HD:
                    q_max_seqlen = q.shape[-3]
                    q_batch = reduce(operator.mul, q.shape[:-3])
                    kv_max_seqlen = k.shape[-4]
                    kv_batch = reduce(operator.mul, k.shape[:-4])
                case NVTE_QKV_Layout.NVTE_THD_THD_THD:
                    q_max_seqlen = q.shape[-3]
                    q_batch = reduce(operator.mul, q.shape[:-3])
                    kv_max_seqlen = k.shape[-3]
                    kv_batch = reduce(operator.mul, k.shape[:-3])

            # Gather valid q_seqlen, which is greater than 0
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, -1, -1, -1, -1]]
            q_seqlen = _fix_len_take(q_seqlen, q_seqlen > 0)
            kv_seqlen = _fix_len_take(kv_seqlen, kv_seqlen > 0)

            # Flatten the offset calculation
            # max_seqlen = 8, [[0, 3, 5, -1], [0, 2, 4, -1]] -> [[0, 3, 5, -1], [8, 11, 13, -1]]
            q_seq_offsets = convert_to_2d(q_seq_offsets, q_batch, q_max_seqlen)
            k_seq_offsets = convert_to_2d(k_seq_offsets, kv_batch, kv_max_seqlen)
            # Gather valid q_seq_offsets, which is greater and equal to 0
            # [[0, 3, 5, -1], [8, 11, 13, -1]] -> [[0, 3, 5, 8], [11, 13, -1, -1]]
            q_seq_offsets = _fix_len_take(q_seq_offsets, q_seq_offsets >= 0)
            k_seq_offsets = _fix_len_take(k_seq_offsets, k_seq_offsets >= 0)

            # Set the unused position to max size (batch * max_seqlen)
            # [[0, 3, 5, 8], [11, 13, -1, -1]] -> [[0, 3, 5, 8], [11, 13, b*s, b*s]]
            q_seq_offsets = jnp.where(q_seq_offsets < 0, q_batch * q_max_seqlen, q_seq_offsets)
            k_seq_offsets = jnp.where(k_seq_offsets < 0, kv_batch * kv_max_seqlen, k_seq_offsets)

        q_cu_seqlen = generate_cu_seqlen(q_seqlen.flatten())
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen.flatten())

        output, softmax_aux, rng_state, _ = FusedAttnFwdPrimitive.inner_primitive.bind(
            q,
            k,
            v,
            bias,
            q_cu_seqlen,
            kv_cu_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
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
        max_segments_per_seq,
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
                max_segments_per_seq=max_segments_per_seq,
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
        max_segments_per_seq,
        mesh,
        arg_infos,
        result_infos,
    ):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, max_segments_per_seq, result_infos
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        match qkv_layout:
            case NVTE_QKV_Layout.NVTE_BS3HD | NVTE_QKV_Layout.NVTE_T3HD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec[:-3], *q_spec[-2:]))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-4], q_spec[-2], q_spec[-4], None)
                )
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD | NVTE_QKV_Layout.NVTE_THD_T2HD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                # k_spec = (...batch, kv_seqlen, 2, num_gqa_groups, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], k_spec[-4])
                )
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD | NVTE_QKV_Layout.NVTE_THD_THD_THD:
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
        max_segments_per_seq,
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
            max_segments_per_seq=max_segments_per_seq,
        )
        return mesh, impl, out_shardings, arg_shardings


register_primitive(FusedAttnFwdPrimitive)


class FusedAttnBwdPrimitive(BasePrimitive):
    """
    Fused Attention Backward Primitive
    """

    name = "te_fused_attn_backward"
    multiple_results = True
    impl_static_args = (12, 13, 14, 15, 16, 17, 18)
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
        q_seqlen_or_cu_seqlen_aval,
        kv_seqlen_or_cu_seqlen_aval,
        _q_seq_offsets,
        _k_seq_offsets,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
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
        assert q_seqlen_or_cu_seqlen_aval.dtype == kv_seqlen_or_cu_seqlen_aval.dtype

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
            max_segments_per_seq,
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
        q_seq_offsets,
        k_seq_offsets,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
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
            q_seq_offsets,
            k_seq_offsets,
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
            max_segments_per_seq,
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
        q_seq_offsets,
        k_seq_offsets,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
    ):
        assert FusedAttnBwdPrimitive.inner_primitive is not None

        if nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format.NVTE_THD:

            def _fix_len_take(x, condition):
                x_shape = x.shape
                x = x.flatten()
                size = x.size
                indices = jnp.nonzero(condition.flatten(), size=size, fill_value=size)[0]
                # TODO(rewang): try indices_are_sorted
                y = jnp.take(x, indices, fill_value=-1)
                return jnp.reshape(y, x_shape)

            def convert_to_2d(offsets, batch, max_seqlen):
                offsets_2d = jnp.where(
                    offsets >= 0,
                    offsets + (jnp.arange(batch) * max_seqlen)[..., jnp.newaxis],
                    offsets,
                )
                return offsets_2d

            match qkv_layout:
                case NVTE_QKV_Layout.NVTE_T3HD:
                    kv_max_seqlen = q_max_seqlen = q.shape[-4]
                    kv_batch = q_batch = reduce(operator.mul, q.shape[:-4])
                case NVTE_QKV_Layout.NVTE_THD_T2HD:
                    q_max_seqlen = q.shape[-3]
                    q_batch = reduce(operator.mul, q.shape[:-3])
                    kv_max_seqlen = k.shape[-4]
                    kv_batch = reduce(operator.mul, k.shape[:-4])
                case NVTE_QKV_Layout.NVTE_THD_THD_THD:
                    q_max_seqlen = q.shape[-3]
                    q_batch = reduce(operator.mul, q.shape[:-3])
                    kv_max_seqlen = k.shape[-3]
                    kv_batch = reduce(operator.mul, k.shape[:-3])

            # Gather valid q_seqlen, which is greater than 0
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, -1, -1, -1, -1]]
            q_seqlen = _fix_len_take(q_seqlen, q_seqlen > 0)
            kv_seqlen = _fix_len_take(kv_seqlen, kv_seqlen > 0)

            # Flatten the offset calculation
            # max_seqlen = 8, [[0, 3, 5, -1], [0, 2, 4, -1]] -> [[0, 3, 5, -1], [8, 11, 13, -1]]
            q_seq_offsets = convert_to_2d(q_seq_offsets, q_batch, q_max_seqlen)
            k_seq_offsets = convert_to_2d(k_seq_offsets, kv_batch, kv_max_seqlen)
            # Gather valid q_seq_offsets, which is greater and equal to 0
            # [[0, 3, 5, -1], [8, 11, 13, -1]] -> [[0, 3, 5, 8], [11, 13, -1, -1]]
            q_seq_offsets = _fix_len_take(q_seq_offsets, q_seq_offsets >= 0)
            k_seq_offsets = _fix_len_take(k_seq_offsets, k_seq_offsets >= 0)

            # Set the unused position to max size (batch * max_seqlen)
            # [[0, 3, 5, 8], [11, 13, -1, -1]] -> [[0, 3, 5, 8], [11, 13, b*s, b*s]]
            q_seq_offsets = jnp.where(q_seq_offsets < 0, q_batch * q_max_seqlen, q_seq_offsets)
            k_seq_offsets = jnp.where(k_seq_offsets < 0, kv_batch * kv_max_seqlen, k_seq_offsets)

        q_cu_seqlen = generate_cu_seqlen(q_seqlen.flatten())
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen.flatten())

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
            q_seq_offsets,
            k_seq_offsets,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
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
        max_segments_per_seq,
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
                max_segments_per_seq=max_segments_per_seq,
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
        max_segments_per_seq,
        mesh,
        arg_infos,
        result_infos,
    ):
        del attn_bias_type, attn_mask_type, qkv_layout, scaling_factor, max_segments_per_seq
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
        max_segments_per_seq,
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
            q_seq_offsets,
            k_seq_offsets,
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
                q_seq_offsets,
                k_seq_offsets,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                qkv_layout=qkv_layout,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_probability,
                is_training=is_training,
                max_segments_per_seq=max_segments_per_seq,
            )
            global_dbias = local_dbias
            if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias)
            return local_dq, local_dk, local_dv, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(FusedAttnBwdPrimitive)


def fused_attn_fwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    q_seq_offsets: Optional[jnp.ndarray],
    kv_seq_offsets: Optional[jnp.ndarray],
    seed: Optional[jnp.ndarray],
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
) -> jnp.ndarray:
    """
    Perform the forward pass of with cuDNN fused attention implementations.

    This function implements the following formula:
        BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    Args:
        qkv (Tuple[jnp.ndarray, ...]): A tuple containing query, key, and value tensors.
        It supports three formats:
            - `(qkv_packed,)`: For interleaved QKV packed format, typically used when query, key,
              and value have the same shape (e.g., self-attention).
            - `(query, kv_packed)`: For separate query and KV packed format, typically used when
              query has a different shape (e.g., cross-attention).
            - `(query, key, value)`: For separate query, key, and value tensors.
        bias (Optional[jnp.ndarray]): An optional bias tensor to be added to the attention scores.
        q_seqlen (jnp.ndarray): Sequence lengths for the query, with shape [batch,].
        kv_seqlen (jnp.ndarray): Sequence lengths for the key and value, with shape [batch,].
        q_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch + 1,].
        kv_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch + 1,].
        seed (Optional[jnp.ndarray]): Optional random seed for dropout.
        attn_bias_type (NVTE_Bias_Type): Type of attention bias.
        attn_mask_type (NVTE_Mask_Type): Type of attention mask.
        qkv_layout (NVTE_QKV_Layout): Layout of the QKV tensors.
        scaling_factor (float): Scaling factor for the attention scores.
        dropout_probability (float): Dropout probability to apply during attention.
        is_training (bool): Flag indicating whether the model is in training mode.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.
    """
    seed = _FusedAttnRNGStateChecker().check_seed(seed, dropout_probability, is_training)

    assert (q_seq_offsets is None) == (
        kv_seq_offsets is None
    ), "Both q_seq_offsets and kv_seq_offsets must be either None or have values."
    is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format.NVTE_THD

    # For optional tensors, which custom calls doesn't support None
    _not_used = jnp.zeros(0, dtype=qkv[0].dtype)
    match qkv_layout:
        case NVTE_QKV_Layout.NVTE_BS3HD | NVTE_QKV_Layout.NVTE_T3HD:
            assert len(qkv) == 1, f"qkv=(packed_qkv,) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = [*qkv, _not_used, _not_used]
        case NVTE_QKV_Layout.NVTE_BSHD_BS2HD | NVTE_QKV_Layout.NVTE_THD_T2HD:
            assert (
                len(qkv) == 2
            ), f"qkv=(query, packed_kv) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = [*qkv, _not_used]
        case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD | NVTE_QKV_Layout.NVTE_THD_THD_THD:
            assert (
                len(qkv) == 3
            ), f"qkv=(query, key, value) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = qkv

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv[0].dtype)

    return FusedAttnFwdPrimitive.outer_primitive.bind(
        *qkv_for_primitive,
        bias,
        q_seqlen,
        kv_seqlen,
        q_seq_offsets if is_ragged else _not_used,
        kv_seq_offsets if is_ragged else _not_used,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
    )


def fused_attn_bwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    q_seq_offsets: Optional[jnp.ndarray],
    kv_seq_offsets: Optional[jnp.ndarray],
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
):
    """
    Perform the backward pass of the cuDNN fused attention implementations.

    Args:
        qkv (Tuple[jnp.ndarray, ...]): A tuple containing the original query, key, and value tensors
        used in the forward pass. It supports three formats:
            - `(qkv_packed,)`: For interleaved QKV packed format, typically used when query, key,
              and value have the same shape (e.g., self-attention).
            - `(query, kv_packed)`: For separate query and KV packed format, typically used when
              query has a different shape (e.g., cross-attention).
            - `(query, key, value)`: For separate query, key, and value tensors.
        bias (Optional[jnp.ndarray]): An optional bias tensor to be added to the attention scores.
        softmax_aux (jnp.ndarray): Auxiliary tensors from the softmax step used in the forward pass.
        rng_state (jnp.ndarray): Auxiliary tensors to save the random state in the forward pass.
        output (jnp.ndarray): The output tensor from the forward pass.
        doutput (jnp.ndarray): The gradient with respect to the output.
        q_seqlen (jnp.ndarray): Sequence lengths for the query, with shape [batch,].
        kv_seqlen (jnp.ndarray): Sequence lengths for the key and value, with shape [batch,].
        q_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch + 1,].
        kv_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch + 1,].
        attn_bias_type (NVTE_Bias_Type): Type of attention bias.
        attn_mask_type (NVTE_Mask_Type): Type of attention mask.
        qkv_layout (NVTE_QKV_Layout): Layout of the QKV tensors.
        scaling_factor (float): Scaling factor for the attention scores.
        dropout_probability (float): Dropout probability to apply during attention.
        is_training (bool): Flag indicating whether the model is in training mode.

    Returns:
        Tuple[jnp.ndarray, ...], jnp.ndarray:
        - The first tuple contains the gradients with respect to the input `qkv` tensors in the
          same format as the input `qkv`.
        - The second value is the gradient with respect to `bias`, or `None` if `bias` is `None`.
    """

    assert (q_seq_offsets is None) == (
        kv_seq_offsets is None
    ), "Both q_seq_offsets and kv_seq_offsets must be either None or have values."
    is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format.NVTE_THD

    # For optional tensors, which custom calls doesn't support None
    _not_used = jnp.zeros(0, dtype=qkv[0].dtype)

    match qkv_layout:
        case NVTE_QKV_Layout.NVTE_BS3HD | NVTE_QKV_Layout.NVTE_T3HD:
            assert len(qkv) == 1, f"qkv=(packed_qkv,) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = [*qkv, _not_used, _not_used]
        case NVTE_QKV_Layout.NVTE_BSHD_BS2HD | NVTE_QKV_Layout.NVTE_THD_T2HD:
            assert (
                len(qkv) == 2
            ), f"qkv=(query, packed_kv) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = [*qkv, _not_used]
        case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD | NVTE_QKV_Layout.NVTE_THD_THD_THD:
            assert (
                len(qkv) == 3
            ), f"qkv=(query, key, value) is expected with {qkv_layout=} but got {qkv=}"
            qkv_for_primitive = qkv

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv[0].dtype)

    *qkv_grads, bias_grad = FusedAttnBwdPrimitive.outer_primitive.bind(
        *qkv_for_primitive,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        q_seq_offsets if is_ragged else _not_used,
        kv_seq_offsets if is_ragged else _not_used,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
    )
    return tuple(qkv_grads[: len(qkv)]), bias_grad


class _FusedRingAttnStatusTracker:

    workload_balacne_of_causal_mask: bool = False

    @staticmethod
    def _balance_workload_along_seq(*inputs, seq_dim: int = 1):
        cp_size = get_mesh_axis_size(global_mesh_resource().cp_resource)

        def reorder_tensor(tensor):
            ori_tensor_shape = tensor.shape
            tensor = tensor.reshpae(
                (
                    *ori_tensor_shape[:seq_dim],
                    2,
                    cp_size,
                    ori_tensor_shape[seq_dim] // 2 // cp_size,
                    *ori_tensor_shape[seq_dim + 1 :],
                )
            )
            tensor = tensor.transpose(
                (seq_dim, seq_dim + 1, seq_dim + 1, *[i for i in range(len(tensor.shape) - 3)])
            )
            reorded_tensor = jnp.zeros(
                (
                    cp_size,
                    ori_tensor_shape[seq_dim] // cp_size,
                    *ori_tensor_shape[:seq_dim] * ori_tensor_shape[seq_dim + 1 :],
                ),
                dtype=tensor.dtype,
            )
            for i in range(cp_size):
                reorded_tensor = reorded_tensor.at[i, ...].set(
                    jnp.concat([tensor[0, i, ...], tensor[1, cp_size - i - 1, ...]], axis=0)
                )
            reorded_tensor = reorded_tensor.transpose(
                (
                    *[i + 2 for i in range(seq_dim)],
                    0,
                    1,
                    *[i for i in range(seq_dim + 2, len(reorded_tensor.shape))],
                )
            )
            reorded_tensor = reorded_tensor.reshape(tensor.shape)
            return reorded_tensor

        outputs = tree_util.tree_map(reorder_tensor, inputs)
        return outputs

    @staticmethod
    @contextmanager
    def workload_balance_of_context_parallel(*inputs, seq_dim: int = 1):
        try:
            outputs = _FusedRingAttnStatusTracker._balance_workload_along_seq(
                *inputs, seq_dim=seq_dim
            )
            _FusedRingAttnStatusTracker.workload_balacne_of_causal_mask = True
            yield outputs
        finally:
            _FusedRingAttnStatusTracker.workload_balacne_of_causal_mask = False

    @staticmethod
    def is_workload_balanced():
        return _FusedRingAttnStatusTracker.workload_balacne_of_causal_mask


class FusedRingAttnFwdPrimitive(FusedAttnFwdPrimitive):
    """
    Fused Ring Attention Forward Primitive
    """

    name = "te_fused_ring_attn_forward"
    multiple_results = True
    impl_static_args = (9, 10, 11, 12, 13, 14, 15)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def correct_softmax_aux(softmax_aux, softmax_aux_per_step):
        max_scale = jnp.maximum(softmax_aux, softmax_aux_per_step)
        min_scale = jnp.minimum(softmax_aux, softmax_aux_per_step)
        new_softmax_aux = max_scale + jnp.log(1 + jnp.exp(min_scale - max_scale))
        return new_softmax_aux

    @staticmethod
    def adjust_seqlen(seqlen, max_seqlen, idx):
        seqlen_of_curr_step = seqlen - max_seqlen * idx
        seqlen_of_curr_step = jnp.where(seqlen_of_curr_step < 0, 0, seqlen_of_curr_step)
        seqlen_per_step = jnp.where(
            seqlen_of_curr_step < max_seqlen, seqlen_of_curr_step, max_seqlen
        )
        return seqlen_per_step

    @staticmethod
    def partition(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
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

        assert dropout_probability <= 0, "Ring Attention currently does not support dropout."
        assert (
            attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS
        ), "Ring Attention currently only support no_bias."
        assert (
            qkv_layout == NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
        ), "Ring Attention currently only support BSHD_BSHD_BSHD QKV layout."

        # arg_shardings[0] is Q sharding
        # Assume index-1 is the sequence dimension
        cp_mesh_axis_name = arg_shardings[0].spec.axis_names[1]

        partial_fmha_fwd_no_mask_impl = partial(
            FusedAttnFwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=NVTE_Mask_Type.NVTE_NO_MASK,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        partial_fmha_fwd_causal_mask_impl = partial(
            FusedAttnFwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        partial_fmha_fwd_regular_mask_impl = partial(
            FusedAttnFwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        def ring_attn_fwd_impl(
            q,
            k,
            v,
            bias,
            q_seqlen,
            kv_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            seed,
        ):
            batch, q_max_seqlen, head, _ = q.shape
            _, kv_max_seqlen, _, _ = k.shape

            cp_rank = lax.axis_index(cp_mesh_axis_name)
            cp_size = get_mesh_axis_size(cp_mesh_axis_name)

            output_per_steps = jnp.zeros((cp_size, *q.shape), dtype=jnp.float32)
            softmax_aux_per_steps = jnp.zeros(
                (cp_size, batch, head, q_max_seqlen, 1), dtype=jnp.float32
            )
            softmax_aux = jnp.full((batch, head, q_max_seqlen, 1), -jnp.inf, dtype=jnp.float32)
            rng_state = jnp.zeros(result_infos[2].shape).astype(result_infos[2].dtype)

            def scan_kv_block(carry, idx):
                k_curr, v_curr, output_per_steps, softmax_aux_per_steps, softmax_aux, rng_state = (
                    carry
                )

                def causal_mask_compute():
                    q_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    output_per_step, softmax_aux_per_step, rng_state_per_step = (
                        partial_fmha_fwd_causal_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                            seed,
                        )
                    )
                    return output_per_step, softmax_aux_per_step, rng_state_per_step

                def half_kv_no_mask_compute():
                    q_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = (
                        FusedRingAttnFwdPrimitive.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx) // 2
                    )
                    k_part = k_curr[:, : kv_max_seqlen // 2, :, :]
                    v_part = v_curr[:, : kv_max_seqlen // 2, :, :]
                    output_per_step, softmax_aux_per_step, rng_state_per_step = (
                        partial_fmha_fwd_no_mask_impl(
                            q,
                            k_part,
                            v_part,
                            bias,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                            seed,
                        )
                    )
                    return output_per_step, softmax_aux_per_step, rng_state_per_step

                def half_q_no_mask_compute():
                    q_seqlen_per_step = (
                        FusedRingAttnFwdPrimitive.adjust_seqlen(q_seqlen, q_max_seqlen, idx) // 2
                    )
                    kv_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    q_part = q[:, q_max_seqlen // 2 :, :, :]
                    output_per_step, softmax_aux_per_step, rng_state_per_step = (
                        partial_fmha_fwd_no_mask_impl(
                            q_part,
                            k_curr,
                            v_curr,
                            bias,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                            seed,
                        )
                    )
                    output_per_step = jnp.concat(
                        [jnp.zeros(q_part.shape, dtype=q_part.dtype), output_per_step], axis=1
                    )
                    softmax_aux_per_step = jnp.concat(
                        [
                            jnp.full(
                                (batch, head, q_part.shape[1], 1), -jnp.inf, dtype=jnp.float32
                            ),
                            softmax_aux_per_step,
                        ],
                        axis=2,
                    )
                    return output_per_step, softmax_aux_per_step, rng_state_per_step

                def no_mask_compute():
                    q_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    output_per_step, softmax_aux_per_step, rng_state_per_step = (
                        partial_fmha_fwd_no_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                            seed,
                        )
                    )
                    return output_per_step, softmax_aux_per_step, rng_state_per_step

                def skip_compute():
                    output_per_step = jnp.zeros(q.shape, dtype=q.dtype)
                    softmax_aux_per_step = jnp.full(
                        (batch, head, q.shape[1], 1), -jnp.inf, dtype=jnp.float32
                    )
                    rng_state_per_step = jnp.zeros(rng_state.shape).astype(rng_state.dtype)
                    return output_per_step, softmax_aux_per_step, rng_state_per_step

                if attn_mask_type == NVTE_Mask_Type.NVTE_CAUSAL_MASK:
                    # This is for nested jax.lax.cond
                    def jax_cond_wrap():
                        if _FusedRingAttnStatusTracker.is_workload_balanced():
                            return lax.cond(
                                (idx <= cp_rank), half_kv_no_mask_compute, half_q_no_mask_compute
                            )
                        return lax.cond((idx <= cp_rank), no_mask_compute, skip_compute)

                    output_per_step, softmax_aux_per_step, rng_state_per_step = lax.cond(
                        idx == 0, causal_mask_compute, jax_cond_wrap
                    )
                else:
                    q_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnFwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    output_per_step, softmax_aux_per_step, rng_state = (
                        partial_fmha_fwd_regular_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                            seed,
                        )
                    )

                softmax_aux = FusedRingAttnFwdPrimitive.correct_softmax_aux(
                    softmax_aux, softmax_aux_per_step
                )
                output_per_steps = output_per_steps.at[idx].set(output_per_step.astype(jnp.float32))
                softmax_aux_per_steps = softmax_aux_per_steps.at[idx].set(softmax_aux_per_step)

                k_curr = lax_paral_op(
                    k_curr,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )
                v_curr = lax_paral_op(
                    v_curr,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )

                return (
                    k_curr,
                    v_curr,
                    softmax_aux,
                    output_per_steps,
                    softmax_aux_per_steps,
                    rng_state_per_step,
                ), None

            k_curr = k
            v_curr = v

            (k_curr, v_curr, softmax_aux, output_per_steps, softmax_aux_per_steps, rng_state), _ = (
                lax.scan(
                    scan_kv_block,
                    init=(
                        k_curr,
                        v_curr,
                        softmax_aux,
                        output_per_steps,
                        softmax_aux_per_steps,
                        rng_state,
                    ),
                    xs=jnp.arange(0, cp_size),
                )
            )

            output = jnp.zeros(q.shape).astype(jnp.float32)
            for idx in range(cp_size):
                output = output + output_per_steps[idx].astype(jnp.float32) * jnp.exp(
                    softmax_aux_per_steps[idx] - softmax_aux
                ).transpose(0, 2, 1, 3)
            output = output.astype(q.dtype)
            return output, softmax_aux, rng_state

        return mesh, ring_attn_fwd_impl, out_shardings, arg_shardings


register_primitive(FusedRingAttnFwdPrimitive)


class FusedRingAttnBwdPrimitive(FusedAttnBwdPrimitive):
    """
    Fused Ring Attention Backward Primitive
    """

    name = "te_fused_ring_attn_backward"
    multiple_results = True
    impl_static_args = (12, 13, 14, 15, 16, 17, 18)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def partition(
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
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

        assert dropout_probability <= 0, "Ring Attention currently does not support dropout."
        assert (
            attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS
        ), "Ring Attention currently only support no_bias."
        assert (
            qkv_layout == NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
        ), "Ring Attention currently only support BSHD_BSHD_BSHD QKV layout."

        # Assume index-1 is the sequence dimension
        cp_mesh_axis_name = q_spec[1]

        partial_fmha_bwd_causal_mask_impl = partial(
            FusedAttnBwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        partial_fmha_bwd_no_mask_impl = partial(
            FusedAttnBwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=NVTE_Mask_Type.NVTE_NO_MASK,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        partial_fmha_bwd_regular_mask_impl = partial(
            FusedAttnBwdPrimitive.impl,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
        )

        def ring_attn_bwd_impl(
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
            q_seq_offsets,
            k_seq_offsets,
        ):
            _, q_max_seqlen, _, _ = q.shape
            _, kv_max_seqlen, _, _ = k.shape

            cp_rank = lax.axis_index(cp_mesh_axis_name)
            cp_size = get_mesh_axis_size(cp_mesh_axis_name)

            dq = jnp.zeros(q.shape, dtype=q.dtype)
            dk = jnp.zeros(k.shape, dtype=k.dtype)
            dv = jnp.zeros(v.shape, dtype=v.dtype)
            dbias = jnp.zeros(bias.shape, dtype=bias.dtype)

            def scan_kv_block(carry, idx):

                k_curr, v_curr, dq, dk, dv, dbias = carry

                def causal_mask_compute():
                    q_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = (
                        partial_fmha_bwd_causal_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            softmax_aux,
                            rng_state,
                            output,
                            doutput,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                        )
                    )
                    return dq_per_step, dk_per_step, dv_per_step, dbias_per_step

                def half_kv_no_mask_compute():
                    q_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = (
                        FusedRingAttnBwdPrimitive.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx) // 2
                    )
                    k_part = k_curr[:, : kv_max_seqlen // 2, :, :]
                    v_part = v_curr[:, : kv_max_seqlen // 2, :, :]
                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = (
                        partial_fmha_bwd_no_mask_impl(
                            q,
                            k_part,
                            v_part,
                            bias,
                            softmax_aux,
                            rng_state,
                            output,
                            doutput,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                        )
                    )
                    dk_per_step = jnp.concat(
                        [dk_per_step, jnp.zeros(k_part.shape, dtype=k_part.dtype)], axis=1
                    )
                    dv_per_step = jnp.concat(
                        [dv_per_step, jnp.zeros(v_part.shape, dtype=v_part.dtype)], axis=1
                    )
                    return dq_per_step, dk_per_step, dv_per_step, dbias_per_step

                def half_q_no_mask_compute():
                    q_seqlen_per_step = (
                        FusedRingAttnBwdPrimitive.adjust_seqlen(q_seqlen, q_max_seqlen, idx) // 2
                    )
                    kv_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    doutput_part = doutput[:, q_max_seqlen // 2 :, :, :]
                    output_part = output[:, q_max_seqlen // 2 :, :, :]
                    softmax_aux_part = softmax_aux[:, :, q_max_seqlen // 2 :, 1]
                    q_part = q[:, q_max_seqlen // 2 :, :, :]
                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = (
                        partial_fmha_bwd_no_mask_impl(
                            q_part,
                            k_curr,
                            v_curr,
                            bias,
                            softmax_aux_part,
                            rng_state,
                            output_part,
                            doutput_part,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                        )
                    )
                    dq_per_step = jnp.concat(
                        [jnp.zeros(q_part.shape, dtype=q_part.dtype), dq_per_step], axis=1
                    )
                    return dq_per_step, dk_per_step, dv_per_step, dbias_per_step

                def no_mask_compute():
                    q_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )
                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = (
                        partial_fmha_bwd_no_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            softmax_aux,
                            rng_state,
                            output,
                            doutput,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                        )
                    )
                    return dq_per_step, dk_per_step, dv_per_step, dbias_per_step

                def skip_compute():
                    dq_per_step = jnp.zeros(q.shape, dtype=q.dtype)
                    dk_per_step = jnp.zeros(k.shape, dtype=k.dtype)
                    dv_per_step = jnp.zeros(v.shape, dtype=v.dtype)
                    dbias_per_step = jnp.zeros(bias.shape, dtype=bias.dtype)
                    return dq_per_step, dk_per_step, dv_per_step, dbias_per_step

                if attn_mask_type == NVTE_Mask_Type.NVTE_CAUSAL_MASK:
                    # This is for nested jax.lax.cond
                    def jax_cond_wrap():
                        if _FusedRingAttnStatusTracker.is_workload_balanced():
                            return lax.cond(
                                (idx <= cp_rank), half_kv_no_mask_compute, half_q_no_mask_compute
                            )
                        return lax.cond((idx <= cp_rank), no_mask_compute, skip_compute)

                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = lax.cond(
                        idx == 0, causal_mask_compute, jax_cond_wrap
                    )
                else:
                    q_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        q_seqlen, q_max_seqlen, idx
                    )
                    kv_seqlen_per_step = FusedRingAttnBwdPrimitive.adjust_seqlen(
                        kv_seqlen, kv_max_seqlen, idx
                    )

                    dq_per_step, dk_per_step, dv_per_step, dbias_per_step = (
                        partial_fmha_bwd_regular_mask_impl(
                            q,
                            k_curr,
                            v_curr,
                            bias,
                            softmax_aux,
                            rng_state,
                            output,
                            doutput,
                            q_seqlen_per_step,
                            kv_seqlen_per_step,
                            q_seq_offsets,
                            k_seq_offsets,
                        )
                    )

                dq = dq + dq_per_step
                dk = dk + dk_per_step
                dv = dv + dv_per_step
                dbias = dbias + dbias_per_step

                k_curr = lax_paral_op(
                    k_curr,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )
                v_curr = lax_paral_op(
                    v_curr,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )
                dk = lax_paral_op(
                    dk,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )
                dv = lax_paral_op(
                    dv,
                    lax.ppermute,
                    cp_mesh_axis_name,
                    perm=[(i, (i + 1) % cp_size) for i in range(cp_size)],
                )

                return (k_curr, v_curr, dq, dk, dv, dbias), None

            k_curr = k
            v_curr = v

            (k_curr, v_curr, dq, dk, dv, dbias), _ = lax.scan(
                scan_kv_block, init=(k_curr, v_curr, dq, dk, dv, dbias), xs=jnp.arange(0, cp_size)
            )

            global_dbias = dbias
            if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(dbias)
            return dq, dk, dv, global_dbias

        return mesh, ring_attn_bwd_impl, out_shardings, arg_shardings


register_primitive(FusedAttnBwdPrimitive)


def fused_ring_attn_fwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    q_seq_offsets: Optional[jnp.ndarray],
    kv_seq_offsets: Optional[jnp.ndarray],
    seed: Optional[jnp.ndarray],
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
) -> jnp.ndarray:

    assert dropout_probability <= 0, "Ring Attention currently does not support dropout."
    assert (
        attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS
    ), "Ring Attention currently only support no_bias."
    assert (
        qkv_layout == NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
    ), "Ring Attention currently only support BSHD_BSHD_BSHD QKV layout."

    seed = _FusedAttnRNGStateChecker().check_seed(seed, dropout_probability, is_training)

    assert (q_seq_offsets is None) == (
        kv_seq_offsets is None
    ), "Both q_seq_offsets and kv_seq_offsets must be either None or have values."

    # For optional tensors, which custom calls doesn't support None
    _not_used = jnp.zeros(0, dtype=qkv[0].dtype)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv[0].dtype)

    return FusedRingAttnFwdPrimitive.outer_primitive.bind(
        *qkv,
        bias,
        q_seqlen,
        kv_seqlen,
        _not_used,
        _not_used,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
    )


def fused_ring_attn_bwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    q_seqlen: jnp.ndarray,
    kv_seqlen: jnp.ndarray,
    q_seq_offsets: Optional[jnp.ndarray],
    kv_seq_offsets: Optional[jnp.ndarray],
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
):

    assert dropout_probability <= 0, "Ring Attention currently does not support dropout."
    assert (
        attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS
    ), "Ring Attention currently only support no_bias."
    assert (
        qkv_layout == NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
    ), "Ring Attention currently only support BSHD_BSHD_BSHD QKV layout."

    assert (q_seq_offsets is None) == (
        kv_seq_offsets is None
    ), "Both q_seq_offsets and kv_seq_offsets must be either None or have values."

    # For optional tensors, which custom calls doesn't support None
    _not_used = jnp.zeros(0, dtype=qkv[0].dtype)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv[0].dtype)

    *qkv_grads, bias_grad = FusedRingAttnBwdPrimitive.outer_primitive.bind(
        *qkv,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        _not_used,
        _not_used,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
    )
    return tuple(qkv_grads[: len(qkv)]), bias_grad
