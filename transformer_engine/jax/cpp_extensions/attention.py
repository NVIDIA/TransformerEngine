# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for attention"""
from dataclasses import dataclass
from functools import partial, reduce
import operator
import os
from typing import Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
from jax import dtypes, lax
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding
from jax import ffi

import transformer_engine_jax
from transformer_engine_jax import (
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_QKV_Layout,
    NVTE_QKV_Format,
    NVTE_Fused_Attn_Backend,
    nvte_get_qkv_format,
)
from transformer_engine.jax.attention import CPStrategy, SequenceDescriptor

from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    te_dtype_to_jax_dtype,
    get_padded_spec,
    get_cudnn_version,
    is_ffi_enabled,
    get_xla_flag,
)
from ..sharding import (
    global_mesh_resource,
    lax_paral_op,
    all_reduce_sum_along_dp_fsdp,
    get_mesh_axis_size,
    get_mesh_axis_rank,
    get_all_mesh_axes,
    num_of_devices,
)


__all__ = [
    "FusedAttnHelper",
    "fused_attn_fwd",
    "fused_attn_bwd",
]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[
        "attn_bias_type",
        "attn_mask_type",
        "qkv_layout",
        "scaling_factor",
        "dropout_probability",
        "is_training",
        "max_segments_per_seq",
        "window_size",
        "context_parallel_load_balanced",
        "cp_axis",
    ],
)
@dataclass(frozen=True)
class _FusedAttnConfig:
    """
    Passes static configuration properties of fused attention.
    """

    attn_bias_type: NVTE_Bias_Type
    attn_mask_type: NVTE_Mask_Type
    qkv_layout: NVTE_QKV_Layout
    scaling_factor: float
    dropout_probability: float
    is_training: bool
    max_segments_per_seq: int
    window_size: Tuple[int, int]
    context_parallel_load_balanced: bool
    cp_axis: str


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
    window_size: Tuple[int, int]

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
            self.window_size[0],
            self.window_size[1],
        )

    @staticmethod
    def is_non_deterministic_allowed():
        """Check if non-deterministic kernels are allowed"""
        return bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

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
    actual_seqlen = jnp.where(actual_seqlen < 0, 0, actual_seqlen)
    cu_seqlen = jnp.cumulative_sum(actual_seqlen, include_initial=True)
    return cu_seqlen


class FusedAttnFwdPrimitive(BasePrimitive):
    """
    Fused Attention Forward Primitive
    """

    name = "te_fused_attn_forward"
    multiple_results = True
    impl_static_args = (13,)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        q_aval,
        k_aval,
        v_aval,
        bias_aval,
        seed_aval,
        q_seqlen_or_cu_seqlen_aval,
        kv_seqlen_or_cu_seqlen_aval,
        _q_seq_offsets,
        _k_seq_offsets,
        _q_segment_ids,
        _kv_segment_ids,
        _q_segment_pos,
        _kv_segment_pos,
        *,
        config: _FusedAttnConfig,
    ):
        """
        Fused attention fwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q_aval.dtype)
        k_dtype = dtypes.canonicalize_dtype(k_aval.dtype)
        v_dtype = dtypes.canonicalize_dtype(v_aval.dtype)
        bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        assert (
            q_dtype == k_dtype == v_dtype == bias_dtype
        ), f"q_dtype={q_dtype}, k_dtype={k_dtype}, v_dtype={v_dtype}, bias_dtype={bias_dtype}"
        assert q_seqlen_or_cu_seqlen_aval.dtype == kv_seqlen_or_cu_seqlen_aval.dtype, (
            f"q_seqlen_or_cu_seqlen_aval={q_seqlen_or_cu_seqlen_aval},"
            f" kv_seqlen_or_cu_seqlen_aval={kv_seqlen_or_cu_seqlen_aval}"
        )

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, config.qkv_layout)
        )

        output_shape = (*batch_shape, q_max_seqlen, attn_heads, head_dim)
        out_aval = q_aval.update(shape=output_shape, dtype=q_dtype)

        # backend determines the softmax buffer shape/dtype
        backend = FusedAttnHelper(
            q_dtype,
            k_dtype,
            config.qkv_layout,
            config.attn_bias_type,
            config.attn_mask_type,
            config.dropout_probability,
            attn_heads,
            num_gqa_groups,
            q_max_seqlen,
            kv_max_seqlen,
            head_dim,
            config.window_size,
        ).get_fused_attn_backend()

        if backend == NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen:
            softmax_shape = (*batch_shape, attn_heads, q_max_seqlen, kv_max_seqlen)
            softmax_dtype = q_dtype
        elif backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            # cuDNN 9.6 reduces the required softmax shape
            if get_cudnn_version() >= (9, 6, 0):
                softmax_shape = (*batch_shape, attn_heads, q_max_seqlen, 1)
            else:
                softmax_shape = (
                    *batch_shape,
                    attn_heads,
                    q_max_seqlen,
                    config.max_segments_per_seq,
                )
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

        if config.attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
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
            config.scaling_factor,
            config.dropout_probability,
            config.attn_bias_type,
            config.attn_mask_type,
            config.qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            config.is_training,
            config.max_segments_per_seq,
            config.window_size[0],
            config.window_size[1],
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
        seed,
        q_cu_seqlen,
        kv_cu_seqlen,
        q_seq_offsets,
        k_seq_offsets,
        _q_segment_ids,
        _kv_segment_ids,
        _q_segment_pos,
        _kv_segment_pos,
        *,
        config: _FusedAttnConfig,
    ):
        """
        Fused attention fwd lowering rules
        """
        q_aval, k_aval, v_aval, bias_aval, *_ = ctx.avals_in

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, config.qkv_layout)
        )

        input_batch = reduce(operator.mul, batch_shape)

        if config.attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        if is_ffi_enabled():
            name = "te_fused_attn_forward_ffi"
            out = ffi.ffi_lowering(name)(
                ctx,
                q,
                k,
                v,
                bias,
                seed,
                q_cu_seqlen,
                kv_cu_seqlen,
                q_seq_offsets,
                k_seq_offsets,
                _q_segment_ids,
                _kv_segment_ids,
                _q_segment_pos,
                _kv_segment_pos,  # ffi_lowering needs number of parameters meets primitive.lowering
                input_batch=input_batch,
                bias_batch=bias_batch,
                q_max_seqlen=q_max_seqlen,
                kv_max_seqlen=kv_max_seqlen,
                attn_heads=attn_heads,
                num_gqa_groups=num_gqa_groups,
                bias_heads=bias_heads,
                head_dim=head_dim,
                max_segments_per_seq=config.max_segments_per_seq,
                scaling_factor=float(config.scaling_factor),
                dropout_probability=float(config.dropout_probability),
                bias_type=int(config.attn_bias_type),
                mask_type=int(config.attn_mask_type),
                qkv_layout=int(config.qkv_layout),
                is_training=config.is_training,
                deterministic=not FusedAttnHelper.is_non_deterministic_allowed(),
                window_size_left=config.window_size[0],
                window_size_right=config.window_size[1],
            )
        else:
            operands = [
                q,
                k,
                v,
                bias,
                seed,
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
                config.max_segments_per_seq,
                wkspace_aval.size,
                config.scaling_factor,
                config.dropout_probability,
                config.attn_bias_type,
                config.attn_mask_type,
                config.qkv_layout,
                jax_dtype_to_te_dtype(q_aval.dtype),
                jax_dtype_to_te_dtype(wkspace_aval.dtype),
                config.is_training,
                not FusedAttnHelper.is_non_deterministic_allowed(),
                config.window_size[0],
                config.window_size[1],
            )

            out = custom_caller(FusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(
        q,
        k,
        v,
        bias,
        seed,
        q_seqlen,
        kv_seqlen,
        q_seq_offsets,
        k_seq_offsets,
        _q_segment_ids,
        _kv_segment_ids,
        _q_segment_pos,
        _kv_segment_pos,
        config: _FusedAttnConfig,
    ):
        assert FusedAttnFwdPrimitive.inner_primitive is not None

        sequence_descriptor = SequenceDescriptor(
            seqlens=(q_seqlen, kv_seqlen),
            seq_offsets=(q_seq_offsets, k_seq_offsets),
            segment_ids=(_q_segment_ids, _kv_segment_ids),
            segment_pos=(_q_segment_pos, _kv_segment_pos),
        )

        (q_seqlen, kv_seqlen), (q_seq_offsets, k_seq_offsets) = (
            sequence_descriptor.get_seqlens_and_offsets(
                config.attn_mask_type,
                config.qkv_layout,
                config.window_size,
                config.max_segments_per_seq,
            )
        )

        if nvte_get_qkv_format(config.qkv_layout) == NVTE_QKV_Format.NVTE_THD:

            def _fix_len_take(x, condition, fill_value=-1):
                x_shape = x.shape
                x = x.flatten()
                size = x.size
                indices = jnp.nonzero(condition.flatten(), size=size, fill_value=size)[0]
                y = jnp.take(x, indices, fill_value=fill_value)
                return jnp.reshape(y, x_shape)

            def convert_to_2d(offsets, batch, max_seqlen):
                offsets_2d = jnp.where(
                    offsets >= 0,
                    offsets + (jnp.arange(batch) * max_seqlen)[..., jnp.newaxis],
                    offsets,
                )
                return offsets_2d

            match config.qkv_layout:
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
            # cuDNN version < 9.3.0:
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, -1, -1, -1, -1]]
            # cuDNN version >= 9.3.0, which supports act_seqlen = 0
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, 0, 0, 0, 0]]
            if get_cudnn_version() >= (9, 3, 0):
                fill_value = 0
            else:
                fill_value = -1

            q_seqlen = _fix_len_take(q_seqlen, q_seqlen > 0, fill_value=fill_value)
            kv_seqlen = _fix_len_take(kv_seqlen, kv_seqlen > 0, fill_value=fill_value)

            # Flatten the offset calculation
            # max_seqlen = 8, [[0, 3, 5, -1], [0, 2, 4, -1]] -> [[0, 3, 5, -1], [8, 11, 13, -1]]
            q_seq_offsets = convert_to_2d(q_seq_offsets, q_batch, q_max_seqlen)
            k_seq_offsets = convert_to_2d(k_seq_offsets, kv_batch, kv_max_seqlen)

            # Gather valid q_seq_offsets, which is greater and equal to 0
            # [[0, 3, 5, -1], [8, 11, 13, -1]] -> [[0, 3, 5, 8], [11, 13, -1, -1]]
            # And set the unused position to max size (batch * max_seqlen)
            # [[0, 3, 5, 8], [11, 13, -1, -1]] -> [[0, 3, 5, 8], [11, 13, b*s, b*s]]
            q_seq_offsets = _fix_len_take(
                q_seq_offsets, q_seq_offsets >= 0, fill_value=q_batch * q_max_seqlen
            )
            k_seq_offsets = _fix_len_take(
                k_seq_offsets, k_seq_offsets >= 0, fill_value=kv_batch * kv_max_seqlen
            )

        q_cu_seqlen = generate_cu_seqlen(q_seqlen.flatten())
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen.flatten())

        output, softmax_aux, rng_state, _ = FusedAttnFwdPrimitive.inner_primitive.bind(
            q,
            k,
            v,
            bias,
            seed,
            q_cu_seqlen,
            kv_cu_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
            config=config,
        )
        return output, softmax_aux, rng_state

    @staticmethod
    def batcher(batched_args, batch_dims, *, config):
        check_valid_batch_dims(batch_dims)
        assert FusedAttnFwdPrimitive.outer_primitive is not None
        q_bdim, _, _, _, seed_bdim, *_ = batch_dims

        out_bdims = q_bdim, q_bdim, seed_bdim
        return (
            FusedAttnFwdPrimitive.outer_primitive.bind(*batched_args, config=config),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(config, mesh, arg_infos, result_infos):
        del result_infos
        q_spec = get_padded_spec(arg_infos[0])
        match config.qkv_layout:
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
                    mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], None)
                )
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD | NVTE_QKV_Layout.NVTE_THD_THD_THD:
                # q_spec = (...batch, q_seqlen, head, hidden)
                # k_spec = (...batch, kv_seqlen, num_gqa_groups, hidden)
                out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
                softmax_aux_sharding = NamedSharding(
                    mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], None)
                )
            case _:
                raise ValueError(f"Unsupported {config.qkv_layout=}")
        rng_state_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        return (out_sharding, softmax_aux_sharding, rng_state_sharding)

    @staticmethod
    def partition(config, mesh, arg_infos, result_infos):
        out_sharding = result_infos[0].sharding
        softmax_aux_sharding = result_infos[1].sharding
        rng_state_sharding = seed_sharding = NamedSharding(
            mesh, PartitionSpec(get_all_mesh_axes(), None)
        )
        arg_shardings = [arg_i.sharding for arg_i in arg_infos]
        arg_shardings[4] = seed_sharding
        arg_shardings[-1] = arg_shardings[-3]
        arg_shardings[-2] = arg_shardings[-4]
        arg_shardings = tuple(arg_shardings)
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)
        impl = partial(FusedAttnFwdPrimitive.impl, config=config)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(FusedAttnFwdPrimitive)


class FusedAttnBwdPrimitive(BasePrimitive):
    """
    Fused Attention Backward Primitive
    """

    name = "te_fused_attn_backward"
    multiple_results = True
    impl_static_args = (16,)
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
        _q_segment_ids,
        _kv_segment_ids,
        _q_segment_pos,
        _kv_segment_pos,
        *,
        config,
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
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, config.qkv_layout)
        )

        if config.attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        deterministic = not FusedAttnHelper.is_non_deterministic_allowed()

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
            config.scaling_factor,
            config.dropout_probability,
            config.attn_bias_type,
            config.attn_mask_type,
            config.qkv_layout,
            jax_dtype_to_te_dtype(q_aval.dtype),
            config.is_training,
            deterministic,
            config.max_segments_per_seq,
            config.window_size[0],
            config.window_size[1],
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
        q_segment_ids,
        kv_segment_ids,
        q_segment_pos,
        kv_segment_pos,
        *,
        config,
    ):
        """
        Fused attention bwd lowering rules
        """
        q_aval, k_aval, v_aval, bias_aval, *_ = ctx.avals_in

        batch_shape, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, head_dim = (
            FusedAttnHelper.parse_qkv_aval(q_aval, k_aval, v_aval, config.qkv_layout)
        )

        input_batch = reduce(operator.mul, batch_shape)

        if config.attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_batch = bias_heads = 0
        else:
            *bias_batch_shape, bias_heads, _, _ = bias_aval.shape
            bias_batch = reduce(operator.mul, bias_batch_shape)

        if is_ffi_enabled():
            name = "te_fused_attn_backward_ffi"
            out = ffi.ffi_lowering(name)(
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
                q_segment_ids,
                kv_segment_ids,
                q_segment_pos,
                kv_segment_pos,  # ffi_lowering needs number of parameters meets primitive.lowering
                input_batch=input_batch,
                bias_batch=bias_batch,
                q_max_seqlen=q_max_seqlen,
                kv_max_seqlen=kv_max_seqlen,
                attn_heads=attn_heads,
                num_gqa_groups=num_gqa_groups,
                bias_heads=bias_heads,
                head_dim=head_dim,
                max_segments_per_seq=config.max_segments_per_seq,
                scaling_factor=float(config.scaling_factor),
                dropout_probability=float(config.dropout_probability),
                bias_type=int(config.attn_bias_type),
                mask_type=int(config.attn_mask_type),
                qkv_layout=int(config.qkv_layout),
                is_training=config.is_training,
                deterministic=not FusedAttnHelper.is_non_deterministic_allowed(),
                window_size_left=config.window_size[0],
                window_size_right=config.window_size[1],
            )
        else:
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
                config.max_segments_per_seq,
                wkspace_aval.size,
                config.scaling_factor,
                config.dropout_probability,
                config.attn_bias_type,
                config.attn_mask_type,
                config.qkv_layout,
                jax_dtype_to_te_dtype(q_aval.dtype),
                jax_dtype_to_te_dtype(wkspace_aval.dtype),
                config.is_training,
                not FusedAttnHelper.is_non_deterministic_allowed(),
                config.window_size[0],
                config.window_size[1],
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
        _q_segment_ids,
        _kv_segment_ids,
        _q_segment_pos,
        _kv_segment_pos,
        config,
    ):
        assert FusedAttnBwdPrimitive.inner_primitive is not None

        sequence_descriptor = SequenceDescriptor(
            seqlens=(q_seqlen, kv_seqlen),
            seq_offsets=(q_seq_offsets, k_seq_offsets),
            segment_ids=(_q_segment_ids, _kv_segment_ids),
            segment_pos=(_q_segment_pos, _kv_segment_pos),
        )

        (q_seqlen, kv_seqlen), (q_seq_offsets, k_seq_offsets) = (
            sequence_descriptor.get_seqlens_and_offsets(
                config.attn_mask_type,
                config.qkv_layout,
                config.window_size,
                config.max_segments_per_seq,
            )
        )

        if nvte_get_qkv_format(config.qkv_layout) == NVTE_QKV_Format.NVTE_THD:

            def _fix_len_take(x, condition, fill_value=-1):
                x_shape = x.shape
                x = x.flatten()
                size = x.size
                indices = jnp.nonzero(condition.flatten(), size=size, fill_value=size)[0]
                # TODO(rewang): try indices_are_sorted
                y = jnp.take(x, indices, fill_value=fill_value)
                return jnp.reshape(y, x_shape)

            def convert_to_2d(offsets, batch, max_seqlen):
                offsets_2d = jnp.where(
                    offsets >= 0,
                    offsets + (jnp.arange(batch) * max_seqlen)[..., jnp.newaxis],
                    offsets,
                )
                return offsets_2d

            match config.qkv_layout:
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
            # cuDNN version < 9.3.0:
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, -1, -1, -1, -1]]
            # cuDNN version >= 9.3.0, which supports act_seqlen = 0
            # [[3, 5, 7, -1, -1], [2, 4, 6, -1, -1]] -> [[3, 5, 7, 2, 4], [6, 0, 0, 0, 0]]
            if get_cudnn_version() >= (9, 3, 0):
                fill_value = 0
            else:
                fill_value = -1
            q_seqlen = _fix_len_take(q_seqlen, q_seqlen > 0, fill_value=fill_value)
            kv_seqlen = _fix_len_take(kv_seqlen, kv_seqlen > 0, fill_value=fill_value)

            # Flatten the offset calculation
            # max_seqlen = 8, [[0, 3, 5, -1], [0, 2, 4, -1]] -> [[0, 3, 5, -1], [8, 11, 13, -1]]
            q_seq_offsets = convert_to_2d(q_seq_offsets, q_batch, q_max_seqlen)
            k_seq_offsets = convert_to_2d(k_seq_offsets, kv_batch, kv_max_seqlen)

            # Gather valid q_seq_offsets, which is greater and equal to 0
            # [[0, 3, 5, -1], [8, 11, 13, -1]] -> [[0, 3, 5, 8], [11, 13, -1, -1]]
            # And set the unused position to max size (batch * max_seqlen)
            # [[0, 3, 5, 8], [11, 13, -1, -1]] -> [[0, 3, 5, 8], [11, 13, b*s, b*s]]
            q_seq_offsets = _fix_len_take(
                q_seq_offsets, q_seq_offsets >= 0, fill_value=q_batch * q_max_seqlen
            )
            k_seq_offsets = _fix_len_take(
                k_seq_offsets, k_seq_offsets >= 0, fill_value=kv_batch * kv_max_seqlen
            )

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
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
            config=config,
        )
        return dq, dk, dv, dbias

    @staticmethod
    def batcher(batched_args, batch_dims, *, config):
        check_valid_batch_dims(batch_dims)
        assert FusedAttnBwdPrimitive.outer_primitive is not None
        q_bdim, k_bdim, v_bdim, *_ = batch_dims

        out_bdims = q_bdim, k_bdim, v_bdim, q_bdim
        return (
            FusedAttnBwdPrimitive.outer_primitive.bind(*batched_args, config=config),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(config, mesh, arg_infos, result_infos):
        del config, result_infos
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
    def partition(config, mesh, arg_infos, result_infos):
        del result_infos
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        v_spec = get_padded_spec(arg_infos[2])
        bias_spec = get_padded_spec(arg_infos[3])
        dq_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        dk_sharding = NamedSharding(mesh, PartitionSpec(*k_spec))
        dv_sharding = NamedSharding(mesh, PartitionSpec(*v_spec))
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))
        arg_shardings = [arg_i.sharding for arg_i in arg_infos]
        arg_shardings[-1] = arg_shardings[-3]
        arg_shardings[-2] = arg_shardings[-4]
        arg_shardings = tuple(arg_shardings)
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
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
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
                _q_segment_ids,
                _kv_segment_ids,
                _q_segment_pos,
                _kv_segment_pos,
                config=config,
            )
            global_dbias = local_dbias
            if config.attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias, mesh)
            return local_dq, local_dk, local_dv, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(FusedAttnBwdPrimitive)


def reorder_causal_load_balancing(tensor, cp_size: int, seq_dim: int, to_contiguous: bool):
    """Reorders a tensor for load balancing the compute of causal attention."""
    if cp_size == 1:
        return tensor

    if cp_size % 2 != 0:
        raise ValueError(f"{cp_size=} must be a multiple of 2.")

    # Need to ensure we have 2 pairs to swap for balancing between cp ranks
    if tensor.shape[seq_dim] % (cp_size * 2) != 0:
        raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

    # [B, S, H, D] -> [B, 2*cp_size, S/2*cp_size, D]
    # [S, B, H, D] -> [2*cp_size, S/2*cp_size, B, H, D]
    ori_tensor_shape = tensor.shape
    tensor = tensor.reshape(
        (
            *ori_tensor_shape[:seq_dim],
            2 * cp_size,
            ori_tensor_shape[seq_dim] // (2 * cp_size),
            *ori_tensor_shape[seq_dim + 1 :],
        )
    )

    parts = []
    if not to_contiguous:
        for cp_rank in range(cp_size):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            index = jnp.array([cp_rank, (2 * cp_size - cp_rank - 1)])
            parts.append(jnp.take(tensor, index, axis=seq_dim))
    else:
        for cp_rank in range(cp_size // 2):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            base = 4 * cp_rank
            index = jnp.array([base, base + 2])
            parts.append(jnp.take(tensor, index, axis=seq_dim))
        for cp_rank in range(cp_size // 2):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            base = 2 * cp_size - 1 - 4 * cp_rank
            index = jnp.array([base, base - 2])
            parts.append(jnp.take(tensor, index, axis=seq_dim))

    # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D]
    # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D]
    combined = jnp.stack(parts, axis=seq_dim)

    return combined.reshape(ori_tensor_shape)


@dataclass(frozen=True)
class _FusedAttnCPWithAllGatherHelper:
    """Helper class to assist with running the all-gather strategy for CP attention."""

    mesh: jax.sharding.Mesh
    config: _FusedAttnConfig

    def check_supported(self):
        """Checks if the context parallel implementation is supported by the given arguments."""
        header = "Context parallel fused attention"

        allowed_layouts = [NVTE_QKV_Layout.NVTE_BSHD_BS2HD, NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD]
        if self.config.qkv_layout not in allowed_layouts:
            raise ValueError(
                f"{header} only supports layouts:"
                f" {','.join(map(str, allowed_layouts))} got: {self.config.qkv_layout}"
            )

        if self.config.attn_bias_type != NVTE_Bias_Type.NVTE_NO_BIAS:
            raise ValueError(f"{header} does not support bias got: {self.config.attn_bias_type}")

        allowed_masks = [NVTE_Mask_Type.NVTE_NO_MASK, NVTE_Mask_Type.NVTE_CAUSAL_MASK]
        if self.config.attn_mask_type not in allowed_masks:
            raise ValueError(
                f"{header} only supports masking types: "
                f" {','.join(map(str, allowed_masks))} got: {self.config.attn_mask_type}"
            )

        if self.config.max_segments_per_seq != 1:
            raise ValueError(
                f"{header} only supports max_segments_per_seq == 1 got:"
                f" {self.config.max_segments_per_seq}"
            )

        if self.config.dropout_probability != 0.0:
            raise ValueError(f"{header} does not support dropout")

    def get_adjusted_mask(self):
        """Converts the mask for context parallelism."""
        if self.config.attn_mask_type == NVTE_Mask_Type.NVTE_CAUSAL_MASK:
            return NVTE_Mask_Type.NVTE_CAUSAL_BOTTOM_RIGHT_MASK
        return self.config.attn_mask_type

    def get_step_config(self) -> _FusedAttnConfig:
        """Returns a _FusedAttnConfig for single CP step call to fused attention."""
        return _FusedAttnConfig(
            attn_bias_type=self.config.attn_bias_type,
            attn_mask_type=self.get_adjusted_mask(),
            qkv_layout=self.config.qkv_layout,
            scaling_factor=self.config.scaling_factor,
            dropout_probability=self.config.dropout_probability,
            is_training=self.config.is_training,
            max_segments_per_seq=self.config.max_segments_per_seq,
            window_size=self.config.window_size,
            context_parallel_load_balanced=self.config.context_parallel_load_balanced,
            cp_axis=self.config.cp_axis,
        )

    def all_gather_kv(self, k, v):
        """Performs a all-gather of k and v over context parallel ranks."""

        def ag(x):
            x = lax_paral_op(
                x, lax.all_gather, self.config.cp_axis, mesh=self.mesh, axis=1, tiled=True
            )
            if self.config.context_parallel_load_balanced:
                cp_size = get_mesh_axis_size(self.config.cp_axis, self.mesh)
                x = reorder_causal_load_balancing(x, cp_size, 1, to_contiguous=True)
            return x

        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                return ag(k), v
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                return ag(k), ag(v)

        return k, v  # fall through

    def reduce_scatter_dkv(self, dk, dv):
        """Performs a reduce-scatter of dk and dv over context parallel ranks."""

        def rs(x):
            if self.config.context_parallel_load_balanced:
                cp_size = get_mesh_axis_size(self.config.cp_axis, self.mesh)
                x = reorder_causal_load_balancing(x, cp_size, 1, to_contiguous=False)

            return lax_paral_op(
                x,
                lax.psum_scatter,
                self.config.cp_axis,
                mesh=self.mesh,
                scatter_dimension=1,
                tiled=True,
            )

        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                return rs(dk), dv
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                return rs(dk), rs(dv)

        return dk, dv  # fall through

    def kv_seqlens_for_rank(self, cp_rank, kv_max_seqlen, kv_seqlen_per_subrank):
        """Returns sequence lengths of KV to use for each sub rank of the given cp_rank.

        Example: CP=4, MaxLen = 1024, Unbalanced
           cp_rank 0: [128, 256]
           cp_rank 1: [384, 512]
           cp_rank 2: [640, 768]
           cp_rank 3: [896, 1024]

        Example: CP=4, MaxLen = 1024, Balanced
           cp_rank 0: [128, 1024]
           cp_rank 1: [256, 896]
           cp_rank 2: [384, 768]
           cp_rank 3: [512, 640]
        """
        if self.config.context_parallel_load_balanced:
            kv_seq_this_rank = [
                (cp_rank + 1) * kv_seqlen_per_subrank,
                kv_max_seqlen - cp_rank * kv_seqlen_per_subrank,
            ]
        else:
            kv_seq_this_rank = [
                (cp_rank * 2 + 1) * kv_seqlen_per_subrank,
                (cp_rank * 2 + 2) * kv_seqlen_per_subrank,
            ]
        return kv_seq_this_rank

    def slice_kv(self, k, v, slice_seq_len):
        """Slices k and v tensors to a sequence length of slice_seq_len."""

        def sliced(x):
            return lax.dynamic_slice_in_dim(x, 0, slice_seq_len, axis=1)

        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                return sliced(k), v
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                return sliced(k), sliced(v)

        return k, v  # fall through

    def pad_kv(self, dk, dv, pad_seq_len):
        """Pads dk and dv tensors to a sequence length of pad_seq_len."""

        def pad(x, npad):
            return jnp.pad(x, npad, "constant", constant_values=0.0)

        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                npad = [[0, 0], [0, pad_seq_len], [0, 0], [0, 0], [0, 0]]
                return pad(dk, npad), dv
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                npad = [[0, 0], [0, pad_seq_len], [0, 0], [0, 0]]
                return pad(dk, npad), pad(dv, npad)

        return dk, dv  # fall through


class FusedAttnCPWithAllGatherFwdPrimitive(FusedAttnFwdPrimitive):
    """
    Fused Attention Forward with Context Parallelism Primitive

    This context parallel implementation uses all-gather to collect KV inputs from context parallel ranks.
    """

    @staticmethod
    def partition(config, mesh, arg_infos, result_infos):
        # Call base implementation for non-context parallel mesh to avoid unecessary work.
        is_context_parallel = get_mesh_axis_size(config.cp_axis, mesh) > 1
        assert (
            not is_context_parallel or config.window_size[0] == -1
        ), "Sliding window attention is not supported when context parallelism is enabled"
        if not is_context_parallel:
            return FusedAttnFwdPrimitive.partition(config, mesh, arg_infos, result_infos)

        helper = _FusedAttnCPWithAllGatherHelper(mesh, config)
        helper.check_supported()

        out_sharding = result_infos[0].sharding
        softmax_aux_sharding = result_infos[1].sharding
        rng_state_sharding = seed_sharding = NamedSharding(
            mesh, PartitionSpec(get_all_mesh_axes(), None)
        )
        arg_shardings = [arg_i.sharding for arg_i in arg_infos]
        arg_shardings[4] = seed_sharding
        arg_shardings = tuple(arg_shardings)
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)

        def impl(
            q,
            k,
            v,
            bias,
            seed,
            q_seqlen,
            kv_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
        ):
            cp_size = get_mesh_axis_size(config.cp_axis, mesh)
            cp_rank = get_mesh_axis_rank(config.cp_axis, mesh)

            # cuDNN does not support right-aligned masking with dynamic sequence length padding.
            # Therefore we must explicitly instantiate each CP rank slicing and use a runtime switch
            # to select the appropriate computation. Each case generates a [..., SEQ/CP, ..] tensor
            # meeting the expectation of the SPMD model.
            # TODO(mgoldfarb-nvidia): When cuDNN supports we should be able to make use of a padding
            # mask/sequence length tensor to avoid this unrolled loop.
            def _cross_attn(idx, q, k, v, bias, q_seqlen, kv_seqlen, seed):
                kv_max_seqlen = k.shape[1]
                kv_seqlen_per_subrank = kv_max_seqlen // (cp_size * 2)
                assert kv_max_seqlen % cp_size == 0, "sequence length must evenly divide cp size"

                q_split = jnp.split(q, 2, axis=1)

                kv_seqlens_for_rank = helper.kv_seqlens_for_rank(
                    idx, kv_max_seqlen, kv_seqlen_per_subrank
                )

                results = []
                for sub_idx in range(2):
                    if config.attn_mask_type == NVTE_Mask_Type.NVTE_NO_MASK:
                        k_unmasked, v_unmasked = k, v  # full kv used for unmasked
                    else:
                        k_unmasked, v_unmasked = helper.slice_kv(k, v, kv_seqlens_for_rank[sub_idx])

                    q_seqlen_for_step = q_seqlen / (cp_size * 2)
                    num_kv_chunks = kv_max_seqlen // kv_seqlens_for_rank[sub_idx]
                    kv_seqlen_for_step = (kv_seqlen / (cp_size * 2)) * num_kv_chunks

                    output, softmax_aux, rng_state = FusedAttnFwdPrimitive.impl(
                        q_split[sub_idx],
                        k_unmasked,
                        v_unmasked,
                        bias,
                        seed,
                        q_seqlen_for_step,
                        kv_seqlen_for_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(),
                    )
                    results.append((output, softmax_aux, rng_state))

                output = jnp.concatenate((results[0][0], results[1][0]), axis=1)
                softmax_aux = jnp.concatenate((results[0][1], results[1][1]), axis=2)
                rng_state = results[1][2]  # Use the final RNG state

                return output, softmax_aux, rng_state

            k_ag, v_ag = helper.all_gather_kv(k, v)

            functions = [
                partial(_cross_attn, idx, q, k_ag, v_ag, bias, q_seqlen, kv_seqlen, seed)
                for idx in range(cp_size)
            ]

            return lax.switch(cp_rank, functions)

        return mesh, impl, out_shardings, arg_shardings


register_primitive(FusedAttnCPWithAllGatherFwdPrimitive)


class FusedAttnCPWithAllGatherBwdPrimitive(FusedAttnBwdPrimitive):
    """
    Fused Attention Backward with Context Parallelism Primitive.

    This context parallel implementation uses all-gather to collect KV and dKV inputs from context parallel ranks.
    The gradients are subsequently reduce-scattered back to each context parallel rank.
    """

    @staticmethod
    def partition(config, mesh, arg_infos, result_infos):
        # Call base implementation for non-context parallel mesh to avoid unecessary work.
        is_context_parallel = get_mesh_axis_size(config.cp_axis, mesh) > 1
        assert (
            not is_context_parallel or config.window_size[0] == -1
        ), "Sliding window attention is not supported when context parallelism is enabled"
        if not is_context_parallel:
            return FusedAttnBwdPrimitive.partition(config, mesh, arg_infos, result_infos)

        # Ensure we can support this configuration with context parallelism.
        helper = _FusedAttnCPWithAllGatherHelper(mesh, config)
        helper.check_supported()

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
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
        ):
            cp_size = get_mesh_axis_size(config.cp_axis, mesh)
            cp_rank = get_mesh_axis_rank(config.cp_axis, mesh)

            # See comment in FusedAttnCPFwdPrimitive.partition for why we define this function.
            def _cross_attn_bwd(
                idx,
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
                _q_segment_ids,
                _kv_segment_ids,
                _q_segment_pos,
                _kv_segment_pos,
            ):
                kv_max_seqlen = k.shape[1]
                kv_seqlen_per_subrank = kv_max_seqlen // (cp_size * 2)
                assert kv_max_seqlen % cp_size == 0, "sequence length must evenly divide cp size"

                q_split = jnp.split(q, 2, axis=1)
                output_split = jnp.split(output, 2, axis=1)
                doutput_split = jnp.split(doutput, 2, axis=1)
                softmax_aux_split = jnp.split(softmax_aux, 2, axis=2)

                kv_seqlens_for_rank = helper.kv_seqlens_for_rank(
                    idx, kv_max_seqlen, kv_seqlen_per_subrank
                )

                results = []
                for sub_idx in range(2):
                    if config.attn_mask_type == NVTE_Mask_Type.NVTE_NO_MASK:
                        k_unmasked, v_unmasked = k, v  # full kv used for unmasked
                    else:
                        k_unmasked, v_unmasked = helper.slice_kv(k, v, kv_seqlens_for_rank[sub_idx])

                    q_seqlen_for_step = q_seqlen // (cp_size * 2)
                    num_kv_chunks = kv_max_seqlen // kv_seqlens_for_rank[sub_idx]
                    kv_seqlen_for_step = (kv_seqlen // (cp_size * 2)) * num_kv_chunks

                    dq_local, dk_local, dv_local, dbias_local = FusedAttnBwdPrimitive.impl(
                        q_split[sub_idx],
                        k_unmasked,
                        v_unmasked,
                        bias,
                        softmax_aux_split[sub_idx],
                        rng_state,
                        output_split[sub_idx],
                        doutput_split[sub_idx],
                        q_seqlen_for_step,
                        kv_seqlen_for_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(),
                    )

                    # pad dk/dv to be unsliced shape so we can reduce scatter over all ranks.
                    if config.attn_mask_type != NVTE_Mask_Type.NVTE_NO_MASK:
                        pad_length = kv_max_seqlen - kv_seqlens_for_rank[sub_idx]
                        dk_local, dv_local = helper.pad_kv(dk_local, dv_local, pad_length)

                    results.append((dq_local, dk_local, dv_local, dbias_local))

                dq_local = jnp.concatenate((results[0][0], results[1][0]), axis=1)
                dk_local_pad = results[0][1] + results[1][1]
                dv_local_pad = results[0][2] + results[1][2]
                return dq_local, dk_local_pad, dv_local_pad, results[1][3]

            k_ag, v_ag = helper.all_gather_kv(k, v)

            functions = [
                partial(
                    _cross_attn_bwd,
                    idx,
                    q,
                    k_ag,
                    v_ag,
                    bias,
                    softmax_aux,
                    rng_state,
                    output,
                    doutput,
                    q_seqlen,
                    kv_seqlen,
                    _q_segment_ids,
                    _kv_segment_ids,
                    _q_segment_pos,
                    _kv_segment_pos,
                )
                for idx in range(cp_size)
            ]

            dq, dk_local, dv_local, dbias = lax.switch(cp_rank, functions)
            dk, dv = helper.reduce_scatter_dkv(dk_local, dv_local)

            return dq, dk, dv, dbias

        return mesh, impl, out_shardings, arg_shardings


register_primitive(FusedAttnCPWithAllGatherBwdPrimitive)


@dataclass(frozen=True)
class _FusedAttnCPWithP2PHelper:
    """Helper class to assist with running the P2P ring strategy for CP attention."""

    mesh: jax.sharding.Mesh
    config: _FusedAttnConfig

    @staticmethod
    def use_scanloop():
        """Returns true if the implementation will use a scan loop for iteration."""
        use_scan = bool(int(os.getenv("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "1")))

        # nvbug(4675071): Disable the HLO verifier for channel ID checks.
        # A WAR was added to XLA: https://github.com/openxla/xla/pull/16779
        def truthy(val):
            return val.lower() in ["1", "true"]

        x = use_scan and get_xla_flag("--xla_ignore_channel_id", default=True, cast=truthy)
        return x

    def check_supported(self):
        """Checks if the context parallel implementation is supported by the given arguments."""
        header = "Context parallel fused ring attention"

        allowed_layouts = [NVTE_QKV_Layout.NVTE_BSHD_BS2HD, NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD]
        if self.config.qkv_layout not in allowed_layouts:
            raise ValueError(
                f"{header} only supports layouts:"
                f" {','.join(map(str, allowed_layouts))} got: {self.config.qkv_layout}"
            )

        if self.config.attn_bias_type != NVTE_Bias_Type.NVTE_NO_BIAS:
            raise ValueError(f"{header} does not support bias got: {self.config.attn_bias_type}")

        allowed_masks = [NVTE_Mask_Type.NVTE_NO_MASK, NVTE_Mask_Type.NVTE_CAUSAL_MASK]
        if self.config.attn_mask_type not in allowed_masks:
            raise ValueError(
                f"{header} only supports masking types: "
                f" {','.join(map(str, allowed_masks))} got: {self.config.attn_mask_type}"
            )

        if self.config.max_segments_per_seq != 1:
            raise ValueError(
                f"{header} only supports max_segments_per_seq == 1 got:"
                f" {self.config.max_segments_per_seq}"
            )

        if self.config.dropout_probability != 0.0:
            raise ValueError(f"{header} does not support dropout")

        # We want to encourage use of scan loop to minimize unrolling and ensure more
        # predictable scheduling from XLA. The unrolled flavor will be supported but
        # not the prefered implementation.
        if not self.use_scanloop():
            warnings.warn(
                "Scan loop is disabled for fused ring attention. To enable set"
                " NVTE_FUSED_RING_ATTENTION_USE_SCAN=1 in your environment and"
                " add --xla_experimental_ignore_channel_id=true to XLA_FLAGS."
            )

    def get_step_config(self, attn_mask_type) -> _FusedAttnConfig:
        """Returns a _FusedAttnConfig for single CP step call to fused attention."""
        return _FusedAttnConfig(
            attn_bias_type=self.config.attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BS2HD,
            scaling_factor=self.config.scaling_factor,
            dropout_probability=self.config.dropout_probability,
            is_training=self.config.is_training,
            max_segments_per_seq=self.config.max_segments_per_seq,
            window_size=self.config.window_size,
            context_parallel_load_balanced=self.config.context_parallel_load_balanced,
            cp_axis=self.config.cp_axis,
        )

    def stack_kv(self, k, v):
        """Stacks k and v tensors if not stacked."""
        _not_used = jnp.zeros(0, dtype=k.dtype)
        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                return k
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                return jnp.stack([k, v], axis=2)
        return _not_used

    def unstack_kv(self, kv):
        """Un-stacks k and v tensors if not stacked."""
        _not_used = jnp.zeros(0, dtype=kv.dtype)
        match self.config.qkv_layout:
            case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
                return kv, _not_used
            case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
                return jnp.unstack(kv, axis=2)
        return _not_used, _not_used  # fall through

    def permute_kv(self, kv, cp_perm):
        """Permutes kv around the ring as described by cp_perm."""
        return lax_paral_op(kv, lax.ppermute, self.config.cp_axis, mesh=self.mesh, perm=cp_perm)

    @staticmethod
    def correct_output_and_softmax_aux(output, softmax_aux, partial_output, partial_softmax_aux):
        """
        Corrects the output and softmax_aux tensor after each iteration of ring attention.

        See https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795 for
        derivation of this equation.
        """
        new_out = output - jax.nn.sigmoid(partial_softmax_aux - softmax_aux).transpose(
            0, 2, 1, 3
        ) * (output - partial_output)
        new_aux = softmax_aux - jax.nn.log_sigmoid(softmax_aux - partial_softmax_aux)
        return new_out, new_aux

    def adjust_seqlen(self, seqlen, max_seqlen, idx):
        """Adjust the sequence length per step."""
        seqlen_of_curr_step = seqlen - max_seqlen * idx
        seqlen_of_curr_step = jnp.where(seqlen_of_curr_step < 0, 0, seqlen_of_curr_step)
        seqlen_per_step = jnp.where(
            seqlen_of_curr_step < max_seqlen, seqlen_of_curr_step, max_seqlen
        )
        return seqlen_per_step


class FusedRingAttnFwdPrimitive(FusedAttnFwdPrimitive):
    """
    Fused Ring Attention Forward Primitive
    """

    @staticmethod
    def partition(config, mesh, arg_infos, result_infos):
        is_context_parallel = get_mesh_axis_size(config.cp_axis, mesh) > 1
        assert (
            not is_context_parallel or config.window_size[0] == -1
        ), "Sliding window attention is not supported when context parallelism is enabled"
        if not is_context_parallel:
            return FusedAttnFwdPrimitive.partition(config, mesh, arg_infos, result_infos)

        helper = _FusedAttnCPWithP2PHelper(mesh, config)
        helper.check_supported()

        out_sharding = result_infos[0].sharding
        softmax_aux_sharding = result_infos[1].sharding
        rng_state_sharding = seed_sharding = NamedSharding(
            mesh, PartitionSpec(get_all_mesh_axes(), None)
        )
        arg_shardings = [arg_i.sharding for arg_i in arg_infos]
        arg_shardings[4] = seed_sharding
        arg_shardings = tuple(arg_shardings)
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)

        def ring_attn_fwd_impl(
            q,
            k,
            v,
            bias,
            seed,
            q_seqlen,
            kv_seqlen,
            q_seq_offsets,
            k_seq_offsets,
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
        ):
            _not_used = jnp.zeros(0, dtype=v.dtype)

            # Combine KV tensors if separate for better permute scheduling and performance.
            # Eventually XLA should perform this automatically.
            kv = helper.stack_kv(k, v)

            batch, q_max_seqlen, head, _ = q.shape
            kv_max_seqlen = k.shape[1]

            cp_size = get_mesh_axis_size(config.cp_axis, mesh)
            cp_rank = get_mesh_axis_rank(config.cp_axis, mesh)
            cp_perm = [(i, (i + 1) % cp_size) for i in range(cp_size)]

            output = jnp.zeros(q.shape).astype(jnp.float32)
            softmax_aux = jnp.full((batch, head, q_max_seqlen, 1), -jnp.inf, dtype=jnp.float32)

            # RNG shape should be the shared shape. This is unused for ring attention as we do not
            # support dropout currently.
            rng_state_shape = (result_infos[2].shape[0] // mesh.size, *result_infos[2].shape[1:])
            rng_state = jnp.zeros(rng_state_shape).astype(result_infos[2].dtype)

            def scan_kv_block(idx, carry):
                kv, output, softmax_aux = carry

                # Send KV block to next step so we can overlap compute.
                kv_next = helper.permute_kv(kv, cp_perm)

                def mask_compute(attn_mask_type):
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx)
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx)
                    output_per_step, softmax_aux_per_step, _ = FusedAttnFwdPrimitive.impl(
                        q,
                        kv,
                        _not_used,
                        bias,
                        seed,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(attn_mask_type),
                    )
                    return output_per_step, softmax_aux_per_step

                causal_mask_compute = partial(mask_compute, NVTE_Mask_Type.NVTE_CAUSAL_MASK)
                no_mask_compute = partial(mask_compute, NVTE_Mask_Type.NVTE_NO_MASK)

                def half_kv_no_mask_compute():
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx)
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx) // 2
                    kv_part = lax.slice_in_dim(kv, 0, kv.shape[1] // 2, axis=1)
                    output_per_step, softmax_aux_per_step, _ = FusedAttnFwdPrimitive.impl(
                        q,
                        kv_part,
                        _not_used,
                        bias,
                        seed,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(NVTE_Mask_Type.NVTE_NO_MASK),
                    )
                    return output_per_step, softmax_aux_per_step

                def half_q_no_mask_compute():
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx) // 2
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx)
                    q_part = lax.slice_in_dim(q, q_max_seqlen // 2, q_max_seqlen, axis=1)
                    output_per_step, softmax_aux_per_step, _ = FusedAttnFwdPrimitive.impl(
                        q_part,
                        kv,
                        _not_used,
                        bias,
                        seed,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(NVTE_Mask_Type.NVTE_NO_MASK),
                    )
                    output_per_step = jnp.concat([jnp.zeros_like(q_part), output_per_step], axis=1)
                    softmax_aux_per_step = jnp.concat(
                        [
                            jnp.full_like(softmax_aux_per_step, -jnp.inf),
                            softmax_aux_per_step,
                        ],
                        axis=2,
                    )
                    return output_per_step, softmax_aux_per_step

                def skip_compute():
                    output_per_step = jnp.zeros_like(q)
                    softmax_aux_per_step = jnp.full(
                        (batch, head, q.shape[1], 1), -jnp.inf, dtype=jnp.float32
                    )
                    return output_per_step, softmax_aux_per_step

                if config.attn_mask_type == NVTE_Mask_Type.NVTE_CAUSAL_MASK:
                    # This is for nested jax.lax.cond
                    def jax_cond_wrap():
                        if config.context_parallel_load_balanced:
                            return lax.cond(
                                (idx <= cp_rank), half_kv_no_mask_compute, half_q_no_mask_compute
                            )
                        return lax.cond((idx <= cp_rank), no_mask_compute, skip_compute)

                    output_per_step, softmax_aux_per_step = lax.cond(
                        idx == 0, causal_mask_compute, jax_cond_wrap
                    )
                else:
                    output_per_step, softmax_aux_per_step = no_mask_compute()

                def skip_correction(output, softmax_aux, output_per_step, softmax_aux_per_step):
                    # No correction done here but we cast outputs to float32 and perform reduction
                    # in full precision.
                    # pylint: disable=unused-argument
                    return output_per_step.astype(jnp.float32), softmax_aux_per_step

                def correction(output, softmax_aux, output_per_step, softmax_aux_per_step):
                    return helper.correct_output_and_softmax_aux(
                        output, softmax_aux, output_per_step, softmax_aux_per_step
                    )

                # first step there is no correction we get initial output and stats
                output, softmax_aux = lax.cond(
                    (idx == 0),
                    skip_correction,
                    correction,
                    output,
                    softmax_aux,
                    output_per_step,
                    softmax_aux_per_step,
                )

                return (kv_next, output, softmax_aux)

            carry = (kv, output, softmax_aux)
            if helper.use_scanloop():
                carry = lax.fori_loop(0, cp_size, scan_kv_block, carry)
            else:
                for i in range(0, cp_size):
                    carry = scan_kv_block(i, carry)
            (kv, output, softmax_aux) = carry

            output = output.astype(q.dtype)
            return output, softmax_aux, rng_state

        return mesh, ring_attn_fwd_impl, out_shardings, arg_shardings


register_primitive(FusedRingAttnFwdPrimitive)


class FusedRingAttnBwdPrimitive(FusedAttnBwdPrimitive):
    """
    Fused Ring Attention Backward Primitive
    """

    @staticmethod
    def partition(config, mesh, arg_infos, result_infos):
        is_context_parallel = get_mesh_axis_size(config.cp_axis, mesh) > 1
        assert (
            not is_context_parallel or config.window_size[0] == -1
        ), "Sliding window attention is not supported when context parallelism is enabled"
        if not is_context_parallel:
            return FusedAttnBwdPrimitive.partition(config, mesh, arg_infos, result_infos)

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

        helper = _FusedAttnCPWithP2PHelper(mesh, config)
        helper.check_supported()

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
            _q_segment_ids,
            _kv_segment_ids,
            _q_segment_pos,
            _kv_segment_pos,
        ):
            _not_used = jnp.zeros(0, dtype=output.dtype)

            # Combine KV tensors if separate for better permute scheduling and performance.
            # Eventually XLA should perform this automatically.
            kv = helper.stack_kv(k, v)

            q_max_seqlen = q.shape[1]
            kv_max_seqlen = k.shape[1]

            cp_size = get_mesh_axis_size(config.cp_axis, mesh)
            cp_rank = get_mesh_axis_rank(config.cp_axis, mesh)
            cp_perm = [(i, (i + 1) % cp_size) for i in range(cp_size)]

            dq = jnp.zeros_like(q)
            dk_dv = helper.stack_kv(jnp.zeros_like(k), jnp.zeros_like(v))
            dbias = jnp.zeros_like(bias)

            def scan_kv_block(idx, carry):

                kv, dq, dk_dv, dbias = carry

                # Start communication that feeds the next iteraton.
                # We further combine the tensors to improve overlap.

                kv_dk_dv = jnp.stack([kv, dk_dv])
                kv_dk_dv = helper.permute_kv(kv_dk_dv, cp_perm)

                def mask_compute(attn_mask_type):
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx)
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx)
                    dq_per_step, dk_dv_per_step, _, dbias_per_step = FusedAttnBwdPrimitive.impl(
                        q,
                        kv,
                        _not_used,
                        bias,
                        softmax_aux,
                        rng_state,
                        output,
                        doutput,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(attn_mask_type),
                    )
                    return dq_per_step, dk_dv_per_step, dbias_per_step

                causal_mask_compute = partial(mask_compute, NVTE_Mask_Type.NVTE_CAUSAL_MASK)
                no_mask_compute = partial(mask_compute, NVTE_Mask_Type.NVTE_NO_MASK)

                def half_kv_no_mask_compute():
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx)
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx) // 2
                    kv_part = lax.slice_in_dim(kv, 0, kv_max_seqlen // 2, axis=1)
                    dq_per_step, dk_dv_per_step, _, dbias_per_step = FusedAttnBwdPrimitive.impl(
                        q,
                        kv_part,
                        _not_used,
                        bias,
                        softmax_aux,
                        rng_state,
                        output,
                        doutput,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(NVTE_Mask_Type.NVTE_NO_MASK),
                    )
                    dk_dv_per_step = jnp.concat(
                        [dk_dv_per_step, jnp.zeros_like(dk_dv_per_step)], axis=1
                    )
                    return dq_per_step, dk_dv_per_step, dbias_per_step

                def half_q_no_mask_compute():
                    q_seqlen_per_step = helper.adjust_seqlen(q_seqlen, q_max_seqlen, idx) // 2
                    kv_seqlen_per_step = helper.adjust_seqlen(kv_seqlen, kv_max_seqlen, idx)

                    q_part = lax.slice_in_dim(q, q_max_seqlen // 2, q_max_seqlen, axis=1)
                    doutput_part = lax.slice_in_dim(
                        doutput, q_max_seqlen // 2, q_max_seqlen, axis=1
                    )
                    output_part = lax.slice_in_dim(output, q_max_seqlen // 2, q_max_seqlen, axis=1)

                    softmax_aux_part = lax.slice_in_dim(
                        softmax_aux, q_max_seqlen // 2, q_max_seqlen, axis=2
                    )

                    dq_per_step, dk_dv_per_step, _, dbias_per_step = FusedAttnBwdPrimitive.impl(
                        q_part,
                        kv,
                        _not_used,
                        bias,
                        softmax_aux_part,
                        rng_state,
                        output_part,
                        doutput_part,
                        q_seqlen_per_step,
                        kv_seqlen_per_step,
                        q_seq_offsets,
                        k_seq_offsets,
                        _q_segment_ids,
                        _kv_segment_ids,
                        _q_segment_pos,
                        _kv_segment_pos,
                        config=helper.get_step_config(NVTE_Mask_Type.NVTE_NO_MASK),
                    )
                    dq_per_step = jnp.concat([jnp.zeros_like(dq_per_step), dq_per_step], axis=1)
                    return dq_per_step, dk_dv_per_step, dbias_per_step

                def skip_compute():
                    return jnp.zeros_like(q), jnp.zeros_like(kv), jnp.zeros_like(bias)

                if config.attn_mask_type == NVTE_Mask_Type.NVTE_CAUSAL_MASK:
                    # This is for nested jax.lax.cond
                    def jax_cond_wrap():
                        if config.context_parallel_load_balanced:
                            return lax.cond(
                                (idx <= cp_rank), half_kv_no_mask_compute, half_q_no_mask_compute
                            )
                        return lax.cond((idx <= cp_rank), no_mask_compute, skip_compute)

                    dq_per_step, dk_dv_per_step, dbias_per_step = lax.cond(
                        idx == 0, causal_mask_compute, jax_cond_wrap
                    )
                else:
                    dq_per_step, dk_dv_per_step, dbias_per_step = no_mask_compute()

                kv_next, dk_dv = jnp.unstack(kv_dk_dv)
                dq = dq + dq_per_step
                dk_dv = dk_dv + dk_dv_per_step
                if config.attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                    dbias = dbias + dbias_per_step

                return (kv_next, dq, dk_dv, dbias)

            carry = (kv, dq, dk_dv, dbias)
            if helper.use_scanloop():
                carry = lax.fori_loop(0, cp_size, scan_kv_block, carry)
            else:
                for i in range(0, cp_size):
                    carry = scan_kv_block(i, carry)
            (kv, dq, dk_dv, dbias) = carry

            # Final permute to put gradients back to their final resting place.
            dk_dv = helper.permute_kv(dk_dv, cp_perm)

            global_dbias = dbias
            if config.attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(dbias, mesh)

            dk, dv = helper.unstack_kv(dk_dv)
            return dq, dk, dv, global_dbias

        return mesh, ring_attn_bwd_impl, out_shardings, arg_shardings


register_primitive(FusedRingAttnBwdPrimitive)


def _maybe_context_parallel_axis(cp_axis: str):
    if not cp_axis:
        gmr = global_mesh_resource()
        if gmr is not None:
            cp_axis = gmr.cp_resource
        else:
            cp_axis = ""
    return cp_axis


def fused_attn_fwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    sequence_descriptor: SequenceDescriptor,
    seed: Optional[jnp.ndarray],
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
    window_size: Optional[Tuple[int, int]] = None,
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
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
        max_segments_per_seq (int):
            Indicating the maximum number of segments inside a sequence. This parameter is to
            constrain the limit usage and need to be static during the e2e training. The XLA compile
            time and memory consumption is proportional to `max_segments_per_seq`.
        window_size (Optional[Tuple[int, int]]): Sliding window size.
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.
    """
    seed = _FusedAttnRNGStateChecker().check_seed(seed, dropout_probability, is_training)
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

    fused_config = _FusedAttnConfig(
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=(-1, -1) if window_size is None else window_size,
        context_parallel_load_balanced=context_parallel_causal_load_balanced,
        cp_axis=_maybe_context_parallel_axis(context_parallel_axis),
    )

    primitive = None
    match context_parallel_strategy:
        case CPStrategy.DEFAULT | CPStrategy.ALL_GATHER:
            primitive = FusedAttnCPWithAllGatherFwdPrimitive.outer_primitive
        case CPStrategy.RING:
            primitive = FusedRingAttnFwdPrimitive.outer_primitive

    seq_desc_flatten, _ = jax.tree.flatten(sequence_descriptor)
    return primitive.bind(
        *qkv_for_primitive,
        bias,
        seed,
        *seq_desc_flatten,
        config=fused_config,
    )


def fused_attn_bwd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    softmax_aux: jnp.ndarray,
    rng_state: jnp.ndarray,
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    sequence_descriptor: SequenceDescriptor,
    attn_bias_type: NVTE_Bias_Type,
    attn_mask_type: NVTE_Mask_Type,
    qkv_layout: NVTE_QKV_Layout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
    window_size: Optional[Tuple[int, int]] = None,
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
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
        max_segments_per_seq (int):
            Indicating the maximum number of segments inside a sequence. This parameter is to
            constrain the limit usage and need to be static during the e2e training. The XLA compile
            time and memory consumption is proportional to `max_segments_per_seq`.
        window_size (Optional[Tuple[int, int]]): Sliding window size .
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        Tuple[jnp.ndarray, ...], jnp.ndarray:
        - The first tuple contains the gradients with respect to the input `qkv` tensors in the
          same format as the input `qkv`.
        - The second value is the gradient with respect to `bias`, or `None` if `bias` is `None`.
    """
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

    fused_config = _FusedAttnConfig(
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=(-1, -1) if window_size is None else window_size,
        context_parallel_load_balanced=context_parallel_causal_load_balanced,
        cp_axis=_maybe_context_parallel_axis(context_parallel_axis),
    )

    primitive = None
    match context_parallel_strategy:
        case CPStrategy.DEFAULT | CPStrategy.ALL_GATHER:
            primitive = FusedAttnCPWithAllGatherBwdPrimitive.outer_primitive
        case CPStrategy.RING:
            primitive = FusedRingAttnBwdPrimitive.outer_primitive

    seq_desc_flatten, _ = jax.tree.flatten(sequence_descriptor)

    *qkv_grads, bias_grad = primitive.bind(
        *qkv_for_primitive,
        bias,
        softmax_aux,
        rng_state,
        output,
        doutput,
        *seq_desc_flatten,
        config=fused_config,
    )
    return tuple(qkv_grads[: len(qkv)]), bias_grad
