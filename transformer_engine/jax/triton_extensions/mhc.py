# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX/TE custom ops for mHC (manifold Hyper-Connection) using Triton kernels."""

import math
import os
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental.custom_partitioning import SdyShardingRule
import triton

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive
from transformer_engine.jax.cpp_extensions.misc import get_padded_spec, NamedSharding
from transformer_engine.common.triton.mhc import (
    _mhc_projection_fwd_fused,
    _mhc_projection_bwd_fused_dx,
    _mhc_projection_bwd_fused_dphi,
    _mhc_scale_fwd_fused,
    _mhc_scale_bwd_fused,
    _mhc_sinkhorn_fwd_fused_recompute,
    _mhc_sinkhorn_bwd_fused_recompute,
    _mhc_sinkhorn_fwd_fused,
    _mhc_sinkhorn_bwd_fused,
    _mhc_aggregate_fwd,
    _mhc_aggregate_bwd,
    _mhc_expand_combine_fwd,
    _mhc_expand_combine_bwd,
)
from .utils import triton_call_lowering

__all__ = [
    "mhc_projection_fwd",
    "mhc_projection_bwd",
    "mhc_scale_fwd",
    "mhc_scale_bwd",
    "mhc_sinkhorn_fwd",
    "mhc_sinkhorn_bwd",
    "mhc_aggregate_fwd",
    "mhc_aggregate_bwd",
    "mhc_expand_combine_fwd",
    "mhc_expand_combine_bwd",
]

ENFORCE_DETERMINISTIC = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "0"

# H, grad_H and the scale output are padded to 32 columns, of which only N = 2n + n*n are valid.
H_PADDED_DIM = 32
# scale_config fixes BLOCK_SIZE_M, which sizes the deterministic grad_alpha/grad_beta workspaces.
SCALE_BLOCK_SIZE_M = 128
# expand_combine_prune_bwd fixes BLOCK_SIZE_M, which sizes the deterministic grad_bias workspace.
EXPAND_COMBINE_BWD_BLOCK_SIZE_M = 4
# RMSNorm epsilon of the PyTorch API.
DEFAULT_NORM_EPS = float(jnp.finfo(jnp.float32).eps)


def check_deterministic(operator: str, use_split_k: bool = False):
    """Split-K/M reductions use atomic adds, which NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 disallows."""
    if use_split_k:
        assert not ENFORCE_DETERMINISTIC, (
            f"[{operator}]: use_split_k=True uses atomic add which violates determinism. Either set"
            " use_split_k=False or unset NVTE_ALLOW_NONDETERMINISTIC_ALGO=0."
        )


def _projection_precision(x_dtype, phi_dtype, has_norm_weight, use_tf32):
    """Precision chosen by mHCProjectionOp.forward, also reused by the dx kernel."""
    if has_norm_weight or (not use_tf32 and x_dtype == jnp.bfloat16 and phi_dtype == jnp.bfloat16):
        phi_dtype = jnp.float32
    precision = "tf32" if use_tf32 else "ieee"
    # See https://github.com/triton-lang/triton/issues/10176.
    if precision == "ieee" and x_dtype == jnp.bfloat16 and phi_dtype == jnp.float32:
        precision = "tf32x3"
    return precision


def _flatten_tokens(x, num_trailing):
    """Collapse the leading (token) dims of `x` into one dim of size M."""
    trailing = x.shape[x.ndim - num_trailing :]
    return x.reshape((math.prod(x.shape[: x.ndim - num_trailing]),) + trailing)


def _empty_aliased(dtype):
    """Placeholder for an aliased kernel buffer that the kernel never touches."""
    return jnp.empty((1,), dtype=dtype)


def _unused(dtype):
    """Placeholder for an optional kernel input that the kernel never reads."""
    return jnp.zeros((0,), dtype=dtype)


def _split_layout(layout):
    return (0, layout) if isinstance(layout, int) else layout


def _token_spec(arg_info, layout):
    before, after = _split_layout(layout)
    spec = get_padded_spec(arg_info)
    return tuple(spec[before : len(spec) - after])


def _sharding(mesh, token_spec, info, layout, desc):
    if layout is None:
        spec = (None,) * len(info.shape)
    else:
        before, after = _split_layout(layout)
        spec = (None,) * before + token_spec + (None,) * after
    return NamedSharding(mesh, PartitionSpec(*spec), desc=desc)


def _mesh_axes(token_spec):
    axes = []
    for axis in token_spec:
        if axis is None:
            continue
        axes.extend(axis if isinstance(axis, tuple) else (axis,))
    return tuple(axes)


class _MHCPrimitive(BasePrimitive):
    """Shared sharding rules for mHC primitives.

    Every operand and result is either token-major or replicated. Token-major arrays keep the
    flattened token dims (M = s * b) contiguous, so any of those dims can be sharded while the
    hidden, stream and padded dims stay replicated. Per-token kernels need no communication;
    results listed in `reduced_outputs` are reductions over tokens and are summed across the
    token shards.

    `layouts(*static_args)` returns the layouts of the operands and of the results. A layout is
    None (replicated), an int t (token dims followed by t trailing dims) or a tuple (b, t)
    (b leading dims, then the token dims, then t trailing dims). The first operand must be
    token-major.
    """

    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None
    reduced_outputs = ()

    @staticmethod
    def layouts(*static_args):
        """Operand and result layouts."""
        raise NotImplementedError

    @classmethod
    def partition(cls, *args):
        """Shard the token dims and replicate the rest."""
        *static_args, mesh, arg_infos, result_infos = args
        in_layouts, out_layouts = cls.layouts(*static_args)
        token_spec = _token_spec(arg_infos[0], in_layouts[0])
        arg_shardings = tuple(
            _sharding(mesh, token_spec, info, layout, f"{cls.__name__}.arg{i}")
            for i, (info, layout) in enumerate(zip(arg_infos, in_layouts))
        )
        # Must match the container type of the results (tuple or list).
        out_shardings = type(result_infos)(
            _sharding(mesh, token_spec, info, layout, f"{cls.__name__}.out{i}")
            for i, (info, layout) in enumerate(zip(result_infos, out_layouts))
        )
        axes = _mesh_axes(token_spec)

        def sharded_impl(*arrays):
            outputs = list(cls.impl(*arrays, *static_args))
            if axes:
                for i in cls.reduced_outputs:
                    outputs[i] = jax.lax.psum(outputs[i], axes)
            return type(result_infos)(outputs)

        return mesh, sharded_impl, out_shardings, arg_shardings

    @classmethod
    def shardy_sharding_rule(cls, *args):
        """Shardy rule sharing one factor per token dim across token-major values."""
        *static_args, _, value_types, result_types = args
        in_layouts, out_layouts = cls.layouts(*static_args)
        before, after = _split_layout(in_layouts[0])
        num_token_dims = len(value_types[0].shape) - before - after
        prefix = cls.__name__
        tokens = tuple(f"{prefix}_t{i}" for i in range(num_token_dims))

        def spec(kind, index, value_type, layout):
            ndim = len(value_type.shape)
            if layout is None:
                return tuple(f"{prefix}_{kind}{index}_d{d}" for d in range(ndim))
            before, after = _split_layout(layout)
            return (
                tuple(f"{prefix}_{kind}{index}_b{d}" for d in range(before))
                + tokens
                + tuple(f"{prefix}_{kind}{index}_a{d}" for d in range(after))
            )

        return SdyShardingRule(
            tuple(spec("in", i, v, l) for i, (v, l) in enumerate(zip(value_types, in_layouts))),
            tuple(spec("out", i, v, l) for i, (v, l) in enumerate(zip(result_types, out_layouts))),
        )


class MHCProjectionFwdPrimitive(_MHCPrimitive):
    """H = x @ phi^T (padded to 32 columns) and ms = mean(x^2, dim=-1)."""

    name = "te_mhc_projection_fwd_triton"
    impl_static_args = (2, 3)  # precision, use_split_k

    @staticmethod
    def layouts(precision, use_split_k):
        del precision, use_split_k
        return (1, None), (1, 0)

    @staticmethod
    def abstract(x_aval, phi_aval, h_aval, ms_aval, *, precision, use_split_k):
        """Inner abstract: H and ms are aliased to their pre-zeroed buffers."""
        del x_aval, phi_aval, precision, use_split_k
        return (
            jax.core.ShapedArray(h_aval.shape, h_aval.dtype),
            jax.core.ShapedArray(ms_aval.shape, ms_aval.dtype),
        )

    @staticmethod
    def outer_abstract(x_aval, phi_aval, *, precision, use_split_k):
        """Outer abstract."""
        del phi_aval, precision, use_split_k
        tokens = x_aval.shape[:-1]
        return (
            jax.core.ShapedArray(tokens + (H_PADDED_DIM,), jnp.float32),
            jax.core.ShapedArray(tokens, jnp.float32),
        )

    @staticmethod
    def impl(x, phi, precision, use_split_k):
        """Allocate the zeroed accumulators and call the inner primitive."""
        assert MHCProjectionFwdPrimitive.inner_primitive is not None
        tokens = x.shape[:-1]
        x_2d = _flatten_tokens(x, 1)
        M = x_2d.shape[0]
        h_buf = jnp.zeros((M, H_PADDED_DIM), dtype=jnp.float32)
        ms_buf = jnp.zeros((M,), dtype=jnp.float32)
        h, ms = MHCProjectionFwdPrimitive.inner_primitive.bind(
            x_2d, phi, h_buf, ms_buf, precision=precision, use_split_k=use_split_k
        )
        return h.reshape(tokens + (H_PADDED_DIM,)), ms.reshape(tokens)

    @staticmethod
    def lowering(ctx, x, phi, h_buf, ms_buf, *, precision, use_split_k):
        """MLIR lowering using triton_call_lowering."""
        M, K = ctx.avals_in[0].shape
        N = ctx.avals_in[1].shape[0]

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))

        return triton_call_lowering(
            ctx,
            _mhc_projection_fwd_fused,
            x,
            phi,
            h_buf,
            ms_buf,
            grid=grid,
            input_output_aliases={2: 0, 3: 1},
            constexprs={
                "M": M,
                "N": N,
                "K": K,
                "stride_xm": K,
                "stride_xk": 1,
                "stride_phin": K,
                "stride_phik": 1,
                "stride_hm": H_PADDED_DIM,
                "stride_hn": 1,
                "stride_ms": 1,
                "stride_norm_weight": 1,
                "BLOCK_SIZE_N": H_PADDED_DIM,
                "precision": precision,
                "USE_SPLIT_K": use_split_k,
                # TMA not supported as global scratch/allocator is required for device-side TMA descriptors
                # Also, host-side descriptors require real pointers which aren't available when compiling JAX/XLA.
                "USE_TMA": False,
            },
        )


register_primitive(MHCProjectionFwdPrimitive)


class MHCProjectionBwdDxPrimitive(_MHCPrimitive):
    """grad_x = grad_H @ (phi * norm_weight) + 2 * x * grad_ms / K."""

    name = "te_mhc_projection_bwd_dx_triton"
    impl_static_args = (6, 7, 8)  # precision, has_norm_weight, fuse_grad_x_acc

    @staticmethod
    def layouts(precision, has_norm_weight, fuse_grad_x_acc):
        del precision, has_norm_weight
        return (1, None, None, 1, 0, 1 if fuse_grad_x_acc else None), (1,)

    @staticmethod
    def abstract(
        x_aval,
        grad_x_aval,
        phi_aval,
        norm_weight_aval,
        grad_h_aval,
        grad_ms_aval,
        *,
        precision,
        has_norm_weight,
        fuse_grad_x_acc,
    ):
        """Inner abstract: grad_x is aliased to its buffer."""
        del x_aval, phi_aval, norm_weight_aval, grad_h_aval, grad_ms_aval
        del precision, has_norm_weight, fuse_grad_x_acc
        return (jax.core.ShapedArray(grad_x_aval.shape, grad_x_aval.dtype),)

    @staticmethod
    def outer_abstract(
        x_aval,
        phi_aval,
        norm_weight_aval,
        grad_h_aval,
        grad_ms_aval,
        grad_x_acc_aval,
        *,
        precision,
        has_norm_weight,
        fuse_grad_x_acc,
    ):
        """Outer abstract."""
        del phi_aval, norm_weight_aval, grad_h_aval, grad_ms_aval, grad_x_acc_aval
        del precision, has_norm_weight
        dtype = jnp.float32 if fuse_grad_x_acc else x_aval.dtype
        return (jax.core.ShapedArray(x_aval.shape, dtype),)

    @staticmethod
    def impl(
        x,
        phi,
        norm_weight,
        grad_h,
        grad_ms,
        grad_x_acc,
        precision,
        has_norm_weight,
        fuse_grad_x_acc,
    ):
        """Allocate (or reuse the accumulation buffer for) grad_x and call the inner primitive."""
        assert MHCProjectionBwdDxPrimitive.inner_primitive is not None
        x_2d = _flatten_tokens(x, 1)
        M, K = x_2d.shape
        if fuse_grad_x_acc:
            grad_x_buf = grad_x_acc.reshape(M, K)
        else:
            grad_x_buf = jnp.empty((M, K), dtype=x.dtype)
        (grad_x,) = MHCProjectionBwdDxPrimitive.inner_primitive.bind(
            x_2d,
            grad_x_buf,
            phi,
            norm_weight,
            grad_h.reshape(M, H_PADDED_DIM),
            grad_ms.reshape(M),
            precision=precision,
            has_norm_weight=has_norm_weight,
            fuse_grad_x_acc=fuse_grad_x_acc,
        )
        return (grad_x.reshape(x.shape),)

    @staticmethod
    def lowering(
        ctx,
        x,
        grad_x_buf,
        phi,
        norm_weight,
        grad_h,
        grad_ms,
        *,
        precision,
        has_norm_weight,
        fuse_grad_x_acc,
    ):
        """MLIR lowering using triton_call_lowering."""
        M, K = ctx.avals_in[0].shape
        N = ctx.avals_in[2].shape[0]

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))

        return triton_call_lowering(
            ctx,
            _mhc_projection_bwd_fused_dx,
            x,
            grad_x_buf,
            phi,
            norm_weight,
            grad_h,
            grad_ms,
            grid=grid,
            input_output_aliases={1: 0},
            constexprs={
                "M": M,
                "N": N,
                "K": K,
                "stride_xm": K,
                "stride_xk": 1,
                "stride_grad_xm": K,
                "stride_grad_xk": 1,
                "stride_phin": K,
                "stride_phik": 1,
                "stride_norm_weight": 1,
                "stride_grad_phin": K,
                "stride_grad_phik": 1,
                "stride_grad_hm": H_PADDED_DIM,
                "stride_grad_hn": 1,
                "stride_grad_ms": 1,
                "BLOCK_SIZE_N": H_PADDED_DIM,
                "precision": precision,
                "FUSE_GRAD_X_ACC": fuse_grad_x_acc,
                "HAS_NORM_WEIGHT": has_norm_weight,
            },
        )


register_primitive(MHCProjectionBwdDxPrimitive)


class MHCProjectionBwdDphiPrimitive(_MHCPrimitive):
    """grad_phi = (grad_H^T @ x) * norm_weight and grad_norm_weight = sum((grad_H^T @ x) * phi, 0)."""

    name = "te_mhc_projection_bwd_dphi_triton"
    impl_static_args = (4, 5)  # precision, use_split_m
    reduced_outputs = (0, 1)

    @staticmethod
    def layouts(precision, use_split_m):
        del precision, use_split_m
        return (1, 1, None, None), (None, None)

    @staticmethod
    def abstract(
        x_aval,
        grad_h_aval,
        phi_aval,
        norm_weight_aval,
        *buf_avals,
        precision,
        use_split_m,
    ):
        """Inner abstract: with split-M, both gradients are aliased to pre-zeroed fp32 buffers."""
        del x_aval, grad_h_aval, precision
        assert len(buf_avals) == (2 if use_split_m else 0)
        return (
            jax.core.ShapedArray(phi_aval.shape, jnp.float32),
            jax.core.ShapedArray(norm_weight_aval.shape, jnp.float32),
        )

    @staticmethod
    def outer_abstract(x_aval, grad_h_aval, phi_aval, norm_weight_aval, *, precision, use_split_m):
        """Outer abstract."""
        del x_aval, grad_h_aval, precision, use_split_m
        return (
            jax.core.ShapedArray(phi_aval.shape, jnp.float32),
            jax.core.ShapedArray(norm_weight_aval.shape, jnp.float32),
        )

    @staticmethod
    def impl(x, grad_h, phi, norm_weight, precision, use_split_m):
        """Call the inner primitive, with zeroed fp32 accumulators for the split-M atomic adds."""
        assert MHCProjectionBwdDphiPrimitive.inner_primitive is not None
        x_2d = _flatten_tokens(x, 1)
        M = x_2d.shape[0]
        # Without split-M every element is stored once, so the outputs need no initialization.
        bufs = (
            (
                jnp.zeros(phi.shape, dtype=jnp.float32),
                jnp.zeros(norm_weight.shape, dtype=jnp.float32),
            )
            if use_split_m
            else ()
        )
        return MHCProjectionBwdDphiPrimitive.inner_primitive.bind(
            x_2d,
            grad_h.reshape(M, H_PADDED_DIM),
            phi,
            norm_weight,
            *bufs,
            precision=precision,
            use_split_m=use_split_m,
        )

    @staticmethod
    def lowering(
        ctx,
        x,
        grad_h,
        phi,
        norm_weight,
        *bufs,
        precision,
        use_split_m,
    ):
        """MLIR lowering using triton_call_lowering."""
        x_aval = ctx.avals_in[0]
        M, K = x_aval.shape
        N = ctx.avals_in[2].shape[0]

        def grid(meta):
            return (triton.cdiv(K, meta["BLOCK_SIZE_K"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

        # grad_phi_ptr and grad_norm_weight_ptr are the last pointer args, so without the
        # split-M buffers they bind directly to the two outputs.
        return triton_call_lowering(
            ctx,
            _mhc_projection_bwd_fused_dphi,
            x,
            grad_h,
            phi,
            norm_weight,
            *bufs,
            grid=grid,
            input_output_aliases={4: 0, 5: 1} if use_split_m else None,
            constexprs={
                "M": M,
                "N": N,
                "K": K,
                "stride_xm": K,
                "stride_xk": 1,
                "stride_grad_Hm": H_PADDED_DIM,
                "stride_grad_Hn": 1,
                "stride_phin": K,
                "stride_phik": 1,
                "stride_norm_weight": 1,
                "stride_grad_phin": K,
                "stride_grad_phik": 1,
                "stride_grad_norm_weight": 1,
                "BLOCK_SIZE_N": H_PADDED_DIM,
                "precision": precision,
                "USE_SPLIT_M": use_split_m,
            },
        )


register_primitive(MHCProjectionBwdDphiPrimitive)


class MHCScaleFwdPrimitive(_MHCPrimitive):
    """RMSNorm scaling, bias and activations producing [H_pre, H_post, H_res] (padded to 32)."""

    name = "te_mhc_scale_fwd_triton"
    impl_static_args = (4, 5)  # n, eps

    @staticmethod
    def layouts(n, eps):
        del n, eps
        return (1, None, None, 0), (1,)

    @staticmethod
    def abstract(h_aval, alpha_aval, beta_aval, ms_aval, *, n, eps):
        """Abstract."""
        del alpha_aval, beta_aval, ms_aval, n, eps
        return (jax.core.ShapedArray(h_aval.shape, jnp.float32),)

    @staticmethod
    def outer_abstract(h_aval, alpha_aval, beta_aval, ms_aval, *, n, eps):
        """Outer abstract."""
        return MHCScaleFwdPrimitive.abstract(h_aval, alpha_aval, beta_aval, ms_aval, n=n, eps=eps)

    @staticmethod
    def impl(h, alpha, beta, ms, n, eps):
        """Flatten the token dims and call the inner primitive."""
        assert MHCScaleFwdPrimitive.inner_primitive is not None
        h_2d = _flatten_tokens(h, 1)
        (out,) = MHCScaleFwdPrimitive.inner_primitive.bind(
            h_2d, alpha, beta, ms.reshape(h_2d.shape[0]), n=n, eps=eps
        )
        return (out.reshape(h.shape),)

    @staticmethod
    def lowering(ctx, h, alpha, beta, ms, *, n, eps):
        """MLIR lowering using triton_call_lowering."""
        M = ctx.avals_in[0].shape[0]

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)

        return triton_call_lowering(
            ctx,
            _mhc_scale_fwd_fused,
            h,
            alpha,
            beta,
            ms,
            grid=grid,
            constexprs={
                "M": M,
                "n": n,
                "stride_hm": H_PADDED_DIM,
                "stride_hn": 1,
                "stride_a": 1,
                "stride_b": 1,
                "stride_ms": 1,
                "stride_out_m": H_PADDED_DIM,
                "stride_out_n": 1,
                "BLOCK_SIZE_N": H_PADDED_DIM,
                "eps": eps,
            },
        )


register_primitive(MHCScaleFwdPrimitive)


class MHCScaleBwdPrimitive(_MHCPrimitive):
    """Gradients of the fused scale with respect to H, alpha, beta and ms."""

    name = "te_mhc_scale_bwd_triton"
    impl_static_args = (5, 6, 7)  # n, eps, deterministic
    reduced_outputs = (1, 2)

    @staticmethod
    def layouts(n, eps, deterministic):
        del n, eps, deterministic
        return (1, 1, 1, None, 0), (1, None, None, 0)

    @staticmethod
    def abstract(
        grad_out_aval,
        out_aval,
        grad_h_aval,
        h_aval,
        grad_a_aval,
        a_aval,
        grad_b_aval,
        grad_ms_aval,
        ms_aval,
        ws_grad_a_aval,
        ws_grad_b_aval,
        *,
        n,
        eps,
        deterministic,
    ):
        """Inner abstract: every output is aliased to its buffer."""
        del grad_out_aval, out_aval, h_aval, a_aval, ms_aval, n, eps, deterministic
        return tuple(
            jax.core.ShapedArray(aval.shape, aval.dtype)
            for aval in (
                grad_h_aval,
                grad_a_aval,
                grad_b_aval,
                grad_ms_aval,
                ws_grad_a_aval,
                ws_grad_b_aval,
            )
        )

    @staticmethod
    def outer_abstract(
        grad_out_aval, out_aval, h_aval, alpha_aval, ms_aval, *, n, eps, deterministic
    ):
        """Outer abstract."""
        del grad_out_aval, out_aval, alpha_aval, ms_aval, eps, deterministic
        return (
            jax.core.ShapedArray(h_aval.shape, jnp.float32),
            jax.core.ShapedArray((3,), jnp.float32),
            jax.core.ShapedArray((2 * n + n * n,), jnp.float32),
            jax.core.ShapedArray(h_aval.shape[:-1], jnp.float32),
        )

    @staticmethod
    def impl(grad_out, out, h, alpha, ms, n, eps, deterministic):
        """Allocate the gradient buffers and workspaces, then reduce the workspaces."""
        assert MHCScaleBwdPrimitive.inner_primitive is not None
        tokens = h.shape[:-1]
        h_2d = _flatten_tokens(h, 1)
        M = h_2d.shape[0]
        N = 2 * n + n * n
        if deterministic:
            grid_m = triton.cdiv(M, SCALE_BLOCK_SIZE_M)
            ws_grad_a = jnp.empty((grid_m, 4), dtype=jnp.float32)
            ws_grad_b = jnp.empty((grid_m, H_PADDED_DIM), dtype=jnp.float32)
        else:
            ws_grad_a = _empty_aliased(jnp.float32)
            ws_grad_b = _empty_aliased(jnp.float32)

        grad_h, grad_alpha, grad_beta, grad_ms, ws_grad_a, ws_grad_b = (
            MHCScaleBwdPrimitive.inner_primitive.bind(
                grad_out.reshape(M, H_PADDED_DIM),
                out.reshape(M, H_PADDED_DIM),
                jnp.zeros((M, H_PADDED_DIM), dtype=jnp.float32),
                h_2d,
                jnp.zeros((3,), dtype=jnp.float32),
                alpha,
                jnp.zeros((H_PADDED_DIM,), dtype=jnp.float32),
                jnp.zeros((M,), dtype=jnp.float32),
                ms.reshape(M),
                ws_grad_a,
                ws_grad_b,
                n=n,
                eps=eps,
                deterministic=deterministic,
            )
        )
        if deterministic:
            grad_alpha = ws_grad_a.sum(axis=0)[:3]
            grad_beta = ws_grad_b.sum(axis=0)
        return grad_h.reshape(h.shape), grad_alpha, grad_beta[:N], grad_ms.reshape(tokens)

    @staticmethod
    def lowering(
        ctx,
        grad_out,
        out,
        grad_h,
        h,
        grad_a,
        a,
        grad_b,
        grad_ms,
        ms,
        ws_grad_a,
        ws_grad_b,
        *,
        n,
        eps,
        deterministic,
    ):
        """MLIR lowering using triton_call_lowering."""
        M = ctx.avals_in[0].shape[0]

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)

        return triton_call_lowering(
            ctx,
            _mhc_scale_bwd_fused,
            grad_out,
            out,
            grad_h,
            h,
            grad_a,
            a,
            grad_b,
            grad_ms,
            ms,
            ws_grad_a,
            ws_grad_b,
            grid=grid,
            input_output_aliases={2: 0, 4: 1, 6: 2, 7: 3, 9: 4, 10: 5},
            constexprs={
                "M": M,
                "n": n,
                "stride_grad_out_m": H_PADDED_DIM,
                "stride_grad_out_n": 1,
                "stride_out_m": H_PADDED_DIM,
                "stride_out_n": 1,
                "stride_grad_hm": H_PADDED_DIM,
                "stride_grad_hn": 1,
                "stride_hm": H_PADDED_DIM,
                "stride_hn": 1,
                "stride_grad_a": 1,
                "stride_a": 1,
                "stride_grad_b": 1,
                "stride_grad_ms": 1,
                "stride_ms": 1,
                "BLOCK_SIZE_N": H_PADDED_DIM,
                "eps": eps,
                "DETERMINISTIC": deterministic,
            },
        )


register_primitive(MHCScaleBwdPrimitive)


class MHCSinkhornFwdPrimitive(_MHCPrimitive):
    """Log-space Sinkhorn normalization of H_res into a doubly stochastic matrix."""

    name = "te_mhc_sinkhorn_fwd_triton"
    impl_static_args = (1, 2, 3)  # n, iters, recompute_hist

    @staticmethod
    def layouts(n, iters, recompute_hist):
        del n, iters
        if recompute_hist:
            return (2,), (2,)
        return (2,), (2, (1, 1), (1, 1))

    @staticmethod
    def abstract(x_aval, *, n, iters, recompute_hist):
        """Abstract."""
        M = x_aval.shape[0]
        out_aval = jax.core.ShapedArray((M, n * n), jnp.float32)
        if recompute_hist:
            return (out_aval,)
        hist_aval = jax.core.ShapedArray((iters + 1, M, n), jnp.float32)
        return out_aval, hist_aval, hist_aval

    @staticmethod
    def outer_abstract(x_aval, *, n, iters, recompute_hist):
        """Outer abstract."""
        tokens = x_aval.shape[:-2]
        out_aval = jax.core.ShapedArray(x_aval.shape, jnp.float32)
        if recompute_hist:
            return (out_aval,)
        hist_aval = jax.core.ShapedArray((iters + 1,) + tokens + (n,), jnp.float32)
        return out_aval, hist_aval, hist_aval

    @staticmethod
    def impl(x, n, iters, recompute_hist):
        """Flatten the token dims and call the inner primitive."""
        assert MHCSinkhornFwdPrimitive.inner_primitive is not None
        tokens = x.shape[:-2]
        outputs = MHCSinkhornFwdPrimitive.inner_primitive.bind(
            x.reshape(math.prod(tokens), n * n), n=n, iters=iters, recompute_hist=recompute_hist
        )
        out = outputs[0].reshape(x.shape)
        if recompute_hist:
            return (out,)
        hist_shape = (iters + 1,) + tokens + (n,)
        return out, outputs[1].reshape(hist_shape), outputs[2].reshape(hist_shape)

    @staticmethod
    def lowering(ctx, x, *, n, iters, recompute_hist):
        """MLIR lowering using triton_call_lowering."""
        M = ctx.avals_in[0].shape[0]
        kernel = _mhc_sinkhorn_fwd_fused_recompute if recompute_hist else _mhc_sinkhorn_fwd_fused

        def grid(meta):
            return (triton.cdiv(M * n * n, meta["BLOCK_SIZE"]),)

        return triton_call_lowering(
            ctx,
            kernel,
            x,
            grid=grid,
            constexprs={
                "stride_xm": n * n,
                "stride_xn": 1,
                "stride_out_m": n * n,
                "stride_out_n": 1,
                "M": M,
                "n": n,
                "iters": iters,
            },
        )


register_primitive(MHCSinkhornFwdPrimitive)


class MHCSinkhornBwdPrimitive(_MHCPrimitive):
    """Backward of the Sinkhorn normalization, recomputing the f/g history if not provided."""

    name = "te_mhc_sinkhorn_bwd_triton"
    impl_static_args = (5, 6, 7)  # n, iters, recompute_hist

    @staticmethod
    def layouts(n, iters, recompute_hist):
        del n, iters
        hist_layout = None if recompute_hist else (1, 1)
        return (2, 2, 2, hist_layout, hist_layout), (2,)

    @staticmethod
    def abstract(
        grad_out_aval,
        out_aval,
        grad_x_aval,
        x_aval,
        hist_f_aval,
        hist_g_aval,
        *,
        n,
        iters,
        recompute_hist,
    ):
        """Inner abstract: grad_x (and the recomputed history scratch) are aliased."""
        del grad_out_aval, out_aval, x_aval, n, iters
        avals = (grad_x_aval, hist_f_aval, hist_g_aval) if recompute_hist else (grad_x_aval,)
        return tuple(jax.core.ShapedArray(aval.shape, aval.dtype) for aval in avals)

    @staticmethod
    def outer_abstract(
        grad_out_aval, out_aval, x_aval, hist_f_aval, hist_g_aval, *, n, iters, recompute_hist
    ):
        """Outer abstract."""
        del grad_out_aval, out_aval, hist_f_aval, hist_g_aval, n, iters, recompute_hist
        return (jax.core.ShapedArray(x_aval.shape, jnp.float32),)

    @staticmethod
    def impl(grad_out, out, x, hist_f, hist_g, n, iters, recompute_hist):
        """Allocate grad_x (and the history scratch) and call the inner primitive."""
        assert MHCSinkhornBwdPrimitive.inner_primitive is not None
        M = math.prod(x.shape[:-2])
        if recompute_hist:
            hist_f = jnp.empty((iters + 1, M, n), dtype=jnp.float32)
            hist_g = jnp.empty((iters + 1, M, n), dtype=jnp.float32)
        else:
            hist_f = hist_f.reshape(iters + 1, M, n)
            hist_g = hist_g.reshape(iters + 1, M, n)
        outputs = MHCSinkhornBwdPrimitive.inner_primitive.bind(
            grad_out.reshape(M, n * n),
            out.reshape(M, n * n),
            jnp.empty((M, n * n), dtype=jnp.float32),
            x.reshape(M, n * n),
            hist_f,
            hist_g,
            n=n,
            iters=iters,
            recompute_hist=recompute_hist,
        )
        return (outputs[0].reshape(x.shape),)

    @staticmethod
    def lowering(ctx, grad_out, out, grad_x, x, hist_f, hist_g, *, n, iters, recompute_hist):
        """MLIR lowering using triton_call_lowering."""
        M = ctx.avals_in[0].shape[0]
        if recompute_hist:
            kernel = _mhc_sinkhorn_bwd_fused_recompute
            input_output_aliases = {2: 0, 4: 1, 5: 2}
        else:
            kernel = _mhc_sinkhorn_bwd_fused
            input_output_aliases = {2: 0}

        def grid(meta):
            return (triton.cdiv(M * n * n, meta["BLOCK_SIZE"]),)

        return triton_call_lowering(
            ctx,
            kernel,
            grad_out,
            out,
            grad_x,
            x,
            hist_f,
            hist_g,
            grid=grid,
            input_output_aliases=input_output_aliases,
            constexprs={
                "stride_grad_out_m": n * n,
                "stride_grad_out_n": 1,
                "stride_out_m": n * n,
                "stride_out_n": 1,
                "stride_grad_xm": n * n,
                "stride_grad_xn": 1,
                "stride_xm": n * n,
                "stride_xn": 1,
                "M": M,
                "n": n,
                "iters": iters,
            },
        )


register_primitive(MHCSinkhornBwdPrimitive)


class MHCAggregateFwdPrimitive(_MHCPrimitive):
    """out = x @ H_pre: (M, C, n) @ (M, n, 1) -> (M, C)."""

    name = "te_mhc_aggregate_fwd_triton"
    impl_static_args = ()

    @staticmethod
    def layouts():
        return (2, 1), (1,)

    @staticmethod
    def abstract(x_aval, h_pre_aval):
        """Abstract."""
        del h_pre_aval
        return (jax.core.ShapedArray(x_aval.shape[:-1], x_aval.dtype),)

    @staticmethod
    def outer_abstract(x_aval, h_pre_aval):
        """Outer abstract."""
        return MHCAggregateFwdPrimitive.abstract(x_aval, h_pre_aval)

    @staticmethod
    def impl(x, h_pre):
        """Flatten the token dims and call the inner primitive."""
        assert MHCAggregateFwdPrimitive.inner_primitive is not None
        (out,) = MHCAggregateFwdPrimitive.inner_primitive.bind(
            _flatten_tokens(x, 2), _flatten_tokens(h_pre, 1)
        )
        return (out.reshape(x.shape[:-1]),)

    @staticmethod
    def lowering(ctx, x, h_pre):
        """MLIR lowering using triton_call_lowering."""
        M, C, n = ctx.avals_in[0].shape

        def grid(meta):
            return (triton.cdiv(C, meta["BLOCK_SIZE_C"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

        return triton_call_lowering(
            ctx,
            _mhc_aggregate_fwd,
            x,
            h_pre,
            grid=grid,
            constexprs={
                "M": M,
                "C": C,
                "n": n,
                "stride_xm": n * C,
                "stride_xCn": 1,
                "stride_output_m": C,
                "stride_output_c": 1,
            },
        )


register_primitive(MHCAggregateFwdPrimitive)


class MHCAggregateBwdPrimitive(_MHCPrimitive):
    """grad_x = grad_out * H_pre and grad_H_pre = sum_C(grad_out * x)."""

    name = "te_mhc_aggregate_bwd_triton"
    impl_static_args = (4, 5)  # precision, fuse_grad_x_acc

    @staticmethod
    def layouts(precision, fuse_grad_x_acc):
        del precision
        return (1, 2, 1, 2 if fuse_grad_x_acc else None), (2, 1)

    @staticmethod
    def abstract(
        grad_out_aval,
        h_pre_aval,
        grad_h_pre_aval,
        x_aval,
        grad_x_aval,
        *,
        precision,
        fuse_grad_x_acc,
    ):
        """Inner abstract: grad_H_pre and grad_x are aliased to their buffers."""
        del grad_out_aval, h_pre_aval, x_aval, precision, fuse_grad_x_acc
        return (
            jax.core.ShapedArray(grad_h_pre_aval.shape, grad_h_pre_aval.dtype),
            jax.core.ShapedArray(grad_x_aval.shape, grad_x_aval.dtype),
        )

    @staticmethod
    def outer_abstract(
        grad_out_aval, x_aval, h_pre_aval, grad_x_acc_aval, *, precision, fuse_grad_x_acc
    ):
        """Outer abstract."""
        del grad_out_aval, grad_x_acc_aval, precision
        return (
            jax.core.ShapedArray(x_aval.shape, jnp.float32 if fuse_grad_x_acc else x_aval.dtype),
            jax.core.ShapedArray(h_pre_aval.shape, jnp.float32),
        )

    @staticmethod
    def impl(grad_out, x, h_pre, grad_x_acc, precision, fuse_grad_x_acc):
        """Allocate the gradient buffers and call the inner primitive."""
        assert MHCAggregateBwdPrimitive.inner_primitive is not None
        x_3d = _flatten_tokens(x, 2)
        M, C, n = x_3d.shape
        if fuse_grad_x_acc:
            grad_x_buf = grad_x_acc.reshape(M, C, n)
        else:
            grad_x_buf = jnp.empty((M, C, n), dtype=x.dtype)
        grad_h_pre, grad_x = MHCAggregateBwdPrimitive.inner_primitive.bind(
            grad_out.reshape(M, C),
            h_pre.reshape(M, n),
            jnp.zeros((M, n), dtype=jnp.float32),
            x_3d,
            grad_x_buf,
            precision=precision,
            fuse_grad_x_acc=fuse_grad_x_acc,
        )
        return grad_x.reshape(x.shape), grad_h_pre.reshape(h_pre.shape)

    @staticmethod
    def lowering(ctx, grad_out, h_pre, grad_h_pre, x, grad_x, *, precision, fuse_grad_x_acc):
        """MLIR lowering using triton_call_lowering."""
        M, C, n = ctx.avals_in[3].shape

        def grid(meta):
            return (triton.cdiv(C, meta["BLOCK_SIZE_C"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

        return triton_call_lowering(
            ctx,
            _mhc_aggregate_bwd,
            grad_out,
            h_pre,
            grad_h_pre,
            x,
            grad_x,
            grid=grid,
            input_output_aliases={2: 0, 4: 1},
            constexprs={
                "M": M,
                "C": C,
                "n": n,
                "stride_grad_output_m": C,
                "stride_grad_output_c": 1,
                "stride_xm": n * C,
                "stride_xCn": 1,
                "stride_grad_xm": n * C,
                "stride_grad_xCn": 1,
                "precision": precision,
                "FUSE_GRAD_X_ACC": fuse_grad_x_acc,
            },
        )


register_primitive(MHCAggregateBwdPrimitive)


class MHCExpandCombineFwdPrimitive(_MHCPrimitive):
    """out = (f [+ bias]) @ H_post + x @ H_res -> (M, C, n)."""

    name = "te_mhc_expand_combine_fwd_triton"
    impl_static_args = (5,)  # has_bias

    @staticmethod
    def layouts(has_bias):
        del has_bias
        return (1, None, 1, 2, 2), (2,)

    @staticmethod
    def abstract(f_aval, bias_aval, h_post_aval, x_aval, h_res_aval, *, has_bias):
        """Abstract."""
        del f_aval, bias_aval, h_post_aval, h_res_aval, has_bias
        return (jax.core.ShapedArray(x_aval.shape, x_aval.dtype),)

    @staticmethod
    def outer_abstract(f_aval, bias_aval, h_post_aval, x_aval, h_res_aval, *, has_bias):
        """Outer abstract."""
        return MHCExpandCombineFwdPrimitive.abstract(
            f_aval, bias_aval, h_post_aval, x_aval, h_res_aval, has_bias=has_bias
        )

    @staticmethod
    def impl(f, bias, h_post, x, h_res, has_bias):
        """Flatten the token dims and call the inner primitive."""
        assert MHCExpandCombineFwdPrimitive.inner_primitive is not None
        (out,) = MHCExpandCombineFwdPrimitive.inner_primitive.bind(
            _flatten_tokens(f, 1),
            bias,
            _flatten_tokens(h_post, 1),
            _flatten_tokens(x, 2),
            _flatten_tokens(h_res, 2),
            has_bias=has_bias,
        )
        return (out.reshape(x.shape),)

    @staticmethod
    def lowering(ctx, f, bias, h_post, x, h_res, *, has_bias):
        """MLIR lowering using triton_call_lowering."""
        M, C, n = ctx.avals_in[3].shape

        def grid(meta):
            return (triton.cdiv(C, meta["BLOCK_SIZE_C"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

        return triton_call_lowering(
            ctx,
            _mhc_expand_combine_fwd,
            f,
            bias,
            h_post,
            x,
            h_res,
            grid=grid,
            constexprs={
                "M": M,
                "C": C,
                "n": n,
                "stride_fm": C,
                "stride_fc": 1,
                "stride_bias": 1,
                "stride_xm": C * n,
                "stride_xCn": 1,
                "stride_output_m": C * n,
                "stride_output_Cn": 1,
                "HAS_BIAS": has_bias,
            },
        )


register_primitive(MHCExpandCombineFwdPrimitive)


class MHCExpandCombineBwdPrimitive(_MHCPrimitive):
    """Gradients of expand-combine with respect to f, bias, H_post, x and H_res."""

    name = "te_mhc_expand_combine_bwd_triton"
    impl_static_args = (7, 8, 9, 10)  # precision, has_bias, fuse_grad_x_acc, deterministic
    reduced_outputs = (1,)

    @staticmethod
    def layouts(precision, has_bias, fuse_grad_x_acc, deterministic):
        del precision, has_bias, deterministic
        return (
            (2, 1, None, 1, 2, 2, 2 if fuse_grad_x_acc else None),
            (1, None, 1, 2, 2),
        )

    @staticmethod
    def abstract(
        grad_out_aval,
        f_aval,
        bias_aval,
        h_post_aval,
        x_aval,
        h_res_aval,
        grad_h_post_aval,
        grad_f_aval,
        grad_bias_aval,
        grad_bias_ws_aval,
        grad_h_res_aval,
        grad_x_aval,
        *,
        precision,
        has_bias,
        fuse_grad_x_acc,
        deterministic,
    ):
        """Inner abstract: every output is aliased to its buffer."""
        del grad_out_aval, f_aval, bias_aval, h_post_aval, x_aval, h_res_aval
        del precision, has_bias, fuse_grad_x_acc, deterministic
        return tuple(
            jax.core.ShapedArray(aval.shape, aval.dtype)
            for aval in (
                grad_h_post_aval,
                grad_f_aval,
                grad_bias_aval,
                grad_bias_ws_aval,
                grad_h_res_aval,
                grad_x_aval,
            )
        )

    @staticmethod
    def outer_abstract(
        grad_out_aval,
        f_aval,
        bias_aval,
        h_post_aval,
        x_aval,
        h_res_aval,
        grad_x_acc_aval,
        *,
        precision,
        has_bias,
        fuse_grad_x_acc,
        deterministic,
    ):
        """Outer abstract."""
        del grad_out_aval, grad_x_acc_aval, precision, deterministic
        return (
            jax.core.ShapedArray(f_aval.shape, f_aval.dtype),
            jax.core.ShapedArray(bias_aval.shape if has_bias else (1,), jnp.float32),
            jax.core.ShapedArray(h_post_aval.shape, h_post_aval.dtype),
            jax.core.ShapedArray(x_aval.shape, jnp.float32 if fuse_grad_x_acc else x_aval.dtype),
            jax.core.ShapedArray(h_res_aval.shape, h_res_aval.dtype),
        )

    @staticmethod
    def impl(
        grad_out,
        f,
        bias,
        h_post,
        x,
        h_res,
        grad_x_acc,
        precision,
        has_bias,
        fuse_grad_x_acc,
        deterministic,
    ):
        """Allocate the gradient buffers and workspace, then reduce the workspace."""
        assert MHCExpandCombineBwdPrimitive.inner_primitive is not None
        x_3d = _flatten_tokens(x, 2)
        M, C, n = x_3d.shape
        if fuse_grad_x_acc:
            grad_x_buf = grad_x_acc.reshape(M, C, n)
        else:
            grad_x_buf = jnp.empty((M, C, n), dtype=x.dtype)
        if has_bias:
            grad_bias_buf = jnp.zeros(bias.shape, dtype=jnp.float32)
        else:
            grad_bias_buf = _empty_aliased(jnp.float32)
        if has_bias and deterministic:
            grad_bias_ws = jnp.empty(
                (triton.cdiv(M, EXPAND_COMBINE_BWD_BLOCK_SIZE_M), C), dtype=jnp.float32
            )
        else:
            grad_bias_ws = _empty_aliased(jnp.float32)

        grad_h_post, grad_f, grad_bias, grad_bias_ws, grad_h_res, grad_x = (
            MHCExpandCombineBwdPrimitive.inner_primitive.bind(
                grad_out.reshape(M, C, n),
                f.reshape(M, C),
                bias,
                h_post.reshape(M, n),
                x_3d,
                h_res.reshape(M, n, n),
                jnp.empty((M, n), dtype=h_post.dtype),
                jnp.empty((M, C), dtype=f.dtype),
                grad_bias_buf,
                grad_bias_ws,
                jnp.empty((M, n, n), dtype=h_res.dtype),
                grad_x_buf,
                precision=precision,
                has_bias=has_bias,
                fuse_grad_x_acc=fuse_grad_x_acc,
                deterministic=deterministic,
            )
        )
        if has_bias and deterministic:
            grad_bias = grad_bias_ws.sum(axis=0)
        return (
            grad_f.reshape(f.shape),
            grad_bias,
            grad_h_post.reshape(h_post.shape),
            grad_x.reshape(x.shape),
            grad_h_res.reshape(h_res.shape),
        )

    @staticmethod
    def lowering(
        ctx,
        grad_out,
        f,
        bias,
        h_post,
        x,
        h_res,
        grad_h_post,
        grad_f,
        grad_bias,
        grad_bias_ws,
        grad_h_res,
        grad_x,
        *,
        precision,
        has_bias,
        fuse_grad_x_acc,
        deterministic,
    ):
        """MLIR lowering using triton_call_lowering."""
        M, C, n = ctx.avals_in[4].shape

        def grid(meta):
            return (triton.cdiv(C, meta["BLOCK_SIZE_C"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

        return triton_call_lowering(
            ctx,
            _mhc_expand_combine_bwd,
            grad_out,
            f,
            bias,
            h_post,
            x,
            h_res,
            grad_h_post,
            grad_f,
            grad_bias,
            grad_bias_ws,
            grad_h_res,
            grad_x,
            grid=grid,
            input_output_aliases={6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5},
            constexprs={
                "M": M,
                "C": C,
                "n": n,
                "stride_grad_output_m": n * C,
                "stride_grad_output_Cn": 1,
                "stride_fm": C,
                "stride_fc": 1,
                "stride_bias": 1,
                "stride_xm": n * C,
                "stride_xCn": 1,
                "stride_grad_fm": C,
                "stride_grad_fc": 1,
                "stride_grad_bias": 1,
                "stride_grad_bias_ws_m": C,
                "stride_grad_bias_ws_c": 1,
                "stride_grad_xm": n * C,
                "stride_grad_xCn": 1,
                "precision": precision,
                "HAS_BIAS": has_bias,
                "FUSE_GRAD_X_ACC": fuse_grad_x_acc,
                "DETERMINISTIC": deterministic,
            },
        )


register_primitive(MHCExpandCombineBwdPrimitive)


def _check_grad_x_acc(grad_x_acc, x):
    if grad_x_acc is not None:
        assert grad_x_acc.dtype == jnp.float32, "grad_x_acc must be fp32"
        assert grad_x_acc.shape == x.shape, "grad_x_acc must have the same shape as x"


def mhc_projection_fwd(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused projection: H = x @ (phi * norm_weight)^T and ms = mean(x^2, dim=-1).

    Parameters
    ----------
    x : jnp.ndarray
        Input of shape (..., K), where the leading dims are tokens and K = n * C.
    phi : jnp.ndarray
        Projection matrix of shape (N, K), where N = 2n + n*n (= 24 for n = 4).
    norm_weight : Optional[jnp.ndarray]
        RMSNorm weight of shape (K,), absorbed into phi.
    use_tf32 : bool
        Whether to use TF32 for the matmul. If False, uses IEEE (or TF32x3 for bf16 x and fp32 phi).
    use_split_k : bool
        Whether to reduce over split-K blocks with atomic adds (non-deterministic).

    Returns
    -------
    H : jnp.ndarray
        fp32 array of shape (..., 32), where only the first N columns are valid.
    ms : jnp.ndarray
        fp32 mean square of shape (...,).
    """
    assert (
        phi.shape[0] == 24
    ), "Currently only n=4 is supported, which means phi should have 24 in its first dimension"
    check_deterministic("mhc_projection_fwd", use_split_k)
    precision = _projection_precision(x.dtype, phi.dtype, norm_weight is not None, use_tf32)
    if norm_weight is not None:
        phi = phi * norm_weight.astype(jnp.float32)
    elif not use_tf32 and x.dtype == jnp.bfloat16 and phi.dtype == jnp.bfloat16:
        # tl.dot ignores input_precision for bf16 x bf16, so upcast phi to honor tf32x3.
        phi = phi.astype(jnp.float32)
    return MHCProjectionFwdPrimitive.outer_primitive.bind(
        x, phi, precision=precision, use_split_k=use_split_k
    )


def mhc_projection_bwd(
    grad_h: jnp.ndarray,
    grad_ms: jnp.ndarray,
    x: jnp.ndarray,
    phi: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
    grad_x_acc: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Backward of `mhc_projection_fwd`.

    Parameters
    ----------
    grad_h : jnp.ndarray
        Gradient of H, of shape (..., 32).
    grad_ms : jnp.ndarray
        Gradient of ms, of shape (...,).
    x, phi, norm_weight, use_tf32, use_split_k :
        The arguments passed to `mhc_projection_fwd`.
    grad_x_acc : Optional[jnp.ndarray]
        fp32 buffer of x's shape that grad_x is accumulated into. If given, the returned grad_x
        is the updated fp32 buffer.

    Returns
    -------
    grad_x, grad_phi, grad_norm_weight (None if norm_weight is None).
    """
    check_deterministic("mhc_projection_bwd", use_split_k)
    _check_grad_x_acc(grad_x_acc, x)
    N = phi.shape[0]
    grad_h = grad_h.astype(jnp.float32)
    grad_ms = grad_ms.astype(jnp.float32)
    has_norm_weight = norm_weight is not None

    if has_norm_weight:
        grad_phi, grad_norm_weight = MHCProjectionBwdDphiPrimitive.outer_primitive.bind(
            x,
            grad_h,
            phi,
            norm_weight,
            precision="tf32" if use_tf32 else "ieee",
            use_split_m=use_split_k,
        )
        grad_phi = grad_phi.astype(phi.dtype)
        grad_norm_weight = grad_norm_weight.astype(norm_weight.dtype)
    else:
        token_axes = tuple(range(x.ndim - 1))
        grad_phi = jax.lax.dot_general(
            grad_h[..., :N],
            x.astype(grad_h.dtype),
            ((token_axes, token_axes), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(phi.dtype)
        grad_norm_weight = None

    (grad_x,) = MHCProjectionBwdDxPrimitive.outer_primitive.bind(
        x,
        phi,
        norm_weight if has_norm_weight else _unused(phi.dtype),
        grad_h,
        grad_ms,
        grad_x_acc if grad_x_acc is not None else _unused(jnp.float32),
        precision=_projection_precision(x.dtype, phi.dtype, has_norm_weight, use_tf32),
        has_norm_weight=has_norm_weight,
        fuse_grad_x_acc=grad_x_acc is not None,
    )
    return grad_x, grad_phi, grad_norm_weight


def mhc_scale_fwd(
    h: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    ms: jnp.ndarray,
    n: int = 4,
    eps: float = DEFAULT_NORM_EPS,
) -> jnp.ndarray:
    """
    Fused scale producing H_pre, H_post and H_res (padded to 32 columns):

    H_pre  = sigmoid(H[:, 0:n] * alpha[0] / sqrt(ms + eps) + beta[0:n])
    H_post = 2 * sigmoid(H[:, n:2n] * alpha[1] / sqrt(ms + eps) + beta[n:2n])
    H_res  = H[:, 2n:2n+n*n] * alpha[2] / sqrt(ms + eps) + beta[2n:2n+n*n]

    Parameters
    ----------
    h : jnp.ndarray
        H from `mhc_projection_fwd`, of shape (..., 32).
    alpha : jnp.ndarray
        Scaling factors of shape (3,).
    beta : jnp.ndarray
        Bias with 2n + n*n elements, e.g. of shape (1, 2n + n*n).
    ms : jnp.ndarray
        Mean square from `mhc_projection_fwd`, of shape (...,).
    n : int
        Number of hyper connections (only n=4 is supported).
    eps : float
        RMSNorm epsilon. Defaults to the fp32 machine epsilon used by the PyTorch API.

    Returns
    -------
    Array of shape (..., 32) in h's dtype, where only the first 2n + n*n columns are valid.
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    (out,) = MHCScaleFwdPrimitive.outer_primitive.bind(
        h.astype(jnp.float32),
        alpha.astype(jnp.float32),
        beta.astype(jnp.float32).reshape(-1),
        ms.astype(jnp.float32),
        n=n,
        eps=float(eps),
    )
    return out.astype(h.dtype)


def mhc_scale_bwd(
    grad_out: jnp.ndarray,
    h: jnp.ndarray,
    alpha: jnp.ndarray,
    ms: jnp.ndarray,
    out: jnp.ndarray,
    n: int = 4,
    eps: float = DEFAULT_NORM_EPS,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Backward of `mhc_scale_fwd`, where `out` is its output and `eps` its RMSNorm epsilon.

    Returns
    -------
    grad_h (fp32, (..., 32)), grad_alpha ((3,), alpha's dtype),
    grad_beta ((2n + n*n,), alpha's dtype), grad_ms (fp32, (...,)).
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    grad_h, grad_alpha, grad_beta, grad_ms = MHCScaleBwdPrimitive.outer_primitive.bind(
        grad_out.astype(jnp.float32),
        out.astype(jnp.float32),
        h.astype(jnp.float32),
        alpha.astype(jnp.float32),
        ms.astype(jnp.float32),
        n=n,
        eps=float(eps),
        deterministic=ENFORCE_DETERMINISTIC,
    )
    return grad_h, grad_alpha.astype(alpha.dtype), grad_beta.astype(alpha.dtype), grad_ms


def mhc_sinkhorn_fwd(
    h_res: jnp.ndarray,
    n: int = 4,
    recompute_hist: bool = True,
    iters: int = 20,
) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
    """
    Sinkhorn normalization of H_res into a doubly stochastic matrix.

    Parameters
    ----------
    h_res : jnp.ndarray
        Input of shape (..., n, n).
    n : int
        Number of hyper connections (only n=4 is supported).
    recompute_hist : bool
        Whether the backward recomputes the f/g history. If False, the history is returned.
    iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    out : jnp.ndarray
        Doubly stochastic matrix of shape (..., n, n) in h_res's dtype.
    hist : Optional[Tuple[jnp.ndarray, jnp.ndarray]]
        fp32 (hist_f, hist_g), each of shape (iters + 1, ..., n), or None if recompute_hist.
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    outputs = MHCSinkhornFwdPrimitive.outer_primitive.bind(
        h_res.astype(jnp.float32), n=n, iters=iters, recompute_hist=recompute_hist
    )
    out = outputs[0].astype(h_res.dtype)
    hist = None if recompute_hist else (outputs[1], outputs[2])
    return out, hist


def mhc_sinkhorn_bwd(
    grad_out: jnp.ndarray,
    h_res: jnp.ndarray,
    out: jnp.ndarray,
    hist: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    n: int = 4,
    iters: int = 20,
) -> jnp.ndarray:
    """
    Backward of `mhc_sinkhorn_fwd`, where `out` and `hist` are its outputs.

    The f/g history is recomputed if `hist` is None. Returns grad_h_res in h_res's dtype.
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    recompute_hist = hist is None
    if recompute_hist:
        hist_f = hist_g = _unused(jnp.float32)
    else:
        hist_f, hist_g = hist
    (grad_h_res,) = MHCSinkhornBwdPrimitive.outer_primitive.bind(
        grad_out,
        out.astype(jnp.float32),
        h_res.astype(jnp.float32),
        hist_f,
        hist_g,
        n=n,
        iters=iters,
        recompute_hist=recompute_hist,
    )
    return grad_h_res.astype(h_res.dtype)


def mhc_aggregate_fwd(x: jnp.ndarray, h_pre: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregate the n streams: out = x @ H_pre, (..., C, n) @ (..., n, 1) -> (..., C).
    """
    assert x.shape[-1] == 4, "Only n=4 is supported in this implementation"
    (out,) = MHCAggregateFwdPrimitive.outer_primitive.bind(x, h_pre)
    return out


def mhc_aggregate_bwd(
    grad_out: jnp.ndarray,
    x: jnp.ndarray,
    h_pre: jnp.ndarray,
    use_tf32: bool = True,
    grad_x_acc: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Backward of `mhc_aggregate_fwd`.

    If `grad_x_acc` (fp32, x's shape) is given, grad_x is accumulated into it and the updated
    buffer is returned. Returns (grad_x, grad_h_pre in h_pre's dtype).
    """
    assert x.shape[-1] == 4, "Only n=4 is supported in this implementation"
    _check_grad_x_acc(grad_x_acc, x)
    grad_x, grad_h_pre = MHCAggregateBwdPrimitive.outer_primitive.bind(
        grad_out,
        x,
        h_pre,
        grad_x_acc if grad_x_acc is not None else _unused(jnp.float32),
        precision="tf32" if use_tf32 else "ieee",
        fuse_grad_x_acc=grad_x_acc is not None,
    )
    return grad_x, grad_h_pre.astype(h_pre.dtype)


def mhc_expand_combine_fwd(
    f: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    h_post: jnp.ndarray,
    x: jnp.ndarray,
    h_res: jnp.ndarray,
) -> jnp.ndarray:
    """
    Expand and combine: out = (f [+ bias]) @ H_post + x @ H_res.

    Shapes: f (..., C), bias (C,) or None, h_post (..., n), x (..., C, n), h_res (..., n, n).
    Returns an array of shape (..., C, n) in x's dtype.
    """
    assert x.shape[-1] == 4, "Only n=4 is supported in this implementation"
    has_bias = bias is not None
    (out,) = MHCExpandCombineFwdPrimitive.outer_primitive.bind(
        f,
        bias if has_bias else _unused(f.dtype),
        h_post,
        x,
        h_res,
        has_bias=has_bias,
    )
    return out


def mhc_expand_combine_bwd(
    grad_out: jnp.ndarray,
    f: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    h_post: jnp.ndarray,
    x: jnp.ndarray,
    h_res: jnp.ndarray,
    use_tf32: bool = True,
    grad_x_acc: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Backward of `mhc_expand_combine_fwd`.

    If `grad_x_acc` (fp32, x's shape) is given, grad_x is accumulated into it and the updated
    buffer is returned. Returns (grad_f, grad_bias or None, grad_h_post, grad_x, grad_h_res).
    """
    assert x.shape[-1] == 4, "Only n=4 is supported in this implementation"
    _check_grad_x_acc(grad_x_acc, x)
    has_bias = bias is not None
    grad_f, grad_bias, grad_h_post, grad_x, grad_h_res = (
        MHCExpandCombineBwdPrimitive.outer_primitive.bind(
            grad_out,
            f,
            bias if has_bias else _unused(f.dtype),
            h_post,
            x,
            h_res,
            grad_x_acc if grad_x_acc is not None else _unused(jnp.float32),
            precision="tf32" if use_tf32 else "ieee",
            has_bias=has_bias,
            fuse_grad_x_acc=grad_x_acc is not None,
            deterministic=ENFORCE_DETERMINISTIC,
        )
    )
    grad_bias = grad_bias.astype(bias.dtype) if has_bias else None
    return grad_f, grad_bias, grad_h_post, grad_x, grad_h_res
