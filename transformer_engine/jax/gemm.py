# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import warnings
import operator
from functools import partial, reduce
from typing import Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from transformer_engine import transformer_engine_jax as tex
from .fp8 import FP8Helper, FP8MetaPackage
from .cpp_extensions import (
    gemm_impl,
    fp8_gemm_impl,
    cast_transpose,
    dact_lu,
    dbias_cast_transpose,
    dact_lu_dbias_cast_transpose,
    get_num_max_compute_streams,
)

from .cpp_extensions.gemm import sanitize_dims, mirror_dim, copy_into_overlap_buffer
from .cpp_extensions.misc import jax_dtype_is_fp8, jax_dtype_to_te_dtype
from .sharding import get_mesh_axis_size, global_mesh_resource


__all__ = [
    "gemm",
    "fp8_gemm",
    "type_safe_gemm",
    "initialize_comm_gemm_overlaps",
    "destroy_comm_gemm_overlap",
    "get_comm_gemm_overlap_config",
]


_ACTIVE_COMM_GEMM_OVERLAPS = dict()


def gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Optional[ArrayLike] = None,
    contracting_dims: Tuple[int, int] = (-1, -2),
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    comm_overlap_name: Optional[str] = None,
    ag_overlap_skip_copy: bool = False,
) -> ArrayLike:
    """
    Non-FP8 collective/distributed `nvte_cublas_gemm()` with GELU and bias-add fusions.

    Parameters
    ----------
    x : ArrayLike
        LHS operand, sized ([B], M, K) when not transposed.
    kernel : ArrayLike
        RHS operand, sized (K, N) when not transposed.
    bias : Optional[ArrayLike], default = `None`
        Optional bias term to add onto the (LHS x RHS) result.
    contracting_dims : Tuple[int, int], default = `(-1, 0)`
        Contracting dimensions of LHS and RHS, respectively, in the matrix-multiplication.
        The default (-1, 0) describes the fully non-transposed 'NN' layout where LHS contracts in
        the last dimension, and RHS contracts in the first dimension.
    fuse_gelu : bool, default = `False`
        Enable the GELU epilogue for GEMM. This applies GELU after the bias-addition if the bias
        term is not `None`.
    accumulate : bool, default = `False`
    use_split_accumulator : bool, default = `False`
    comm_overlap_name : Optional[str], default = `None`
        Name of the comm+GEMM overlap layer that this GEMM is associated with. Comm+GEMM overlap
        must be initialized with `te.jax.gemm.initialize_comm_gemm_overlaps()` before this
        GEMM call, and the configuration dictionary used in the initialization must include
        the name passed into this function.
    ag_overlap_skip_copy: bool = `False`
        All-gather overlap requires the LHS operand to be copied into the communication buffer.
        If the communication buffer already has the necessary data, setting this flag will
        avoid an unnecessary memcpy operation.
    """
    comm_overlap_config = None
    if comm_overlap_name is not None:
        global _ACTIVE_COMM_GEMM_OVERLAPS
        comm_overlap_layer = (
            comm_overlap_name + "_fprop"
            if comm_overlap_name not in ["ag_gemm", "gemm_rs"]
            else comm_overlap_name
        )
        comm_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(comm_overlap_layer, None)
        if comm_overlap_config is None:
            warnings.warn(
                f"Comm+GEMM overlap for {comm_overlap_name} has not been initialized! "
                + "Sharded operands will trigger XLA collectives instead."
            )

        elif (
            not ag_overlap_skip_copy
            and comm_overlap_config["method"] != "bulk"
            and comm_overlap_config["comm_type"] == tex.CommOverlapType.AG
        ):
            if sanitize_dims(contracting_dims[0], x.ndim) != x.ndim - 1:
                x = jnp.matrix_transpose(x)
            copy_into_overlap_buffer(x, comm_overlap_name, True)

    return _gemm(
        x,
        kernel,
        bias,
        contracting_dims,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Union[ArrayLike, None],
    contracting_dims: Tuple[int, int],
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap_config: dict,
) -> ArrayLike:
    out, _ = _gemm_fwd_rule(
        x,
        kernel,
        bias,
        contracting_dims,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
    )
    return out


def _gemm_fwd_rule(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike,
    contracting_dims: Tuple[int, int],
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap_config: dict,
) -> Tuple[ArrayLike, ...]:
    assert (
        kernel.ndim == 2
    ), "TE/JAX Collective GEMM custom op does not support batched RHS operand in forward mode."

    fuse_bias = bias is not None

    # AG+GEMM: ([B], M/P, K) --(AG)--> ([B], M, K) x (K, N/P) --> ([B], M, N/P)
    #
    # GEMM+AR: ([B], M, K/P) x (K/P, N) --(AR)--> ([B], M, N)
    #
    # GEMM+RS: ([B], M, K/P) x (K/P, N) --(RS)--> ([B], M/P, N)
    out, pre_gelu_out, extra_out = gemm_impl(
        x,
        kernel,
        bias=bias,
        batched_output=(x.ndim > 2),
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=comm_overlap_config,
    )

    final_out = out
    if (
        comm_overlap_config is not None
        and comm_overlap_config["method"] != "bulk"
        and comm_overlap_config["comm_type"] == tex.CommOverlapType.RS
    ):
        # Non-bulk RS overlap output is in extra output, not usual output
        final_out = extra_out

    ctx = (
        x,
        kernel,
        pre_gelu_out if fuse_gelu else None,
        fuse_bias,
    )

    return final_out, ctx


def _gemm_bwd_rule(
    contracting_dims,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
    comm_overlap_config,
    ctx,
    grad,
):
    x, kernel, pre_gelu_out, fuse_bias = ctx
    x_inner_dim, kernel_inner_dim = map(sanitize_dims, contracting_dims, (x.ndim, kernel.ndim))
    x_outer_dim, kernel_outer_dim = map(
        mirror_dim, (x_inner_dim, kernel_inner_dim), (x.ndim, kernel.ndim)
    )

    # Recover DGRAD and WGRAD comm+GEMM overlap configs
    dgrad_overlap_name = None
    dgrad_overlap_config = None
    wgrad_overlap_name = None
    wgrad_overlap_config = None
    if comm_overlap_config is not None:
        dgrad_overlap_name = comm_overlap_config["name"].rstrip("_fprop") + "_dgrad"
        dgrad_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(dgrad_overlap_name, None)
        wgrad_overlap_name = comm_overlap_config["name"].rstrip("_fprop") + "_wgrad"
        wgrad_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(wgrad_overlap_name, None)

    dgrad_pre_rs = None
    if dgrad_overlap_config is not None:
        if dgrad_overlap_config["method"] == "bulk":
            # Set DGRAD output buffer to the comm buffer of WGRAD GEMM in order to do the
            # bulk RS overlap without an extra memcpy.
            assert (
                wgrad_overlap_config is not None
            ), f"Missing comm+GEMM overlap config for {wgrad_overlap_name}!"
            dgrad_pre_rs = tex.get_overlap_buffer(wgrad_overlap_name, False)

            # Copy transposed input into the DGRAD overlap buffer for bulk AG.
            copy_into_overlap_buffer(jnp.matrix_transpose(x), dgrad_overlap_name, True)

    # FWD MODE:
    #     AG+GEMM: ([B], M/P, K) --(AG)--> ([B], M, K) x (K, N/P) ------> ([B], M, N/P)
    #
    #     GEMM+AR: ([B], M, K/P) x (K/P, N) --(AR)--> ([B], M, N)
    #
    #     GEMM+RS: ([B], M, K/P) x (K/P, N) --(RS)--> ([B], M/P, N)

    # DGRAD w/o Overlap:
    #    AG+GEMM: ([B], M, N/P) x (K, N/P)^T ---(AR)---> ([B], M, K)
    #
    #    GEMM+AR: ([B], M, N) x (K/P, N)^T ----> ([B], M, K/P)
    #
    # DGRAD w/ Overlap:
    #    AG+GEMM w/ DGRAD+RS Overlap: ([B], M, N/P) x (K, N/P)^T ---(RS)---> ([B], M/P, K)
    #
    #    AG+GEMM w/ Bulk AG Overlap: ([B], M, N/P) x (K, N/P)^T -----> ([B], M, K) (deferred RS)
    #                                ([B], M, K/P)^T --(Bulk AG)--> ([B], M, K)^T (needed in WGRAD)
    #
    #    GEMM+RS: ([B], M/P, N) --(AG)--> ([B], M, N) x (K/P, N)^T ----> ([B], M, K/P)
    dgrad, dgelu, _, dgrad_extra_out = gemm_impl(
        grad,
        kernel,
        out=dgrad_pre_rs,
        gelu_input=pre_gelu_out,
        batched_output=(x.ndim > 2),
        contracting_dims=(-1, kernel_outer_dim),
        fuse_gelu=fuse_gelu,
        fuse_bias=False,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=dgrad_overlap_config,
    )

    if (
        dgrad_overlap_config is not None
        and dgrad_overlap_config["method"] != "bulk"
        and dgrad_overlap_config["comm_type"] == tex.CommOverlapType.RS
    ):
        # Otherwise, if DGRAD overlap is RS overlap, DGRAD output is the extra output tensor
        dgrad = dgrad_extra_out

    # WGRAD w/o Overlap:
    #    AG+GEMM: ([B], M/P, K)^T --(AG)--> ([B], M, K)^T x ([B], M, N/P) --> (K, N/P)
    #
    #    GEMM+AR: ([B], M, K/P)^T --(AG)--> ([B], M, K)^T x ([B], M, N) ---------> (K, N)
    #
    # WGRAD w/ Overlap:
    #    AG+GEMM w/ DGRAD+RS Overlap: ([B], M, K/P)^T --(AG)--> ([B], M, K)^T x ([B], M, N/P) --> (K, N/P)
    #
    #    AG+GEMM w/ Bulk Overlaps: ([B], M, K)^T x ([B], M, N/P) --> (K, N/P)
    #                              ([B], M, K) --(Bulk RS)--> ([B], M/P, K) (finalize DGRAD)
    #
    #    GEMM+RS: ([B], M, K/P)^T x ([B], M, N) --> (K/P, N) (re-use all-gathered GRAD from DGRAD)
    wgrad_rhs = dgelu if fuse_gelu else (grad if comm_overlap_config is None else dgrad_extra_out)
    wgrad, _, bgrad, wgrad_extra_out = gemm_impl(
        x,
        wgrad_rhs,
        gelu_input=pre_gelu_out,
        batched_output=False,
        contracting_dims=(x_outer_dim, wgrad_rhs.ndim - 2),
        fuse_gelu=False,
        fuse_bias=fuse_bias,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=wgrad_overlap_config,
    )

    if (
        wgrad_overlap_config is not None
        and wgrad_overlap_config["method"] == "bulk"
        and wgrad_overlap_config["comm_type"] == tex.CommOverlapType.RS
    ):
        # DGRAD was reduce-scattered during WGRAD GEMM, so set DGRAD to WGRAD extra output here
        dgrad = wgrad_extra_out

    if not fuse_bias:
        bgrad = None

    return dgrad, wgrad, bgrad


_gemm.defvjp(_gemm_fwd_rule, _gemm_bwd_rule)


def fp8_gemm(
    x: ArrayLike,
    kernel_t: ArrayLike,
    fp8_meta: FP8MetaPackage,
    bias: Optional[ArrayLike] = None,
    out: Optional[ArrayLike] = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    comm_overlap_name: Optional[str] = None,
    ag_overlap_skip_copy: bool = False,
) -> ArrayLike:
    """
    FP8 collective/distributed `nvte_cublas_gemm()` with GELU and bias-add fusions.

    FP8 GEMM requires the LHS operand to be non-transposed, and the RHS operand to be transposed,
    such that the contracting dimensions are always the last dimension for both operands.

    Parameters
    ----------
    x : ArrayLike
        Non-transposed LHS operand, sized ([B], M, K).
    kernel_t : ArrayLike
        Transposed RHS operand, sized (N, K).
    fp8_meta : transformer_engine.jax.fp8.FP8MetaPackage
        FP8MetaPackage object carrying amax, scale and scale_inv information for the GEMM operands.
    bias : Optional[ArrayLike], default = `None`
        Optional bias term to add onto the (LHS x RHS) result.
    out: Optional[ArrayLike], default = `None`
        Optional empty buffer for FP8 GEMM output.
    out_dtype : jnp.dtype, default = `jnp.bfloat16`
        Data type of the FP8 GEMM output. If chosen as an FP8 dtype (i.e. `jnp.float8_e4m3fn` or
        `jnp.float8_e5m2`), the `fp8_meta` must also contain amax and scale information for the
        GEMM output. This option is overridden by the data type of the `out` buffer, if given.
    fuse_gelu : bool, default = `False`
        Enable the GELU epilogue for GEMM. This applies GELU after the bias-addition if the bias
        term is not `None`.
    accumulate : bool, default = `False`
    use_split_accumulator : bool, default = `False`
    comm_overlap_name : Optional[str], default = `None`
        Name of the comm+GEMM overlap layer that this GEMM is associated with. Comm+GEMM overlap
        must be initialized with `te.jax.gemm.initialize_comm_gemm_overlaps()` before this
        GEMM call, and the configuration dictionary used in the initialization must include
        the name passed into this function.
    ag_overlap_skip_copy: bool = `False`
        All-gather overlap requires the LHS operand to be copied into the communication buffer.
        If the communication buffer already has the necessary data, setting this flag will
        avoid an unnecessary memcpy operation.
    """
    comm_overlap_config = None
    if comm_overlap_name is not None:
        comm_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(comm_overlap_name, None)
        if comm_overlap_config is None:
            warnings.warn(
                f"Comm+GEMM overlap for {comm_overlap_name} has not been initialized! "
                + "Sharded operands will trigger XLA collectives instead."
            )

        elif (
            not ag_overlap_skip_copy
            and comm_overlap_config["method"] != "bulk"
            and comm_overlap_config["comm_type"] == tex.CommOverlapType.AG
        ):
            copy_into_overlap_buffer(x, comm_overlap_name, True)

    return _fp8_gemm(
        x,
        kernel_t,
        bias,
        fp8_meta.amax_list,
        fp8_meta.scale_list,
        out_dtype,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def _fp8_gemm(
    x: ArrayLike,
    kernel_t: ArrayLike,
    bias: ArrayLike,
    amax_list: ArrayLike,
    scale_list: ArrayLike,
    out: ArrayLike,
    out_dtype: jnp.dtype,
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap_config: dict,
) -> ArrayLike:
    out, _ = _fp8_gemm_fwd_rule(
        x,
        kernel_t,
        bias,
        amax_list,
        scale_list,
        out_dtype,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
    )
    return out


def _fp8_gemm_fwd_rule(
    x: ArrayLike,
    kernel_t: ArrayLike,
    bias: ArrayLike,
    amax_list: ArrayLike,
    scale_list: ArrayLike,
    out_dtype: jnp.dtype,
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap_config: dict,
) -> Tuple[ArrayLike, ...]:
    assert (
        kernel_t.ndim == 2
    ), "TE/JAX Collective GEMM custom op does not support batched RHS operand in forward mode."

    fuse_bias = bias is not None

    maybe_fm32_to_fp32, maybe_fp32_to_fm32 = FP8Helper.generate_fp8_meta_dtype_converter_pair(
        *amax_list,
        *scale_list,
    )
    amax_list = maybe_fm32_to_fp32(*amax_list)
    scale_list = maybe_fm32_to_fp32(*scale_list)

    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    fp8_dtype_list = [fwd_dtype, fwd_dtype, bwd_dtype, fwd_dtype]
    scale_list, scale_inv_list = FP8MetaPackage.update_fp8_scale(
        amax_list, scale_list, fp8_dtype_list
    )
    amax_list = FP8MetaPackage.update_amax_list(amax_list)

    x_amax = amax_list[FP8MetaPackage.INPUT_IDX][0:1]
    x_scale = scale_list[FP8MetaPackage.INPUT_IDX]
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    if x.dtype not in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        casted_x, casted_x_t, updated_x_amax = cast_transpose(
            x,
            x_amax,
            x_scale,
            x_scale_inv,
            fwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
    else:
        casted_x = x
        casted_x_t = jnp.matrix_transpose(x)
        updated_x_amax = x_amax

    kernel_amax = amax_list[FP8MetaPackage.WEIGHT_IDX][0:1]
    kernel_scale = scale_list[FP8MetaPackage.WEIGHT_IDX]
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    if kernel_t.dtype not in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        casted_kernel_t, casted_kernel, updated_kernel_amax = cast_transpose(
            kernel_t,
            kernel_amax,
            kernel_scale,
            kernel_scale_inv,
            fwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
    else:
        casted_kernel = jnp.matrix_transpose(kernel_t)
        casted_kernel_t = kernel_t
        updated_kernel_amax = kernel_amax

    out_amax = (
        amax_list[FP8MetaPackage.OUTPUT_IDX][0:1]
        if out_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]
        else None
    )
    out_scale = (
        scale_list[FP8MetaPackage.OUTPUT_IDX][0:1]
        if out_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]
        else None
    )

    # Set scale_inv for comm overlap buffer
    buffer_scale_inv = None
    if comm_overlap_config is not None:
        overlap_name = comm_overlap_config["name"]
        if comm_overlap_config["method"] != "bulk" and tex.overlap_buffer_is_fp8(overlap_name):
            if comm_overlap_config["comm_type"] == tex.CommOverlapType.AG:
                buffer_scale_inv = x_scale_inv

            elif comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
                out_dtype = fwd_dtype
                out_scale = scale_list[FP8MetaPackage.OUTPUT_IDX][0:1]
                buffer_scale_inv = jnp.reciprocal(out_scale)

            tex.set_overlap_buffer_scale_inverse(
                overlap_name,
                jax.dlpack.to_dlpack(buffer_scale_inv),
            )

    out, updated_out_amax, updated_out_scale, pre_gelu_out, extra_out = fp8_gemm_impl(
        casted_x,
        x_scale_inv,
        casted_kernel_t,
        kernel_scale_inv,
        bias=bias,
        out_amax=out_amax,
        out_scale=out_scale,
        out_dtype=out_dtype,
        batched_output=(x.ndim > 2),
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=comm_overlap_config,
    )

    # Update returned and saved arrays based on comm+GEMM overlap config
    final_out = out
    if comm_overlap_config is not None:
        if comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
            # RS overlap puts the reduce-scattered sharded output into extra_out
            final_out = extra_out

    if not jax_dtype_is_fp8(final_out):
        updated_out_amax = None
        updated_out_scale = None

    ctx = (
        casted_x_t,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        updated_out_amax,
        pre_gelu_out if fuse_gelu else None,
        fuse_bias,
        maybe_fp32_to_fm32,
        (x.ndim > 2),
    )

    return (final_out, updated_out_amax, updated_out_scale), ctx


def _fp8_gemm_bwd_rule(
    out_dtype,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
    comm_overlap_config,
    ctx,
    grad,
):
    (
        casted_x_t,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        updated_out_amax,
        pre_gelu_out,
        fuse_bias,
        maybe_fp32_to_fm32,
        batched_input,
    ) = ctx
    del out_dtype
    bwd_dtype = FP8Helper.BWD_DTYPE

    # Recover DGRAD and WGRAD comm+GEMM overlap configs
    dgrad_overlap_name = None
    dgrad_overlap_config = None
    wgrad_overlap_name = None
    wgrad_overlap_config = None
    if comm_overlap_config is not None:
        dgrad_overlap_name = comm_overlap_config["name"].rstrip("_fprop") + "_dgrad"
        dgrad_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(dgrad_overlap_name, None)
        wgrad_overlap_name = comm_overlap_config["name"].rstrip("_fprop") + "_wgrad"
        wgrad_overlap_config = _ACTIVE_COMM_GEMM_OVERLAPS.get(wgrad_overlap_name, None)

    # Cast-transpose grad with potential fusions
    grad_amax = amax_list[FP8MetaPackage.GRAD_IDX][0:1]
    grad_scale = scale_list[FP8MetaPackage.GRAD_IDX]
    grad_scale_inv = scale_inv_list[FP8MetaPackage.GRAD_ID]
    if fuse_gelu:
        if fuse_bias:
            # Fuse dbias into this dGELU.
            casted_grad, casted_grad_t, bgrad, updated_grad_amax = dact_lu_dbias_cast_transpose(
                grad,
                pre_gelu_out,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
                activation_type=("gelu",),
            )
        else:
            # No bias to fuse so we just do dGELU.
            casted_grad, casted_grad_t, updated_grad_amax = dact_lu(grad, pre_gelu_out, ("gelu",))
            bgrad = None
    else:
        if fuse_bias:
            # Since there is no GELU fusion, we need to fuse dbias into this cast_transpose.
            casted_grad, casted_grad_t, bgrad, updated_grad_amax = dbias_cast_transpose(
                grad,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
            )
        else:
            # If both bias and GELU is fused into the forward pass, we will fuse dbias later with
            # dGELU. No need to do it here.
            casted_grad, casted_grad_t, updated_grad_amax = cast_transpose(
                grad,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
            )
            bgrad = None

    # Set scale_inv for comm overlap buffer
    dgrad_amax = None
    dgrad_scale = None
    if dgrad_overlap_config is not None:
        if dgrad_overlap_config["method"] == "bulk":
            assert (
                wgrad_overlap_config is not None
            ), f"Missing comm+GEMM overlap config for {wgrad_overlap_name}!"
            # Set WGRAD buffer as output of DGRAD in order to avoid a memcpy for bulk RS overlap
            dgrad_pre_rs = jax.dlpack.from_dlpack(tex.get_overlap_buffer(wgrad_overlap_name, False))
            # Copy input into overlap buffer for all-gather
            copy_into_overlap_buffer(casted_x_t, dgrad_overlap_name, True)

        elif tex.overlap_buffer_is_fp8(dgrad_overlap_name):
            # Non-bulk RS DGRAD overlap needs output amax and scale if buffer type is FP8
            dgrad_amax = grad_amax
            dgrad_scale = grad_scale
            tex.set_overlap_buffer_scale_inverse(
                dgrad_overlap_name,
                jax.dlpack.to_dlpack(grad_scale_inv),
            )

    # DGRAD: ([B], M, N) x (K, N)^T = ([B], M, K)
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    dgrad, *_, dgrad_extra_out = fp8_gemm_impl(
        casted_grad,
        grad_scale_inv,
        casted_kernel,
        kernel_scale_inv,
        out=dgrad_pre_rs,
        out_amax=dgrad_amax,
        out_scale=dgrad_scale,
        batched_output=batched_input,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=dgrad_overlap_config,
    )

    # If dgrad overlapped reduce-scatter, set it to the RS output
    if (
        dgrad_overlap_config is not None
        and dgrad_overlap_config["method"] != "bulk"
        and dgrad_overlap_config["comm_type"] == tex.CommOverlapType.RS
    ):
        dgrad = dgrad_extra_out

    # Prepare comm+GEMM overlap for WGRAD
    if wgrad_overlap_config is not None:
        if wgrad_overlap_config["method"] == "bulk":
            # Get all-gathered input from DGRAD bulk overlap
            casted_x_t = jax.dlpack.from_dlpack(tex.get_overlap_buffer(dgrad_overlap_name, False))

        elif tex.overlap_buffer_is_fp8(wgrad_overlap_name):
            # Set FP8 scale inverse for non-bulk AG overlap
            tex.set_overlap_buffer_scale_inverse(
                wgrad_overlap_name, jax.dlpack.to_dlpack(x_scale_inv)
            )

    # WGRAD: ([B], N, M) x ([B], K, M)^T = (N, K)
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    wgrad, *_, wgrad_extra_out = fp8_gemm_impl(
        casted_x_t,
        x_scale_inv,
        casted_grad_t,
        grad_scale_inv,
        out_dtype=jnp.bfloat16,
        batched_output=False,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=wgrad_overlap_config,
    )

    # If wgrad overlapped reduce-scatter, set it to the RS output
    if (
        wgrad_overlap_config is not None
        and wgrad_overlap_config["method"] != "bulk"
        and wgrad_overlap_config["comm_type"] == tex.CommOverlapType.RS
    ):
        dgrad = wgrad_extra_out

    amax_list[FP8MetaPackage.INPUT_IDX] = (
        amax_list[FP8MetaPackage.INPUT_IDX].at[0].set(updated_x_amax[0])
    )
    amax_list[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_amax[0])
    )
    amax_list[FP8MetaPackage.GRAD_IDX] = (
        amax_list[FP8MetaPackage.GRAD_IDX].at[0].set(updated_grad_amax[0])
    )
    if updated_out_amax is not None:
        amax_list[FP8MetaPackage.OUTPUT_IDX] = (
            amax_list[FP8MetaPackage.OUTPUT_IDX].at[0].set(updated_out_amax[0])
        )

    amax_list = maybe_fp32_to_fm32(*amax_list)
    scale_list = maybe_fp32_to_fm32(*scale_list)

    return dgrad, wgrad, bgrad, amax_list, scale_list


_fp8_gemm.defvjp(_fp8_gemm_fwd_rule, _fp8_gemm_bwd_rule)


def type_safe_gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Optional[ArrayLike] = None,
    out: Optional[ArrayLike] = None,
    out_dtype: Optional[jnp.dtype] = None,
    fp8_meta: Optional[FP8MetaPackage] = None,
    contracting_dims: Tuple[int, int] = (-1, -2),
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    comm_overlap_name: Optional[str] = None,
) -> ArrayLike:
    if jax_dtype_is_fp8(x.dtype) or jax_dtype_is_fp8(kernel.dtype):
        assert fp8_meta is not None, "GEMM operands have FP8 dtypes but FP8MetaPackage is None."

    if fp8_meta is not None:
        x_inner_dim, kernel_inner_dim = map(sanitize_dims, contracting_dims, (x.ndim, kernel.ndim))
        assert x_inner_dim == x.ndim - 1 and kernel_inner_dim == kernel.ndim - 1, (
            "FP8 GEMM requires non-transposed X (LHS) and transposed kernel (RHS), "
            + "i.e. contracting_dims=(-1, -1)."
        )
        return fp8_gemm(
            x,
            kernel,
            fp8_meta,
            bias=bias,
            out=out,
            out_dtype=out_dtype,
            fuse_gelu=fuse_gelu,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_name=comm_overlap_name,
        )
    else:
        return gemm(
            x,
            kernel,
            bias=bias,
            contracting_dims=contracting_dims,
            fuse_gelu=fuse_gelu,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_name=comm_overlap_name,
        )


def initialize_comm_gemm_overlaps(
    buffer_shape: Sequence[int],
    mesh: jax.sharding.Mesh,
    myrank: int,
    numranks: int,
    **kwargs: Optional[dict],
) -> None:
    """
    Initialize Comm+GEMM overlap communicators and buffers.

    .. warning::
       Communication buffer allocations for this functionality are outside the XLA memory pool
       and can cause OOM errors if XLA's memory margin is not reduced.

    Parameters
    ----------
    buffer_shape : Sequence[int]
        Shape of the communication buffer. This should be sized to match the global shape of the
        input/activation tensor.
    mesh : jax.sharding.Mesh
        JAX Mesh with a `tp_resource` axis.
    myrank: int
        Global rank of the calling process.
    numranks: int
        Global number of processes.
    tp_resource : Optional[str] = None
        Tensor-parallel mesh axis name. If not given, defaults to the TP resource in the global
        te.sharding.MeshResource context.
    tp_size : Optional[int] = None
        Size of the tensor-parallel axis in the mesh. If not given, defaults to the size of the
        tensor-parallel axis in `jax.interpreters.pxla.thread_resources`.
    use_fp8 : bool = False
        Flag for allocating an FP8 communication buffer. This is not supported for reduce-scatter
        overlaps with the `pipeline` method.
    overlap_configs: Optional[dict] = None,
        Dictionary of configs for comm+GEMM overlaps by layer name.
    """
    assert tex.ubuf_built_with_mpi(), (
        "Comm+GEMM overlap in TE/JAX requires Transformer Engine to be compiled with "
        + "`NVTE_UB_WITH_MPI=1` and `MPI_HOME=/path/to/mpi` variables."
    )
    if not tex.device_supports_multicast():
        assert bool(int(os.getenv("UB_SKIPMC", "0"))), (
            "CUDA device, driver and/or toolkit version does not support comm+GEMM overlap with "
            + "CUDA Multicast. Launch with UB_SKIPMC=1 to try CUDA IPC instead."
        )
    # Extract kwargs
    tp_resource = kwargs.get("tp_resource", global_mesh_resource().tp_resource)
    tp_size = kwargs.get("tp_size", get_mesh_axis_size(tp_resource, mesh=mesh))
    use_fp8 = kwargs.get("use_fp8", False)
    overlap_configs = kwargs.get("overlap_configs", None)

    # Layers that support comm+GEMM overlap
    layers_all_gather_overlap = [
        "ag_gemm",
        "qkv_fprop",
        "qkv_dgrad",
        "proj_dgrad",
        "fc1_fprop",
        "fc1_dgrad",
        "fc2_dgrad",
    ]
    layers_reduce_scatter_overlap = [
        "gemm_rs",
        "proj_fprop",
        "fc2_fprop",
        "qkv_wgrad",
        "fc1_wgrad",
    ]
    dgrad_reduce_scatter_overlap = ["qkv_dgrad", "fc1_dgrad"]

    # Default overlap methods for layers
    methods = {
        "ring_exchange": [
            "ag_gemm",
            "gemm_rs",
            "qkv_fprop",
            "fc1_fprop",
            "proj_dgrad",
            "fc2_dgrad",
        ],
        "pipeline": ["proj_fprop", "fc2_fprop"],
        "bulk": ["qkv_dgrad", "qkv_wgrad", "fc1_dgrad", "fc1_wgrad"],
    }

    # AG-RS overlap pairs of layers forming a tensor-parallel block
    ag_rs_pairs = {
        "qkv_fprop": "proj_fprop",
        "fc1_fprop": "fc2_fprop",
    }
    rs_ag_pairs = {v: k for k, v in ag_rs_pairs.items()}
    global layers_atomic_ring_exchange
    layers_atomic_ring_exchange = []

    def get_method(name):
        for method, names in methods.items():
            if name in names:
                return method
        raise KeyError(f"Given layer name {name} does not exist.")

    def get_default_config(name):
        method = get_method(name)
        default_cfg = {
            "mesh": mesh,
            "tp_resource": tp_resource,
            "tp_size": tp_size,
            "name": name,
            "method": method,
            "comm_type": (
                tex.CommOverlapType.AG
                if name in layers_all_gather_overlap
                else tex.CommOverlapType.RS
            ),
            "num_sm": 1 if method == "ring_exchange" else 16,
            "num_max_streams": get_num_max_compute_streams(),
            "cga_size": 1 if method == "ring_exchange" else 2,
            "set_sm_margin": False,
            "num_splits": 4 if method == "pipeline" else tp_size,
            "aggregate": False,
            "atomic_gemm": False,
            "pipeline_rs_overlap_first_gemm": False,
            "use_ce": True,
            "fp8_buf": name in layers_all_gather_overlap,
        }
        return default_cfg

    def add_new_comm_gemm_overlap(
        shape: Sequence[int],
        kwargs: dict,
    ) -> None:
        overlap_name = kwargs["name"]
        assert (
            overlap_name not in _ACTIVE_COMM_GEMM_OVERLAPS
        ), f"Duplicate initialization for `{overlap_name}` overlap!"

        overlap_method = kwargs["method"]
        overlap_atomic_gemm = kwargs["atomic_gemm"]
        if overlap_atomic_gemm:
            warnings.warn(
                "Atomic GEMM uses a beta API from cublas and is not tested for all use cases."
            )
            assert use_fp8, "Atomic GEMM overlap supported only for FP8 GEMM."
            if overlap_method == "bulk":
                warnings.warn(
                    f"At {overlap_name}, atoimic GEMM not is supported for a bulk overlap."
                    "Defaulting to `atomic_gemm=False`."
                )
                overlap_atomic_gemm = False
        kwargs["atomic_gemm"] = overlap_atomic_gemm
        if overlap_method == "pipeline" and kwargs["comm_type"] == tex.CommOverlapType.AG:
            raise ValueError(
                f"At {overlap_name}, `pipeline` overlap method is not supported for AllGather."
            )
        # Check if both AG and RS overlaps use `atomic GEMM`` + `p2p ring-exchange`.
        # Using atomic GEMM + p2p ring-exchange in only one of the pair breaks functionality.
        global layers_atomic_ring_exchange
        if (
            overlap_atomic_gemm
            and overlap_method == "ring_exchange"
            and overlap_name in ag_rs_pairs
        ):
            layers_atomic_ring_exchange += [overlap_name, ag_rs_pairs[overlap_name]]
        if overlap_name in rs_ag_pairs:
            assert_message = (
                f"At {overlap_name}, atomic AG-GEMM overlap with `ring_exchange` shuffles GEMM "
                "chunk outputs, and  RS-GEMM overlap un-suffle them. When one of the GEMM-AG and "
                "GEMM-RS overlaps forming a TP block (e.g., qkv_fprop and proj_fprop) uses "
                "`atomic gemm` and `ring_exhcnage`, its pair must use the same overlap config "
                "for functionality."
            )
            if overlap_name in layers_atomic_ring_exchange:
                assert overlap_atomic_gemm and overlap_method == "ring_exchange", assert_message
            else:
                if overlap_atomic_gemm and overlap_method == "ring_exchange":
                    assert rs_ag_pairs[overlap_name] in layers_atomic_ring_exchange, assert_message

        # Reduce buffer shape to 2D here in case the user initialized with batch dims
        buffer_shape = (reduce(operator.mul, shape[:-1], 1), shape[-1])
        tex.bootstrap_comm_gemm_overlap(
            buffer_shape,
            jax_dtype_to_te_dtype(jnp.uint8 if (use_fp8 and fp8_buf) else jnp.bfloat16),
            overlap_name,
            overlap_method,
            kwargs["comm_type"],
            myrank,
            numranks,
            tp_size,
            kwargs["num_splits"],
            get_num_max_compute_streams(),
            kwargs["cga_size"],
            kwargs["num_sm"],
            kwargs["set_sm_margin"],
            kwargs["use_ce"],
            overlap_atomic_gemm,
            kwargs["aggregate"],
            kwargs["pipeline_rs_overlap_first_gemm"],
        )

    if overlap_configs is not None:
        for name in dgrad_reduce_scatter_overlap:
            if (
                name in overlap_configs
                and "method" in overlap_configs[name]
                and overlap_configs[name]["method"] != "bulk"
            ):
                wgrad_name = name.replace("dgrad", "wgrad")
                assert wgrad_name not in overlap_configs
                layers_reduce_scatter_overlap.remove(wgrad_name)
                layers_all_gather_overlap.remove(name)
                layers_reduce_scatter_overlap.append(name)
                methods["bulk"].remove(name)
                methods["bulk"].remove(wgrad_name)
                new_method = overlap_configs[name]["method"]
                methods[new_method].append(name)

    global _ACTIVE_COMM_GEMM_OVERLAPS
    for name in methods["ring_exchange"] + methods["pipeline"] + methods["bulk"]:
        if overlap_configs is not None and name in overlap_configs:
            fp8_buf = (name in layers_all_gather_overlap) or (
                overlap_configs[name].get("fp8_buf", False) and name not in methods["pipeline"]
            )
            final_config = get_default_config(name)
            final_config.update(overlap_configs[name])
            final_config["fp8_buf"] = fp8_buf
            add_new_comm_gemm_overlap(buffer_shape, final_config)
            _ACTIVE_COMM_GEMM_OVERLAPS[name] = final_config


def destroy_comm_gemm_overlaps():
    global _ACTIVE_COMM_GEMM_OVERLAPS
    for name in _ACTIVE_COMM_GEMM_OVERLAPS:
        tex.destroy_comm_gemm_overlap(name)
    _ACTIVE_COMM_GEMM_OVERLAPS = dict()


def get_comm_overlap_config(name):
    global _ACTIVE_COMM_GEMM_OVERLAPS
    assert (
        name in _ACTIVE_COMM_GEMM_OVERLAPS
    ), f"Comm+GEMM overlap for '{name}' has not been initialized!"
    return _ACTIVE_COMM_GEMM_OVERLAPS[name]
