# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Joint fused operation for MoE grouped MLP."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import functools
import os
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Any, Optional

import torch
from packaging.version import Version as PkgVersion

import transformer_engine_torch as tex
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload, start_offload
from ...cpp_extensions import general_gemm, general_grouped_gemm_for_grouped_tensor
from ...module.base import _2X_ACC_WGRAD
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer, NVFP4Tensor, NVFP4TensorStorage, Quantizer
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ...triton.grouped_dbias_dscales import compute_grouped_dbias_dscales
from ...utils import (
    ceil_div,
    clear_tensor_data,
    get_cached_ones_tensor,
    get_device_compute_capability,
    mark_grouped_tensor,
    round_up_to_nearest_multiple,
)
from ..basic import (
    GroupedLinear,
    ScaledClampedQGeGLU,
    ScaledSReLU,
    ScaledSwiGLU,
)
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    is_quantized_tensor,
    maybe_dequantize,
    view_main_grad_as_grouped_buffer,
)


@functools.lru_cache(maxsize=None)
def _cudnn_frontend_version_at_least(min_version: str) -> bool:
    """Check cuDNN frontend package version."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion(min_version)
    except PackageNotFoundError:
        return False


def _cudnn_frontend_version_supported() -> bool:
    """Check cuDNN frontend is at least 1.23.0.

    All grouped MLP fused-kernel features require cuDNN frontend >= 1.23.0.
    """
    return _cudnn_frontend_version_at_least("1.23.0")


def _cudnn_frontend_geglu_runtime_params() -> bool:
    """Check cuDNN frontend is at least 1.24.0.

    Runtime-configurable GeGLU parameters (linear_offset, geglu_alpha,
    glu_clamp_max, glu_clamp_min) require cuDNN frontend >= 1.24.0.
    """
    return _cudnn_frontend_version_at_least("1.24.0")


def _cudnn_frontend_supports_grouped_gemm_srelu() -> bool:
    """Check cuDNN frontend min version for grouped GEMM SReLU kernels."""
    return _cudnn_frontend_version_at_least("1.24.0")


def _cudnn_frontend_supports_grouped_gemm_srelu_hadamard() -> bool:
    """Check cuDNN frontend min version for grouped GEMM SReLU hadamard kernels."""
    return _cudnn_frontend_version_at_least("1.26.0")


def _nvidia_cudnn_frontend_supports_wgrad() -> bool:
    """Check cuDNN FE min version for grouped GEMM wgrad kernel."""
    return _cudnn_frontend_version_supported()


def _wrap_single_nvfp4_as_grouped(
    tensor: torch.Tensor,
    quantized: NVFP4Tensor | NVFP4TensorStorage,
    quantizer: NVFP4Quantizer,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Wrap a single NVFP4 tensor in GroupedTensor storage."""
    with_gemm_swizzled_scales = quantized._with_gemm_swizzled_scales
    if quantizer.optimize_for_gemm:
        tex.swizzle_scales_for_gemm_(quantized)
        with_gemm_swizzled_scales = True

    rowwise_data = quantized._rowwise_data
    rowwise_scale = quantized._rowwise_scale_inv
    columnwise_data = quantized._columnwise_data
    columnwise_scale = quantized._columnwise_scale_inv
    amax = quantized._amax_rowwise
    columnwise_amax = quantized._amax_columnwise

    if split_sizes is None:
        split_sizes = torch.full((1,), tensor.shape[0], dtype=torch.int64, device=tensor.device)
    else:
        split_sizes = split_sizes.to(dtype=torch.int64, device=tensor.device)

    m_dim = tensor.shape[0]
    if rowwise_data is not None:
        k_dim = rowwise_data.shape[-1] * 2
    elif columnwise_data is not None:
        k_dim = columnwise_data.shape[0]
    else:
        k_dim = tensor.shape[-1]

    if tensor_offsets is None:
        tensor_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=tensor.device),
                torch.cumsum(split_sizes * k_dim, dim=0),
            ],
        )

    return GroupedTensor(
        shape=(m_dim, k_dim),
        dtype=tensor.dtype,
        quantizer=quantizer,
        num_tensors=1,
        data=rowwise_data.reshape(-1) if rowwise_data is not None else None,
        columnwise_data=columnwise_data.reshape(-1) if columnwise_data is not None else None,
        scale_inv=rowwise_scale.reshape(-1) if rowwise_scale is not None else None,
        columnwise_scale_inv=columnwise_scale.reshape(-1) if columnwise_scale is not None else None,
        amax=amax,
        columnwise_amax=columnwise_amax,
        first_dims=split_sizes,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )


def _group_quantize_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize into grouped storage."""

    if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer):
        return tex.group_quantize(
            tensor,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets=tensor_offsets,
        )

    quantized = tex.quantize(tensor, quantizer)
    return _wrap_single_nvfp4_as_grouped(
        tensor,
        quantized,
        quantizer,
        split_sizes,
        tensor_offsets=tensor_offsets,
    )


def _group_quantize_with_amax_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    rowwise_amax: torch.Tensor,
    columnwise_amax: torch.Tensor,
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize with precomputed NVFP4 amaxes into grouped storage."""
    if not isinstance(quantizer, NVFP4Quantizer):
        return _group_quantize_for_grouped_mlp(
            tensor,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets=tensor_offsets,
        )

    if num_groups != 1:
        return tex.nvfp4_group_quantize_with_amax(
            tensor,
            quantizer,
            num_groups,
            split_sizes,
            rowwise_amax,
            columnwise_amax,
            tensor_offsets=tensor_offsets,
        )

    quantized = tex.nvfp4_quantize_with_amax(
        tensor, quantizer, rowwise_amax.view(-1)[:1], columnwise_amax.view(-1)[:1]
    )
    return _wrap_single_nvfp4_as_grouped(
        tensor,
        quantized,
        quantizer,
        split_sizes,
        tensor_offsets=tensor_offsets,
    )


def _nvfp4_amax(
    tensors: GroupedTensor | Iterable[NVFP4TensorStorage],
    *,
    columnwise: bool,
) -> torch.Tensor:
    """Get one NVFP4 amax value per group."""
    grouped_attr = "columnwise_amax" if columnwise else "amax"
    tensor_attr = "_amax_columnwise" if columnwise else "_amax_rowwise"

    if hasattr(tensors, grouped_attr):
        amax = getattr(tensors, grouped_attr)
        if amax is None:
            raise RuntimeError(f"NVFP4 GroupedTensor is missing {grouped_attr}.")
        return amax.view(-1)

    amaxes = [getattr(tensor, tensor_attr) for tensor in tensors]
    if any(amax is None for amax in amaxes):
        raise RuntimeError(f"NVFP4 tensor list is missing {tensor_attr}.")
    return torch.cat([amax.view(-1) for amax in amaxes], dim=0)


def _nvfp4_single_tensor_from_grouped(
    grouped: GroupedTensor,
    quantizer: Optional[NVFP4Quantizer] = None,
    *,
    fp4_dtype: Optional[torch.dtype] = None,
) -> NVFP4Tensor:
    """Build a single NVFP4Tensor view over a one-member grouped storage."""
    if quantizer is None:
        quantizer = grouped.quantizer
    if not isinstance(quantizer, NVFP4Quantizer):
        raise TypeError("Expected an NVFP4 GroupedTensor.")

    shape = tuple(grouped.logical_shape)
    rowwise_data = None
    if grouped.rowwise_data is not None:
        rowwise_data = grouped.rowwise_data.view(quantizer.convert_shape_for_fp4(shape))

    rowwise_scale_inv = None
    if grouped.scale_inv is not None:
        rowwise_scale_inv = grouped.scale_inv.view(quantizer.get_scale_shape(shape, False))

    columnwise_data = None
    if grouped.columnwise_data is not None:
        columnwise_shape = quantizer.get_columnwise_shape(shape)
        columnwise_data = grouped.columnwise_data.view(
            quantizer.convert_shape_for_fp4(columnwise_shape)
        )

    columnwise_scale_inv = None
    if grouped.columnwise_scale_inv is not None:
        columnwise_scale_inv = grouped.columnwise_scale_inv.view(
            quantizer.get_scale_shape(shape, True)
        )

    return NVFP4Tensor(
        shape=shape,
        dtype=grouped.get_dtype(),
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=grouped.amax,
        amax_columnwise=grouped.columnwise_amax,
        fp4_dtype=fp4_dtype or quantizer.dtype,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=grouped._with_gemm_swizzled_scales,
    )


def _pack_grouped_linear_bias_for_cudnn(linear_op: GroupedLinear) -> Optional[torch.Tensor]:
    """Bias layout expected by cuDNN grouped GEMM: shape (n, num_groups), stride (1, n)."""
    if not linear_op.has_bias:
        return None
    num_groups = linear_op.num_groups
    grouped_bias = getattr(linear_op, "bias", None)
    if grouped_bias is not None:
        packed = grouped_bias.rowwise_data.view(num_groups, -1)
        return packed.transpose(0, 1)
    rows = [getattr(linear_op, f"bias{group_idx}") for group_idx in range(num_groups)]
    # stack to [num_groups, n] but cuDNN expects [n, num_groups] with stride [1, n].
    return torch.stack(rows, dim=0).transpose(0, 1)


@functools.lru_cache(maxsize=1)
def _grouped_gemm_dsrelu_backward_supported() -> bool:
    """Whether the cuDNN FE grouped GEMM dSReLU backward wrapper is available."""
    if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        return False
    if get_device_compute_capability()[0] != 10:
        return False
    if not _cudnn_frontend_supports_grouped_gemm_srelu():
        return False
    try:
        from cudnn import (
            grouped_gemm_dsrelu_wrapper_sm100,
        )  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return grouped_gemm_dsrelu_wrapper_sm100 is not None


@functools.lru_cache(maxsize=1)
def _use_tmem_post_rht_amax() -> bool:
    """Whether the FC1 GLU+RHT+amax kernel should use TMEM post-RHT amax."""
    return os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP_FC1_GLU_RHT_AMAX_TMEM", "0") == "1"


def _nvfp4_single_group_wgrad_gemm(
    grouped_x: GroupedTensor,
    grouped_dy: GroupedTensor,
    wgrad_output,
    *,
    weight_shape: tuple[int, int],
    accumulate: bool,
) -> None:
    """Run one-group NVFP4 wgrad with regular GEMM instead of grouped GEMM."""
    x_single = _nvfp4_single_tensor_from_grouped(grouped_x)
    dy_single = _nvfp4_single_tensor_from_grouped(grouped_dy)
    if isinstance(wgrad_output, GroupedTensor):
        out = wgrad_output.rowwise_data.view(1, *weight_shape)[0]
    else:
        out = wgrad_output[0]

    general_gemm(
        x_single,
        dy_single,
        out_dtype=out.dtype,
        out=out,
        layout="NT",
        accumulate=accumulate,
        use_split_accumulator=_2X_ACC_WGRAD,
    )


def _cudnn_compute_wgrad(
    grouped_x: GroupedTensor,
    grouped_dy: GroupedTensor,
    wgrad_output,
    weight_shape: tuple,
    offsets: torch.Tensor,
    accumulate: bool,
    wgrad_kernel_fn,
    single_grouped_weight: bool,
    use_nvfp4: bool,
    data_dtype: torch.dtype,
    scale_view_dtype: torch.dtype,
    sf_vec_size: int,
    current_stream=None,
):
    """Compute wgrad using the cuDNN CuTe DSL grouped GEMM wgrad kernel.

    The cuDNN wgrad kernel computes:
        wgrad[e] = a[:, tok_start:tok_end] @ b[tok_start:tok_end, :]
    where a = DY^T = (out_features, total_tokens) row-major and
          b = X  = (total_tokens, in_features) column-major.
    """
    if current_stream is None:
        current_stream = torch.cuda.current_stream().cuda_stream

    out_features, in_features = weight_shape
    total_tokens = grouped_dy.logical_shape[0]

    sfa_leading_dim = round_up_to_nearest_multiple(out_features, 128)
    sfb_leading_dim = round_up_to_nearest_multiple(in_features, 128)

    if total_tokens == 0:
        # A workaround for the case with zero-token experts.
        # Even for this case, cuteDSL still requires the same
        # stride requirements for the input and scale tensors.
        device = grouped_dy.columnwise_data.device
        a_tensor = torch.empty_strided(
            (out_features, 0),
            (16, 1),
            dtype=data_dtype,
            device=device,
        )
        b_tensor = torch.empty_strided(
            (0, in_features),
            (in_features, 1),
            dtype=data_dtype,
            device=device,
        )
        sfa_tensor = torch.empty_strided(
            (sfa_leading_dim, 0),
            (16, 1),
            dtype=scale_view_dtype,
            device=device,
        )
        sfb_tensor = torch.empty_strided(
            (sfb_leading_dim, 0),
            (16, 1),
            dtype=scale_view_dtype,
            device=device,
        )
    elif use_nvfp4:
        # NVFP4 columnwise data is stored expert-major as per-expert
        # (logical_K, group_M / 2) chunks. cuDNN consumes that layout directly
        # with input_order="tensor_ragged".
        a_tensor = grouped_dy.columnwise_data.view(dtype=data_dtype).view(
            out_features,
            total_tokens // 2,
        )
        b_tensor = (
            grouped_x.columnwise_data.view(dtype=data_dtype)
            .view(
                in_features,
                total_tokens // 2,
            )
            .T
        )
        sfa_tensor = grouped_dy.columnwise_scale_inv.view(sfa_leading_dim, -1).view(
            dtype=scale_view_dtype
        )
        sfb_tensor = grouped_x.columnwise_scale_inv.view(sfb_leading_dim, -1).view(
            dtype=scale_view_dtype
        )
    else:
        a_tensor = (
            grouped_dy.columnwise_data.view(dtype=data_dtype).view(total_tokens, out_features).T
        )
        b_tensor = grouped_x.columnwise_data.view(dtype=data_dtype).view(total_tokens, in_features)
        sfa_tensor = grouped_dy.columnwise_scale_inv.view(sfa_leading_dim, -1).view(
            dtype=scale_view_dtype
        )
        sfb_tensor = grouped_x.columnwise_scale_inv.view(sfb_leading_dim, -1).view(
            dtype=scale_view_dtype
        )

    common_wgrad_kwargs = {
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
        "sfa_tensor": sfa_tensor,
        "sfb_tensor": sfb_tensor,
        "offsets_tensor": offsets,
        "acc_dtype": torch.float32,
        "sf_vec_size": sf_vec_size,
        "accumulate_on_output": accumulate,
        "current_stream": current_stream,
    }
    if use_nvfp4:
        global_scale_denom = 448.0 * 6.0
        if total_tokens == 0:
            global_scale_shape = (offsets.shape[0],)
            common_wgrad_kwargs["global_scale_a"] = torch.zeros(
                global_scale_shape,
                dtype=torch.float32,
                device=device,
            )
            common_wgrad_kwargs["global_scale_b"] = torch.zeros(
                global_scale_shape,
                dtype=torch.float32,
                device=device,
            )
        else:
            common_wgrad_kwargs["global_scale_a"] = (
                _nvfp4_amax(grouped_dy, columnwise=True).to(torch.float32) / global_scale_denom
            )
            common_wgrad_kwargs["global_scale_b"] = (
                _nvfp4_amax(grouped_x, columnwise=True).to(torch.float32) / global_scale_denom
            )
        common_wgrad_kwargs["input_order"] = "tensor_ragged"

    # Prepare wgrad output
    if single_grouped_weight:
        # Dense mode: single (num_groups, out_features, in_features) tensor
        wgrad_tensor = wgrad_output.rowwise_data.view(offsets.shape[0], out_features, in_features)
        wgrad_kernel_fn(
            **common_wgrad_kwargs,
            output_mode="dense",
            wgrad_tensor=wgrad_tensor,
            wgrad_dtype=wgrad_tensor.dtype,
        )
    else:
        # Discrete mode: per-expert wgrad device pointers
        wgrad_ptrs = tex.copy_data_ptrs_to_device(wgrad_output, wgrad_output[0].device)
        wgrad_kernel_fn(
            **common_wgrad_kwargs,
            output_mode="discrete",
            wgrad_ptrs=wgrad_ptrs,
            wgrad_dtype=wgrad_output[0].dtype,
        )


def _compute_grad_params(
    fc_op,
    ctx,
    num_groups,
    weight_shape,
    grouped_x,
    grouped_dy,
    dtype,
    device,
    bias_grads,
    bias_grad_packed,
    label="",
    *,
    cudnn_wgrad_kernel_fn,
    use_nvfp4,
    data_dtype,
    scale_view_dtype,
    sf_vec_size,
    offsets,
):
    """Compute weight gradients and build grad_params for a GroupedLinear layer.
    Returns the grad_params list in parameter registration order.
    """

    # Allocate grad buffers, determine accumulate flag.
    accumulate_into_main_grad = False
    grouped_wgrad = None
    wgrad_output = None
    op_label = f"Grouped MLP fused backward ({label})" if label else "Grouped MLP fused backward"
    weights = fc_op._get_weight_tensors()
    if fc_op.single_grouped_weight:
        w_list = [None]
        if ctx.weight_requires_grad:
            if fc_op._accumulate_into_main_grad:
                # Main-grad fusion: GEMM writes directly into ``main_grad``.
                # ``overwrite_main_grad`` only flips the GEMM's ``accumulate``
                # flag (overwrite vs. accumulate); it does not change the
                # output buffer.
                main_grad = get_main_grad_from_param(weights[0], op_label=op_label)
                main_grad = view_main_grad_as_grouped_buffer(
                    main_grad, num_groups, weight_shape, label=f"{op_label} weight"
                )
                grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                    num_tensors=num_groups,
                    tensor_shape=weight_shape,
                    rowwise_data=main_grad,
                    dtype=main_grad.dtype,
                )
                accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
            else:
                grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                    num_tensors=num_groups,
                    shapes=[weight_shape] * num_groups,
                    quantizer=None,
                    device=device,
                    dtype=dtype,
                )
            wgrad_output = grouped_wgrad
            w_list = [grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)]
    else:
        w_list = [None] * num_groups
        if ctx.weight_requires_grad:
            if fc_op._accumulate_into_main_grad:
                w_list = [get_main_grad_from_param(w, op_label=op_label) for w in weights]
                accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
            else:
                wgrad_packed = torch.empty(
                    num_groups,
                    *weight_shape,
                    dtype=dtype,
                    device=device,
                )
                w_list = [wgrad_packed[i] for i in range(num_groups)]
            wgrad_output = w_list

    if ctx.weight_requires_grad:
        # Launch or defer the GEMM
        delay_wgrad = fc_op.wgrad_store is not None and fc_op.wgrad_store.delay_wgrad_compute()
        if cudnn_wgrad_kernel_fn is not None:
            offsets = offsets if offsets.dtype == torch.int32 else offsets.to(dtype=torch.int32)
            gemm_fn = functools.partial(
                _cudnn_compute_wgrad,
                weight_shape=weight_shape,
                offsets=offsets,
                accumulate=accumulate_into_main_grad,
                wgrad_kernel_fn=cudnn_wgrad_kernel_fn,
                single_grouped_weight=fc_op.single_grouped_weight,
                use_nvfp4=use_nvfp4,
                data_dtype=data_dtype,
                scale_view_dtype=scale_view_dtype,
                sf_vec_size=sf_vec_size,
            )
        elif (
            num_groups == 1
            and isinstance(grouped_x, GroupedTensor)
            and isinstance(grouped_dy, GroupedTensor)
            and isinstance(grouped_x.quantizer, NVFP4Quantizer)
            and isinstance(grouped_dy.quantizer, NVFP4Quantizer)
        ):
            gemm_fn = functools.partial(
                _nvfp4_single_group_wgrad_gemm,
                weight_shape=weight_shape,
                accumulate=accumulate_into_main_grad,
            )
        else:
            gemm_fn = functools.partial(
                general_grouped_gemm_for_grouped_tensor,
                layout="NT",
                accumulate=accumulate_into_main_grad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        if delay_wgrad:
            fc_op.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], gemm_fn)
        else:
            gemm_fn(grouped_x, grouped_dy, wgrad_output)

        # Need to return dummy wgrads for Megatron-LM wgrad fusion if grad is already added
        if fc_op._accumulate_into_main_grad:
            w_list = get_dummy_wgrads_for_params(weights)
        elif delay_wgrad:
            w_list = [None] if fc_op.single_grouped_weight else [None] * num_groups

    # Assemble grad_params in parameter registration order.
    if not fc_op.has_bias:
        return w_list

    if fc_op.single_grouped_bias:
        return w_list + [bias_grad_packed]

    bias_list = bias_grads if bias_grads is not None else [None] * num_groups
    if fc_op.single_grouped_weight:
        return bias_list + w_list
    return w_list + bias_list


def is_glu_activation(activation_op) -> bool:
    """Whether an activation consumes a GLU-style doubled input."""
    return isinstance(activation_op, (ScaledSwiGLU, ScaledClampedQGeGLU))


def validate_grouped_mlp_dims(fc1, activation_op, fc2) -> None:
    """Validate FC1 / activation / FC2 dimensions for fused grouped MLP."""
    if fc1.in_features % 64 != 0 or fc1.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
            f"in_features={fc1.in_features}, out_features={fc1.out_features})."
        )
    if fc2.in_features % 64 != 0 or fc2.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
            f"in_features={fc2.in_features}, out_features={fc2.out_features})."
        )
    if is_glu_activation(activation_op):
        expected_fc1_out_features = 2 * fc2.in_features
    elif isinstance(activation_op, ScaledSReLU):
        expected_fc1_out_features = fc2.in_features
    else:
        raise TypeError(f"Unsupported grouped MLP activation ({activation_op.__class__.__name__}).")

    if fc1.out_features != expected_fc1_out_features or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match."
        )
    if is_glu_activation(activation_op) and activation_op.glu_interleave_size != 32:
        raise ValueError(
            "Fused kernel requires 32-wide GLU interleaving, "
            f"but got glu_interleave_size={activation_op.glu_interleave_size}."
        )


def fuse_grouped_mlp_ops(
    ops,
    *,
    recipe,
    fused_op_cls,
    activation_op_types=None,
):
    """Sliding-window fusion for GroupedLinear + activation + GroupedLinear.

    Parameters
    ----------
    ops : list of FusibleOperation
        Operations to scan.
    recipe : Recipe or None
        Quantization recipe.
    fused_op_cls : type
        Fused operation class with ``is_supported()`` classmethod and
        constructor accepting ``fc1``, ``activation``, and ``fc2`` keyword args.

    Returns
    -------
    list of FusibleOperation
        Updated operations with matched triples replaced by fused ops.
    """
    if not fused_op_cls.is_supported():
        return ops
    if recipe is None or not (recipe.mxfp8() or recipe.nvfp4()):
        return ops
    # NVFP4 fused grouped MLP uses graph-safe grouped quantize, which currently requires RHT.
    if recipe.nvfp4() and recipe.disable_rht:
        return ops
    if activation_op_types is None:
        activation_op_types = (ScaledSwiGLU, ScaledClampedQGeGLU)

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], activation_op_types)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif (
            isinstance(window[1], ScaledClampedQGeGLU)
            and not _cudnn_frontend_geglu_runtime_params()
            and (
                abs(window[1]._clamped.alpha - 1.702) > 0.001
                or abs(window[1]._clamped.glu_linear_offset - 1.0) > 0.001
                or abs(window[1]._clamped.limit - 7.0) > 0.001
            )
        ):
            matches_pattern = False
        else:
            try:
                validate_grouped_mlp_dims(window[0], window[1], window[2])
            except (TypeError, ValueError):
                matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(
                fc1=window[0],
                activation=window[1],
                fc2=window[2],
            )
            window = [op]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out


class _GroupedMLP_CuTeGEMMBase(FusedOperation):
    """Joint fused op for block-scaled GroupedLinear + activation + GroupedLinear.

    Uses experimental CuTe DSL kernels from cuDNN front-end.

    """

    @classmethod
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, activation, and post-multiplication."""
        raise NotImplementedError

    @classmethod
    def grouped_gemm_dactivation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, activation backward, and scale grad."""
        raise NotImplementedError

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_quant_kernel(cls) -> Callable:
        """Grouped GEMM quant kernel for block-scaled inputs."""
        from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_quant_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_wgrad_kernel(cls) -> Optional[Callable]:
        """CuTe DSL kernel for grouped GEMM wgrad on SM100+.

        Returns ``None`` when the environment variable
        ``NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP`` is set to ``1``.
        """
        if int(os.environ.get("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "0")) >= 1:
            return None
        from cudnn import grouped_gemm_wgrad_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_wgrad_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        if not _cudnn_frontend_version_supported():
            return False
        try:
            cls.grouped_gemm_activation_kernel()
            cls.grouped_gemm_dactivation_kernel()
            cls.grouped_gemm_quant_kernel()
            cls.grouped_gemm_wgrad_kernel()
        except ImportError:
            return False
        return True

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        activation: Optional[FusibleOperation],
        fc2: GroupedLinear,
    ) -> None:
        if activation is None:
            raise TypeError("Expected a grouped MLP activation op.")
        super().__init__((fc1, activation, fc2))
        if not self.is_supported():
            self.grouped_gemm_activation_kernel()  # Try triggering import error
            self.grouped_gemm_dactivation_kernel()
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, activation, fc2)
        if not is_glu_activation(activation):
            # grouped_gemm_srelu_wrapper_sm100 and grouped_gemm_dsrelu_wrapper_sm100 are
            # SReLU-specific and do not take GLU ``act_func`` selectors.
            self._cudnn_act_func: Optional[str] = None
            self._cudnn_dact_func: Optional[str] = None
        else:
            # The cuDNN geglu implementations correspond to ScaledClampedQGeGLU.
            # The act_func strings should be fixed on the cuDNN FE side.
            self._cudnn_act_func = (
                "geglu" if isinstance(activation, ScaledClampedQGeGLU) else "swiglu"
            )
            self._cudnn_dact_func = (
                "dgeglu" if isinstance(activation, ScaledClampedQGeGLU) else "dswiglu"
            )

        # cuDNN-frontend >= 1.24.0 exposes runtime-configurable GeGLU
        # parameters; pass them through when the activation carries
        # non-default values (or always, if available).
        self._pass_geglu_runtime_params: bool = (
            isinstance(activation, ScaledClampedQGeGLU) and _cudnn_frontend_geglu_runtime_params()
        )
        if self._pass_geglu_runtime_params:
            self._cudnn_linear_offset: float = activation._clamped.glu_linear_offset
            self._cudnn_geglu_alpha: float = activation._clamped.alpha
            self._cudnn_glu_clamp_max: float = activation._clamped.limit
            self._cudnn_glu_clamp_min: float = -activation._clamped.limit

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        # Get basic operations
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, _activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        input_ = input_.reshape(-1, fc1_weight_shape[1])
        in_shape = list(input_.size())
        if in_shape[0] % 128 != 0:
            raise ValueError(f"Unsupported input shape for fused grouped MLP ({in_shape=}).")

        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_weight_param.dtype

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = requires_grad and (
            fc1_weight_param.requires_grad or fc2_weight_param.requires_grad
        )

        # Quantizers
        fc1_input_quantizer = fc1_op.get_quantizer("forward", 0)
        fc1_weight_quantizer = fc1_op.get_quantizer("forward", 1)
        fc1_grad_output_quantizer = fc1_op.get_quantizer("backward", 0)
        fc2_input_quantizer = fc2_op.get_quantizer("forward", 0)
        fc2_weight_quantizer = fc2_op.get_quantizer("forward", 1)
        fc2_grad_output_quantizer = fc2_op.get_quantizer("backward", 0)

        # Extract split sizes from extra input
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            fc1_split_sizes.size() != fc2_split_sizes.size()
            or fc1_split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} got different split points for FC1 and FC2."
            )
        split_sizes = fc1_split_sizes
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")

        # Prepare split metadata
        split_sizes, (
            split_points,
            base_split_offsets,
            fc1_x_tensor_offsets,
            fc2_x_tensor_offsets,
            fc2_out_tensor_offsets,
        ) = tex.splits_to_offsets_multi(
            split_sizes,
            device,
            strides=[1, 1, fc1_weight_shape[1], fc2_weight_shape[1], fc2_weight_shape[0]],
            include_leading_zero=[False, True, True, True, True],
            dtypes=[torch.int32, torch.int64, torch.int64, torch.int64, torch.int64],
            bulk_allocate=True,
        )

        # Extract per-row activation probabilities from the middle op.
        scales = basic_op_extra_inputs[1][0]

        # Prepare FC1 grouped weight tensor for fused kernels.
        #  - single_grouped_weight=True: op.weight is already a GroupedTensor
        #  - single_grouped_weight=False: cute DSL kernel works with discrete weight tensors
        #   as long as host pointers for addresses are packed as contiguous device tensor.
        if fc1_op.single_grouped_weight:
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc1_op.weight.quantizer is not None:
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc1_op.weight.quantizer = fc1_weight_quantizer
                grouped_fc1_weight = fc1_op.weight
            else:
                if fc1_op.weight.rowwise_data is None:
                    raise RuntimeError("FC1 grouped weight has no rowwise_data to quantize.")
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc1_weight = _group_quantize_for_grouped_mlp(
                    fc1_op.weight.rowwise_data.view(fc1_op.weight.logical_shape),
                    fc1_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc1_weights = []
            for idx, weight in enumerate(fc1_weights):
                quantizer = fc1_op.get_quantizer("forward", 2 * idx + 1)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc1_weights.append(quantizer(weight))
                else:
                    quantized_fc1_weights.append(weight)
            grouped_fc1_weight = quantized_fc1_weights

        # Prepare FC2 grouped weight tensor for fused kernels.
        if fc2_op.single_grouped_weight:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc2_op.weight.quantizer is not None:
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc2_op.weight.quantizer = fc2_weight_quantizer
                grouped_fc2_weight = fc2_op.weight
            else:
                if fc2_op.weight.rowwise_data is None:
                    raise RuntimeError("FC2 grouped weight has no rowwise_data to quantize.")
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc2_weight = _group_quantize_for_grouped_mlp(
                    fc2_op.weight.rowwise_data.view(fc2_op.weight.logical_shape),
                    fc2_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc2_weights = []
            for idx, weight in enumerate(fc2_weights):
                quantizer = fc2_op.get_quantizer("forward", 2 * idx + 1)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc2_weights.append(quantizer(weight))
                else:
                    quantized_fc2_weights.append(weight)
            grouped_fc2_weight = quantized_fc2_weights

        # Some wrapper-copy paths may drop grouped storage metadata; enforce defaults.
        if isinstance(grouped_fc1_weight, GroupedTensor) and not hasattr(
            grouped_fc1_weight, "_with_gemm_swizzled_scales"
        ):
            grouped_fc1_weight._with_gemm_swizzled_scales = False
        if isinstance(grouped_fc2_weight, GroupedTensor) and not hasattr(
            grouped_fc2_weight, "_with_gemm_swizzled_scales"
        ):
            grouped_fc2_weight._with_gemm_swizzled_scales = False

        # Group-quantize input tensor and convert dtypes if needed
        fc1_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        fc1_input_quantizer.optimize_for_gemm = True
        fc1_input_quantizer.internal = True
        input_quantizer = getattr(input_, "quantizer", None)
        if isinstance(input_, GroupedTensor) and (
            isinstance(fc1_input_quantizer, MXFP8Quantizer)
            and isinstance(input_quantizer, MXFP8Quantizer)
            or isinstance(fc1_input_quantizer, NVFP4Quantizer)
            and isinstance(input_quantizer, NVFP4Quantizer)
        ):
            # GroupedTensor is a torch.Tensor subclass, so the CPU offload
            # infrastructure's prepare_for_saving treats it as a plain tensor
            # and does not decompose it into its component data tensors.  By
            # repacking into a GroupedTensorStorage (not a torch.Tensor), we
            # ensure the fuser's prepare_for_saving call correctly decomposes
            # the activation before save_for_backward.
            grouped_fc1_x = GroupedTensorStorage(
                shape=input_.logical_shape,
                dtype=input_.fake_dtype,
                num_tensors=input_.num_tensors,
                shapes=input_.tensor_shapes,
                quantizer=input_.quantizer,
                data=input_.rowwise_data,
                columnwise_data=input_.columnwise_data,
                scale_inv=input_.scale_inv,
                columnwise_scale_inv=input_.columnwise_scale_inv,
                amax=input_.amax,
                columnwise_amax=input_.columnwise_amax,
                scale=input_.scale,
                first_dims=input_.first_dims,
                last_dims=input_.last_dims,
                tensor_offsets=input_.tensor_offsets,
                offsets=input_.offsets,
                scale_inv_offsets=input_.scale_inv_offsets,
                columnwise_scale_inv_offsets=input_.columnwise_scale_inv_offsets,
                with_gemm_swizzled_scales=input_._with_gemm_swizzled_scales,
                row_scaled_nvfp4=input_.row_scaled_nvfp4,
                nvfp4_use_4over6=input_.nvfp4_use_4over6,
                nvfp4_e4m3_max=input_.nvfp4_e4m3_max,
            )
        else:
            fc1_x = maybe_dequantize(input_, dtype)
            grouped_fc1_x = _group_quantize_for_grouped_mlp(
                fc1_x,
                fc1_input_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc1_x_tensor_offsets,
            )

        use_nvfp4 = isinstance(fc1_input_quantizer, NVFP4Quantizer) or isinstance(
            fc1_weight_param, NVFP4Tensor
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        data_in_k = in_shape[1] // 2 if use_nvfp4 else in_shape[1]
        fc1_weight_k = fc1_weight_shape[1] // 2 if use_nvfp4 else fc1_weight_shape[1]
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        fc1_x_data = grouped_fc1_x.rowwise_data.view(dtype=data_dtype)
        fc1_x_data = fc1_x_data.view(in_shape[0], data_in_k)
        fc1_x_data = fc1_x_data.unsqueeze(0).permute(1, 2, 0)
        fc1_x_scales = grouped_fc1_x.scale_inv
        fc1_x_scales = fc1_x_scales.view(dtype=scale_view_dtype)
        with_gemm_swizzled_scales = grouped_fc1_x._with_gemm_swizzled_scales
        if use_nvfp4 and with_gemm_swizzled_scales:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                ceil_div(data_in_k, k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)
        elif use_nvfp4:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                4,
                32,
                ceil_div(data_in_k, k_sf_divisor),
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 2, 1, 5, 4, 0)
        else:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                ceil_div(in_shape[1], k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)

        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, torch.float32, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        fc1_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc1_op)
        fc2_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc2_op)

        fc1_d_dtype = torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn
        fc1_prob_tensor = (
            scales.detach().to(dtype=torch.float32 if use_nvfp4 else dtype).reshape(-1, 1, 1)
        )
        fc1_norm_const_tensor = None if use_nvfp4 else norm_const_tensor
        if use_nvfp4:
            nvfp4_fp4_max = 6.0
            nvfp4_fp8_max = 448.0
            nvfp4_global_scale_denom = nvfp4_fp4_max * nvfp4_fp8_max
            # cuDNN receives NVFP4 block-scaled inputs without TE's per-group
            # global scale factors, so alpha supplies the product of the two
            # operand global scales.
            fc1_alpha_tensor = (
                _nvfp4_amax(grouped_fc1_x, columnwise=False)
                * _nvfp4_amax(grouped_fc1_weight, columnwise=False)
                / (nvfp4_global_scale_denom**2)
            ).to(torch.float32)
        else:
            fc1_alpha_tensor = alpha_tensor

        use_tmem_post_rht_amax = _use_tmem_post_rht_amax()
        use_fc1_act_hadamard = False
        use_fc1_act_hadamard_srelu = False
        use_nvfp4_rht_amax = (
            use_nvfp4
            and isinstance(fc2_input_quantizer, NVFP4Quantizer)
            and fc2_input_quantizer.with_rht
            and fc2_input_quantizer.with_post_rht_amax
        )
        activation_is_srelu = isinstance(activation_op, ScaledSReLU)
        activation_supports_hadamard = self._cudnn_act_func == "swiglu" or (
            activation_is_srelu and _cudnn_frontend_supports_grouped_gemm_srelu_hadamard()
        )
        if use_nvfp4_rht_amax and activation_supports_hadamard:
            kernel_getter = getattr(self, "grouped_gemm_act_hadamard_kernel", None)
            if kernel_getter is not None:
                use_fc1_act_hadamard = kernel_getter() is not None
                use_fc1_act_hadamard_srelu = use_fc1_act_hadamard and activation_is_srelu

        fc1_activation_kwargs = {
            "a_tensor": fc1_x_data,
            "sfa_tensor": fc1_x_scales,
            "padded_offsets": split_points,
            "alpha_tensor": fc1_alpha_tensor,
            "bias_tensor": fc1_bias_packed,
            "prob_tensor": fc1_prob_tensor,
            "acc_dtype": torch.float32,
            "c_dtype": torch.bfloat16,
            "d_dtype": fc1_d_dtype,
            "cd_major": "n",
            "sf_vec_size": sf_vec_size,
            "current_stream": current_stream,
            "use_dynamic_sched": True,
        }
        if use_fc1_act_hadamard_srelu:
            fc1_activation_kwargs["act_func"] = "srelu"
        elif self._cudnn_act_func is not None:
            fc1_activation_kwargs["act_func"] = self._cudnn_act_func
        if use_fc1_act_hadamard:
            fc1_activation_kwargs["use_tmem_post_rht_amax"] = use_tmem_post_rht_amax
        else:
            fc1_activation_kwargs["norm_const_tensor"] = fc1_norm_const_tensor
            fc1_activation_kwargs["discrete_col_sfd"] = not use_nvfp4
        if self._pass_geglu_runtime_params:
            fc1_activation_kwargs.update(
                linear_offset=self._cudnn_linear_offset,
                geglu_alpha=self._cudnn_geglu_alpha,
                glu_clamp_max=self._cudnn_glu_clamp_max,
                glu_clamp_min=self._cudnn_glu_clamp_min,
            )

        if fc1_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM.
            fc1_weight_for_gemm = grouped_fc1_weight.copy()
            tex.grouped_swizzle_for_gemm(fc1_weight_for_gemm, rowwise=True, columnwise=False)

            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, n, k)
            # Data logical shape: (n, k, num_groups)
            fc1_w_data = fc1_weight_for_gemm.rowwise_data
            fc1_w_data = fc1_w_data.view(dtype=data_dtype)
            fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], fc1_weight_k)
            fc1_w_data = fc1_w_data.permute(1, 2, 0)
            fc1_w_scales = fc1_weight_for_gemm.scale_inv.view(dtype=scale_view_dtype)
            fc1_w_scales = fc1_w_scales.view(
                num_groups,
                ceil_div(fc1_weight_shape[0], 128),
                ceil_div(fc1_weight_shape[1], k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

            fc1_activation_kwargs["b_tensor"] = fc1_w_data
            fc1_activation_kwargs["sfb_tensor"] = fc1_w_scales
        else:
            # Discrete-weight kernel: per-expert data/scale pointers
            fc1_b_ptrs, fc1_sfb_ptrs, _fc1_sfb_buffer = (
                tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                    [w._rowwise_data for w in grouped_fc1_weight],
                    [w._rowwise_scale_inv for w in grouped_fc1_weight],
                    "nvfp4" if use_nvfp4 else "mxfp8_rowwise",
                    device,
                )
            )
            fc1_activation_kwargs["b_ptrs"] = fc1_b_ptrs
            fc1_activation_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
            fc1_activation_kwargs["n"] = fc1_weight_shape[0]
            fc1_activation_kwargs["b_dtype"] = data_dtype
            fc1_activation_kwargs["b_major"] = "k"

        if use_fc1_act_hadamard:
            fc1_kernel_out = self.grouped_gemm_act_hadamard_kernel()(**fc1_activation_kwargs)
        else:
            fc1_kernel_out = self.grouped_gemm_activation_kernel()(**fc1_activation_kwargs)

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m_splits), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m_splits)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (sum(m_splits), k, 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m_splits)/128, 1)
        activation_in = fc1_kernel_out["c_tensor"]
        activation_in = activation_in.view(in_shape[0], fc1_weight_shape[0])

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_weight_shape[0]]
        fc2_scales = basic_op_extra_inputs[2][1] if fc2_op._scale_bias else None

        if use_nvfp4:
            fc2_bias_for_gemm = None
            fc2_bias_scale = None
            if fc2_bias_packed is not None:
                fc2_bias_for_gemm = fc2_op._get_grouped_bias_for_gemm(dtype)
                if fc2_scales is not None:
                    fc2_bias_scale = fc2_scales.reshape(-1)
                    if fc2_bias_scale.dtype != torch.float32:
                        fc2_bias_scale = fc2_bias_scale.to(dtype=torch.float32)

            fc2_in = fc1_kernel_out["d_tensor"]
            fc2_in = fc2_in.view(in_shape[0], fc2_weight_shape[1]).contiguous()
            fc2_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            fc2_input_quantizer.optimize_for_gemm = True
            if use_fc1_act_hadamard:
                grouped_fc2_x = _group_quantize_with_amax_for_grouped_mlp(
                    fc2_in,
                    fc2_input_quantizer,
                    num_groups,
                    split_sizes,
                    fc1_kernel_out["amax_tensor"].view(-1),
                    fc1_kernel_out["post_rht_amax_tensor"].view(-1),
                    tensor_offsets=fc2_x_tensor_offsets,
                )
            else:
                grouped_fc2_x = _group_quantize_for_grouped_mlp(
                    fc2_in,
                    fc2_input_quantizer,
                    num_groups,
                    split_sizes,
                    tensor_offsets=fc2_x_tensor_offsets,
                )

            fc2_out_buf = torch.empty(fc2_out_shape, dtype=dtype, device=device)
            if (
                num_groups == 1
                and grouped_fc2_x.columnwise_data is not None
                and grouped_fc2_x.columnwise_scale_inv is not None
            ):
                if fc2_op.single_grouped_weight:
                    fc2_w_single = grouped_fc2_weight.split_into_quantized_tensors()[0]
                else:
                    fc2_w_single = grouped_fc2_weight[0]
                fc2_x_single = _nvfp4_single_tensor_from_grouped(
                    grouped_fc2_x,
                    fc2_input_quantizer,
                    fp4_dtype=fc2_w_single._fp4_dtype,
                )
                general_gemm(
                    fc2_w_single,
                    fc2_x_single,
                    out_dtype=dtype,
                    out=fc2_out_buf,
                    layout="TN",
                    use_split_accumulator=False,
                )
                if fc2_bias_packed is not None:
                    token_bias = (
                        fc2_bias_packed.transpose(0, 1).contiguous().expand(in_shape[0], -1)
                    )
                    if fc2_scales is not None:
                        fc2_out_buf = fc2_out_buf + token_bias * fc2_scales.view(-1, 1)
                    else:
                        fc2_out_buf = fc2_out_buf + token_bias
            else:
                fc2_out_grouped = GroupedTensorStorage(
                    shape=(in_shape[0], fc2_weight_shape[0]),
                    dtype=dtype,
                    num_tensors=num_groups,
                    quantizer=None,
                    data=fc2_out_buf.view(-1),
                    first_dims=split_sizes,
                    tensor_offsets=fc2_out_tensor_offsets,
                )
                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc2_weight,
                    grouped_fc2_x,
                    fc2_out_grouped,
                    layout="TN",
                    bias=fc2_bias_for_gemm,
                    bias_scale=fc2_bias_scale,
                )
            fc2_out = fc2_out_buf
        else:
            fc2_in_row_data = fc1_kernel_out["d_tensor"]
            fc2_in_row_data = fc2_in_row_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_row_scale = fc1_kernel_out["sfd_row_tensor"]
            fc2_in_row_scale = fc2_in_row_scale.permute(5, 2, 4, 0, 1, 3)

            fc2_in_col_data = fc1_kernel_out["d_col_tensor"]
            fc2_in_col_data = fc2_in_col_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_col_scale = fc1_kernel_out["sfd_col_tensor"]
            fc2_in_col_scale = fc2_in_col_scale.permute(5, 2, 4, 0, 1, 3)

            grouped_fc2_x = GroupedTensorStorage(
                shape=(in_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_input_quantizer,
                data=fc2_in_row_data.reshape(-1),
                columnwise_data=fc2_in_col_data.reshape(-1),
                scale_inv=fc2_in_row_scale.reshape(-1),
                columnwise_scale_inv=fc2_in_col_scale.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

            fc2_scales_tensor = (
                fc2_scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1)
                if fc2_scales is not None
                else torch.ones((in_shape[0], 1, 1), dtype=torch.float32, device=device)
            )
            fc2_quant_kwargs = {
                "a_tensor": fc1_kernel_out["d_tensor"],
                "sfa_tensor": fc1_kernel_out["sfd_row_tensor"],
                "padded_offsets": split_points,
                "alpha_tensor": alpha_tensor,
                "bias_tensor": fc2_bias_packed,
                "norm_const_tensor": None,
                "prob_tensor": fc2_scales_tensor,
                "acc_dtype": torch.float32,
                "d_dtype": dtype,
                "cd_major": "n",
                "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                "current_stream": current_stream,
                "use_dynamic_sched": True,
            }

            if fc2_op.single_grouped_weight:
                # Clone and swizzle scales for GEMM (original stays unmodified for save_for_backward)
                fc2_weight_for_gemm = grouped_fc2_weight.copy()
                tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=True, columnwise=False)

                fc2_w_data = fc2_weight_for_gemm.rowwise_data
                fc2_w_data = fc2_w_data.view(dtype=torch.float8_e4m3fn)
                fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_shape[1])
                fc2_w_data = fc2_w_data.permute(1, 2, 0)

                fc2_w_scales = fc2_weight_for_gemm.scale_inv.view(dtype=torch.float8_e8m0fnu)
                fc2_w_scales = fc2_w_scales.view(
                    num_groups,
                    ceil_div(fc2_weight_shape[0], 128),
                    ceil_div(fc2_weight_shape[1], 128),
                    MXFP8_BLOCK_SCALING_SIZE,
                    4,
                    4,
                )
                fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)
                fc2_quant_kwargs["b_tensor"] = fc2_w_data
                fc2_quant_kwargs["sfb_tensor"] = fc2_w_scales
            else:
                fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sfb_buffer = (
                    tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                        [w._rowwise_data for w in grouped_fc2_weight],
                        [w._rowwise_scale_inv for w in grouped_fc2_weight],
                        "nvfp4" if use_nvfp4 else "mxfp8_rowwise",
                        device,
                    )
                )
                fc2_quant_kwargs["b_ptrs"] = fc2_b_ptrs
                fc2_quant_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
                fc2_quant_kwargs["n"] = fc2_weight_shape[0]
                fc2_quant_kwargs["b_dtype"] = torch.float8_e4m3fn
                fc2_quant_kwargs["b_major"] = "k"

            fc2_kernel_out = self.grouped_gemm_quant_kernel()(**fc2_quant_kwargs)
            fc2_out = fc2_kernel_out["d_tensor"].permute(2, 0, 1).view(fc2_out_shape).contiguous()

        # Save state for backward pass
        if requires_grad:
            mark_grouped_tensor(grouped_fc1_x, activation_in, scales, grouped_fc2_x)
            activation_op = self.basic_ops[1]
            cpu_offloading = is_cpu_offload_enabled()
            activation_is_srelu = isinstance(activation_op, ScaledSReLU)
            activation_recompute_in_mlp = bool(
                getattr(activation_op, "activation_recompute_in_mlp", False)
            )
            recompute_srelu_fc2_x = (
                activation_is_srelu
                and activation_recompute_in_mlp
                and weight_requires_grad
                and _grouped_gemm_dsrelu_backward_supported()
                and _nvidia_cudnn_frontend_supports_wgrad()
            )
            saved_grouped_fc2_x = None if recompute_srelu_fc2_x else grouped_fc2_x

            # MXFP8 wgrad only needs columnwise tiles. NVFP4 generic GEMM fallbacks
            # need the full grouped tensor state, including rowwise data and amax.
            if not use_nvfp4:
                for grouped_fc_x in (grouped_fc1_x, saved_grouped_fc2_x):
                    if grouped_fc_x is not None:
                        grouped_fc_x.rowwise_data = None
                        grouped_fc_x.scale_inv = None

            if cpu_offloading:
                activation_tensors = [
                    t for t in (grouped_fc1_x, activation_in, saved_grouped_fc2_x) if t is not None
                ]
                start_offload(*activation_tensors)
                mark_activation_offload(*activation_tensors)

            # Save an internal layout for this joint fused op. The saved state is
            # intentionally not compatible with the basic GroupedLinear backward.
            fc1_weight_tensors = (
                [grouped_fc1_weight] if fc1_op.single_grouped_weight else grouped_fc1_weight
            )
            fc2_weight_tensors = (
                [grouped_fc2_weight] if fc2_op.single_grouped_weight else grouped_fc2_weight
            )
            fc1_ctx.save_for_backward(
                split_sizes,
                base_split_offsets,
                split_points,
                grouped_fc1_x,
                *fc1_weight_tensors,
                activation_in,
                scales,
                saved_grouped_fc2_x,
                *fc2_weight_tensors,
            )

            fc1_ctx.input_quantizers = [fc1_input_quantizer]
            fc1_ctx.grad_output_quantizers = [fc1_grad_output_quantizer]
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = weight_requires_grad

            fc2_ctx.input_quantizers = [fc2_input_quantizer]
            fc2_ctx.grad_output_quantizers = [fc2_grad_output_quantizer]
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad
            fc2_ctx.recompute_input_from_dsrelu = recompute_srelu_fc2_x

        return fc2_out, [(), (), ()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        fc1_op, activation_op, fc2_op = self.basic_ops
        activation_is_srelu = isinstance(activation_op, ScaledSReLU)
        fc1_ctx, _activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        grad_output = grad_output.reshape(-1, fc2_weight_shape[0])
        out_shape = list(grad_output.size())
        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        dtype = fc1_ctx.dtype

        # Saved tensors from the joint forward.
        # Layout: [split_sizes, base_split_offsets, split_points,
        #          grouped_fc1_x, *fc1_weights,
        #          activation_in, scales,
        #          grouped_fc2_x, *fc2_weights]
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, base_split_offsets, split_points = saved_tensors[:3]
        saved_tensors = saved_tensors[3:]
        grouped_fc1_x, saved_tensors = saved_tensors[0], saved_tensors[1:]
        if fc1_op.single_grouped_weight:
            grouped_fc1_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc1_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        activation_in, scales, saved_tensors = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2:],
        )
        scale_bias = fc2_op._scale_bias and fc2_op.has_bias
        recompute_fc2_x_from_dsrelu = bool(
            getattr(fc2_ctx, "recompute_input_from_dsrelu", False)
        ) and bool(fc2_ctx.weight_requires_grad)

        grouped_fc2_x, saved_tensors = saved_tensors[0], saved_tensors[1:]
        if fc2_op.single_grouped_weight:
            grouped_fc2_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc2_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        # Group splits
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")

        if not fc1_ctx.weight_requires_grad:
            grouped_fc1_x = None
        if not fc2_ctx.weight_requires_grad:
            grouped_fc2_x = None

        # Split grad output tensor and convert dtypes if needed
        fc2_grad_output_quantizer = fc2_ctx.grad_output_quantizers[0]
        fc2_grad_output_quantizer.set_usage(rowwise=True, columnwise=fc2_ctx.weight_requires_grad)
        fc2_grad_output_quantizer.optimize_for_gemm = True
        output_fc2_dbias = fc2_op.has_bias
        fc2_dbias_packed = None
        fc2_dy = None
        grad_output_quantizer = getattr(grad_output, "quantizer", None)
        fc2_grad_output_quantizer_matches = (
            isinstance(fc2_grad_output_quantizer, MXFP8Quantizer)
            and isinstance(grad_output_quantizer, MXFP8Quantizer)
        ) or (
            isinstance(fc2_grad_output_quantizer, NVFP4Quantizer)
            and isinstance(grad_output_quantizer, NVFP4Quantizer)
        )
        if (
            not output_fc2_dbias
            and isinstance(grad_output, GroupedTensor)
            and fc2_grad_output_quantizer_matches
        ):
            grouped_fc2_dy = grad_output
        else:
            fc2_dy = maybe_dequantize(grad_output, dtype)
            if output_fc2_dbias and not scale_bias:
                grouped_fc2_dy, fc2_dbias_packed = tex.bgrad_group_quantize(
                    fc2_dy,
                    fc2_grad_output_quantizer,
                    num_groups,
                    split_sizes,
                )
            else:
                grouped_fc2_dy = _group_quantize_for_grouped_mlp(
                    fc2_dy,
                    fc2_grad_output_quantizer,
                    num_groups,
                    split_sizes,
                    tensor_offsets=base_split_offsets * fc2_weight_shape[0],
                )

        use_nvfp4 = (
            isinstance(fc2_grad_output_quantizer, NVFP4Quantizer)
            or isinstance(fc1_weight_param, NVFP4Tensor)
            or isinstance(fc2_weight_param, NVFP4Tensor)
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        data_k = out_shape[1] // 2 if use_nvfp4 else out_shape[1]
        fc2_weight_k = fc2_weight_shape[1] // 2 if use_nvfp4 else fc2_weight_shape[1]
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        fc2_dy_data = grouped_fc2_dy.rowwise_data.view(dtype=data_dtype)
        fc2_dy_data = fc2_dy_data.view(out_shape[0], data_k)
        fc2_dy_data = fc2_dy_data.unsqueeze(0).permute(1, 2, 0)
        fc2_dy_scales = grouped_fc2_dy.scale_inv
        fc2_dy_scales = fc2_dy_scales.view(dtype=scale_view_dtype)
        with_gemm_swizzled_scales = grouped_fc2_dy._with_gemm_swizzled_scales
        if use_nvfp4 and with_gemm_swizzled_scales:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                ceil_div(out_shape[0], 128),
                ceil_div(data_k, k_sf_divisor),
                32,
                4,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)
        elif use_nvfp4:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                ceil_div(out_shape[0], 128),
                4,
                32,
                ceil_div(data_k, k_sf_divisor),
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 2, 1, 5, 4, 0)
        else:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                ceil_div(out_shape[0], 128),
                ceil_div(out_shape[1], k_sf_divisor),
                32,
                4,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel scaling factors
        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, torch.float32, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        scales_f32 = scales.detach().to(dtype=torch.float32)
        scales_tensor = scales_f32.reshape(-1, 1, 1)
        dscales_tensor = torch.zeros_like(scales_tensor)

        fc2_d_dtype = torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn
        if use_nvfp4:
            nvfp4_fp4_max = 6.0
            nvfp4_fp8_max = 448.0
            nvfp4_global_scale_denom = nvfp4_fp4_max * nvfp4_fp8_max
            fc2_dy_amax = _nvfp4_amax(grouped_fc2_dy, columnwise=False)
            fc2_weight_col_amax = _nvfp4_amax(grouped_fc2_weight, columnwise=True)
            if activation_is_srelu:
                # DSReLU applies alpha once, so pass the full product of the
                # two operand global scales.
                fc2_alpha_tensor = (
                    (fc2_dy_amax * fc2_weight_col_amax / (nvfp4_global_scale_denom**2))
                    .to(torch.float32)
                    .expand(num_groups)
                )
            else:
                # DGLU applies alpha to both gate branches, so the wrapper
                # expects sqrt(product) to recover the same global-scale factor.
                fc2_alpha_tensor = (
                    torch.sqrt(fc2_dy_amax * fc2_weight_col_amax) / nvfp4_global_scale_denom
                ).expand(num_groups)
            fc2_beta_tensor = get_cached_ones_tensor(num_groups, torch.float32, device)
            fc2_norm_const_tensor = None
        else:
            fc2_alpha_tensor = alpha_tensor
            fc2_beta_tensor = alpha_tensor
            fc2_norm_const_tensor = norm_const_tensor

        fc2_dactivation_kwargs = {
            "a_tensor": fc2_dy_data,
            "c_tensor": activation_in.unsqueeze(0).permute(1, 2, 0),
            "sfa_tensor": fc2_dy_scales,
            "padded_offsets": split_points,
            "alpha_tensor": fc2_alpha_tensor,
            "prob_tensor": scales_tensor,
            "dprob_tensor": dscales_tensor,
            "generate_dbias": fc1_op.has_bias,
            "norm_const_tensor": fc2_norm_const_tensor,
            "d_dtype": fc2_d_dtype,
            "cd_major": "n",
            "sf_vec_size": sf_vec_size,
            "current_stream": current_stream,
            "discrete_col_sfd": not use_nvfp4,
            "use_dynamic_sched": True,
        }
        if self._cudnn_dact_func is not None:
            fc2_dactivation_kwargs["beta_tensor"] = fc2_beta_tensor
            fc2_dactivation_kwargs["act_func"] = self._cudnn_dact_func
        else:
            fc2_dactivation_kwargs["use_dsrelu_reuse"] = recompute_fc2_x_from_dsrelu
        if self._pass_geglu_runtime_params:
            fc2_dactivation_kwargs.update(
                linear_offset=self._cudnn_linear_offset,
                geglu_alpha=self._cudnn_geglu_alpha,
                glu_clamp_max=self._cudnn_glu_clamp_max,
                glu_clamp_min=self._cudnn_glu_clamp_min,
            )

        if fc2_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM
            fc2_weight_for_gemm = grouped_fc2_weight.copy()
            tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=False, columnwise=True)
            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, k, n)
            # Data logical shape: (n, k, num_groups)
            fc2_w_data = fc2_weight_for_gemm.columnwise_data
            fc2_w_data = fc2_w_data.view(dtype=data_dtype)
            fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_k)
            fc2_w_data = fc2_w_data.permute(1, 2, 0) if use_nvfp4 else fc2_w_data.permute(2, 1, 0)
            fc2_w_scales = fc2_weight_for_gemm.columnwise_scale_inv.view(dtype=scale_view_dtype)
            fc2_w_scales = fc2_w_scales.view(
                num_groups,
                ceil_div(fc2_weight_shape[1], k_sf_divisor),
                ceil_div(fc2_weight_shape[0], 128),
                32,
                4,
                4,
            )
            fc2_w_scales = (
                fc2_w_scales.permute(3, 4, 2, 5, 1, 0)
                if use_nvfp4
                else fc2_w_scales.permute(3, 4, 1, 5, 2, 0)
            )

            fc2_dactivation_kwargs["b_tensor"] = fc2_w_data
            fc2_dactivation_kwargs["sfb_tensor"] = fc2_w_scales
        else:
            fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sfb_buffer = (
                tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                    [w._columnwise_data for w in grouped_fc2_weight],
                    [w._columnwise_scale_inv for w in grouped_fc2_weight],
                    "nvfp4" if use_nvfp4 else "mxfp8_columnwise",
                    device,
                )
            )
            fc2_dactivation_kwargs["b_ptrs"] = fc2_b_ptrs
            fc2_dactivation_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
            fc2_dactivation_kwargs["n"] = fc2_weight_shape[1]
            fc2_dactivation_kwargs["b_dtype"] = data_dtype
            fc2_dactivation_kwargs["b_major"] = "k" if use_nvfp4 else "n"

        fc2_dgrad_kernel_out = self.grouped_gemm_dactivation_kernel()(**fc2_dactivation_kwargs)

        if use_nvfp4:
            fc1_dy_bf16 = fc2_dgrad_kernel_out["d_row_tensor"]
            fc1_dy_bf16 = fc1_dy_bf16.view(out_shape[0], fc1_weight_shape[0]).contiguous()
            fc1_dy_row_data = None
            fc1_dy_row_scale = None
            fc1_dy_col_data = None
            fc1_dy_col_scale = None
        else:
            fc1_dy_bf16 = None
            fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
            fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0])
            # View scale in their actual swizzled shape
            fc1_dy_row_scale = (
                fc2_dgrad_kernel_out["sfd_row_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
            )
            fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
            fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0])
            # View scale in their actual swizzled shape
            fc1_dy_col_scale = (
                fc2_dgrad_kernel_out["sfd_col_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
            )
        grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].view(-1)

        if recompute_fc2_x_from_dsrelu:
            d_srelu_tensor = fc2_dgrad_kernel_out.get("d_srelu_tensor")
            if d_srelu_tensor is None:
                raise RuntimeError(
                    "SReLU recompute is enabled, but the DSReLU kernel did not return "
                    "the recomputed FC2 input tensor."
                )

            if use_nvfp4:
                # cuDNN's grouped dSReLU FP4 path recomputes FC2 input in BF16.
                # Re-quantize to produce the columnwise FP4 data, scales, and
                # dSReLU-specific amax needed by the NVFP4 wgrad GEMM.
                fc2_x_bf16 = d_srelu_tensor.view(out_shape[0], fc2_weight_shape[1]).contiguous()
                fc2_input_quantizer = fc2_ctx.input_quantizers[0]
                fc2_input_quantizer.set_usage(rowwise=False, columnwise=True)
                fc2_input_quantizer.optimize_for_gemm = True
                grouped_fc2_x = _group_quantize_for_grouped_mlp(
                    fc2_x_bf16,
                    fc2_input_quantizer,
                    num_groups,
                    split_sizes,
                    tensor_offsets=base_split_offsets * fc2_weight_shape[1],
                )
            else:
                sfd_col_d_srelu_tensor = fc2_dgrad_kernel_out.get("sfd_col_d_srelu_tensor")
                if sfd_col_d_srelu_tensor is None:
                    raise RuntimeError(
                        "SReLU recompute is enabled, but the DSReLU kernel did not return "
                        "the recomputed FC2 input column scale tensor."
                    )

                fc2_x_col_data = d_srelu_tensor.view(out_shape[0], fc2_weight_shape[1])
                fc2_x_col_scale = sfd_col_d_srelu_tensor.permute(5, 2, 4, 0, 1, 3)
                grouped_fc2_x = GroupedTensor(
                    shape=(out_shape[0], fc2_weight_shape[1]),
                    dtype=dtype,
                    num_tensors=num_groups,
                    quantizer=fc2_ctx.input_quantizers[0],
                    data=None,
                    columnwise_data=fc2_x_col_data.reshape(-1),
                    scale_inv=None,
                    columnwise_scale_inv=fc2_x_col_scale.reshape(-1),
                    first_dims=split_sizes,
                    tensor_offsets=base_split_offsets * fc2_weight_shape[1],
                    with_gemm_swizzled_scales=True,
                )

        fc2_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc2_bias_grad_packed: Optional[torch.Tensor] = None
        if scale_bias:
            fc2_biases = fc2_op._get_bias_tensors(dtype)
            bias_packed = torch.stack(fc2_biases)
            fc2_dbias_packed_result, grad_scales = compute_grouped_dbias_dscales(
                fc2_dy,
                scales_f32,
                bias_packed,
                offsets=base_split_offsets,
                dscales=grad_scales,
            )
            fc2_dbias_packed_result = fc2_dbias_packed_result.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed_result
            else:
                fc2_bias_grads = [fc2_dbias_packed_result[idx] for idx in range(num_groups)]
        elif fc2_dbias_packed is not None:
            fc2_dbias_packed = fc2_dbias_packed.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed
            else:
                fc2_bias_grads = [fc2_dbias_packed[idx] for idx in range(num_groups)]

        if grad_scales is not None:
            grad_scales = grad_scales.to(dtype=dtype)

        fc1_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc1_bias_grad_packed: Optional[torch.Tensor] = None
        if fc1_op.has_bias:
            dbias_t = fc2_dgrad_kernel_out["dbias_tensor"]
            if dbias_t is not None:
                dbias_2d = dbias_t.squeeze(-1).to(dtype=dtype)
                if fc1_op.single_grouped_bias:
                    fc1_bias_grad_packed = dbias_2d
                else:
                    fc1_bias_grads = [dbias_2d[group_idx] for group_idx in range(num_groups)]

        # FC1 grad output for dgrad and wgrad GEMMs
        fc1_dy_tensor_offsets = base_split_offsets * fc1_weight_shape[0]
        fc1_grad_output_quantizer = fc1_ctx.grad_output_quantizers[0]
        if use_nvfp4:
            fc1_grad_output_quantizer.set_usage(
                rowwise=True,
                columnwise=fc1_ctx.weight_requires_grad,
            )
            fc1_grad_output_quantizer.optimize_for_gemm = True
            grouped_fc1_dy = _group_quantize_for_grouped_mlp(
                fc1_dy_bf16,
                fc1_grad_output_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc1_dy_tensor_offsets,
            )
        else:
            grouped_fc1_dy = GroupedTensor(
                shape=(out_shape[0], fc1_weight_shape[0]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc1_grad_output_quantizer,
                data=fc1_dy_row_data,
                columnwise_data=fc1_dy_col_data,
                scale_inv=fc1_dy_row_scale,
                columnwise_scale_inv=fc1_dy_col_scale,
                first_dims=split_sizes,
                tensor_offsets=fc1_dy_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        # FC2 wgrad GEMM
        wgrad_kernel_fn = self.grouped_gemm_wgrad_kernel()
        fc2_grad_params = _compute_grad_params(
            fc_op=fc2_op,
            ctx=fc2_ctx,
            num_groups=num_groups,
            weight_shape=fc2_weight_shape,
            grouped_x=grouped_fc2_x,
            grouped_dy=grouped_fc2_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc2_bias_grads,
            bias_grad_packed=fc2_bias_grad_packed,
            label="FC2",
            cudnn_wgrad_kernel_fn=wgrad_kernel_fn,
            use_nvfp4=use_nvfp4,
            data_dtype=data_dtype,
            scale_view_dtype=scale_view_dtype,
            sf_vec_size=sf_vec_size,
            offsets=split_points,
        )

        # Clear FC2 input tensor if possible
        if grouped_fc2_x is not None and not (
            fc2_ctx.weight_requires_grad
            and fc2_op.wgrad_store is not None
            and fc2_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc2_x.rowwise_data,
                grouped_fc2_x.columnwise_data,
                grouped_fc2_x.scale_inv,
                grouped_fc2_x.columnwise_scale_inv,
            )

        # FC1 dgrad GEMM
        grad_input = None
        if fc1_ctx.input_requires_grad:
            in_shape = out_shape[:-1] + [fc1_weight_shape[1]]

            if use_nvfp4:
                grad_input = torch.empty(in_shape, dtype=dtype, device=device)
                if num_groups == 1:
                    if fc1_op.single_grouped_weight:
                        fc1_w_single = grouped_fc1_weight.split_into_quantized_tensors()[0]
                    else:
                        fc1_w_single = grouped_fc1_weight[0]
                    fc1_dy_single = _nvfp4_single_tensor_from_grouped(grouped_fc1_dy)
                    general_gemm(
                        fc1_w_single,
                        fc1_dy_single,
                        out_dtype=dtype,
                        out=grad_input,
                        layout="NN",
                    )
                else:
                    fc1_x_tensor_offsets = base_split_offsets * fc1_weight_shape[1]
                    grouped_grad_input = GroupedTensor(
                        shape=(out_shape[0], fc1_weight_shape[1]),
                        dtype=dtype,
                        num_tensors=num_groups,
                        quantizer=None,
                        data=grad_input.view(-1),
                        first_dims=split_sizes,
                        tensor_offsets=fc1_x_tensor_offsets,
                    )
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc1_weight,
                        grouped_fc1_dy,
                        grouped_grad_input,
                        layout="NN",
                    )
            else:
                fc1_dgrad_a_data = fc2_dgrad_kernel_out["d_row_tensor"]
                fc1_dgrad_a_scales = fc2_dgrad_kernel_out["sfd_row_tensor"]

                fc1_dgrad_kwargs = {
                    "a_tensor": fc1_dgrad_a_data,
                    "sfa_tensor": fc1_dgrad_a_scales,
                    "padded_offsets": split_points,
                    "alpha_tensor": alpha_tensor,
                    "norm_const_tensor": None,
                    "prob_tensor": torch.ones(
                        (out_shape[0], 1, 1), dtype=torch.float32, device=device
                    ),
                    "acc_dtype": torch.float32,
                    "d_dtype": dtype,
                    "cd_major": "n",
                    "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                    "current_stream": current_stream,
                    "discrete_col_sfd": True,
                    "use_dynamic_sched": True,
                }

                if fc1_op.single_grouped_weight:
                    # Clone and swizzle scales for GEMM
                    fc1_weight_for_gemm = grouped_fc1_weight.copy()
                    tex.grouped_swizzle_for_gemm(
                        fc1_weight_for_gemm, rowwise=False, columnwise=True
                    )

                    fc1_w_data = fc1_weight_for_gemm.columnwise_data
                    fc1_w_data = fc1_w_data.view(dtype=torch.float8_e4m3fn)
                    fc1_w_data = fc1_w_data.view(
                        num_groups, fc1_weight_shape[0], fc1_weight_shape[1]
                    )
                    fc1_w_data = fc1_w_data.permute(2, 1, 0)
                    fc1_w_scales = fc1_weight_for_gemm.columnwise_scale_inv.view(
                        dtype=torch.float8_e8m0fnu
                    )
                    fc1_w_scales = fc1_w_scales.view(
                        num_groups,
                        ceil_div(fc1_weight_shape[1], 128),
                        ceil_div(fc1_weight_shape[0], 128),
                        MXFP8_BLOCK_SCALING_SIZE,
                        4,
                        4,
                    )
                    fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

                    fc1_dgrad_kwargs["b_tensor"] = fc1_w_data
                    fc1_dgrad_kwargs["sfb_tensor"] = fc1_w_scales
                else:
                    fc1_b_ptrs, fc1_sfb_ptrs, _fc1_sfb_buffer = (
                        tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                            [w._columnwise_data for w in grouped_fc1_weight],
                            [w._columnwise_scale_inv for w in grouped_fc1_weight],
                            "nvfp4" if use_nvfp4 else "mxfp8_columnwise",
                            device,
                        )
                    )

                    fc1_dgrad_kwargs["b_ptrs"] = fc1_b_ptrs
                    fc1_dgrad_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
                    fc1_dgrad_kwargs["n"] = fc1_weight_shape[1]
                    fc1_dgrad_kwargs["b_dtype"] = torch.float8_e4m3fn
                    fc1_dgrad_kwargs["b_major"] = "n"

                fc1_dgrad_kernel_out = self.grouped_gemm_quant_kernel()(**fc1_dgrad_kwargs)
                grad_input = fc1_dgrad_kernel_out["d_tensor"].view(in_shape)

        # FC1 wgrad GEMM
        fc1_grad_params = _compute_grad_params(
            fc_op=fc1_op,
            ctx=fc1_ctx,
            num_groups=num_groups,
            weight_shape=fc1_weight_shape,
            grouped_x=grouped_fc1_x,
            grouped_dy=grouped_fc1_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc1_bias_grads,
            bias_grad_packed=fc1_bias_grad_packed,
            label="FC1",
            cudnn_wgrad_kernel_fn=wgrad_kernel_fn,
            use_nvfp4=use_nvfp4,
            data_dtype=data_dtype,
            scale_view_dtype=scale_view_dtype,
            sf_vec_size=sf_vec_size,
            offsets=split_points,
        )

        # Clear FC1 input tensor if possible
        if grouped_fc1_x is not None and not (
            fc1_ctx.weight_requires_grad
            and fc1_op.wgrad_store is not None
            and fc1_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc1_x.rowwise_data,
                grouped_fc1_x.columnwise_data,
                grouped_fc1_x.scale_inv,
                grouped_fc1_x.columnwise_scale_inv,
            )

        fc2_grad_extra = (None, None) if fc2_op._scale_bias else (None,)
        activation_grad_extra = (grad_scales,) if grad_scales is not None else ()
        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), activation_grad_extra, fc2_grad_extra],
        )


class GroupedMLP_CuTeGEMMGLU(_GroupedMLP_CuTeGEMMBase):
    """Joint fused op for block-scaled GroupedLinear + scaled GLU + GroupedLinear."""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation, and post-multiplication."""
        from cudnn import grouped_gemm_glu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_glu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_act_hadamard_kernel(cls) -> Optional[Callable]:
        """Fused grouped GEMM activation kernel that also emits NVFP4 RHT amaxes."""
        try:
            from cudnn import (
                grouped_gemm_glu_hadamard_wrapper_sm100,
            )  # pylint: disable=no-name-in-module,import-outside-toplevel
        except ImportError:
            return None

        return grouped_gemm_glu_hadamard_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dactivation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation backward, and scale grad."""
        from cudnn import grouped_gemm_dglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dglu_wrapper_sm100


class GroupedMLP_CuTeGEMMUnary(_GroupedMLP_CuTeGEMMBase):
    """Joint fused op for block-scaled GroupedLinear + scaled unary activation + GroupedLinear."""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the SReLU fused operation is supported on the current system."""
        return _cudnn_frontend_supports_grouped_gemm_srelu() and super().is_supported()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, SReLU activation, and post-multiplication."""
        from cudnn import grouped_gemm_srelu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_srelu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_act_hadamard_kernel(cls) -> Optional[Callable]:
        """Fused grouped GEMM activation kernel that also emits NVFP4 RHT amaxes."""
        if not _cudnn_frontend_supports_grouped_gemm_srelu_hadamard():
            return None

        try:
            from cudnn import (
                grouped_gemm_glu_hadamard_wrapper_sm100,
            )  # pylint: disable=no-name-in-module,import-outside-toplevel
        except ImportError:
            return None

        return grouped_gemm_glu_hadamard_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dactivation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM and dSReLU activation backward."""
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dsrelu_wrapper_sm100


def fuse_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply joint GroupedLinear + scaled GLU + GroupedLinear fusion."""

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=GroupedMLP_CuTeGEMMGLU,
    )


def fuse_srelu_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply joint GroupedLinear + ScaledSReLU + GroupedLinear fusion."""

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=GroupedMLP_CuTeGEMMUnary,
        activation_op_types=(ScaledSReLU,),
    )


# Register joint fusions if available.
if GroupedMLP_CuTeGEMMGLU.is_supported():
    register_forward_backward_fusion(fuse_ops, prepend=True)
if GroupedMLP_CuTeGEMMUnary.is_supported():
    register_forward_backward_fusion(fuse_srelu_ops, prepend=True)
