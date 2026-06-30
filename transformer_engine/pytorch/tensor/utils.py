# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions for using fp8/nvfp4 tensors as weights"""

from typing import Optional, Union, List
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import (
    multi_tensor_scale,
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)

from ..quantized_tensor import QuantizedTensor, Quantizer, QuantizedTensorStorage
from .float8_tensor import Float8Tensor, Float8Quantizer, Float8CurrentScalingQuantizer
from .nvfp4_tensor import NVFP4Tensor, NVFP4Quantizer
from .mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from .float8_blockwise_tensor import Float8BlockwiseQTensor, Float8BlockQuantizer
from .hybrid_tensor import HybridQuantizedTensor, HybridQuantizer
from .identity_tensor import IdentityQuantizer
from .storage.identity_tensor_storage import IdentityTensorStorage
from ..optimizers.multi_tensor_apply import multi_tensor_applier
from ..utils import is_non_tn_fp8_gemm_supported
from ..constants import NVFP4_BLOCK_SCALING_SIZE, DType


def replace_raw_data(tensor: QuantizedTensor, new_raw_data: torch.Tensor):
    r"""Change a quantized tensor's data buffer while preserving values

    This function modifies only the address space of the underlying
    raw data and does not alter any other tensor attributes or values.

    This may be used for custom buffer allocations, e.g. packing
    multiple parameter tensors together into a single contiguous
    buffer for ZeRO-2.

    """
    if isinstance(tensor, Float8Tensor):
        old_raw_data = tensor._data
        if old_raw_data.dtype != new_raw_data.dtype:
            raise ValueError(
                "The data types of raw data don't match: "
                f"old dtype={old_raw_data.dtype}, new dtype={new_raw_data.dtype}"
            )
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data
    elif isinstance(tensor, Float8BlockwiseQTensor):
        old_raw_data = tensor._rowwise_data
        if old_raw_data.dtype != new_raw_data.dtype:
            raise ValueError(
                "The data types of raw data don't match: "
                f"old dtype={old_raw_data.dtype}, new dtype={new_raw_data.dtype}"
            )
        new_raw_data.detach().copy_(old_raw_data)
        tensor._rowwise_data = new_raw_data
        del old_raw_data
    elif isinstance(tensor, NVFP4Tensor):
        old_rowwise = tensor._rowwise_data
        if old_rowwise.dtype != new_raw_data.dtype:
            raise ValueError(
                f"The data types of raw data don't match: {old_rowwise.dtype} vs"
                f" {new_raw_data.dtype}"
            )
        new_raw_data.detach().copy_(old_rowwise)
        tensor._rowwise_data = new_raw_data
        del old_rowwise
    elif isinstance(tensor, IdentityTensorStorage):
        old_raw_data = tensor._hp_data
        if old_raw_data is None:
            raise RuntimeError("IdentityTensorStorage has no data")
        if old_raw_data.dtype != new_raw_data.dtype:
            raise ValueError(
                "The data types of raw data don't match: "
                f"old dtype={old_raw_data.dtype}, new dtype={new_raw_data.dtype}"
            )
        new_raw_data.detach().copy_(old_raw_data)
        tensor._hp_data = new_raw_data
        del old_raw_data
    elif isinstance(tensor, MXFP8Tensor):
        raise NotImplementedError("replace_raw_data for MXFP8Tensor is not supported yet")
    elif isinstance(tensor, HybridQuantizedTensor):
        # The distopt all-gather buffer routes at the rowwise sub-storage only;
        # the columnwise sub-storage is refreshed each iteration via
        # ``HybridQuantizer.update_quantized``. The underlying call delegates
        # to the rowwise sub-storage's own ``replace_raw_data`` (which may
        # raise for sub-storage types that don't implement it).
        if tensor._rowwise_storage is None:
            raise NotImplementedError(
                "replace_raw_data for HybridQuantizedTensor without a rowwise "
                "sub-storage is not supported."
            )
        replace_raw_data(tensor._rowwise_storage, new_raw_data)
    else:
        raise ValueError(f"replace_raw_data for {type(tensor)} is not supported yet")


def _is_float8_transpose_only(tensor: QuantizedTensor) -> bool:
    """Whether a Float8 tensor stores its live payload only in _transpose."""
    return (
        isinstance(tensor, Float8Tensor)
        and tensor._data is None
        and tensor._transpose is not None
        and not tensor._transpose_invalid
    )


def _validate_flat_fragment(
    model_weight: QuantizedTensor, master_weight: torch.Tensor, start_offset
):
    """Validate a flat logical shard and return its exclusive end offset."""
    if start_offset is None:
        raise ValueError("start_offset must not be None when master_weight is provided")
    if start_offset < 0:
        raise ValueError(f"start_offset must be non-negative, got {start_offset}")
    end_offset = start_offset + master_weight.numel()
    if end_offset > model_weight.numel():
        raise ValueError(
            f"end_offset ({end_offset}) exceeds model_weight numel ({model_weight.numel()}), "
            f"start_offset={start_offset}, master_weight numel={master_weight.numel()}"
        )
    return end_offset


def _cast_master_weight_to_rowwise_fp8_bytes(
    master_weight: torch.Tensor,
    model_weight: Float8Tensor,
    quantizer: Float8Quantizer,
) -> torch.Tensor:
    """Cast a flat master shard to row-major FP8 bytes using ``quantizer`` scale state."""
    rowwise_quantizer = Float8Quantizer(
        scale=quantizer.scale,
        amax=quantizer.amax,
        fp8_dtype=quantizer.dtype,
        rowwise=True,
        columnwise=False,
    )
    raw = torch.empty((1, master_weight.numel()), dtype=torch.uint8, device=model_weight.device)
    temp = rowwise_quantizer.create_tensor_from_data(raw, model_weight.dtype)
    rowwise_quantizer.update_quantized(master_weight.reshape(1, -1), temp)
    if temp._data is None:
        raise RuntimeError("Expected rowwise Float8 temporary to populate _data")
    return temp._data.reshape(-1)


def _update_transpose_only_float8_flat_fragment(
    model_weight: QuantizedTensor,
    master_weight: torch.Tensor,
    start_offset,
    quantizer: Float8Quantizer,
) -> bool:
    """Update a logical flat shard in a transpose-only Float8 tensor.

    Hopper / L40 columnwise-only Float8 sub-storages keep their live FP8
    bytes in ``_transpose`` with physical shape ``[K, rows]`` for a logical
    ``[rows, K]`` tensor. A row-major logical shard is not contiguous in that
    storage, so cast the shard once and scatter the resulting FP8 bytes by
    logical row into the transpose buffer.
    """
    if not _is_float8_transpose_only(model_weight):
        return False

    _validate_flat_fragment(model_weight, master_weight, start_offset)
    numel = master_weight.numel()
    if numel == 0:
        return True

    shape = tuple(model_weight.shape)
    if len(shape) == 0:
        raise ValueError("Float8 scalar transpose-only flat update is not supported")
    logical_cols = int(shape[-1])
    if logical_cols <= 0 or model_weight.numel() % logical_cols != 0:
        raise ValueError(f"Invalid Float8 logical shape for transpose-only update: {shape}")
    logical_rows = model_weight.numel() // logical_cols

    transpose = model_weight._transpose
    if transpose.numel() != model_weight.numel():
        raise ValueError(
            "Float8 transpose-only storage has unexpected numel: "
            f"transpose={transpose.numel()}, logical={model_weight.numel()}"
        )
    transpose_2d = transpose.reshape(logical_cols, logical_rows)
    fp8_bytes = _cast_master_weight_to_rowwise_fp8_bytes(master_weight, model_weight, quantizer)

    remaining = numel
    logical_offset = start_offset
    src_offset = 0
    while remaining > 0:
        row = logical_offset // logical_cols
        col = logical_offset % logical_cols
        n = min(remaining, logical_cols - col)
        transpose_2d[col : col + n, row].copy_(fp8_bytes[src_offset : src_offset + n])
        logical_offset += n
        src_offset += n
        remaining -= n

    model_weight._transpose_invalid = False
    return True


def quantize_master_weights(
    model_weights,
    master_weights,
    start_offsets,
    group,
    fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False,
):
    r"""Helper function to cast master weights to quantized (FP8/NVFP4) primary weights.

    This is intended for use with ZeRO/FSDP. Each rank has a shard of
    the master weights (possibly empty) and a full copy of the model
    weights. Supports FP8 (delayed, current, blockwise, MXFP8) and NVFP4 quantization.

    Parameters
    ----------
    model_weights  : list of quantized weights (FP8 or NVFP4).
    master_weights : list of master weights. Typically they are FP32 weights.
    start_offsets  : list of integers, the starting index of the master weight in the model weight.
                     master_weight may be smaller than model_weight because it could be distributed
                     across multiple ranks. These offsets indicate which part of the model_weight
                     should be updated.
    group          : The distributed group to do amax reduction. Typically it's the data parallel
                     group.
    fsdp_shard_model_weights : list of FSDP shard model weights. If None, it means that the model weights are
                             not sharded. Otherwise, it means that the model weights are sharded and we get
                             target model weights data storage using the FSDP shard model weights.
    manual_post_all_gather_processing : bool, default = `False`.
                     If False, post processing will be automatically triggered during next forward.
                     If True, the timing of calling post_all_gather_processing is left to the user.
                     Note that users must call `post_all_gather_processing` if it's set to True,
                     otherwise the weights won't be updated correctly.

    """

    delayed_scaling_params = []
    current_scaling_params = []
    blockwise_scaling_params = []
    mxfp8_scaling_params = []
    nvfp4_params = []
    identity_params = []

    if fsdp_shard_model_weights is None:
        use_fsdp_shard_model_weights = False
        fsdp_shard_model_weights = [None] * len(model_weights)
    else:
        use_fsdp_shard_model_weights = True

    for model_weight, master_weight, start_offset, fsdp_shard_model_weight in zip(
        model_weights, master_weights, start_offsets, fsdp_shard_model_weights
    ):
        # Clear `_high_precision_init_val` of model_weight automatically.
        # - Master weights are initialized from model weights, if we use fp8 primary weights to
        #   initialize master weights, the numerical values of master weights are not consistent
        #   with the numerical values when we initialize them from bf16/fp16 weights.
        # - So we add a `_high_precision_init_val` attribute to each model weight to store the
        #   original bf16/fp16 weight on cpu before casting it to fp8. And users can use
        #   `get_high_precision_init_val` to get this cpu tensor.
        # - This cpu tensor is not needed once the master weight is initialized, so users should
        #   call `clear_high_precision_init_val` to remove it after master weight is initialized.
        # - In case users don't call `clear_high_precision_init_val`, we will clear it automatically
        #   here. It's safe to clear the `_high_precision_init_val` at this time because this
        #   function is supposed to be called after the master weights are initialized and updated.
        if hasattr(model_weight, "clear_high_precision_init_val"):
            model_weight.clear_high_precision_init_val()

        if master_weight is not None:
            # When not using fp8/fp4_primary_weights, the master_weight (fp32) is first cast to
            # bf16/fp16, and then cast to fp8 during forward. Although it's not necessary when
            # fp8/fp4_primary_weights is enabled, we still keep this logic to keep numerical
            # consistency. So here we cast the master_weight to model_weight.dtype.
            master_weight = master_weight.to(model_weight.dtype)

        quantizer = model_weight._get_quantizer()

        if isinstance(quantizer, NVFP4Quantizer):
            nvfp4_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, Float8Quantizer):
            delayed_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, Float8CurrentScalingQuantizer):
            current_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, Float8BlockQuantizer):
            blockwise_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, MXFP8Quantizer):
            mxfp8_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, IdentityQuantizer):
            identity_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, HybridQuantizer):
            _route_hybrid_to_buckets(
                model_weight,
                master_weight,
                start_offset,
                fsdp_shard_model_weight,
                delayed_scaling_params=delayed_scaling_params,
                current_scaling_params=current_scaling_params,
                identity_params=identity_params,
            )
        else:
            raise ValueError(f"quantize_master_weights for {type(quantizer)} is not supported yet")

    extra_args = [group, use_fsdp_shard_model_weights, manual_post_all_gather_processing]
    if len(delayed_scaling_params) > 0:
        _cast_master_weights_to_fp8_delayed_scaling(delayed_scaling_params, *extra_args)
    if len(current_scaling_params) > 0:
        _cast_master_weights_to_fp8_current_scaling(current_scaling_params, *extra_args)
    if len(blockwise_scaling_params) > 0:
        _cast_master_weights_to_fp8_blockwise_scaling(blockwise_scaling_params, *extra_args)
    if len(mxfp8_scaling_params) > 0:
        _cast_master_weights_to_fp8_mxfp8_scaling(mxfp8_scaling_params, *extra_args)
    if len(nvfp4_params) > 0:
        _cast_master_weights_to_nvfp4_2d(nvfp4_params, *extra_args)
    if len(identity_params) > 0:
        _cast_master_weights_to_identity(identity_params, *extra_args)


def cast_master_weights_to_fp8(
    model_weights,
    master_weights,
    start_offsets,
    group,
    fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False,
):
    r"""Helper function to cast master weights to FP8 primary weights.

    .. deprecated::
        Use :func:`quantize_master_weights` instead.

    """
    quantize_master_weights(
        model_weights,
        master_weights,
        start_offsets,
        group,
        fsdp_shard_model_weights,
        manual_post_all_gather_processing,
    )


def _cast_master_weights_to_fp8_delayed_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for delayed scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Collect amaxes to do reduce-max among dp group.
    # Collect scales and scale_invs to update scale_invs of the fp8 weights.
    amaxes, scales, scale_invs = [], [], []

    for model_weight, master_weight, start_offset, shard_model_weight_raw in params:
        if not manual_post_all_gather_processing:
            # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight._reset_caches()

        quantizer = model_weight._get_quantizer()

        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # If master weight is not None, start_offset must be a valid value.
        if start_offset is None:
            raise ValueError("start_offset must not be None when master_weight is provided")
        if start_offset < 0:
            raise ValueError(f"start_offset must be non-negative, got {start_offset}")
        end_offset = start_offset + master_weight.numel()
        if end_offset > model_weight.numel():
            raise ValueError(
                f"end_offset ({end_offset}) exceeds model_weight numel ({model_weight.numel()}), "
                f"start_offset={start_offset}, master_weight numel={master_weight.numel()}"
            )

        # master_weight may be smaller than model_weight because it could be distributed across
        # multiple ranks. So we need to create a dummy weight using the raw data from model_weight.
        if not use_fsdp_shard_model_weights:
            if _update_transpose_only_float8_flat_fragment(
                model_weight, master_weight, start_offset, quantizer
            ):
                continue
            shard_model_weight_raw = model_weight._data.view(-1)[start_offset:end_offset]
        shard_model_weight_fp8 = quantizer.create_tensor_from_data(
            shard_model_weight_raw.view(1, -1),
            model_weight.dtype,
        )

        # Cast master weight to fp8.
        quantizer.update_quantized(master_weight.view(1, -1), shard_model_weight_fp8)

    if len(amaxes) > 0:
        dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=amaxes[0].device)

        # Reduce amaxes.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [amaxes, packed_amax_views], 1.0
        )
        torch.distributed.all_reduce(
            packed_amaxes,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
        )
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [packed_amax_views, amaxes], 1.0
        )

        # Update scale_invs.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [scales, packed_scale_views], 1.0
        )
        torch.reciprocal(packed_scales, out=packed_scales)
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [packed_scale_views, scale_invs], 1.0
        )


def _cast_master_weights_to_fp8_current_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for current scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    fp8_dtype = params[0][0]._get_quantizer().dtype
    force_pow_2_scales = params[0][0]._get_quantizer().force_pow_2_scales
    amax_epsilon = params[0][0]._get_quantizer().amax_epsilon

    # Create a dummy overflow buffer, it's needed by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    amaxes = [packed_amaxes[i : i + 1] for i in range(len(params))]

    # Collect scales and scale_invs to update them after amax reduction.
    scales, scale_invs = [], []

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If the master weight is None, the corresponding amax
    #         will be set to 0.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, _, _), amax in zip(params, amaxes):

        # Make sure all the model weights have the same numerical options.
        quantizer = model_weight._get_quantizer()
        if quantizer.dtype != fp8_dtype:
            raise ValueError(
                "All model weights must have the same fp8 dtype, "
                f"expected {fp8_dtype} but got {quantizer.dtype}"
            )
        if quantizer.force_pow_2_scales != force_pow_2_scales:
            raise ValueError(
                "All model weights must have the same force_pow_2_scales, "
                f"expected {force_pow_2_scales} but got {quantizer.force_pow_2_scales}"
            )
        if quantizer.amax_epsilon != amax_epsilon:
            raise ValueError(
                "All model weights must have the same amax_epsilon, "
                f"expected {amax_epsilon} but got {quantizer.amax_epsilon}"
            )

        scales.append(torch.empty(1, dtype=torch.float32, device=device))
        scale_invs.append(model_weight._scale_inv.view(1))

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            tex.compute_amax(master_weight, amax)

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    if fp8_dtype == DType.kFloat8E4M3:
        max_fp8 = 448.0
    elif fp8_dtype == DType.kFloat8E5M2:
        max_fp8 = 57344.0
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        max_fp8,
        force_pow_2_scales,
        amax_epsilon,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, start_offset, model_weight_fragment), scale in zip(
        params, scales
    ):
        if not manual_post_all_gather_processing:
            # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight._reset_caches()

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        quantizer = Float8Quantizer(
            scale=scale,
            amax=torch.Tensor(),
            fp8_dtype=model_weight._fp8_dtype,
        )
        if not use_fsdp_shard_model_weights:
            if _update_transpose_only_float8_flat_fragment(
                model_weight, master_weight, start_offset, quantizer
            ):
                continue
            model_weight_fragment = model_weight.reshape(-1)[start_offset:end_offset]
        if use_fsdp_shard_model_weights and not isinstance(model_weight_fragment, Float8Tensor):
            # NOTE: The fsdp shard model weight may be a unit8 tensor instead of
            # a float8 tensor. We should handle this situation properly.
            model_weight_fragment = quantizer.create_tensor_from_data(
                model_weight_fragment.view(-1),
                model_weight.dtype,
            )
        quantizer.update_quantized(master_weight, model_weight_fragment)


def _cast_master_weights_to_fp8_blockwise_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for blockwise scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    block_len = params[0][0]._get_quantizer().block_len
    fp8_dtype = params[0][0]._get_quantizer().dtype
    force_pow_2_scales = params[0][0]._get_quantizer().force_pow_2_scales
    amax_epsilon = params[0][0]._get_quantizer().amax_epsilon

    # Create a dummy overflow buffer, it's needed by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Get the total number of amax elements in all the model weights.
    cu_amax_sizes = [0]
    for model_weight, _, _, _ in params:
        scale_shape = model_weight._get_quantizer().get_scale_shape(model_weight.shape, False)
        num_amaxes = scale_shape[0] * scale_shape[1]
        cu_amax_sizes.append(cu_amax_sizes[-1] + num_amaxes)

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes, scales, scale_invs = [], [], []
    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):

        # Make sure all the model weights have the same numerical options.
        quantizer = model_weight._get_quantizer()
        if block_len != quantizer.block_len:
            raise ValueError(
                "All model weights must have the same block_len, "
                f"expected {block_len} but got {quantizer.block_len}"
            )
        if fp8_dtype != quantizer.dtype:
            raise ValueError(
                "All model weights must have the same fp8 dtype, "
                f"expected {fp8_dtype} but got {quantizer.dtype}"
            )
        if force_pow_2_scales != quantizer.force_pow_2_scales:
            raise ValueError(
                "All model weights must have the same force_pow_2_scales, "
                f"expected {force_pow_2_scales} but got {quantizer.force_pow_2_scales}"
            )
        if amax_epsilon != quantizer.amax_epsilon:
            raise ValueError(
                "All model weights must have the same amax_epsilon, "
                f"expected {amax_epsilon} but got {quantizer.amax_epsilon}"
            )

        scale_shape = quantizer.get_scale_shape(model_weight.shape, False)
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
        scale_inv = model_weight._rowwise_scale_inv
        if len(scale_shape) != 2:
            raise ValueError(f"scale_shape must be 2D, got {len(scale_shape)}D shape {scale_shape}")
        if len(scale_inv.shape) != 2:
            raise ValueError(
                f"scale_inv must be 2D, got {len(scale_inv.shape)}D shape {scale_inv.shape}"
            )
        if scale_inv.shape[0] != scale_shape[0]:
            raise ValueError(
                f"scale_inv dim 0 mismatch: scale_inv.shape={scale_inv.shape},"
                f" scale_shape={scale_shape}"
            )
        if scale_inv.shape[1] != scale_shape[1]:
            raise ValueError(
                f"scale_inv dim 1 mismatch: scale_inv.shape={scale_inv.shape},"
                f" scale_shape={scale_shape}"
            )

        amaxes.append(amax)
        scales.append(scale)
        scale_invs.append(scale_inv)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            if len(model_weight.shape) != 2:
                raise ValueError(
                    "model_weight must be 2D for blockwise scaling, "
                    f"got {len(model_weight.shape)}D shape {model_weight.shape}"
                )
            h, w = model_weight.shape
            tex.fp8_block_scaling_compute_partial_amax(
                master_weight, amax, h, w, start_offset, block_len
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    if fp8_dtype == DType.kFloat8E4M3:
        max_fp8 = 448.0
    elif fp8_dtype == DType.kFloat8E5M2:
        max_fp8 = 57344.0
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        max_fp8,
        force_pow_2_scales,
        amax_epsilon,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, start_offset, model_weight_fragment), scale in zip(
        params, scales
    ):
        if not manual_post_all_gather_processing:
            # Clear columnwise data for all model weights.
            # We cannot create columnwise data here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated at this moment.
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        if not use_fsdp_shard_model_weights:
            model_weight_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
        if len(model_weight.shape) != 2:
            raise ValueError(
                "model_weight must be 2D for blockwise scaling partial cast, "
                f"got {len(model_weight.shape)}D shape {model_weight.shape}"
            )
        h, w = model_weight.shape
        tex.fp8_block_scaling_partial_cast(
            master_weight, model_weight_fragment, scale, h, w, start_offset, block_len, fp8_dtype
        )


def _cast_master_weights_to_nvfp4_2d(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to NVFP4 2D quantized weights.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    device = params[0][0].device
    block_len = NVFP4_BLOCK_SCALING_SIZE

    cu_amax_sizes = [0]
    tile_shapes: List[tuple[int, int]] = []
    tile_widths: List[int] = []
    scale_targets: List[torch.Tensor] = []
    amax_targets: List[Optional[torch.Tensor]] = []
    for model_weight, _, _, _ in params:
        quantizer = model_weight._get_quantizer()
        if not isinstance(quantizer, NVFP4Quantizer):
            raise TypeError(f"Expected NVFP4Quantizer, got {type(quantizer).__name__}")
        if not quantizer.with_2d_quantization:
            raise ValueError("NVFP4 2D quantization must be enabled.")
        if len(model_weight.shape) != 2:
            raise ValueError(f"Expected 2D model weight, got {len(model_weight.shape)}D")
        h, w = model_weight.shape
        tile_h = (h + block_len - 1) // block_len
        tile_w = (w + block_len - 1) // block_len
        tile_shapes.append((tile_h, tile_w))
        tile_widths.append(tile_w)
        scale_targets.append(model_weight._rowwise_scale_inv)
        amax_targets.append(model_weight._amax_rowwise)
        num_amaxes = tile_h * tile_w
        cu_amax_sizes.append(cu_amax_sizes[-1] + num_amaxes)

    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)
    packed_scales = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    amaxes: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    global_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    global_amax_views: List[torch.Tensor] = [global_amaxes[i : i + 1] for i in range(len(params))]

    # Collect tensors for batched multi-tensor amax computation
    master_weight_list: List[torch.Tensor] = []
    partial_amax_list: List[torch.Tensor] = []
    global_amax_list: List[torch.Tensor] = []
    h_list: List[int] = []
    w_list: List[int] = []
    start_offset_list: List[int] = []

    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        scale_shape = tile_shapes[i]
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = packed_scales[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        global_amax_view = global_amax_views[i]

        if model_weight._rowwise_scale_inv is None:
            raise RuntimeError("model_weight._rowwise_scale_inv must not be None")

        amaxes.append(amax)
        scales.append(scale)

        if master_weight is not None and master_weight.numel() > 0:
            if len(model_weight.shape) != 2:
                raise ValueError(f"Expected 2D model weight, got {len(model_weight.shape)}D")
            h, w = model_weight.shape
            # Collect for batched processing
            master_weight_list.append(master_weight)
            partial_amax_list.append(amax)
            global_amax_list.append(global_amax_view)
            h_list.append(h)
            w_list.append(w)
            start_offset_list.append(start_offset)

    # Batched multi-tensor call for partial and global amax computation
    if master_weight_list:
        tex.nvfp4_multi_tensor_compute_partial_amax(
            master_weight_list,
            partial_amax_list,
            global_amax_list,
            h_list,
            w_list,
            start_offset_list,
            block_len,
        )

    if packed_amaxes.numel() > 0:
        torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    if global_amaxes.numel() > 0:
        torch.distributed.all_reduce(global_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # Use GPU kernel to compute global encode scales from global amaxes
    # This replaces multiple Python tensor operations with a single kernel
    global_scale_tensor = torch.empty_like(global_amaxes)

    tex.nvfp4_compute_global_scale(global_amaxes, global_scale_tensor)
    global_scale_views = [global_scale_tensor[i : i + 1] for i in range(len(params))]

    # Collect tensors for batched fused scale kernel
    fused_scale_block_amax_list: List[torch.Tensor] = []
    fused_scale_global_amax_list: List[torch.Tensor] = []
    fused_scale_per_block_scale_list: List[torch.Tensor] = []
    fused_scale_target_scale_list: List[torch.Tensor] = []
    fused_scale_target_amax_list: List[torch.Tensor] = []
    fused_scale_tile_rows_list: List[int] = []
    fused_scale_tile_cols_list: List[int] = []
    fused_scale_rows_padded_list: List[int] = []

    # Collect tensors for batched partial cast kernel
    partial_cast_inp_list: List[torch.Tensor] = []
    partial_cast_out_list: List[torch.Tensor] = []
    partial_cast_scale_list: List[torch.Tensor] = []
    partial_cast_global_scale_list: List[torch.Tensor] = []
    partial_cast_h_list: List[int] = []
    partial_cast_w_list: List[int] = []
    partial_cast_start_offset_list: List[int] = []

    # First pass: collect all tensors and update usage
    zipped_meta = zip(
        tile_shapes,
        tile_widths,
        scale_targets,
        amax_targets,
        params,
        amaxes,
        scales,
        global_scale_views,
    )
    for idx, (
        tile_shape,
        tile_col_cnt,
        target_scale,
        target_amax,
        (model_weight, master_weight, start_offset, model_weight_fragment),
        block_amax,
        per_block_decode_scale,
        global_scale,
    ) in enumerate(zipped_meta):

        if not manual_post_all_gather_processing:
            # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)

        tile_rows = tile_shape[0]
        rows_padded = target_scale.shape[0]
        global_amax_view = global_amaxes[idx : idx + 1]

        # Collect for fused scale kernel (only if target_amax is not None)
        if target_amax is not None:
            fused_scale_block_amax_list.append(block_amax)
            fused_scale_global_amax_list.append(global_amax_view)
            fused_scale_per_block_scale_list.append(per_block_decode_scale)
            fused_scale_target_scale_list.append(target_scale)
            fused_scale_target_amax_list.append(target_amax)
            fused_scale_tile_rows_list.append(tile_rows)
            fused_scale_tile_cols_list.append(tile_col_cnt)
            fused_scale_rows_padded_list.append(rows_padded)

        # Collect for partial cast kernel (only for layers owned by this rank)
        if master_weight is not None and master_weight.numel() > 0:
            end_offset = start_offset + master_weight.numel()
            if not use_fsdp_shard_model_weights:
                rowwise_bytes = model_weight._rowwise_data.view(-1)
                byte_start = start_offset // 2
                byte_end = (end_offset + 1) // 2
                model_weight_fragment = rowwise_bytes[byte_start:byte_end]
            if len(model_weight.shape) != 2:
                raise ValueError(f"Expected 2D model weight, got {len(model_weight.shape)}D")
            h, w = model_weight.shape

            partial_cast_inp_list.append(master_weight)
            partial_cast_out_list.append(model_weight_fragment)
            partial_cast_scale_list.append(per_block_decode_scale)
            partial_cast_global_scale_list.append(global_scale)
            partial_cast_h_list.append(h)
            partial_cast_w_list.append(w)
            partial_cast_start_offset_list.append(start_offset)

    # Batched multi-tensor call for fused scale
    if fused_scale_block_amax_list:
        tex.nvfp4_multi_tensor_fused_scale(
            fused_scale_block_amax_list,
            fused_scale_global_amax_list,
            fused_scale_per_block_scale_list,
            fused_scale_target_scale_list,
            fused_scale_target_amax_list,
            fused_scale_tile_rows_list,
            fused_scale_tile_cols_list,
            fused_scale_rows_padded_list,
            block_len,
        )

    # Batched multi-tensor call for partial cast
    if partial_cast_inp_list:
        tex.nvfp4_multi_tensor_2d_partial_cast(
            partial_cast_inp_list,
            partial_cast_out_list,
            partial_cast_scale_list,
            partial_cast_global_scale_list,
            partial_cast_h_list,
            partial_cast_w_list,
            partial_cast_start_offset_list,
            block_len,
        )


def _identity_storage_data(tensor):
    if not isinstance(tensor, IdentityTensorStorage):
        raise TypeError(f"Expected IdentityTensorStorage, got {type(tensor).__name__}")
    if tensor._hp_data is None:
        raise RuntimeError("IdentityTensorStorage has no data")
    return tensor._hp_data


def _cast_master_weights_to_identity(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    del group, manual_post_all_gather_processing

    for model_weight, master_weight, start_offset, model_weight_fragment in params:
        if master_weight is None:
            continue
        if start_offset is None:
            raise ValueError("start_offset must not be None when master_weight is provided")
        if start_offset < 0:
            raise ValueError(f"start_offset must be non-negative, got {start_offset}")
        end_offset = start_offset + master_weight.numel()
        if end_offset > model_weight.numel():
            raise ValueError(
                f"end_offset ({end_offset}) exceeds model_weight numel ({model_weight.numel()}), "
                f"start_offset={start_offset}, master_weight numel={master_weight.numel()}"
            )

        if use_fsdp_shard_model_weights:
            target = model_weight_fragment
            if target is None:
                raise RuntimeError("FSDP shard model weight is required for Identity writeback")
            if isinstance(target, IdentityTensorStorage):
                target_flat = _identity_storage_data(target).reshape(-1)
            else:
                target_flat = target.reshape(-1)
            target_slice = target_flat[: master_weight.numel()]
        else:
            target_slice = _identity_storage_data(model_weight).reshape(-1)[start_offset:end_offset]

        if target_slice.numel() != master_weight.numel():
            raise ValueError(
                f"Identity target slice has {target_slice.numel()} elements, "
                f"but master_weight has {master_weight.numel()}"
            )
        target_slice.copy_(master_weight.reshape(-1))


def _cast_master_weights_to_fp8_mxfp8_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):  # pylint: disable=unused-argument
    r"""Helper function to cast master weights to FP8 primary weights for mxfp8 scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    for _, master_weight, _, _ in params:
        if master_weight is not None:
            master_weight_dtype = master_weight.dtype
            break

    # Get the total number of amax elements in all the model weights.
    cu_rowwise_amax_sizes = [0]
    cu_colwise_amax_sizes = [0]
    for model_weight, _, _, _ in params:
        rowwise_shape = model_weight._rowwise_scale_inv.shape
        if len(rowwise_shape) != 2:
            raise ValueError(
                f"rowwise_scale_inv must be 2D, got {len(rowwise_shape)}D shape {rowwise_shape}"
            )
        colwise_shape = model_weight._columnwise_scale_inv.shape
        if len(colwise_shape) != 2:
            raise ValueError(
                f"columnwise_scale_inv must be 2D, got {len(colwise_shape)}D shape {colwise_shape}"
            )
        cu_rowwise_amax_sizes.append(
            cu_rowwise_amax_sizes[-1] + rowwise_shape[0] * rowwise_shape[1]
        )
        cu_colwise_amax_sizes.append(
            cu_colwise_amax_sizes[-1] + colwise_shape[0] * colwise_shape[1]
        )

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(
        cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[-1],
        dtype=master_weight_dtype,
        device=device,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes_rowwise, scale_invs_rowwise = [], []
    amaxes_colwise, scale_invs_colwise = [], []
    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        rowwise_shape = model_weight._rowwise_scale_inv.shape
        colwise_shape = model_weight._columnwise_scale_inv.shape
        rowwise_start = cu_rowwise_amax_sizes[i]
        rowwise_end = cu_rowwise_amax_sizes[i + 1]
        colwise_start = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i]
        colwise_end = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i + 1]
        amax_rowwise = packed_amaxes[rowwise_start:rowwise_end].reshape(rowwise_shape)
        amax_colwise = packed_amaxes[colwise_start:colwise_end].reshape(colwise_shape)
        amaxes_rowwise.append(amax_rowwise)
        amaxes_colwise.append(amax_colwise)
        scale_invs_rowwise.append(model_weight._rowwise_scale_inv)
        scale_invs_colwise.append(model_weight._columnwise_scale_inv)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            if len(model_weight.shape) != 2:
                raise ValueError(
                    "model_weight must be 2D for MXFP8 scaling, "
                    f"got {len(model_weight.shape)}D shape {model_weight.shape}"
                )
            h, w = model_weight.shape
            tex.mxfp8_scaling_compute_partial_amax(
                master_weight, amax_rowwise, amax_colwise, h, w, start_offset
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    multi_tensor_applier(
        multi_tensor_compute_scale_inv_e8m0,
        None,  # dummy_overflow_buf
        [
            amaxes_rowwise + amaxes_colwise,
            scale_invs_rowwise + scale_invs_colwise,
        ],
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (
        (model_weight, master_weight, start_offset, model_weight_fragment),
        scale_inv_rowwise,
        scale_inv_colwise,
    ) in zip(params, scale_invs_rowwise, scale_invs_colwise):
        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        if use_fsdp_shard_model_weights:
            rowwise_fragment = model_weight_fragment[0]
            colwise_fragment = model_weight_fragment[1]
        else:
            rowwise_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
            colwise_fragment = model_weight._columnwise_data.reshape(-1)[start_offset:end_offset]
        if len(model_weight.shape) != 2:
            raise ValueError(
                "model_weight must be 2D for MXFP8 scaling partial cast, "
                f"got {len(model_weight.shape)}D shape {model_weight.shape}"
            )
        h, w = model_weight.shape
        tex.mxfp8_scaling_partial_cast(
            master_weight,
            rowwise_fragment,
            colwise_fragment,
            scale_inv_rowwise,
            scale_inv_colwise,
            h,
            w,
            start_offset,
        )


# ---------------------------------------------------------------------------------------------
# HybridQuantizer helpers for `quantize_master_weights` / `post_all_gather_processing`.
#
# Dispatch is per-direction: `_route_hybrid_to_buckets` iterates over both sub-storages
# of a `HybridQuantizedTensor` and routes each one independently into the per-format
# bucket matching its own sub-quantizer type. Row and col make their own decisions and
# can mix any pair of currently-supported sub-quantizers.
#
#   Supported (per-tensor Float8 or Identity sub-quantizers, any direction):
#     - Float8Quantizer                  (delayed scaling)
#     - Float8CurrentScalingQuantizer    (current scaling)
#     - IdentityQuantizer                (high-precision passthrough)
#
#   Per-tensor Float8 works because `_cast_master_weights_to_fp8_{delayed,current}_scaling`
#   accept any Float8Tensor (single direction is fine — each entry is one Float8Tensor
#   with its own `_scale_inv` and the helper writes that one entry's `_data`). Each
#   hybrid sub-storage IS a single-direction Float8Tensor, so we route them as two
#   independent entries (into the same bucket for same-format, or into different
#   buckets for cross-format Float8 — e.g. delayed row + current col).
#
#   Identity routes to an exact copy bucket. Single-direction hybrid (only one
#   sub-storage populated) routes the present direction only. Both-None hybrids
#   raise ValueError. Per-block sub-quantizers still hit their per-direction TODO.
#
#   Not supported (raise NotImplementedError per-direction + TODO):
#
#     - MXFP8Quantizer as a hybrid sub-quantizer (any direction)
#         TODO(#3158, hybrid-mxfp8-distopt): the distopt cast kernels
#         (`tex.mxfp8_scaling_compute_partial_amax`, `tex.mxfp8_scaling_partial_cast`)
#         are bidirectional — both rowwise and colwise outputs required — so they
#         cannot ingest a single-direction hybrid sub-storage. (Unrelated to the
#         regular `tex.quantize` kernel used by forward/backward, which natively
#         supports single-direction output.) Unblocker: add single-direction
#         variants of the two distopt kernels, then route hybrid sub-storages
#         per-direction into `mxfp8_scaling_params` matching the Float8 path above.
#         Also unlocks cross-format MXFP8 row + <other format> col.
#
#     - NVFP4Quantizer as a hybrid sub-quantizer (any direction)
#         TODO(#3158, hybrid-nvfp4-distopt): load-bearing blocker is the kernel assertion
#         `return_identity || !use_2d_quantization` in
#         `quantize_transpose_vector_blockwise_fp4.cu`, which rejects exactly the
#         columnwise-only 2D configuration that `HybridQuantizer.__init__` produces
#         for the col sub-quantizer. Blocks hybrid 2D NVFP4 weight construction at
#         `quantized_model_init` time. 1D NVFP4 is unaffected. The assertion is an
#         explicitly-marked unwritten code path, not an algorithmic limit (see the
#         kernel author's note above the early-return guard).
#
#         Secondary blocker (gated on the kernel fix): the distopt helper
#         `_cast_master_weights_to_nvfp4_2d` writes only `_rowwise_data` and relies
#         on per-tensor post-AG `_create_columnwise()` — for hybrid, the columnwise
#         data needs to land in a SEPARATE col sub-storage, so the post-AG branch
#         must be made hybrid-aware (derive `col_sub._columnwise_data` from
#         `row_sub`'s gathered rowwise).
#
#     - Float8BlockQuantizer as a hybrid sub-quantizer
#         TODO(#3158, hybrid-fp8-blockwise): same shape as the NVFP4 secondary blocker —
#         `_cast_master_weights_to_fp8_blockwise_scaling` writes only `_rowwise_data`
#         with per-tensor post-AG `_create_columnwise()` that doesn't reach hybrid's
#         separate col sub-storage. Unlike NVFP4, there is no kernel-level
#         construction blocker (the Block FP8 kernel natively supports
#         columnwise-only mode), so hybrid Block FP8 weights construct fine via the
#         non-distopt FusedAdam path today; only the sharded-master distopt cast
#         path is blocked. Unblocker is a Python-side hybrid-aware post-AG branch;
#         no C++ work needed.
#
# ---------------------------------------------------------------------------------------------


def _route_hybrid_to_buckets(
    model_weight,
    master_weight,
    start_offset,
    fsdp_shard_model_weight,
    *,
    delayed_scaling_params,
    current_scaling_params,
    identity_params,
):
    """Decompose a `HybridQuantizedTensor` into per-direction entries and route each
    into the appropriate per-format bucket used by `quantize_master_weights`.

    Per-direction dispatch: each sub-storage routes independently based on its
    own sub-quantizer type. Per-tensor Float8 sub-quantizers (delayed and/or
    current scaling) are supported in any combination per direction; single-
    direction hybrid (one sub-storage dropped via ``update_usage``) is also
    supported. See the TODO block above this helper for the per-block-format
    rejection rationale and unblocker shapes.
    """
    row_sub = model_weight._rowwise_storage
    col_sub = model_weight._columnwise_storage
    sub_q_row = model_weight._quantizer.rowwise_quantizer
    sub_q_col = model_weight._quantizer.columnwise_quantizer

    if row_sub is None and col_sub is None:
        raise ValueError(
            "quantize_master_weights called on HybridQuantizedTensor with both "
            "rowwise and columnwise sub-storages dropped (via update_usage). "
            "Nothing to cast — this is most likely a caller bug."
        )

    # Per-direction routing: each (sub_storage, sub_quantizer) pair selects its
    # own bucket based on the sub-quantizer's type. Directions that have been
    # dropped via ``update_usage`` are silently skipped.
    for direction, sub_storage, sub_q in (
        ("rowwise", row_sub, sub_q_row),
        ("columnwise", col_sub, sub_q_col),
    ):
        if sub_storage is None:
            continue
        shard_fragment = fsdp_shard_model_weight
        if shard_fragment is not None and isinstance(shard_fragment, HybridQuantizedTensor):
            shard_fragment = (
                shard_fragment._rowwise_storage
                if direction == "rowwise"
                else shard_fragment._columnwise_storage
            )
        entry = (sub_storage, master_weight, start_offset, shard_fragment)
        if isinstance(sub_q, Float8Quantizer):
            # Delayed scaling: the per-format helper iterates entries
            # independently and does a per-DP amax all-reduce across the bucket.
            delayed_scaling_params.append(entry)
        elif isinstance(sub_q, Float8CurrentScalingQuantizer):
            current_scaling_params.append(entry)
        elif isinstance(sub_q, IdentityQuantizer):
            identity_params.append(entry)
        elif isinstance(sub_q, MXFP8Quantizer):
            # TODO(#3158, hybrid-mxfp8-distopt): the distopt cast kernels are
            # bidirectional, so a single-direction hybrid sub-storage cannot be
            # fed in. See top-of-file TODO block for the unblocker (single-
            # direction variants of the two distopt kernels).
            raise NotImplementedError(
                "quantize_master_weights for HybridQuantizer with MXFP8Quantizer "
                f"{direction} sub-quantizer is not supported yet. See the TODO "
                "block above _route_hybrid_to_buckets for the unblocker shape."
            )
        elif isinstance(sub_q, NVFP4Quantizer):
            # TODO(#3158, hybrid-nvfp4-distopt): load-bearing blocker is the kernel
            # assertion that rejects columnwise-only 2D NVFP4 — which is
            # exactly what hybrid's col sub-quantizer pin produces. Secondary
            # blocker (gated on the kernel fix) is the per-tensor post-AG
            # `_create_columnwise()` not reaching hybrid's separate col
            # sub-storage. See top-of-file TODO block for details.
            raise NotImplementedError(
                "quantize_master_weights for HybridQuantizer with NVFP4Quantizer "
                f"{direction} sub-quantizer is not supported yet. See the TODO "
                "block above _route_hybrid_to_buckets for details."
            )
        elif isinstance(sub_q, Float8BlockQuantizer):
            # Pending hybrid-fp8-blockwise work (#3158): same shape as the NVFP4
            # secondary blocker (and only that one — no kernel-level construction
            # blocker for Block FP8). Python-side post-AG fix. See the
            # _route_hybrid_to_buckets design note above for details.
            raise NotImplementedError(
                "quantize_master_weights for HybridQuantizer with Float8BlockQuantizer "
                f"{direction} sub-quantizer is not supported yet. See the TODO "
                "block above _route_hybrid_to_buckets for details."
            )
        else:
            raise NotImplementedError(
                "quantize_master_weights for HybridQuantizer with "
                f"{type(sub_q).__name__} {direction} sub-quantizer is not supported yet."
            )


def post_all_gather_processing(model_weights: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Post-processing after all-gather for weights in distributed optimizer.
    - Float8Tensor: may need to create a transposed view to match backend GEMM.
    - Float8BlockwiseQTensor: create column-wise storage.
    - Plain pytorch tensor: noop.

    For NVFP4 tensors, uses batched multi-tensor processing to reduce CPU overhead.

    For `HybridQuantizedTensor`, recurses per-direction so each present
    sub-storage runs its native post-processing. Identity sub-storages are no-op.
    """
    if not isinstance(model_weights, list):
        model_weights = [model_weights]

    # Collect NVFP4 tensors for batched processing
    nvfp4_tensors = []

    for model_weight in model_weights:
        if isinstance(model_weight, Float8Tensor):
            # Delayed scaling and per-tensor current scaling: if backend does not support
            # non-transposed FP8 GEMM, pre-create the transpose.
            if not is_non_tn_fp8_gemm_supported():
                model_weight._create_transpose()
        elif isinstance(model_weight, Float8BlockwiseQTensor):
            # Blockwise scaling: create column-wise storage.
            model_weight._create_columnwise()
        elif isinstance(model_weight, NVFP4Tensor):
            # Collect for batched processing
            nvfp4_tensors.append(model_weight)
        elif isinstance(model_weight, MXFP8Tensor):
            # MXFP8 scaling: no need to do anything.
            pass
        elif isinstance(model_weight, IdentityTensorStorage):
            pass
        elif isinstance(model_weight, HybridQuantizedTensor):
            for sub in (model_weight._rowwise_storage, model_weight._columnwise_storage):
                if sub is not None:
                    post_all_gather_processing(sub)
        elif isinstance(model_weight, QuantizedTensor):
            raise ValueError(f"post_processing for {type(model_weight)} is not supported")

    # Batch process all NVFP4 tensors with multi-tensor approach
    if nvfp4_tensors:
        _nvfp4_2d_multi_tensor_transpose(nvfp4_tensors)


def _nvfp4_2d_multi_tensor_transpose(nvfp4_tensors: List[NVFP4Tensor]):
    """
    Batched columnwise creation for multiple NVFP4 tensors.
    Reduces CPU overhead by collecting all tensor metadata and dispatching to C++.
    """
    # Prepare tensor lists for batched C++ call
    rowwise_data_list = []
    columnwise_data_list = []
    rowwise_scale_inv_list = []
    columnwise_scale_inv_list = []
    M_list = []
    K_list = []

    for tensor in nvfp4_tensors:
        rowwise_data = tensor._rowwise_data
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
            tensor._rowwise_data = rowwise_data

        logical_shape = tensor.size()
        M, K = logical_shape[0], logical_shape[-1]

        # Allocate columnwise_data if needed
        if tensor._columnwise_data is None:
            # Output shape: [K, M/2] packed bytes
            columnwise_data = torch.empty(
                (K, M // 2),
                dtype=torch.uint8,
                device=rowwise_data.device,
            )
            tensor._columnwise_data = columnwise_data
        else:
            columnwise_data = tensor._columnwise_data

        # Allocate columnwise_scale_inv if needed
        if tensor._columnwise_scale_inv is None:
            if tensor._quantizer is None:
                raise RuntimeError("tensor._quantizer must not be None")
            columnwise_scale_inv_shape = tensor._quantizer.get_scale_shape(logical_shape, True)
            columnwise_scale_inv = torch.empty(
                columnwise_scale_inv_shape,
                dtype=tensor._rowwise_scale_inv.dtype,
                device=tensor._rowwise_scale_inv.device,
            )
            tensor._columnwise_scale_inv = columnwise_scale_inv
        else:
            columnwise_scale_inv = tensor._columnwise_scale_inv

        rowwise_data_list.append(rowwise_data)
        columnwise_data_list.append(columnwise_data)
        rowwise_scale_inv_list.append(tensor._rowwise_scale_inv)
        columnwise_scale_inv_list.append(columnwise_scale_inv)
        M_list.append(M)
        K_list.append(K)

        # Copy amax if needed
        if tensor._amax_columnwise is None and tensor._amax_rowwise is not None:
            tensor._amax_columnwise = tensor._amax_rowwise.clone()
        elif tensor._amax_rowwise is not None:
            tensor._amax_columnwise.copy_(tensor._amax_rowwise)

    # Dispatch to C++ multi-tensor kernel
    tex.nvfp4_2d_multi_tensor_transpose(
        rowwise_data_list,
        columnwise_data_list,
        rowwise_scale_inv_list,
        columnwise_scale_inv_list,
        M_list,
        K_list,
    )


def clear_columnwise_cache(tensor: QuantizedTensorStorage) -> None:
    """Clear the columnwise cache of a quantized tensor.
    Use-case: FSDP2, where TE allocates allgathered
    columnwise data(by deriving it out of allgathered rowwise data)
    in fsdp2 hooks. And so FSDP2 cant deallocate it when it's done with it"""
    if hasattr(tensor, "_columnwise_data"):
        tensor._columnwise_data = None
    if hasattr(tensor, "_columnwise_scale_inv"):
        tensor._columnwise_scale_inv = None
    if hasattr(tensor, "_transpose"):
        tensor._transpose = None
    if hasattr(tensor, "_transpose_invalid"):
        tensor._transpose_invalid = True


def is_custom(x: Optional[Union[Quantizer, QuantizedTensorStorage]] = None) -> bool:
    """Check if an object is custom.

    Returns False if x is a torch.Tensor.
    """
    if x is None or isinstance(x, torch.Tensor):
        return False
    if not isinstance(x, (Quantizer, QuantizedTensorStorage)):
        raise AssertionError("Object must be a Quantizer or QuantizedTensorStorage instance")
    return hasattr(x, "custom") and x.custom
