# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped split-quantization helpers used by :mod:`GroupedLinear`."""

from typing import List, Optional, Sequence, Tuple, Union, cast

import torch

import transformer_engine_torch as tex

from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from ..tensor import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    HybridQuantizer,
    IdentityQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
from ..tensor.storage.hybrid_tensor_storage import HybridQuantizedTensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..utils import cast_if_needed
from ...debug.pytorch.debug_quantization import DebugQuantizer

_NATIVE_SPLIT_QUANTIZER_TYPES = frozenset(
    {
        Float8Quantizer,
        Float8CurrentScalingQuantizer,
        Float8BlockQuantizer,
        MXFP8Quantizer,
        NVFP4Quantizer,
    }
)

_NATIVE_BGRAD_QUANTIZER_TYPES = frozenset(
    {
        Float8Quantizer,
        Float8CurrentScalingQuantizer,
        MXFP8Quantizer,
    }
)

_DYNAMIC_QUANTIZER_FIELDS = frozenset(
    {
        "rowwise_usage",
        "columnwise_usage",
        "internal",
        "optimize_for_gemm",
    }
)


def _supports_native_split_quantize(quantizer: Quantizer) -> bool:
    """Whether ``tex.split_quantize`` has an exact converter for this quantizer."""
    return type(quantizer) in _NATIVE_SPLIT_QUANTIZER_TYPES


def _prefers_native_bgrad_quantize(quantizer: Quantizer) -> bool:
    """Whether per-split ``bgrad_quantize`` is preferred over bulk quantization."""
    return type(quantizer) in _NATIVE_BGRAD_QUANTIZER_TYPES


def _uses_identity_quantizer(quantizer: Optional[Quantizer]) -> bool:
    """Whether a quantizer, including a hybrid sub-quantizer, is Identity-backed."""
    if quantizer is None:
        return False
    if isinstance(quantizer, IdentityQuantizer):
        return True
    if isinstance(quantizer, HybridQuantizer):
        return _uses_identity_quantizer(quantizer.rowwise_quantizer) or _uses_identity_quantizer(
            quantizer.columnwise_quantizer
        )
    return False


def _identity_quantizer_signature(quantizer: Optional[Quantizer]) -> Tuple[bool, bool]:
    """Identity usage per GEMM direction: ``(rowwise, columnwise)``."""
    if isinstance(quantizer, HybridQuantizer):
        return (
            _uses_identity_quantizer(quantizer.rowwise_quantizer),
            _uses_identity_quantizer(quantizer.columnwise_quantizer),
        )
    identity = isinstance(quantizer, IdentityQuantizer)
    return (identity, identity)


def _backend_quantizer_signature(quantizer: Optional[Quantizer]):
    """Return backend configuration that grouped kernels require to be uniform."""
    if quantizer is None:
        return None

    # Identity is not registered as a torch.compile value quantizer, but its
    # dtype changes the grouped GEMM input type and therefore must be uniform.
    if isinstance(quantizer, IdentityQuantizer):
        return (type(quantizer), (("dtype", quantizer.dtype),))

    fields = quantizer._value_fields()
    if fields is None:
        # Delayed-scaling Float8Quantizer carries per-expert scale/amax tensors,
        # which are intentionally different, but its emitted FP8 dtype is a
        # group-wide backend choice. Other unregistered/custom quantizers retain
        # the conservative exact-family behavior until they expose value fields.
        fields = ("dtype",) if isinstance(quantizer, Float8Quantizer) else ()

    config = []
    for name in fields:
        if name in _DYNAMIC_QUANTIZER_FIELDS:
            continue
        value = getattr(quantizer, name)
        if name == "dtype":
            value = int(value)
        config.append((name, value))
    return (type(quantizer), tuple(config))


def _validate_backend_match(
    reference: Quantizer,
    quantizer: Quantizer,
    operand_name: str,
    direction: str,
    expert_index: int,
) -> None:
    """Validate one expert against the group's reference backend."""
    if type(quantizer) is not type(reference):
        raise ValueError(
            f"GroupedLinear {operand_name} quantizers use incompatible {direction} backend"
            f" families across experts: expert 0 uses {type(reference).__name__}, but expert"
            f" {expert_index} uses {type(quantizer).__name__}. Grouped operands require one"
            " quantizer family per direction."
        )
    reference_signature = _backend_quantizer_signature(reference)
    quantizer_signature = _backend_quantizer_signature(quantizer)
    if quantizer_signature != reference_signature:
        raise ValueError(
            f"GroupedLinear {operand_name} quantizers use incompatible {direction} backend"
            f" configurations across experts: expert 0 uses {reference_signature}, but expert"
            f" {expert_index} uses {quantizer_signature}. Grouped operands require the same"
            " backend-relevant configuration per direction."
        )


def validate_grouped_quantizer_list(
    quantizers: Sequence[Optional[Quantizer]],
    *,
    operand_name: str = "operand",
) -> None:
    """Validate that one grouped operand has compatible expert quantizers."""
    if not quantizers:
        return

    reference = quantizers[0]
    reference_is_hybrid = isinstance(reference, HybridQuantizer)
    reference_identity = _identity_quantizer_signature(reference)

    for expert_index, quantizer in enumerate(quantizers[1:], start=1):
        if (quantizer is None) != (reference is None):
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix None and concrete quantizers"
                f" across experts: expert 0 is {type(reference).__name__}, but expert"
                f" {expert_index} is {type(quantizer).__name__}."
            )
        if reference is None:
            continue

        quantizer_is_hybrid = isinstance(quantizer, HybridQuantizer)
        if quantizer_is_hybrid != reference_is_hybrid:
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix HybridQuantizer and non-hybrid"
                f" quantizers across experts: expert 0 is {type(reference).__name__}, but expert"
                f" {expert_index} is {type(quantizer).__name__}."
            )

        identity = _identity_quantizer_signature(quantizer)
        if identity != reference_identity:
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix Identity-backed and quantized"
                f" directions across experts: expert 0 uses {reference_identity}, but expert"
                f" {expert_index} uses {identity}."
            )

        if reference_is_hybrid:
            _validate_backend_match(
                reference.rowwise_quantizer,
                quantizer.rowwise_quantizer,
                operand_name,
                "rowwise",
                expert_index,
            )
            _validate_backend_match(
                reference.columnwise_quantizer,
                quantizer.columnwise_quantizer,
                operand_name,
                "columnwise",
                expert_index,
            )
            if quantizer.columnwise_source != reference.columnwise_source:
                raise ValueError(
                    f"GroupedLinear {operand_name} HybridQuantizer list has mixed columnwise"
                    " source policies across experts: expert 0 uses"
                    f" {reference.columnwise_source!r}, but expert {expert_index} uses"
                    f" {quantizer.columnwise_source!r}."
                )
        else:
            _validate_backend_match(
                reference,
                quantizer,
                operand_name,
                "plain",
                expert_index,
            )


def _can_batch_nvfp4_rowwise(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    quantizers: Sequence[Quantizer],
) -> bool:
    """Whether independent row scales allow quantizing the concatenation once."""
    if not quantizers:
        return False
    if type(quantizers[0]) is not NVFP4Quantizer:  # pylint: disable=unidiomatic-typecheck
        return False
    reference = quantizers[0]
    if (  # pylint: disable=too-many-boolean-expressions
        tensor.ndim != 2
        or tensor.device.type != "cuda"
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or tensor.shape[1] == 0
        or tensor.shape[1] % 64 != 0
        or tensor.shape[0] == 0
        or len(split_sizes) < 2
        or len(split_sizes) != len(quantizers)
        or sum(split_sizes) != tensor.shape[0]
        or any(rows < 0 or rows % 32 != 0 for rows in split_sizes)
    ):
        return False
    if (  # pylint: disable=too-many-boolean-expressions
        not reference.row_scaled_nvfp4
        or reference.with_rht
        or reference.with_post_rht_amax
        or reference.with_2d_quantization
        or reference.stochastic_rounding
        or reference.with_amax_reduction
        or reference.nvfp4_use_4over6
        or reference.disable_second_level_scale
    ):
        return False
    signature = _backend_quantizer_signature(reference)
    return all(
        type(quantizer) is NVFP4Quantizer  # pylint: disable=unidiomatic-typecheck
        and quantizer.internal
        and quantizer.rowwise_usage
        and not quantizer.columnwise_usage
        and _backend_quantizer_signature(quantizer) == signature
        for quantizer in quantizers
    )


def _batch_nvfp4_rowwise(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    quantizers: Sequence[NVFP4Quantizer],
) -> Sequence[NVFP4TensorStorage]:
    """Quantize independent rows once and expose disjoint per-expert storage."""
    # Match split_quantize's unswizzled layout. Swizzling the concatenation
    # before splitting would mix experts that are not 128-row aligned.
    quantizer = quantizers[0].copy()
    quantizer.optimize_for_gemm = False
    whole = tex.quantize(tensor, quantizer)
    # Each expert needs its own 128-row scale padding.
    padded_sizes = [(rows + 127) // 128 * 128 for rows in split_sizes]
    if list(split_sizes) == padded_sizes:
        scales = torch.split(whole._rowwise_scale_inv, split_sizes)
    else:
        buffer = whole._rowwise_scale_inv.new_zeros(
            (sum(padded_sizes), whole._rowwise_scale_inv.shape[1])
        )
        scales = torch.split(buffer, padded_sizes)
        live_scales = torch.split(whole._rowwise_scale_inv[: tensor.shape[0]], split_sizes)
        # Padding must not alias another expert's live scales: a later in-place
        # quantization of one result must leave every other result intact.
        torch._foreach_copy_(  # pylint: disable=protected-access
            [scale[:rows] for scale, rows in zip(scales, split_sizes)], live_scales
        )
    data = torch.split(whole._rowwise_data, split_sizes)
    amaxes = torch.split(whole._amax_rowwise, split_sizes)
    outputs = []
    for payload, scale, amax, quantizer in zip(data, scales, amaxes, quantizers):
        metadata = whole.get_metadata()
        metadata.update(
            rowwise_data=payload,
            rowwise_scale_inv=scale,
            amax_rowwise=amax,
            quantizer=quantizer,
        )
        outputs.append(NVFP4TensorStorage(**metadata))
    return outputs


def _split_quantize_non_hybrid(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    quantizers: Sequence[Quantizer],
    dtype: torch.dtype,
    *,
    disable_bulk_allocation: bool = False,
    allow_identity_views: bool = True,
) -> Sequence[Union[torch.Tensor, QuantizedTensorStorage]]:
    """Split and quantize one homogeneous, non-Hybrid quantizer list."""
    reference = quantizers[0]
    if not disable_bulk_allocation and _can_batch_nvfp4_rowwise(tensor, split_sizes, quantizers):
        return _batch_nvfp4_rowwise(tensor, split_sizes, quantizers)
    if _supports_native_split_quantize(reference):
        return tex.split_quantize(
            tensor,
            split_sizes,
            quantizers,
            disable_bulk_allocation=disable_bulk_allocation,
        )

    tensor = cast_if_needed(tensor, dtype)
    if (
        allow_identity_views
        # Only the base IdentityQuantizer can bypass quantization; subclasses
        # may override its behavior and must go through their normal call path.
        and type(reference) is IdentityQuantizer  # pylint: disable=unidiomatic-typecheck
        and (reference.dtype is None or reference.dtype == dtype)
    ):
        return torch.split(tensor, split_sizes)

    return [
        quantizer(tensor_part)
        for tensor_part, quantizer in zip(torch.split(tensor, split_sizes), quantizers)
    ]


def _split_quantize_hybrid(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    quantizers: Sequence[HybridQuantizer],
    *,
    disable_bulk_allocation: bool = False,
) -> Sequence[HybridQuantizedTensorStorage]:
    """Split and quantize an all-hybrid, generation-validated operand."""
    reference = quantizers[0]
    rowwise_enabled = reference.rowwise_usage
    columnwise_enabled = reference.columnwise_usage
    columnwise_source = reference.columnwise_source
    rowwise_quantizers = [quantizer.rowwise_quantizer for quantizer in quantizers]
    columnwise_quantizers = [quantizer.columnwise_quantizer for quantizer in quantizers]

    needs_rowwise_result = rowwise_enabled or (
        columnwise_enabled and columnwise_source == "rowwise_dequantized"
    )
    row_results = (
        _split_quantize_non_hybrid(
            tensor,
            split_sizes,
            rowwise_quantizers,
            tensor.dtype,
            disable_bulk_allocation=disable_bulk_allocation,
            allow_identity_views=False,
        )
        if needs_rowwise_result
        else [None] * len(quantizers)
    )

    columnwise_src = tensor
    if columnwise_enabled and columnwise_source == "rowwise_dequantized":
        # Assemble the exact grouped row results in split order. NVFP4 padding
        # and scale layout can differ from independently quantizing each split.
        columnwise_src = torch.cat(
            [result.dequantize(dtype=tensor.dtype) for result in row_results],
            dim=0,
        )
    col_results = (
        _split_quantize_non_hybrid(
            columnwise_src,
            split_sizes,
            columnwise_quantizers,
            tensor.dtype,
            disable_bulk_allocation=disable_bulk_allocation,
            allow_identity_views=False,
        )
        if columnwise_enabled
        else [None] * len(quantizers)
    )

    return [
        HybridQuantizedTensorStorage(
            rowwise_storage=row if rowwise_enabled else None,
            columnwise_storage=col,
            quantizer=quantizer,
            fake_dtype=tensor.dtype,
        )
        for row, col, quantizer in zip(row_results, col_results, quantizers)
    ]


def _split_quantize(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    quantizers: Optional[Sequence[Optional[Quantizer]]],
    activation_dtype: torch.dtype,
    *,
    with_quantized_output: bool = True,
    compute_dbias: bool = False,
    disable_bulk_allocation: bool = False,
) -> Tuple[
    Sequence[Union[torch.Tensor, QuantizedTensorStorage]],
    Optional[List[torch.Tensor]],
]:
    """Split a grouped operand, quantizing when quantizers are provided.

    Native, hybrid, Identity, debug, and Python fallback dispatch are internal
    implementation choices. ``dbiases`` is ``None`` when ``compute_dbias`` is
    false and otherwise contains one reduction result per split. Quantizer lists
    must be homogeneous; dispatch intentionally uses expert 0 as the reference.
    """
    if quantizers is not None and len(quantizers) != len(split_sizes):
        raise ValueError(
            "Grouped split quantizer count does not match the number of tensor splits "
            f"({len(quantizers)} != {len(split_sizes)})"
        )

    reference = quantizers[0] if with_quantized_output and quantizers else None
    if reference is None:
        outputs = torch.split(cast_if_needed(tensor, activation_dtype), split_sizes)
        dbiases = (
            [tensor_part.sum(dim=0) for tensor_part in torch.split(tensor, split_sizes)]
            if compute_dbias
            else None
        )
        return outputs, dbiases

    concrete_quantizers = cast(Sequence[Quantizer], quantizers)
    if compute_dbias and _prefers_native_bgrad_quantize(reference):
        outputs = []
        dbiases = []
        for tensor_part, quantizer in zip(torch.split(tensor, split_sizes), concrete_quantizers):
            dbias, output = tex.bgrad_quantize(tensor_part, quantizer)
            dbiases.append(dbias)
            outputs.append(output)
        return outputs, dbiases

    dbiases = (
        [tensor_part.sum(dim=0) for tensor_part in torch.split(tensor, split_sizes)]
        if compute_dbias
        else None
    )
    if isinstance(reference, DebugQuantizer):
        outputs = DebugQuantizer.multi_tensor_quantize(
            tensor,
            concrete_quantizers,
            split_sizes,
            activation_dtype,
        )
    elif isinstance(reference, HybridQuantizer):
        outputs = _split_quantize_hybrid(
            tensor,
            split_sizes,
            cast(Sequence[HybridQuantizer], concrete_quantizers),
            disable_bulk_allocation=disable_bulk_allocation,
        )
    else:
        outputs = _split_quantize_non_hybrid(
            tensor,
            split_sizes,
            concrete_quantizers,
            activation_dtype,
            disable_bulk_allocation=disable_bulk_allocation,
        )
    return outputs, dbiases


__all__ = ["_split_quantize", "validate_grouped_quantizer_list"]
