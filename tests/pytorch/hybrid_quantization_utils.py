# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared explicit quantizer factories for hybrid quantization tests.

These are intentionally ordinary module-level functions rather than a public
factory abstraction. Keeping them module-level makes ``CustomRecipe`` objects
that reference them picklable in distributed checkpoint tests.
"""

import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes.quantization_factory_base import (
    current_scaling_quantizer_factory,
    float8_block_scaling_quantizer_factory,
    mxfp8_quantizer_factory,
    nvfp4_quantizer_factory,
)

_LINEAR_MODULE_TYPES = ("linear", "grouped_linear")
_FORWARD_TENSOR_TYPES = ("input", "weight", "output")
_GRAD_TENSOR_TYPES = ("grad_output", "grad_input")


def _is_linear_role(role):
    return role is not None and role.module_type in _LINEAR_MODULE_TYPES


def _make_fp8_current(*, fp8_dtype=te.DType.kFloat8E4M3):
    return te.Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device="cuda")


def _make_mxfp8(*, fp8_dtype=te.DType.kFloat8E4M3):
    return te.MXFP8Quantizer(fp8_dtype=fp8_dtype)


def fp8_e4m3_factory():
    """Construct the default E4M3 current-scaling test quantizer."""
    return te.Float8CurrentScalingQuantizer(te.DType.kFloat8E4M3, device="cuda")


def fp8_e5m2_factory():
    """Construct the default E5M2 current-scaling test quantizer."""
    return te.Float8CurrentScalingQuantizer(te.DType.kFloat8E5M2, device="cuda")


def mxfp8_e4m3_factory():
    """Construct the default E4M3 MXFP8 test quantizer."""
    return te.MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)


def make_fp8_quantizer(*, rowwise=True, columnwise=True):
    """Construct the standard current-scaling FP8 test quantizer."""
    return te.Float8CurrentScalingQuantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        device="cuda",
        rowwise=rowwise,
        columnwise=columnwise,
    )


def make_nvfp4_quantizer(*, rowwise=True, columnwise=True):
    """Construct the standard NVFP4 test quantizer."""
    return te.NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def make_hybrid_quantizer_fp8_row_fp4_col():
    """Construct a hybrid quantizer with FP8 rowwise and NVFP4 columnwise."""
    return te.HybridQuantizer(
        rowwise_quantizer=make_fp8_quantizer(),
        columnwise_quantizer=make_nvfp4_quantizer(),
    )


def as_data_tensor_tuple(storage):
    """Return a storage's raw buffers as a tuple without copying them."""
    if storage is None:
        return ()
    tensors = storage.get_data_tensors()
    return tensors if isinstance(tensors, tuple) else (tensors,)


def snapshot_storage_tensor_metadata(storage, *, clone=False):
    """Capture every tensor-valued concrete-storage metadata field."""
    if storage is None:
        return None
    tensor_metadata = {}
    for name, value in storage.get_metadata().items():
        if isinstance(value, torch.Tensor) or value is None:
            snapshot_value = value.detach().clone() if clone and value is not None else value
            if (
                value is not None
                and getattr(storage, "_is_2D_scaled", False)
                and name in ("rowwise_scale_inv", "columnwise_scale_inv")
            ):
                # Float8Block 2D scales pad one tile dimension. Kernels do not
                # initialize or consume that padding, so canonicalize it.
                snapshot_value = value.detach().clone()
                m, n = storage._fsdp_logical_mn()
                block_len = storage._FSDP_BLOCK_LEN
                m_tiles = (m + block_len - 1) // block_len
                n_tiles = (n + block_len - 1) // block_len
                if name == "rowwise_scale_inv":
                    snapshot_value[:, n_tiles:] = 0
                else:
                    snapshot_value[:, m_tiles:] = 0
            tensor_metadata[name] = snapshot_value
    return {
        "storage_type": type(storage).__name__,
        "tensor_metadata": tensor_metadata,
    }


def assert_nested_state_exact(actual, expected, *, path="state"):
    """Recursively compare nested state, using zero tolerance for tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected Tensor, got {type(actual)}"
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, msg=path)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert actual.keys() == expected.keys(), f"{path}: dictionary keys differ"
        for key in expected:
            assert_nested_state_exact(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(
            actual, type(expected)
        ), f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: sequence lengths differ"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_nested_state_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_storage_data_exact(actual, expected, *, context):
    """Assert every tensor-valued data, scale, and amax metadata field exactly."""
    assert_nested_state_exact(
        snapshot_storage_tensor_metadata(actual),
        snapshot_storage_tensor_metadata(expected),
        path=context,
    )


def assert_hybrid_tensor_exact(actual, expected, *, context):
    """Compare both hybrid directions, including metadata and dequantization."""
    for direction in ("rowwise", "columnwise"):
        actual_storage = getattr(actual, f"{direction}_sub_storage")
        expected_storage = getattr(expected, f"{direction}_sub_storage")
        assert_storage_data_exact(
            actual_storage,
            expected_storage,
            context=f"{context} {direction}",
        )
        if expected_storage is None:
            continue
        try:
            expected_dequantized = expected_storage.dequantize()
        except NotImplementedError:
            continue
        torch.testing.assert_close(
            actual_storage.dequantize(),
            expected_dequantized,
            rtol=0.0,
            atol=0.0,
            msg=f"{context} {direction} dequantized value differs",
        )


def make_role_aware_quantizer(factory, role):
    """Construct a quantizer with the standard Block-FP8 GEMM geometry."""
    quantizer = factory()
    if isinstance(quantizer, te.Float8BlockQuantizer):
        is_weight = (
            role is not None
            and role.module_type in _LINEAR_MODULE_TYPES
            and role.tensor_type == "weight"
        )
        quantizer.block_scaling_dim = 2 if is_weight else 1
    return quantizer


def hybrid_custom_recipe(row_factory, col_factory, grad_factory=None):
    """Build a CustomRecipe with hybrid forward and configurable grad quantizers."""
    if grad_factory is None:
        grad_factory = col_factory

    def qfactory(role):
        is_linear = _is_linear_role(role)
        if is_linear and role.tensor_type in _FORWARD_TENSOR_TYPES:
            return te.HybridQuantizer(
                rowwise_quantizer=make_role_aware_quantizer(row_factory, role),
                columnwise_quantizer=make_role_aware_quantizer(col_factory, role),
            )
        if is_linear and role.tensor_type in _GRAD_TENSOR_TYPES:
            return make_role_aware_quantizer(grad_factory, role)
        return make_role_aware_quantizer(row_factory, role)

    return recipe.CustomRecipe(qfactory=qfactory)


def hybrid_fp8_current_qfactory(role):
    """FP8 current scaling in both hybrid directions for forward tensor roles."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=current_scaling_quantizer_factory(role),
            columnwise_quantizer=current_scaling_quantizer_factory(role),
        )
    return current_scaling_quantizer_factory(role)


def hybrid_fp8_current_e5m2_grads_qfactory(role):
    """FP8 current-scaling hybrid with E5M2 for both grad boundary roles."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=_make_fp8_current(),
            columnwise_quantizer=_make_fp8_current(),
        )
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return _make_fp8_current(fp8_dtype=te.DType.kFloat8E5M2)
    return _make_fp8_current()


def hybrid_mxfp8_qfactory(role):
    """MXFP8 in both hybrid directions for forward tensor roles."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=mxfp8_quantizer_factory(role),
        )
    return mxfp8_quantizer_factory(role)


def hybrid_float8_block_qfactory(role):
    """Float8 block scaling in both hybrid directions for forward roles."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=float8_block_scaling_quantizer_factory(role),
            columnwise_quantizer=float8_block_scaling_quantizer_factory(role),
        )
    return float8_block_scaling_quantizer_factory(role)


def hybrid_block_fp8_e4m3_qfactory(role):
    """Hybrid E4M3 Block-FP8 with role-aware 1D/2D block geometry."""
    is_linear = _is_linear_role(role)
    is_weight = is_linear and role.tensor_type == "weight"
    block_scaling_dim = 2 if is_weight else 1

    def make_quantizer():
        return te.Float8BlockQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

    if is_linear and role.tensor_type in _GRAD_TENSOR_TYPES:
        return make_quantizer()
    return te.HybridQuantizer(
        rowwise_quantizer=make_quantizer(),
        columnwise_quantizer=make_quantizer(),
    )


def hybrid_mixed_mxfp8_fp8_qfactory(role):
    """MXFP8 rowwise plus FP8 current-scaling columnwise."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=current_scaling_quantizer_factory(role),
        )
    return current_scaling_quantizer_factory(role)


def hybrid_fp8_current_identity_qfactory(role):
    """FP8 current-scaling forward plus Identity backward."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=current_scaling_quantizer_factory(role),
            columnwise_quantizer=te.IdentityQuantizer(),
        )
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return te.IdentityQuantizer()
    return current_scaling_quantizer_factory(role)


def hybrid_mxfp8_identity_qfactory(role):
    """MXFP8 forward plus Identity backward."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=te.IdentityQuantizer(),
        )
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return te.IdentityQuantizer()
    return mxfp8_quantizer_factory(role)


def identity_qfactory(role):  # pylint: disable=unused-argument
    """High-precision passthrough for every quantizer slot."""
    return te.IdentityQuantizer()


def hybrid_nvfp4_qfactory(role):
    """NVFP4 in both hybrid directions for forward tensor roles."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=nvfp4_quantizer_factory(role),
            columnwise_quantizer=nvfp4_quantizer_factory(role),
        )
    return nvfp4_quantizer_factory(role)


def hybrid_tp_mxfp8_nvfp4_qfactory(role):
    """TP/SP MXFP8 rowwise plus NVFP4 columnwise, including boundary roles."""
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return nvfp4_quantizer_factory(role)
    return te.HybridQuantizer(
        rowwise_quantizer=mxfp8_quantizer_factory(role),
        columnwise_quantizer=nvfp4_quantizer_factory(role),
    )


def hybrid_fp8_mxfp8_qfactory(role):
    """FP8 current-scaling rowwise plus MXFP8 columnwise for CPU-offload tests."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=_make_fp8_current(),
            columnwise_quantizer=_make_mxfp8(),
        )
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return _make_mxfp8(fp8_dtype=te.DType.kFloat8E5M2)
    return _make_fp8_current()


def hybrid_mxfp8_nvfp4_qfactory(role):
    """MXFP8 rowwise plus NVFP4 columnwise for CPU-offload tests."""
    if _is_linear_role(role) and role.tensor_type in _FORWARD_TENSOR_TYPES:
        return te.HybridQuantizer(
            rowwise_quantizer=_make_mxfp8(),
            columnwise_quantizer=nvfp4_quantizer_factory(role),
        )
    if _is_linear_role(role) and role.tensor_type in _GRAD_TENSOR_TYPES:
        return nvfp4_quantizer_factory(role)
    return _make_mxfp8()
