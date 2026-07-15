# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for hybrid quantization (mixed rowwise/columnwise formats)."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes.quantization_factory_base import (
    nvfp4_quantizer_factory,
)
from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
    mxfp8_fwd_nvfp4_bwd_quantizer_factory,
)
from transformer_engine.pytorch import (
    autocast,
    quantized_model_init,
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    TransformerLayer,
    GroupedLinear,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    Float8BlockQuantizer,
    NVFP4Quantizer,
    HybridQuantizer,
    HybridQuantizedTensor,
    IdentityQuantizer,
    HybridQuantizedTensorStorage,
    Float8Tensor,
    Float8TensorStorage,
    NVFP4Tensor,
    NVFP4TensorStorage,
    QuantizedTensor,
)
from transformer_engine.pytorch.cpp_extensions.gemm import (
    _unwrap_hybrid_A,
    _unwrap_hybrid_B,
)

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)

requires_fp8 = pytest.mark.skipif(
    not fp8_available,
    reason=f"FP8: {reason_for_no_fp8}",
)

requires_nvfp4 = pytest.mark.skipif(
    not nvfp4_available,
    reason=f"NVFP4: {reason_for_no_nvfp4}",
)

requires_fp8_and_nvfp4 = pytest.mark.skipif(
    not (fp8_available and nvfp4_available),
    reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
)

requires_mxfp8_and_nvfp4 = pytest.mark.skipif(
    not (mxfp8_available and nvfp4_available),
    reason=f"MXFP8: {reason_for_no_mxfp8}; NVFP4: {reason_for_no_nvfp4}",
)


def _make_fp8_quantizer(*, rowwise=True, columnwise=True):
    return Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _make_nvfp4_quantizer(*, rowwise=True, columnwise=True):
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _make_hybrid_quantizer_fp8_row_fp4_col():
    """FP8 rowwise + NVFP4 columnwise."""
    return HybridQuantizer(
        rowwise_quantizer=_make_fp8_quantizer(),
        columnwise_quantizer=_make_nvfp4_quantizer(),
    )


def _make_hybrid_quantizer_fp4_row_fp8_col():
    """NVFP4 rowwise + FP8 columnwise (reversed direction)."""
    return HybridQuantizer(
        rowwise_quantizer=_make_nvfp4_quantizer(),
        columnwise_quantizer=_make_fp8_quantizer(),
    )


def _as_data_tensor_tuple(storage):
    """Return a storage's raw buffers as a tuple without copying them."""
    if storage is None:
        return ()
    tensors = storage.get_data_tensors()
    return tensors if isinstance(tensors, tuple) else (tensors,)


def _snapshot_storage_tensor_metadata(storage, *, clone=False):
    """Capture every tensor-valued concrete-storage metadata field without mutation."""
    if storage is None:
        return None
    metadata = storage.get_metadata()
    tensor_metadata = {}
    for name, value in metadata.items():
        # Optional tensor fields are represented by ``None``. Including them
        # makes missing rowwise/columnwise data, scales, and amax explicit.
        if isinstance(value, torch.Tensor) or value is None:
            snapshot_value = value.detach().clone() if clone and value is not None else value
            if (
                value is not None
                and getattr(storage, "_is_2D_scaled", False)
                and name in ("rowwise_scale_inv", "columnwise_scale_inv")
            ):
                # Float8Block 2D scales pad one tile dimension to a multiple
                # of four. Kernels never initialize or consume that padding,
                # and FSDP deliberately strips then zero-repads it. Canonicalize
                # only those non-logical elements while still comparing the
                # complete tensor field, shape, presence, and every valid scale.
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


def _assert_storage_data_exact(actual, expected, *, context):
    """Assert every tensor-valued data/scale/amax metadata field exactly."""
    _assert_nested_state_exact(
        _snapshot_storage_tensor_metadata(actual),
        _snapshot_storage_tensor_metadata(expected),
        path=context,
    )


def _clone_nested_state(value):
    """Clone tensors in nested checkpoint state so later steps cannot mutate snapshots."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested_state(item) for item in value)
    return value


def _assert_nested_state_exact(actual, expected, *, path="state"):
    """Recursively compare checkpoint state, using zero tolerance for tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected Tensor, got {type(actual)}"
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, msg=path)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert actual.keys() == expected.keys(), f"{path}: dictionary keys differ"
        for key in expected:
            _assert_nested_state_exact(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(
            actual, type(expected)
        ), f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: sequence lengths differ"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_state_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_hybrid_tensor_exact(actual, expected, *, context):
    """Compare both hybrid directions, including all metadata and native dequantization."""
    for direction in ("rowwise", "columnwise"):
        actual_storage = getattr(actual, f"{direction}_sub_storage")
        expected_storage = getattr(expected, f"{direction}_sub_storage")
        _assert_storage_data_exact(
            actual_storage,
            expected_storage,
            context=f"{context} {direction}",
        )
        if expected_storage is None:
            continue
        try:
            expected_dequantized = expected_storage.dequantize()
        except NotImplementedError:
            # Some direction-only formats (notably NVFP4 columnwise) expose
            # bytes/scales for GEMM but intentionally do not dequantize.
            continue
        actual_dequantized = actual_storage.dequantize()
        torch.testing.assert_close(
            actual_dequantized,
            expected_dequantized,
            rtol=0.0,
            atol=0.0,
            msg=f"{context} {direction} dequantized value differs",
        )


def _snapshot_model_parameters(model):
    """Capture normal values and all tensor metadata in both hybrid directions."""
    snapshot = {}
    for name, param in model.named_parameters():
        if isinstance(param, HybridQuantizedTensor):
            snapshot[name] = {
                "rowwise": _snapshot_storage_tensor_metadata(param.rowwise_sub_storage, clone=True),
                "columnwise": _snapshot_storage_tensor_metadata(
                    param.columnwise_sub_storage, clone=True
                ),
            }
        else:
            snapshot[name] = param.detach().clone()
    return snapshot


class _CountingIdentityQuantizer(IdentityQuantizer):
    """Identity quantizer that counts quantize calls for regression tests."""

    def __init__(self, counter, *, dtype=None, rowwise=True, columnwise=True):
        super().__init__(dtype=dtype, rowwise=rowwise, columnwise=columnwise)
        self.counter = counter

    def copy(self):
        quantizer = _CountingIdentityQuantizer(
            self.counter,
            dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def quantize_impl(self, tensor):
        self.counter["calls"] += 1
        return super().quantize_impl(tensor)


@requires_mxfp8_and_nvfp4
class TestComposerStyleFactory:
    """Composer 2-style row-scaled NVFP4 forward + MXFP8 backward dispatch."""

    @staticmethod
    def _factory(role):
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
            nvfp4_row_scaled_fwd_dequantized_mxfp8_bwd_quantizer_factory,
        )

        return nvfp4_row_scaled_fwd_dequantized_mxfp8_bwd_quantizer_factory(role)

    @pytest.mark.parametrize(
        "tensor_type,row_scaled",
        [("input", True), ("weight", False)],
    )
    def test_grouped_linear_forward_roles_use_nvfp4_rowwise_mxfp8_columnwise(
        self, tensor_type, row_scaled
    ):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(
            QuantizerRole(module_type="grouped_linear", tensor_type=tensor_type)
        )

        assert isinstance(quantizer, HybridQuantizer)
        assert quantizer.columnwise_source == "rowwise_dequantized"
        assert isinstance(quantizer.rowwise_quantizer, NVFP4Quantizer)
        assert isinstance(quantizer.columnwise_quantizer, MXFP8Quantizer)
        assert quantizer.rowwise_quantizer.row_scaled_nvfp4 is row_scaled
        assert quantizer.rowwise_quantizer.with_rht is False
        assert quantizer.rowwise_quantizer.with_post_rht_amax is False
        assert quantizer.rowwise_quantizer.with_2d_quantization is False
        assert quantizer.rowwise_quantizer.stochastic_rounding is False
        assert quantizer.rowwise_quantizer.rowwise_usage is True
        assert quantizer.rowwise_quantizer.columnwise_usage is False
        assert quantizer.columnwise_quantizer.rowwise_usage is False
        assert quantizer.columnwise_quantizer.columnwise_usage is True

    @pytest.mark.parametrize("tensor_type", ["input", "output", "weight"])
    def test_regular_linear_forward_roles_fall_back_to_mxfp8(self, tensor_type):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type="linear", tensor_type=tensor_type))

        assert isinstance(quantizer, MXFP8Quantizer)

    @pytest.mark.parametrize("module_type", ["linear", "grouped_linear"])
    @pytest.mark.parametrize("tensor_type", ["grad_output", "grad_input"])
    def test_backward_roles_use_mxfp8(self, module_type, tensor_type):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type=module_type, tensor_type=tensor_type))

        assert isinstance(quantizer, MXFP8Quantizer)

    def test_non_linear_roles_fall_back_to_mxfp8(self):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type="dpa", tensor_type="qkv"))

        assert isinstance(quantizer, MXFP8Quantizer)


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
class TestMXFP8FwdDequantizedBwdFactory:
    """MXFP8 forward + dequantized backward qfactory dispatch."""

    @staticmethod
    def _factory(role):
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
            mxfp8_fwd_dequantized_bwd_quantizer_factory,
        )

        return mxfp8_fwd_dequantized_bwd_quantizer_factory(role)

    @pytest.mark.parametrize("module_type", ["linear", "grouped_linear"])
    @pytest.mark.parametrize("tensor_type", ["input", "weight"])
    def test_forward_roles_require_saved_forward_quantized_value(self, module_type, tensor_type):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type=module_type, tensor_type=tensor_type))

        assert isinstance(quantizer, HybridQuantizer)
        assert quantizer.columnwise_source == "rowwise_dequantized"
        assert isinstance(quantizer.rowwise_quantizer, MXFP8Quantizer)
        assert isinstance(quantizer.columnwise_quantizer, IdentityQuantizer)
        assert quantizer.allows_save_original_input_for_backward() is False

    @pytest.mark.parametrize("module_type", ["linear", "grouped_linear"])
    def test_grad_output_role_allows_save_original_input(self, module_type):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type=module_type, tensor_type="grad_output"))

        assert isinstance(quantizer, IdentityQuantizer)
        assert quantizer.allows_save_original_input_for_backward() is True


@requires_nvfp4
class TestNVFP4WeightDoubleQuantFactory:
    """Base NVFP4 recipe with W.T sourced from dequantized 1D W."""

    @staticmethod
    def _factory(role):
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
            nvfp4_1d_double_quantized_weight_quantizer_factory,
        )

        return nvfp4_1d_double_quantized_weight_quantizer_factory(role)

    @staticmethod
    def _assert_plain_1d_nvfp4(quantizer):
        assert isinstance(quantizer, NVFP4Quantizer)
        assert quantizer.row_scaled_nvfp4 is False
        assert quantizer.with_rht is False
        assert quantizer.with_post_rht_amax is False
        assert quantizer.with_2d_quantization is False
        assert quantizer.stochastic_rounding is False

    @pytest.mark.parametrize("module_type", ["linear", "grouped_linear"])
    def test_weight_roles_use_rowwise_dequantized_source(self, module_type):
        from transformer_engine.pytorch.quantization import QuantizerRole

        quantizer = self._factory(QuantizerRole(module_type=module_type, tensor_type="weight"))

        assert isinstance(quantizer, HybridQuantizer)
        assert quantizer.columnwise_source == "rowwise_dequantized"
        self._assert_plain_1d_nvfp4(quantizer.rowwise_quantizer)
        self._assert_plain_1d_nvfp4(quantizer.columnwise_quantizer)
        assert quantizer.rowwise_quantizer.rowwise_usage is True
        assert quantizer.rowwise_quantizer.columnwise_usage is False
        assert quantizer.columnwise_quantizer.rowwise_usage is False
        assert quantizer.columnwise_quantizer.columnwise_usage is True

    @staticmethod
    def _assert_matches_base_nvfp4_recipe(quantizer, expected):
        assert isinstance(quantizer, NVFP4Quantizer)
        assert isinstance(expected, NVFP4Quantizer)
        assert quantizer.dtype == expected.dtype
        assert quantizer.rowwise_usage == expected.rowwise_usage
        assert quantizer.columnwise_usage == expected.columnwise_usage
        assert quantizer.with_rht == expected.with_rht
        assert quantizer.with_post_rht_amax == expected.with_post_rht_amax
        assert quantizer.with_2d_quantization == expected.with_2d_quantization
        assert quantizer.stochastic_rounding == expected.stochastic_rounding
        assert quantizer.row_scaled_nvfp4 == expected.row_scaled_nvfp4

    @pytest.mark.parametrize("module_type", ["linear", "grouped_linear"])
    @pytest.mark.parametrize("tensor_type", ["input", "output", "grad_output", "grad_input"])
    def test_non_weight_linear_roles_match_base_nvfp4_recipe(self, module_type, tensor_type):
        from transformer_engine.pytorch.quantization import QuantizerRole
        from transformer_engine.pytorch.custom_recipes.quantization_factory_base import (
            nvfp4_quantizer_factory,
        )

        role = QuantizerRole(module_type=module_type, tensor_type=tensor_type)
        quantizer = self._factory(role)
        expected = nvfp4_quantizer_factory(role)

        self._assert_matches_base_nvfp4_recipe(quantizer, expected)

    def test_non_linear_roles_match_base_nvfp4_recipe(self):
        from transformer_engine.pytorch.quantization import QuantizerRole
        from transformer_engine.pytorch.custom_recipes.quantization_factory_base import (
            nvfp4_quantizer_factory,
        )

        role = QuantizerRole(module_type="dpa", tensor_type="qkv")
        quantizer = self._factory(role)
        expected = nvfp4_quantizer_factory(role)

        self._assert_matches_base_nvfp4_recipe(quantizer, expected)

    def test_weight_quantizer_produces_both_nvfp4_storages(self):
        from transformer_engine.pytorch.quantization import QuantizerRole

        torch.manual_seed(2026)
        src = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        quantizer = self._factory(QuantizerRole(module_type="linear", tensor_type="weight"))

        out = quantizer.quantize(src)

        assert isinstance(out.rowwise_sub_storage, (NVFP4TensorStorage, NVFP4Tensor))
        assert isinstance(out.columnwise_sub_storage, (NVFP4TensorStorage, NVFP4Tensor))


@requires_fp8_and_nvfp4
class TestHybridQuantizerConstruction:
    """Test construction and basic properties of hybrid quantizer."""

    def test_creation(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        assert isinstance(hq, HybridQuantizer)
        assert hq.rowwise_usage is True
        assert hq.columnwise_usage is True
        assert isinstance(hq.rowwise_quantizer, Float8CurrentScalingQuantizer)
        assert isinstance(hq.columnwise_quantizer, NVFP4Quantizer)

    def test_hybrid_storage_and_tensor_require_parent_quantizer(self):
        with pytest.raises(TypeError, match="requires a parent HybridQuantizer"):
            HybridQuantizedTensorStorage(
                rowwise_storage=None,
                columnwise_storage=None,
                quantizer=None,
            )
        with pytest.raises(TypeError, match="requires a parent HybridQuantizer"):
            HybridQuantizedTensor(
                shape=(1, 1),
                dtype=torch.bfloat16,
                rowwise_storage=None,
                columnwise_storage=None,
                quantizer=None,
            )

    def test_rejects_same_sub_quantizer_instance_for_both_directions(self):
        quantizer = _make_fp8_quantizer()

        with pytest.raises(ValueError, match="requires distinct rowwise and columnwise"):
            HybridQuantizer(rowwise_quantizer=quantizer, columnwise_quantizer=quantizer)

        assert quantizer.rowwise_usage is True
        assert quantizer.columnwise_usage is True

    def test_compatible_recipe_is_custom_recipe(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        assert hq._get_compatible_recipe() is recipe.CustomRecipe

    def test_supports_only_rowwise_all_gather_nvfp4_columnwise(self):
        """NVFP4 columnwise sub-quantizer forces rowwise-only AG.

        ``NVFP4Tensor.dequantize()`` raises ``NotImplementedError`` for
        columnwise-only data, so the BF16 fallback in
        ``gather_along_first_dim`` cannot operate on a columnwise-only
        NVFP4 hybrid sub-storage. ``HybridQuantizer.supports_only_rowwise_all_gather``
        must return True in this case so ``_linear_forward_impl`` /
        ``_linear_backward`` preserve rowwise data (which NVFP4 can
        dequantize) instead.
        """
        hq = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_nvfp4_quantizer(),
        )
        assert hq.supports_only_rowwise_all_gather() is True

    def test_supports_only_rowwise_all_gather_mxfp8_both(self):
        """MXFP8 in both directions → columnwise dequant works → default
        False so the save-columnwise (for wgrad) path stays active."""
        if not mxfp8_available:
            pytest.skip(f"MXFP8: {reason_for_no_mxfp8}")
        hq = HybridQuantizer(
            rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            columnwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
        )
        assert hq.supports_only_rowwise_all_gather() is False

    def test_supports_only_rowwise_all_gather_fp8_current_propagates(self):
        """Float8CurrentScalingQuantizer returns True for its own flag;
        hybrid must propagate (not swallow) that semantics."""
        hq = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        assert hq.supports_only_rowwise_all_gather() is True

    def test_supports_only_rowwise_all_gather_nvfp4_both(self):
        """NVFP4 in both directions → columnwise sub-quantizer is NVFP4
        → forces rowwise-only AG regardless of rowwise flag."""
        hq = HybridQuantizer(
            rowwise_quantizer=_make_nvfp4_quantizer(),
            columnwise_quantizer=_make_nvfp4_quantizer(),
        )
        assert hq.supports_only_rowwise_all_gather() is True


@requires_fp8
class TestHybridColumnwiseSource:
    """Test columnwise source provenance in HybridQuantizer."""

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(90210)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    @staticmethod
    def _make_quantizer(columnwise_source="original"):
        return HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source=columnwise_source,
        )

    def test_default_columnwise_source_original(self, input_tensor):
        hq = self._make_quantizer()
        out = hq.quantize(input_tensor)

        assert hq.columnwise_source == "original"
        torch.testing.assert_close(
            out.columnwise_sub_storage.dequantize(), input_tensor, rtol=0.0, atol=0.0
        )

    def test_invalid_columnwise_source_raises(self):
        with pytest.raises(ValueError, match="columnwise_source"):
            HybridQuantizer(
                rowwise_quantizer=IdentityQuantizer(),
                columnwise_quantizer=IdentityQuantizer(),
                columnwise_source="dequantized",
            )

    def test_default_quantizer_allows_save_original_input_for_backward(self):
        assert IdentityQuantizer().allows_save_original_input_for_backward() is True

    @pytest.mark.parametrize(
        "columnwise_source,allowed",
        [("original", True), ("rowwise_dequantized", False)],
    )
    def test_hybrid_save_original_input_policy(self, columnwise_source, allowed):
        hq = self._make_quantizer(columnwise_source=columnwise_source)

        assert hq.allows_save_original_input_for_backward() is allowed

    def test_copy_preserves_columnwise_source(self):
        hq = self._make_quantizer(columnwise_source="rowwise_dequantized")
        hq.set_usage(rowwise=False, columnwise=True)

        copied = hq.copy()

        assert copied.columnwise_source == "rowwise_dequantized"
        assert copied.rowwise_usage is False
        assert copied.columnwise_usage is True

    def test_rowwise_dequantized_identity_columnwise_matches_rowwise(self, input_tensor):
        hq = self._make_quantizer(columnwise_source="rowwise_dequantized")
        out = hq.quantize(input_tensor)

        rowwise_dq = out.rowwise_sub_storage.dequantize()
        columnwise_dq = out.columnwise_sub_storage.dequantize()
        torch.testing.assert_close(columnwise_dq, rowwise_dq, rtol=0.0, atol=0.0)

    def test_columnwise_only_uses_transient_rowwise_source(self, input_tensor):
        hq = self._make_quantizer(columnwise_source="rowwise_dequantized")
        hq.set_usage(rowwise=False, columnwise=True)
        expected = _make_fp8_quantizer().quantize(input_tensor).dequantize()

        out = hq.quantize(input_tensor)

        assert out.rowwise_sub_storage is None
        assert out.columnwise_sub_storage is not None
        torch.testing.assert_close(
            out.columnwise_sub_storage.dequantize(), expected, rtol=0.0, atol=0.0
        )

    def test_update_quantized_columnwise_only_uses_transient_rowwise_source(self, input_tensor):
        hq = self._make_quantizer(columnwise_source="rowwise_dequantized")
        dst = hq.quantize(input_tensor)
        rowwise_before = dst.rowwise_sub_storage.dequantize().clone()
        new_src = torch.randn_like(input_tensor) * 8
        expected = _make_fp8_quantizer().quantize(new_src).dequantize()

        hq.set_usage(rowwise=False, columnwise=True)
        hq.update_quantized(new_src, dst)

        torch.testing.assert_close(
            dst.rowwise_sub_storage.dequantize(), rowwise_before, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            dst.columnwise_sub_storage.dequantize(), expected, rtol=0.0, atol=0.0
        )

    def test_update_quantized_uses_updated_rowwise_storage(self, input_tensor):
        hq = self._make_quantizer(columnwise_source="rowwise_dequantized")
        dst = hq.quantize(input_tensor)
        new_src = torch.randn_like(input_tensor) * 8

        hq.set_usage(rowwise=True, columnwise=True)
        hq.update_quantized(new_src, dst)

        torch.testing.assert_close(
            dst.columnwise_sub_storage.dequantize(),
            dst.rowwise_sub_storage.dequantize(),
            rtol=0.0,
            atol=0.0,
        )


@requires_fp8
class TestHybridSaveOriginalInputPolicy:
    """Module-level save_original_input policy for hybrid qfactory inputs."""

    @staticmethod
    def _counting_identity_hybrid_recipe(counter):
        def factory(role):
            if role is not None and role.module_type == "linear" and role.tensor_type == "input":
                return HybridQuantizer(
                    rowwise_quantizer=_CountingIdentityQuantizer(counter),
                    columnwise_quantizer=IdentityQuantizer(),
                    columnwise_source="rowwise_dequantized",
                )
            return IdentityQuantizer()

        return recipe.CustomRecipe(qfactory=factory)

    def test_linear_save_original_input_veto_uses_saved_forward_quantized_input(self):
        torch.manual_seed(205)
        counter = {"calls": 0}
        model = Linear(
            128,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            save_original_input=True,
        ).cuda()
        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        with pytest.warns(UserWarning, match="Ignoring save_original_input=True"):
            with autocast(
                enabled=True,
                recipe=self._counting_identity_hybrid_recipe(counter),
            ):
                out = model(inp)

        calls_after_forward = counter["calls"]
        assert calls_after_forward > 0

        out.float().sum().backward()

        assert counter["calls"] == calls_after_forward


@requires_fp8_and_nvfp4
class TestHybridQuantize:
    """Test quantization via HybridQuantizer."""

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def test_quantize_returns_hybrid_tensor(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        assert isinstance(result, HybridQuantizedTensor)

    def test_quantize_shape_preserved(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        assert result.shape == input_tensor.shape

    def test_quantize_dtype_preserved(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        assert result.dtype == input_tensor.dtype

    def test_sub_storage_types_fp8_row_fp4_col(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        row_storage = result.rowwise_sub_storage
        col_storage = result.columnwise_sub_storage
        assert isinstance(row_storage, (Float8TensorStorage, Float8Tensor))
        assert isinstance(col_storage, (NVFP4TensorStorage, NVFP4Tensor))

    def test_sub_storage_types_reversed(self, input_tensor):
        hq = _make_hybrid_quantizer_fp4_row_fp8_col()
        result = hq.quantize(input_tensor)
        row_storage = result.rowwise_sub_storage
        col_storage = result.columnwise_sub_storage
        assert isinstance(row_storage, (NVFP4TensorStorage, NVFP4Tensor))
        assert isinstance(col_storage, (Float8TensorStorage, Float8Tensor))

    def test_quantize_internal_returns_storage(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.internal = True
        result = hq.quantize(input_tensor)
        assert isinstance(result, HybridQuantizedTensorStorage)
        assert not isinstance(result, HybridQuantizedTensor)
        hq.internal = False


@requires_fp8_and_nvfp4
class TestHybridDequantize:
    """Test dequantization round-trip."""

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def test_dequantize_shape(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize()
        assert dequantized.shape == input_tensor.shape

    def test_dequantize_dtype(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize()
        assert dequantized.dtype == input_tensor.dtype

    def test_dequantize_explicit_dtype(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize(dtype=torch.float32)
        assert dequantized.dtype == torch.float32
        assert dequantized.shape == input_tensor.shape

    def test_dequantize_close_to_original(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize()
        torch.testing.assert_close(
            dequantized.float(), input_tensor.float(), rtol=0.125, atol=0.0675
        )

    def test_dequantize_reversed_close_to_original(self, input_tensor):
        hq = _make_hybrid_quantizer_fp4_row_fp8_col()
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize()
        torch.testing.assert_close(dequantized.float(), input_tensor.float(), rtol=0.5, atol=1.0)

    def test_storage_dequantize(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.internal = True
        result = hq.quantize(input_tensor)
        dequantized = result.dequantize(dtype=torch.bfloat16)
        assert dequantized.shape == input_tensor.shape
        hq.internal = False


class TestHybridUpdateUsage:
    """Test update_usage semantics and sub-storage cleanup."""

    @pytest.fixture
    def hybrid_tensor(self):
        inp = torch.randn(4, 8)
        hq = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        return hq.quantize(inp)

    def test_initial_usages(self, hybrid_tensor):
        usages = hybrid_tensor.get_usages()
        assert usages["rowwise"] is True
        assert usages["columnwise"] is True

    def test_drop_rowwise(self, hybrid_tensor):
        hybrid_tensor.update_usage(rowwise_usage=False)
        assert hybrid_tensor.rowwise_sub_storage is None
        assert hybrid_tensor.columnwise_sub_storage is not None
        usages = hybrid_tensor.get_usages()
        assert usages["rowwise"] is False
        assert usages["columnwise"] is True

    def test_drop_columnwise(self, hybrid_tensor):
        hybrid_tensor.update_usage(columnwise_usage=False)
        assert hybrid_tensor.columnwise_sub_storage is None
        assert hybrid_tensor.rowwise_sub_storage is not None
        usages = hybrid_tensor.get_usages()
        assert usages["rowwise"] is True
        assert usages["columnwise"] is False

    def test_drop_both(self, hybrid_tensor):
        hybrid_tensor.update_usage(rowwise_usage=False, columnwise_usage=False)
        usages = hybrid_tensor.get_usages()
        assert usages["rowwise"] is False
        assert usages["columnwise"] is False

    def test_request_true_is_noop(self, hybrid_tensor):
        row_before = hybrid_tensor.rowwise_sub_storage
        col_before = hybrid_tensor.columnwise_sub_storage
        hybrid_tensor.update_usage(rowwise_usage=True, columnwise_usage=True)
        assert hybrid_tensor.rowwise_sub_storage is row_before
        assert hybrid_tensor.columnwise_sub_storage is col_before

    def test_request_missing_columnwise_raises(self, hybrid_tensor):
        hybrid_tensor.update_usage(columnwise_usage=False)

        with pytest.raises(RuntimeError, match="no columnwise sub-storage"):
            hybrid_tensor.update_usage(columnwise_usage=True)

        assert hybrid_tensor.rowwise_sub_storage is not None
        assert hybrid_tensor.columnwise_sub_storage is None

    def test_request_missing_rowwise_raises(self, hybrid_tensor):
        hybrid_tensor.update_usage(rowwise_usage=False)

        with pytest.raises(RuntimeError, match="no rowwise sub-storage"):
            hybrid_tensor.update_usage(rowwise_usage=True)

        assert hybrid_tensor.rowwise_sub_storage is None
        assert hybrid_tensor.columnwise_sub_storage is not None

    def test_missing_direction_request_is_atomic(self, hybrid_tensor):
        hybrid_tensor.update_usage(columnwise_usage=False)
        row_before = hybrid_tensor.rowwise_sub_storage

        with pytest.raises(RuntimeError, match="no columnwise sub-storage"):
            hybrid_tensor.update_usage(rowwise_usage=False, columnwise_usage=True)

        assert hybrid_tensor.rowwise_sub_storage is row_before
        assert hybrid_tensor.columnwise_sub_storage is None

    def test_none_preserves_missing_direction(self, hybrid_tensor):
        hybrid_tensor.update_usage(columnwise_usage=False)
        row_before = hybrid_tensor.rowwise_sub_storage

        hybrid_tensor.update_usage(rowwise_usage=None, columnwise_usage=None)

        assert hybrid_tensor.rowwise_sub_storage is row_before
        assert hybrid_tensor.columnwise_sub_storage is None

    def test_repr_after_drop(self, hybrid_tensor):
        hybrid_tensor.update_usage(rowwise_usage=False)
        r = repr(hybrid_tensor)
        assert "HybridQuantizedTensor" in r
        assert "rowwise=None" in r

        hybrid_tensor.update_usage(columnwise_usage=False)
        r = repr(hybrid_tensor)
        assert "rowwise=None" in r
        assert "columnwise=None" in r


requires_mxfp8 = pytest.mark.skipif(
    not mxfp8_available,
    reason=f"MXFP8: {reason_for_no_mxfp8}",
)


@requires_fp8_and_nvfp4
class TestHybridClear:
    """Test HybridQuantizedTensorStorage.clear() — buffer deallocation.

    ``clear()`` is invoked by cpu_offload_v1 after the offloader has taken
    its own reference to the extracted buffers, to free the GPU-resident
    originals. HybridQuantizedTensorStorage delegates to each sub-storage's
    own clear(), which replaces primary data buffers with empty tensors.
    """

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    @staticmethod
    def _primary_data_numels(sub_storage):
        """Collect numel() of primary data buffers on a sub-storage.

        After ``clear()`` every entry should be 0 (native sub-storages set
        ``t.data = _empty_tensor()`` on the primary buffers).
        """
        if sub_storage is None:
            return []
        data = sub_storage.get_data_tensors()
        if not isinstance(data, tuple):
            data = (data,)
        return [t.numel() for t in data if t is not None]

    def test_clear_delegates_to_both_sub_storages(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)

        row_before = self._primary_data_numels(ht.rowwise_sub_storage)
        col_before = self._primary_data_numels(ht.columnwise_sub_storage)
        assert row_before and all(n > 0 for n in row_before)
        assert col_before and all(n > 0 for n in col_before)

        ht.clear()

        row_after = self._primary_data_numels(ht.rowwise_sub_storage)
        col_after = self._primary_data_numels(ht.columnwise_sub_storage)
        assert all(n == 0 for n in row_after)
        assert all(n == 0 for n in col_after)

    @requires_mxfp8
    def test_clear_delegates_mxfp8_nvfp4(self, input_tensor):
        """Per-block sub-storage path (MXFP8 rowwise + NVFP4 columnwise)."""
        hq = HybridQuantizer(
            rowwise_quantizer=_make_mxfp8_quantizer(),
            columnwise_quantizer=_make_nvfp4_quantizer(),
        )
        ht = hq.quantize(input_tensor)
        ht.clear()
        for sub in (ht.rowwise_sub_storage, ht.columnwise_sub_storage):
            for n in self._primary_data_numels(sub):
                assert n == 0

    def test_clear_with_rowwise_only(self, input_tensor):
        """columnwise sub-storage is None — clear() must not crash."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        ht.update_usage(columnwise_usage=False)
        assert ht.columnwise_sub_storage is None

        ht.clear()

        assert all(n == 0 for n in self._primary_data_numels(ht.rowwise_sub_storage))

    def test_clear_with_columnwise_only(self, input_tensor):
        """rowwise sub-storage is None — clear() must not crash."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        ht.update_usage(rowwise_usage=False)
        assert ht.rowwise_sub_storage is None

        ht.clear()

        assert all(n == 0 for n in self._primary_data_numels(ht.columnwise_sub_storage))

    def test_clear_with_both_sub_storages_dropped(self, input_tensor):
        """Both sub-storages are None — clear() must not crash."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        ht.update_usage(rowwise_usage=False, columnwise_usage=False)
        assert ht.rowwise_sub_storage is None
        assert ht.columnwise_sub_storage is None

        ht.clear()  # must not raise

    def test_clear_is_idempotent(self, input_tensor):
        """Calling clear() twice must not raise and leaves buffers empty."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        ht.clear()
        ht.clear()
        for sub in (ht.rowwise_sub_storage, ht.columnwise_sub_storage):
            for n in self._primary_data_numels(sub):
                assert n == 0


@requires_fp8
class TestHybridTensorShapeOps:
    """Shape ops that preserve supported Hybrid sub-storages."""

    def test_fp8_current_non_noop_slice_and_narrow_preserve_hybrid(self):
        torch.manual_seed(42)
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
        quantizer = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        tensor = quantizer.quantize(x)
        dequantized = tensor.dequantize()

        sliced = torch.ops.aten.slice.Tensor(tensor, 0, 0, 32, 1)
        narrowed = tensor.narrow(0, 16, 32)

        assert isinstance(sliced, HybridQuantizedTensor)
        assert isinstance(narrowed, HybridQuantizedTensor)
        torch.testing.assert_close(sliced.dequantize(), dequantized[:32], rtol=0, atol=0)
        torch.testing.assert_close(narrowed.dequantize(), dequantized[16:48], rtol=0, atol=0)

    def test_full_span_step_slice_is_not_treated_as_noop(self):
        torch.manual_seed(42)
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
        quantizer = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        tensor = quantizer.quantize(x)
        dequantized = tensor.dequantize()

        sliced = torch.ops.aten.slice.Tensor(tensor, 0, 0, tensor.size(0), 2)

        assert isinstance(sliced, HybridQuantizedTensor)
        torch.testing.assert_close(sliced.dequantize(), dequantized[::2], rtol=0, atol=0)

    def test_same_shape_as_strided_with_offset_is_not_treated_as_noop(self):
        torch.manual_seed(42)
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
        quantizer = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        tensor = quantizer.quantize(x)
        dequantized = tensor.dequantize()

        base_view = torch.ops.aten.slice.Tensor(tensor, 0, 0, 63, 1)
        shifted = torch.ops.aten.as_strided.default(
            base_view, base_view.shape, base_view.stride(), x.stride(0)
        )

        assert isinstance(shifted, HybridQuantizedTensor)
        torch.testing.assert_close(shifted.dequantize(), dequantized[1:], rtol=0, atol=0)


@requires_fp8_and_nvfp4
class TestHybridDetachIsolation:
    """``HybridQuantizedTensor.detach()`` must produce a hybrid whose
    sub-storage wrappers are NOT shared with ``self`` — so that
    ``detached.prepare_for_saving()`` does not null out fields on the
    original.

    This is the property cpu_offload_v2 relies on at
    ``cpu_offload.py:378-382``:

        tensor_copy = tensor.detach()
        saved_tensors, _ = tensor_copy.prepare_for_saving()
    """

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def test_detach_produces_distinct_sub_storage_wrappers(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        detached = ht.detach()

        assert detached is not ht
        assert detached._rowwise_storage is not ht._rowwise_storage
        assert detached._columnwise_storage is not ht._columnwise_storage

    def test_detach_prepare_for_saving_does_not_affect_original(self, input_tensor):
        """prepare_for_saving on the detach() copy must not null the original.

        This is the specific invariant the cpu_offload_v2 push/reload cycle
        depends on. Without it, a second push on the same tensor — or even
        a bare ``.device`` read during offload eligibility checks — hits
        ``<native sub-storage> has no data!``.
        """
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)

        detached = ht.detach()
        _ = detached.prepare_for_saving()

        # Original must still be usable: dequantize, .device, repeated clone
        _ = ht.dequantize()
        _ = ht.device

    def test_detach_shares_underlying_buffers(self, input_tensor):
        """Buffer tensors are shared (no GPU allocation) — only wrappers differ."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        ht = hq.quantize(input_tensor)
        detached = ht.detach()

        orig_row_buffers = ht._rowwise_storage.get_data_tensors()
        new_row_buffers = detached._rowwise_storage.get_data_tensors()
        if not isinstance(orig_row_buffers, tuple):
            orig_row_buffers = (orig_row_buffers,)
            new_row_buffers = (new_row_buffers,)
        for a, b in zip(orig_row_buffers, new_row_buffers):
            if a is None and b is None:
                continue
            assert a is b, "detach() must share buffer tensors, not copy them"


@requires_fp8_and_nvfp4
class TestHybridSaveRestore:
    """Test prepare_for_saving / restore_from_saved round-trip."""

    @pytest.fixture
    def hybrid_tensor(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        return hq.quantize(inp)

    def test_save_restore_roundtrip(self, hybrid_tensor):
        dq_before = hybrid_tensor.dequantize()
        expected_metadata = {
            "rowwise": _snapshot_storage_tensor_metadata(
                hybrid_tensor.rowwise_sub_storage, clone=True
            ),
            "columnwise": _snapshot_storage_tensor_metadata(
                hybrid_tensor.columnwise_sub_storage, clone=True
            ),
        }
        buffers_before = _as_data_tensor_tuple(hybrid_tensor)
        tensors, obj = hybrid_tensor.prepare_for_saving()
        assert isinstance(tensors, list)
        assert all(t is None or isinstance(t, torch.Tensor) for t in tensors)

        remainder = obj.restore_from_saved(tensors)
        assert isinstance(remainder, list)
        assert len(remainder) == 0

        dq_after = hybrid_tensor.dequantize()
        buffers_after = _as_data_tensor_tuple(hybrid_tensor)
        assert len(buffers_after) == len(buffers_before)
        for before, after in zip(buffers_before, buffers_after):
            assert after is before, "Direct save/restore must reattach the exact saved buffer"
        torch.testing.assert_close(dq_before, dq_after, rtol=0.0, atol=0.0)
        for direction in ("rowwise", "columnwise"):
            actual_metadata = _snapshot_storage_tensor_metadata(
                getattr(hybrid_tensor, f"{direction}_sub_storage"), clone=False
            )
            _assert_nested_state_exact(
                actual_metadata,
                expected_metadata[direction],
                path=f"save/restore {direction}",
            )

    def test_save_clears_data(self, hybrid_tensor):
        tensors, obj = hybrid_tensor.prepare_for_saving()
        row_storage = hybrid_tensor.rowwise_sub_storage
        row_data_tensors = row_storage.get_data_tensors()
        if isinstance(row_data_tensors, tuple):
            assert all(t is None for t in row_data_tensors)
        else:
            assert row_data_tensors is None
        # Restore to clean up
        obj.restore_from_saved(tensors)


@requires_fp8_and_nvfp4
class TestHybridMakeEmpty:
    """Test HybridQuantizer.make_empty()."""

    def test_make_empty_shape(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        shape = (128, 256)
        empty = hq.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        assert isinstance(empty, HybridQuantizedTensor)
        assert empty.shape == torch.Size(shape)

    def test_make_empty_dtype(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        shape = (128, 256)
        empty = hq.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        assert empty.dtype == torch.bfloat16

    def test_make_empty_has_sub_storages(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        shape = (128, 256)
        empty = hq.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        assert empty.rowwise_sub_storage is not None
        assert empty.columnwise_sub_storage is not None


@requires_fp8_and_nvfp4
class TestHybridUsageFlagsRespected:
    """``HybridQuantizer`` must skip directions whose parent usage flag is
    False. Native quantizers honor ``rowwise_usage`` / ``columnwise_usage``
    inside the C++ kernel; hybrid sub-quantizers are pinned to one direction
    in ``__init__``, so the parent's flags never reach C++ — the equivalent
    skip lives in the Python composition layer. Modules call ``set_usage``
    extensively before each ``quantize`` (inference, output / grad_input
    quantizers, AG paths), so honoring the flags avoids 2x quantization waste.
    """

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    # ── quantize_impl ────────────────────────────────────────────

    def test_quantize_rowwise_only(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=True, columnwise=False)
        out = hq.quantize(input_tensor)
        assert out.rowwise_sub_storage is not None
        assert out.columnwise_sub_storage is None

    def test_quantize_columnwise_only(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=False, columnwise=True)
        out = hq.quantize(input_tensor)
        assert out.rowwise_sub_storage is None
        assert out.columnwise_sub_storage is not None

    def test_quantize_both_false(self, input_tensor):
        """``set_usage(False, False)`` mirrors ``update_usage(False, False)`` —
        both produce an empty hybrid. No defensive assert (matches native)."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=False, columnwise=False)
        out = hq.quantize(input_tensor)
        assert out.rowwise_sub_storage is None
        assert out.columnwise_sub_storage is None

    def test_quantize_both_true_default(self, input_tensor):
        """Default state (both flags True) keeps both directions populated."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        out = hq.quantize(input_tensor)
        assert out.rowwise_sub_storage is not None
        assert out.columnwise_sub_storage is not None

    def test_quantize_internal_storage_rowwise_only(self, input_tensor):
        """Internal storage path (used by FSDP2 / make_like flows) also
        honors the gate."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=True, columnwise=False)
        hq.internal = True
        try:
            out = hq.quantize(input_tensor)
            assert isinstance(out, HybridQuantizedTensorStorage)
            assert out.rowwise_sub_storage is not None
            assert out.columnwise_sub_storage is None
        finally:
            hq.internal = False

    def test_quantize_flag_change_between_calls(self, input_tensor):
        """A single quantizer can be re-used with different flags across
        calls (which is exactly how modules use one ``input_quantizer`` /
        ``weight_quantizer`` across forward / backward phases)."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()

        hq.set_usage(rowwise=True, columnwise=False)
        out_row = hq.quantize(input_tensor)
        assert out_row.rowwise_sub_storage is not None
        assert out_row.columnwise_sub_storage is None

        hq.set_usage(rowwise=False, columnwise=True)
        out_col = hq.quantize(input_tensor)
        assert out_col.rowwise_sub_storage is None
        assert out_col.columnwise_sub_storage is not None

        hq.set_usage(rowwise=True, columnwise=True)
        out_both = hq.quantize(input_tensor)
        assert out_both.rowwise_sub_storage is not None
        assert out_both.columnwise_sub_storage is not None

    # ── make_empty ───────────────────────────────────────────────

    def test_make_empty_rowwise_only(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=True, columnwise=False)
        empty = hq.make_empty((128, 256), dtype=torch.bfloat16, device="cuda")
        assert empty.rowwise_sub_storage is not None
        assert empty.columnwise_sub_storage is None

    def test_make_empty_columnwise_only(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=False, columnwise=True)
        empty = hq.make_empty((128, 256), dtype=torch.bfloat16, device="cuda")
        assert empty.rowwise_sub_storage is None
        assert empty.columnwise_sub_storage is not None

    def test_make_empty_both_false(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.set_usage(rowwise=False, columnwise=False)
        empty = hq.make_empty((128, 256), dtype=torch.bfloat16, device="cuda")
        assert empty.rowwise_sub_storage is None
        assert empty.columnwise_sub_storage is None

    # ── update_quantized ─────────────────────────────────────────
    #
    # Comparison strategy: snapshot raw data buffers via ``get_data_tensors()``
    # and compare bytes pre/post-update (same pattern as ``TestHybridClear``).
    # Avoids per-format ``dequantize()`` limitations (NVFP4 columnwise raises
    # NotImplementedError) and is a strictly stronger check — if the kernel
    # writes, raw bytes differ regardless of whether dequant is reversible.

    @staticmethod
    def _clone_data_tensors(sub_storage):
        """Deep-clone the primary data buffers of a sub-storage."""
        if sub_storage is None:
            return ()
        data = sub_storage.get_data_tensors()
        if not isinstance(data, tuple):
            data = (data,)
        return tuple(t.clone() if t is not None else None for t in data)

    @staticmethod
    def _assert_data_tensors_equal(snapshot, sub_storage):
        """Assert sub-storage's current data buffers byte-match a prior snapshot."""
        assert sub_storage is not None
        current = sub_storage.get_data_tensors()
        if not isinstance(current, tuple):
            current = (current,)
        assert len(snapshot) == len(
            current
        ), f"Buffer count changed: {len(snapshot)} → {len(current)}"
        for before, after in zip(snapshot, current):
            if before is None:
                assert after is None
                continue
            assert after is not None
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    @staticmethod
    def _assert_data_tensors_differ(snapshot, sub_storage):
        """Assert at least one buffer changed bytes vs the prior snapshot."""
        assert sub_storage is not None
        current = sub_storage.get_data_tensors()
        if not isinstance(current, tuple):
            current = (current,)
        any_changed = False
        for before, after in zip(snapshot, current):
            if before is None or after is None:
                continue
            if not torch.equal(before, after):
                any_changed = True
                break
        assert any_changed, "Expected at least one data buffer to change but none did"

    def test_update_quantized_rowwise_only_preserves_columnwise_data(self, input_tensor):
        """``update_quantized`` must not refresh a direction whose parent flag
        is False, even if the dst storage has that direction allocated.

        Mirrors how native ``tex.quantize(src, quantizer, dst, noop_flag)``
        skips a direction when ``quantizer.rowwise_usage=False`` even if the
        dst storage has that direction allocated.
        """
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        # Fully populate both directions
        dst = hq.quantize(input_tensor)
        # Snapshot the columnwise raw buffers before the targeted rowwise-only update
        col_before = self._clone_data_tensors(dst._columnwise_storage)

        # Switch to rowwise-only refresh and feed a substantially different src
        hq.set_usage(rowwise=True, columnwise=False)
        new_src = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") * 100
        hq.update_quantized(new_src, dst)

        # Both sub-storage objects survive in-place; columnwise bytes untouched
        self._assert_data_tensors_equal(col_before, dst._columnwise_storage)

    def test_update_quantized_columnwise_only_preserves_rowwise_data(self, input_tensor):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        dst = hq.quantize(input_tensor)
        row_before = self._clone_data_tensors(dst._rowwise_storage)

        hq.set_usage(rowwise=False, columnwise=True)
        new_src = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") * 100
        hq.update_quantized(new_src, dst)

        self._assert_data_tensors_equal(row_before, dst._rowwise_storage)

    def test_update_quantized_both_false_is_noop(self, input_tensor):
        """``set_usage(False, False)`` then ``update_quantized`` must leave
        both sub-storages' bytes untouched."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        dst = hq.quantize(input_tensor)
        row_before = self._clone_data_tensors(dst._rowwise_storage)
        col_before = self._clone_data_tensors(dst._columnwise_storage)

        hq.set_usage(rowwise=False, columnwise=False)
        new_src = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") * 100
        hq.update_quantized(new_src, dst)

        self._assert_data_tensors_equal(row_before, dst._rowwise_storage)
        self._assert_data_tensors_equal(col_before, dst._columnwise_storage)

    def test_update_quantized_actually_refreshes_requested(self, input_tensor):
        """Sanity check: when the parent flag is True, the corresponding
        sub-storage IS refreshed (otherwise the previous tests would pass
        vacuously by not refreshing anything)."""
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        dst = hq.quantize(input_tensor)
        row_before = self._clone_data_tensors(dst._rowwise_storage)

        hq.set_usage(rowwise=True, columnwise=False)
        new_src = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") * 100
        hq.update_quantized(new_src, dst)

        # Rowwise bytes must differ — confirms update_quantized actually ran
        self._assert_data_tensors_differ(row_before, dst._rowwise_storage)

    # ── te.Linear integration: inference path takes rowwise-only ─

    def test_te_linear_inference_workspace_rowwise_only(self):
        """``te.Linear`` forward under ``torch.no_grad()`` with a hybrid
        ``CustomRecipe`` must produce a rowwise-only weight workspace.
        ``linear.py:266-274`` sets ``weight_quantizer.set_usage(columnwise=False)``
        in inference; without the parent-flag gate, hybrid would still allocate
        both directions.
        """
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        )
        torch.manual_seed(2026)
        model = Linear(128, 256, bias=False, params_dtype=torch.bfloat16).cuda()
        x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")

        # is_first_microbatch=True forces the cache_name="weight" path
        # (see linear.py:1631) so the hybrid workspace persists in
        # model._fp8_workspaces and we can inspect its sub-storages.
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                _ = model(x, is_first_microbatch=True)

        ws = model._fp8_workspaces.get("weight")
        assert isinstance(
            ws, HybridQuantizedTensorStorage
        ), f"Expected hybrid weight workspace, got {type(ws).__name__}"
        assert ws.rowwise_sub_storage is not None, "Rowwise sub-storage must be populated for fprop"
        assert ws.columnwise_sub_storage is None, (
            "Inference forward must produce rowwise-only hybrid weight workspace; "
            "columnwise quantization should have been skipped per "
            "weight_quantizer.set_usage(rowwise=True, columnwise=False)."
        )


@requires_fp8_and_nvfp4
class TestHybridTorchDispatch:
    """Test torch dispatch operations."""

    @pytest.fixture
    def hybrid_tensor(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        return hq.quantize(inp)

    def test_detach(self, hybrid_tensor):
        detached = hybrid_tensor.detach()
        assert isinstance(detached, HybridQuantizedTensor)
        assert not detached.requires_grad

    def test_repr(self, hybrid_tensor):
        r = repr(hybrid_tensor)
        assert "HybridQuantizedTensor" in r


@requires_fp8_and_nvfp4
class TestHybridGetDataTensors:
    """Test get_data_tensors returns data from both sub-storages."""

    def test_get_data_tensors(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(inp)
        data_tensors = result.get_data_tensors()
        row_tensors = _as_data_tensor_tuple(result.rowwise_sub_storage)
        col_tensors = _as_data_tensor_tuple(result.columnwise_sub_storage)

        assert isinstance(data_tensors, tuple)
        assert row_tensors and col_tensors
        assert len(data_tensors) == len(row_tensors) + len(col_tensors)
        assert all(
            actual is expected for actual, expected in zip(data_tensors, row_tensors + col_tensors)
        ), "Hybrid get_data_tensors must concatenate both sub-storages in direction order"


@requires_fp8_and_nvfp4
class TestHybridDeviceAndSize:
    """Test device and size properties."""

    def test_device(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        result = hq.quantize(inp)
        assert result.device.type == "cuda"

    def test_size_from_storage(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hq.internal = True
        result = hq.quantize(inp)
        size = result.size()
        assert size == torch.Size([128, 256])
        hq.internal = False


@requires_fp8
class TestHybridGemmBitwiseIdentical:
    """Hybrid quantizer with same FP8 format in both directions must produce
    bitwise-identical results to the vanilla Float8CurrentScaling recipe."""

    def test_linear_fwd_bwd_matches_vanilla_fp8(self):
        torch.manual_seed(123)

        in_features = 64
        out_features = 64
        batch = 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.Float8CurrentScaling()
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        loss_ref = out_ref.float().sum()
        loss_ref.backward()

        def hybrid_fp8_factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight", "output")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E4M3,
                        device="cuda",
                    ),
                    columnwise_quantizer=Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E4M3,
                        device="cuda",
                    ),
                )
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3,
                device="cuda",
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_fp8_factory)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_hybrid = model_hybrid(inp_hybrid)
        loss_hybrid = out_hybrid.float().sum()
        loss_hybrid.backward()

        # Forward outputs must be bitwise identical
        assert torch.equal(
            out_ref, out_hybrid
        ), f"Forward mismatch: max diff = {(out_ref - out_hybrid).abs().max().item()}"

        # Input gradients must be bitwise identical
        assert inp_ref.grad is not None and inp_hybrid.grad is not None
        assert torch.equal(
            inp_ref.grad, inp_hybrid.grad
        ), f"Input grad mismatch: max diff = {(inp_ref.grad - inp_hybrid.grad).abs().max().item()}"

        # Parameter gradients must be bitwise identical
        ref_params = dict(model_ref.named_parameters())
        hybrid_params = dict(model_hybrid.named_parameters())
        for name, p_ref in ref_params.items():
            p_hyb = hybrid_params[name]
            assert (
                p_ref.grad is not None and p_hyb.grad is not None
            ), f"Missing gradient for param '{name}'"
            assert torch.equal(p_ref.grad, p_hyb.grad), (
                f"Param '{name}' grad mismatch: max diff = "
                f"{(p_ref.grad - p_hyb.grad).abs().max().item()}"
            )


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
class TestHybridGemmBitwiseIdenticalMXFP8:
    """Hybrid quantizer with MXFP8 in both directions must produce
    bitwise-identical results to the vanilla MXFP8BlockScaling recipe."""

    def test_linear_fwd_bwd_matches_vanilla_mxfp8(self):
        torch.manual_seed(200)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.MXFP8BlockScaling()
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_mxfp8_factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
            return HybridQuantizer(
                rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
                columnwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_mxfp8_factory)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_hybrid = model_hybrid(inp_hybrid)
        out_hybrid.float().sum().backward()

        assert torch.equal(
            out_ref, out_hybrid
        ), f"Forward mismatch: max diff = {(out_ref - out_hybrid).abs().max().item()}"
        assert torch.equal(
            inp_ref.grad, inp_hybrid.grad
        ), f"Input grad mismatch: max diff = {(inp_ref.grad - inp_hybrid.grad).abs().max().item()}"
        for name, p_ref in dict(model_ref.named_parameters()).items():
            p_hyb = dict(model_hybrid.named_parameters())[name]
            assert (
                p_ref.grad is not None and p_hyb.grad is not None
            ), f"Missing gradient for param '{name}'"
            assert torch.equal(p_ref.grad, p_hyb.grad), (
                f"Param '{name}' grad mismatch: max diff = "
                f"{(p_ref.grad - p_hyb.grad).abs().max().item()}"
            )

    def test_dequantized_bwd_qfactory_save_original_input_matches_base_recipe_bitwise(self):
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
            mxfp8_fwd_dequantized_bwd_quantizer_factory,
        )

        torch.manual_seed(204)
        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            save_original_input=False,
        ).cuda()
        model_qfactory = Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            save_original_input=True,
        ).cuda()
        model_qfactory.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_qfactory = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.MXFP8BlockScaling()
        ref_recipe.backward_override = "dequantized"
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)

        qfactory_recipe = recipe.CustomRecipe(qfactory=mxfp8_fwd_dequantized_bwd_quantizer_factory)
        with pytest.warns(UserWarning, match="Ignoring save_original_input=True"):
            with autocast(enabled=True, recipe=qfactory_recipe):
                out_qfactory = model_qfactory(inp_qfactory)

        assert torch.equal(
            out_ref, out_qfactory
        ), f"Forward mismatch: max diff = {(out_ref - out_qfactory).abs().max().item()}"

        out_ref.float().sum().backward()
        out_qfactory.float().sum().backward()

        assert inp_ref.grad is not None and inp_qfactory.grad is not None
        assert torch.equal(inp_ref.grad, inp_qfactory.grad), (
            "Input grad mismatch: max diff = "
            f"{(inp_ref.grad - inp_qfactory.grad).abs().max().item()}"
        )
        for name, p_ref in dict(model_ref.named_parameters()).items():
            p_qfactory = dict(model_qfactory.named_parameters())[name]
            assert (
                p_ref.grad is not None and p_qfactory.grad is not None
            ), f"Missing gradient for param {name!r}"
            assert torch.equal(p_ref.grad, p_qfactory.grad), (
                f"Param {name!r} grad mismatch: max diff = "
                f"{(p_ref.grad - p_qfactory.grad).abs().max().item()}"
            )


@requires_fp8
class TestCustomDPALocalRecipeCache:
    """Custom-DPA native recipe labels track the quantizer rebuild."""

    def test_inference_runs_once_per_quantizer_state_and_clears_stale_labels(self, monkeypatch):
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa_module,
        )
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        custom_recipe = recipe.CustomRecipe(qfactory=lambda _role: IdentityQuantizer())
        monkeypatch.setattr(
            FP8GlobalStateManager,
            "get_fp8_recipe",
            classmethod(lambda _cls: custom_recipe),
        )

        state = [object()]
        quantizer = [object()]

        def fake_base_init(module, num_gemms=1):  # pylint: disable=unused-argument
            module.fp8_meta["scaling_fwd"] = state[0]
            module.quantizers["scaling_fwd"] = [quantizer[0]]

        monkeypatch.setattr(
            TransformerEngineBaseModule,
            "init_fp8_metadata",
            fake_base_init,
        )

        inferred_labels = [recipe.MXFP8BlockScaling()]
        mutated_recipe_labels = [recipe.MXFP8BlockScaling(fp8_mha=True)]
        inference_results = iter((inferred_labels, mutated_recipe_labels, None))
        inference_calls = 0

        def fake_infer(*_args, **_kwargs):
            nonlocal inference_calls
            inference_calls += 1
            return next(inference_results)

        monkeypatch.setattr(dpa_module, "_infer_custom_dpa_local_recipes", fake_infer)

        dpa = te.DotProductAttention(
            num_attention_heads=2,
            kv_channels=16,
            attention_dropout=0.0,
        )
        dpa.init_fp8_metadata()
        assert dpa.fp8_meta["local_recipes"] is inferred_labels
        dpa.init_fp8_metadata()
        assert inference_calls == 1
        assert dpa.fp8_meta["local_recipes"] is inferred_labels

        # Native labels also copy these mutable fields from CustomRecipe. They
        # must refresh even when the quantizer generation itself is unchanged.
        custom_recipe.fp8_mha = True
        dpa.init_fp8_metadata()
        assert inference_calls == 2
        assert dpa.fp8_meta["local_recipes"] is mutated_recipe_labels
        dpa.init_fp8_metadata()
        assert inference_calls == 2
        assert dpa.fp8_meta["local_recipes"] is mutated_recipe_labels

        # A rebuilt recipe state/quantizer list invalidates the cache. If the
        # new family has no native label, the old label must not survive.
        state[0] = object()
        quantizer[0] = object()
        dpa.init_fp8_metadata()
        assert inference_calls == 3
        assert "local_recipes" not in dpa.fp8_meta
        dpa.init_fp8_metadata()
        assert inference_calls == 3
        assert "local_recipes" not in dpa.fp8_meta


@requires_fp8_and_nvfp4
class TestAttentionFactoryNativeRecipeParity:
    """Linear + DPA qfactories should match native DPA recipe switches bitwise."""

    batch = 2
    seq_len = 128
    hidden_size = 128
    num_heads = 4
    kv_channels = hidden_size // num_heads

    class _LinearDPALinear(torch.nn.Module):
        def __init__(self, hidden_size, num_heads, kv_channels):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.kv_channels = kv_channels
            self.qkv_proj = Linear(
                hidden_size,
                3 * hidden_size,
                params_dtype=torch.bfloat16,
                bias=False,
                name="qkv",
            ).cuda()
            self.dpa = te.DotProductAttention(
                num_heads,
                kv_channels,
                attention_dropout=0.0,
                qkv_format="bshd",
                name="core_attention",
            ).cuda()
            self.out_proj = Linear(
                hidden_size,
                hidden_size,
                params_dtype=torch.bfloat16,
                bias=False,
                name="proj",
            ).cuda()

        def forward(self, inp):
            batch, seq_len, _ = inp.shape
            qkv = self.qkv_proj(inp).view(
                batch,
                seq_len,
                3,
                self.num_heads,
                self.kv_channels,
            )
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            attn_out = self.dpa(q, k, v, qkv_format="bshd").reshape(
                batch,
                seq_len,
                self.hidden_size,
            )
            return self.out_proj(attn_out)

    @staticmethod
    def _set_native_dpa_recipe(monkeypatch, recipe_name):
        from transformer_engine.pytorch.attention import multi_head_attention as mha_module
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa_module,
        )

        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        monkeypatch.setenv("NVTE_DPA_FP8_RECIPE", recipe_name)
        monkeypatch.setenv("NVTE_DPA_FP8_FORMAT", "HYBRID")
        monkeypatch.setenv("NVTE_DPA_FP8DS_AMAX_ALGO", "most_recent")
        monkeypatch.setenv("NVTE_DPA_FP8DS_AMAX_HISTLEN", "1")
        monkeypatch.setenv("NVTE_DPA_FP8DS_REDUCE_AMAX", "1")
        monkeypatch.setattr(dpa_module, "_dpa_fp8_recipe", recipe_name)
        monkeypatch.setattr(dpa_module, "_dpa_fp8_format", recipe.Format.HYBRID)
        monkeypatch.setattr(dpa_module, "_dpa_fp8ds_amax_algo", "most_recent")
        monkeypatch.setattr(dpa_module, "_dpa_fp8ds_amax_histlen", 1)
        monkeypatch.setattr(dpa_module, "_dpa_fp8ds_reduce_amax", True)
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe", recipe_name)
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe_dpa", False)
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe_mha", False)

    @staticmethod
    def _clear_native_dpa_recipe(monkeypatch):
        from transformer_engine.pytorch.attention import multi_head_attention as mha_module
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa_module,
        )

        monkeypatch.delenv("NVTE_DPA_FP8_RECIPE", raising=False)
        monkeypatch.delenv("NVTE_DPA_FP8_FORMAT", raising=False)
        monkeypatch.delenv("NVTE_DPA_FP8DS_AMAX_ALGO", raising=False)
        monkeypatch.delenv("NVTE_DPA_FP8DS_AMAX_HISTLEN", raising=False)
        monkeypatch.delenv("NVTE_DPA_FP8DS_REDUCE_AMAX", raising=False)
        monkeypatch.setattr(dpa_module, "_dpa_fp8_recipe", "")
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe", "")
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe_dpa", False)
        monkeypatch.setattr(mha_module, "_dpa_fp8_recipe_mha", False)

    @staticmethod
    def _assert_equal(actual, expected, label):
        assert torch.equal(
            actual, expected
        ), f"{label} mismatch: max diff = {(actual.float() - expected.float()).abs().max().item()}"

    def _run_model(self, model, inp, grad, fp8_recipe, seed):
        run_inp = inp.clone().detach().requires_grad_(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with autocast(enabled=True, recipe=fp8_recipe):
            out = model(run_inp)
        out.backward(grad)
        local_recipes = [type(r).__name__ for r in model.dpa.fp8_meta.get("local_recipes", [])]
        return (
            out.detach().clone(),
            run_inp.grad.detach().clone(),
            {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            },
            local_recipes,
        )

    @pytest.mark.parametrize(
        "case_name,native_dpa_recipe,qfactory_name",
        [
            (
                "fp8_dpa",
                "Float8CurrentScaling",
                "nvfp4_linear_fp8_dpa_factory",
            ),
            (
                "mxfp8_dpa",
                "MXFP8BlockScaling",
                "nvfp4_linear_mxfp8_dpa_factory",
            ),
        ],
    )
    def test_linear_dpa_linear_matches_native_env_recipe_bitwise(
        self,
        monkeypatch,
        case_name,
        native_dpa_recipe,
        qfactory_name,
    ):
        if case_name == "mxfp8_dpa" and not mxfp8_available:
            pytest.skip(f"MXFP8: {reason_for_no_mxfp8}")

        from transformer_engine.pytorch.custom_recipes import quantization_factory_zoo
        from transformer_engine.pytorch.utils import get_device_compute_capability

        cc = get_device_compute_capability()
        if cc < (9, 0) or cc >= (12, 0):
            pytest.skip(f"FP8 attention not supported on sm{cc[0] * 10 + cc[1]}")

        self._set_native_dpa_recipe(monkeypatch, native_dpa_recipe)

        torch.manual_seed(2201)
        model_native = self._LinearDPALinear(
            self.hidden_size,
            self.num_heads,
            self.kv_channels,
        )
        model_qfactory = self._LinearDPALinear(
            self.hidden_size,
            self.num_heads,
            self.kv_channels,
        )
        model_qfactory.load_state_dict(model_native.state_dict())

        torch.manual_seed(2202)
        base_inp = torch.randn(
            self.batch,
            self.seq_len,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad = torch.randn_like(base_inp)

        native_recipe = recipe.NVFP4BlockScaling(fp8_dpa=True)
        qfactory = getattr(quantization_factory_zoo, qfactory_name)
        qfactory_recipe = recipe.CustomRecipe(qfactory=qfactory, fp8_dpa=True)

        native_out, native_dx, native_grads, native_local_recipes = self._run_model(
            model_native,
            base_inp,
            grad,
            native_recipe,
            seed=2203,
        )
        self._clear_native_dpa_recipe(monkeypatch)
        qfactory_out, qfactory_dx, qfactory_grads, qfactory_local_recipes = self._run_model(
            model_qfactory,
            base_inp,
            grad,
            qfactory_recipe,
            seed=2203,
        )

        expected_local_recipes = (
            ["Float8CurrentScaling", "DelayedScaling"]
            if native_dpa_recipe == "Float8CurrentScaling"
            else ["MXFP8BlockScaling"]
        )
        assert native_local_recipes == expected_local_recipes
        assert qfactory_local_recipes == expected_local_recipes

        self._assert_equal(qfactory_out, native_out, f"{case_name} output")
        self._assert_equal(qfactory_dx, native_dx, f"{case_name} input grad")
        assert qfactory_grads.keys() == native_grads.keys()
        for name, native_grad in native_grads.items():
            self._assert_equal(
                qfactory_grads[name],
                native_grad,
                f"{case_name} param grad {name}",
            )

    def _run_mha_model(self, model, inp, grad, fp8_recipe, seed):
        run_inp = inp.clone().detach().requires_grad_(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with autocast(enabled=True, recipe=fp8_recipe):
            out = model(run_inp, attn_mask_type="no_mask")
            if isinstance(out, tuple):
                out = out[0]
        out.backward(grad)
        return (
            out.detach().clone(),
            run_inp.grad.detach().clone(),
            {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            },
        )

    @pytest.mark.parametrize(
        "case_name,native_dpa_recipe,qfactory_name,expected_flags",
        [
            (
                "fp8_dpa",
                "Float8CurrentScaling",
                "nvfp4_linear_fp8_dpa_factory",
                (False, False, False),
            ),
            (
                "mxfp8_dpa",
                "MXFP8BlockScaling",
                "nvfp4_linear_mxfp8_dpa_factory",
                (False, False, False),
            ),
        ],
    )
    def test_multihead_attention_matches_native_env_recipe_bitwise(
        self,
        monkeypatch,
        case_name,
        native_dpa_recipe,
        qfactory_name,
        expected_flags,
    ):
        if case_name == "mxfp8_dpa" and not mxfp8_available:
            pytest.skip(f"MXFP8: {reason_for_no_mxfp8}")

        from transformer_engine.pytorch.attention import multi_head_attention as mha_module
        from transformer_engine.pytorch.custom_recipes import quantization_factory_zoo
        from transformer_engine.pytorch.utils import get_device_compute_capability

        cc = get_device_compute_capability()
        if cc < (9, 0) or cc >= (12, 0):
            pytest.skip(f"FP8 attention not supported on sm{cc[0] * 10 + cc[1]}")

        recorded_flags = []
        orig_update_roles = mha_module.MultiheadAttention._update_output_quantizer_roles

        def _recording_update_roles(self, qkv_fp8_output, proj_fp8_grad, dpa_fp8_output):
            recorded_flags.append((qkv_fp8_output, dpa_fp8_output, proj_fp8_grad))
            return orig_update_roles(self, qkv_fp8_output, proj_fp8_grad, dpa_fp8_output)

        monkeypatch.setattr(
            mha_module.MultiheadAttention,
            "_update_output_quantizer_roles",
            _recording_update_roles,
        )

        self._set_native_dpa_recipe(monkeypatch, native_dpa_recipe)

        torch.manual_seed(2301)
        model_native = te.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            kv_channels=self.kv_channels,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            params_dtype=torch.bfloat16,
            bias=False,
            qkv_format="sbhd",
            name="mha",
        ).cuda()
        model_qfactory = te.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            kv_channels=self.kv_channels,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            params_dtype=torch.bfloat16,
            bias=False,
            qkv_format="sbhd",
            name="mha",
        ).cuda()
        model_qfactory.load_state_dict(model_native.state_dict())

        torch.manual_seed(2302)
        base_inp = torch.randn(
            self.seq_len,
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad = torch.randn_like(base_inp)

        native_recipe = recipe.NVFP4BlockScaling(fp8_dpa=True)
        qfactory = getattr(quantization_factory_zoo, qfactory_name)
        qfactory_recipe = recipe.CustomRecipe(qfactory=qfactory, fp8_dpa=True)

        native_out, native_dx, native_grads = self._run_mha_model(
            model_native,
            base_inp,
            grad,
            native_recipe,
            seed=2303,
        )
        native_flags = recorded_flags[-1]
        self._clear_native_dpa_recipe(monkeypatch)
        qfactory_out, qfactory_dx, qfactory_grads = self._run_mha_model(
            model_qfactory,
            base_inp,
            grad,
            qfactory_recipe,
            seed=2303,
        )
        qfactory_flags = recorded_flags[-1]

        assert native_flags == expected_flags
        assert qfactory_flags == expected_flags
        self._assert_equal(qfactory_out, native_out, f"{case_name} MHA output")
        self._assert_equal(qfactory_dx, native_dx, f"{case_name} MHA input grad")
        assert qfactory_grads.keys() == native_grads.keys()
        for name, native_grad in native_grads.items():
            self._assert_equal(
                qfactory_grads[name],
                native_grad,
                f"{case_name} MHA param grad {name}",
            )

    def test_update_output_quantizer_roles_wires_independent_boundaries(self):
        from transformer_engine.pytorch.quantization import QuantizerRole

        model = te.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            kv_channels=self.kv_channels,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            params_dtype=torch.bfloat16,
            bias=False,
            qkv_format="sbhd",
            name="mha",
        ).cuda()
        qkv = model.layernorm_qkv if model.input_layernorm else model.qkv

        expected_qkv = QuantizerRole(
            module_type="dpa",
            tensor_type="qkv",
            name=model.core_attention.name or "",
        )
        expected_do = QuantizerRole(
            module_type="dpa",
            tensor_type="do",
            name=model.core_attention.name or "",
        )
        expected_o = QuantizerRole(
            module_type="linear",
            tensor_type="input",
            name=model.proj.name or "",
        )
        expected_dqkv = QuantizerRole(
            module_type="linear",
            tensor_type="grad_output",
            name=qkv.name or "",
        )

        def boundary_roles():
            return (
                qkv.output_quantizer_role,
                model.proj.grad_input_quantizer_role,
                model.core_attention.output_quantizer_role,
                model.core_attention.grad_input_quantizer_role,
            )

        model._update_output_quantizer_roles(True, False, False)
        assert boundary_roles() == (expected_qkv, None, None, None)

        model._update_output_quantizer_roles(False, True, False)
        assert boundary_roles() == (None, expected_do, None, None)

        model._update_output_quantizer_roles(False, False, True)
        assert boundary_roles() == (None, None, expected_o, expected_dqkv)

        model._update_output_quantizer_roles(False, False, False)
        assert boundary_roles() == (None, None, None, None)

    def test_mxfp8_qfactory_uses_plain_bf16_mha_boundaries(self, monkeypatch):
        """MXFP8 DPA stays internal; MHA boundary tensors remain plain BF16."""
        if not mxfp8_available:
            pytest.skip(f"MXFP8: {reason_for_no_mxfp8}")

        from transformer_engine.pytorch.attention.dot_product_attention import (
            backends as dpa_backends,
        )
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
            nvfp4_linear_mxfp8_dpa_factory,
        )
        from transformer_engine.pytorch.utils import get_device_compute_capability

        cc = get_device_compute_capability()
        if cc < (9, 0) or cc >= (12, 0):
            pytest.skip(f"FP8 attention not supported on sm{cc[0] * 10 + cc[1]}")

        self._clear_native_dpa_recipe(monkeypatch)
        torch.manual_seed(2401)
        model = te.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            kv_channels=self.kv_channels,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            params_dtype=torch.bfloat16,
            bias=False,
            qkv_format="sbhd",
            name="mha",
        ).cuda()

        boundary_tensors = {}
        saved_dpa_tensors = {}
        orig_prepare_for_saving = dpa_backends.prepare_for_saving

        def _record_prepare_for_saving(*tensors, **kwargs):
            saved_dpa_tensors["fp8_o"] = tensors[3]
            saved_dpa_tensors["f16_o"] = tensors[7]
            return orig_prepare_for_saving(*tensors, **kwargs)

        monkeypatch.setattr(dpa_backends, "prepare_for_saving", _record_prepare_for_saving)

        def _record_grad(name):
            def _hook(grad):
                boundary_tensors[name] = grad
                return grad

            return _hook

        def _record_dpa_inputs(_module, inputs):
            for name, tensor in zip(("q", "k", "v"), inputs[:3]):
                boundary_tensors[name] = tensor
                tensor.register_hook(_record_grad(f"d{name}"))

        def _record_dpa_output(_module, _inputs, output):
            boundary_tensors["o"] = output
            output.register_hook(_record_grad("do"))

        def _record_projection_input(_module, inputs):
            boundary_tensors["proj_input"] = inputs[0]
            boundary_tensors["o_is_proj_input"] = boundary_tensors["o"] is inputs[0]

        handles = (
            model.core_attention.register_forward_pre_hook(_record_dpa_inputs),
            model.core_attention.register_forward_hook(_record_dpa_output),
            model.proj.register_forward_pre_hook(_record_projection_input),
        )

        torch.manual_seed(2402)
        inp = torch.randn(
            self.seq_len,
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad = torch.randn_like(inp)
        qfactory_recipe = recipe.CustomRecipe(
            qfactory=nvfp4_linear_mxfp8_dpa_factory,
            fp8_dpa=True,
        )
        try:
            self._run_mha_model(model, inp, grad, qfactory_recipe, seed=2403)
        finally:
            for handle in handles:
                handle.remove()

        assert boundary_tensors["o_is_proj_input"]
        assert saved_dpa_tensors["fp8_o"] is None
        saved_o = saved_dpa_tensors["f16_o"]
        assert type(saved_o) is torch.Tensor
        assert saved_o.dtype is torch.bfloat16
        assert (
            saved_o.untyped_storage().data_ptr()
            == boundary_tensors["o"].untyped_storage().data_ptr()
        )

        for name in ("q", "k", "v", "o", "proj_input", "dq", "dk", "dv", "do"):
            tensor = boundary_tensors[name]
            assert type(tensor) is torch.Tensor, f"{name} is {type(tensor)}"
            assert tensor.dtype is torch.bfloat16, f"{name} has dtype {tensor.dtype}"


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
class TestHybridGemmBitwiseIdenticalBlockFP8:
    """Hybrid quantizer with Block FP8 in both directions must produce
    bitwise-identical results to the vanilla Float8BlockScaling recipe."""

    def test_linear_fwd_bwd_matches_vanilla_block_fp8(self):
        torch.manual_seed(201)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.Float8BlockScaling()
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_block_fp8_factory(role):
            is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
            is_weight = is_linear and role.tensor_type == "weight"
            dim = 2 if is_weight else 1
            if is_linear and role.tensor_type in ("grad_output", "grad_input"):
                return Float8BlockQuantizer(
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,
                    block_scaling_dim=dim,
                )
            return HybridQuantizer(
                rowwise_quantizer=Float8BlockQuantizer(
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,
                    block_scaling_dim=dim,
                ),
                columnwise_quantizer=Float8BlockQuantizer(
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,
                    block_scaling_dim=dim,
                ),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_block_fp8_factory)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_hybrid = model_hybrid(inp_hybrid)
        out_hybrid.float().sum().backward()

        assert torch.equal(
            out_ref, out_hybrid
        ), f"Forward mismatch: max diff = {(out_ref - out_hybrid).abs().max().item()}"
        assert torch.equal(
            inp_ref.grad, inp_hybrid.grad
        ), f"Input grad mismatch: max diff = {(inp_ref.grad - inp_hybrid.grad).abs().max().item()}"
        for name, p_ref in dict(model_ref.named_parameters()).items():
            p_hyb = dict(model_hybrid.named_parameters())[name]
            assert (
                p_ref.grad is not None and p_hyb.grad is not None
            ), f"Missing gradient for param '{name}'"
            assert torch.equal(p_ref.grad, p_hyb.grad), (
                f"Param '{name}' grad mismatch: max diff = "
                f"{(p_ref.grad - p_hyb.grad).abs().max().item()}"
            )


@pytest.mark.skipif(
    not (fp8_available and nvfp4_available),
    reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
)
class TestHybridGemmBitwiseIdenticalNVFP4:
    """Same-format hybrid NVFP4 must match vanilla with seeded SR."""

    def test_linear_fwd_bwd_matches_vanilla_nvfp4(self):
        torch.manual_seed(202)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.NVFP4BlockScaling()
        torch.manual_seed(1202)
        torch.cuda.manual_seed_all(1202)
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_nvfp4_factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type == "grad_output"
            ):
                return nvfp4_quantizer_factory(role)
            return HybridQuantizer(
                rowwise_quantizer=nvfp4_quantizer_factory(role),
                columnwise_quantizer=nvfp4_quantizer_factory(role),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_nvfp4_factory)
        torch.manual_seed(1202)
        torch.cuda.manual_seed_all(1202)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_hybrid = model_hybrid(inp_hybrid)
        out_hybrid.float().sum().backward()

        assert torch.equal(
            out_ref, out_hybrid
        ), f"Forward mismatch: max diff = {(out_ref - out_hybrid).abs().max().item()}"
        assert torch.equal(
            inp_ref.grad, inp_hybrid.grad
        ), f"Input grad mismatch: max diff = {(inp_ref.grad - inp_hybrid.grad).abs().max().item()}"
        for name, p_ref in dict(model_ref.named_parameters()).items():
            p_hyb = dict(model_hybrid.named_parameters())[name]
            assert (
                p_ref.grad is not None and p_hyb.grad is not None
            ), f"Missing gradient for param '{name}'"
            assert torch.equal(p_ref.grad, p_hyb.grad), (
                f"Param '{name}' grad mismatch: max diff = "
                f"{(p_ref.grad - p_hyb.grad).abs().max().item()}"
            )

    def test_linear_fwd_bwd_all_roles_hybrid(self):
        """Exercise grad_output as a HybridQuantizer too."""
        torch.manual_seed(203)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.NVFP4BlockScaling()
        torch.manual_seed(1203)
        torch.cuda.manual_seed_all(1203)
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_nvfp4_all_roles_factory(role):
            return HybridQuantizer(
                rowwise_quantizer=nvfp4_quantizer_factory(role),
                columnwise_quantizer=nvfp4_quantizer_factory(role),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_nvfp4_all_roles_factory)
        torch.manual_seed(1203)
        torch.cuda.manual_seed_all(1203)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_hybrid = model_hybrid(inp_hybrid)
        out_hybrid.float().sum().backward()

        assert torch.equal(
            out_ref, out_hybrid
        ), f"Forward mismatch: max diff = {(out_ref - out_hybrid).abs().max().item()}"
        assert torch.equal(
            inp_ref.grad, inp_hybrid.grad
        ), f"Input grad mismatch: max diff = {(inp_ref.grad - inp_hybrid.grad).abs().max().item()}"
        for name, p_ref in dict(model_ref.named_parameters()).items():
            p_hyb = dict(model_hybrid.named_parameters())[name]
            assert (
                p_ref.grad is not None and p_hyb.grad is not None
            ), f"Missing gradient for param '{name}'"
            assert torch.equal(p_ref.grad, p_hyb.grad), (
                f"Param '{name}' grad mismatch: max diff = "
                f"{(p_ref.grad - p_hyb.grad).abs().max().item()}"
            )


@requires_fp8_and_nvfp4
class TestHybridGemmMixedFormat:
    """FP8 rowwise + NVFP4 columnwise through te.Linear forward+backward."""

    def test_linear_fwd_bwd_fp8_row_nvfp4_col(self):
        torch.manual_seed(42)

        in_features = 128
        out_features = 128
        batch = 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch,
            in_features,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        def mixed_factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return _make_nvfp4_quantizer()
            return _make_fp8_quantizer()

        mixed_recipe = recipe.CustomRecipe(qfactory=mixed_factory)

        with autocast(enabled=True, recipe=mixed_recipe):
            out = model(inp)

        assert out.shape == (batch, out_features)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None, "Input gradient is None"
        assert inp.grad.shape == inp.shape
        assert not torch.isnan(inp.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(inp.grad).any(), "Input gradient contains Inf"

        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"
            assert not torch.isnan(p.grad).any(), f"Gradient for '{name}' contains NaN"
            assert not torch.isinf(p.grad).any(), f"Gradient for '{name}' contains Inf"

    def test_numerical_sanity_against_bf16(self):
        """Mixed-format output should be within reasonable tolerance of BF16 baseline."""
        torch.manual_seed(42)

        in_features = 128
        out_features = 128
        batch = 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)

        # BF16 baseline (no quantization)
        with torch.no_grad():
            out_bf16 = model(inp)

        def mixed_factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            return _make_fp8_quantizer()

        mixed_recipe = recipe.CustomRecipe(qfactory=mixed_factory)
        with torch.no_grad():
            with autocast(enabled=True, recipe=mixed_recipe):
                out_mixed = model(inp)

        # FP8/FP4 quantization introduces error, but the result should be
        # in the same ballpark as BF16
        torch.testing.assert_close(
            out_mixed.float(),
            out_bf16.float(),
            rtol=0.25,
            atol=0.5,
        )


@requires_fp8_and_nvfp4
class TestUnwrapHybridDirection:
    """Test per-operand unwrap selects the correct sub-storage.

    Operand A: transposed (layout[0]=='T') → rowwise, else → columnwise
    Operand B: not-transposed (layout[1]=='N') → rowwise, else → columnwise
    """

    @pytest.fixture
    def hybrid_tensor(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        return hq.quantize(inp)

    def test_A_tn_returns_rowwise(self, hybrid_tensor):
        assert _unwrap_hybrid_A(hybrid_tensor, "TN") is hybrid_tensor.rowwise_sub_storage

    def test_A_nn_returns_columnwise(self, hybrid_tensor):
        assert _unwrap_hybrid_A(hybrid_tensor, "NN") is hybrid_tensor.columnwise_sub_storage

    def test_A_nt_returns_columnwise(self, hybrid_tensor):
        assert _unwrap_hybrid_A(hybrid_tensor, "NT") is hybrid_tensor.columnwise_sub_storage

    def test_B_tn_returns_rowwise(self, hybrid_tensor):
        assert _unwrap_hybrid_B(hybrid_tensor, "TN") is hybrid_tensor.rowwise_sub_storage

    def test_B_nn_returns_rowwise(self, hybrid_tensor):
        assert _unwrap_hybrid_B(hybrid_tensor, "NN") is hybrid_tensor.rowwise_sub_storage

    def test_B_nt_returns_columnwise(self, hybrid_tensor):
        assert _unwrap_hybrid_B(hybrid_tensor, "NT") is hybrid_tensor.columnwise_sub_storage

    def test_tn_sub_storage_type(self, hybrid_tensor):
        assert isinstance(
            _unwrap_hybrid_A(hybrid_tensor, "TN"),
            (Float8TensorStorage, Float8Tensor),
        )

    def test_nt_sub_storage_type(self, hybrid_tensor):
        assert isinstance(
            _unwrap_hybrid_B(hybrid_tensor, "NT"),
            (NVFP4TensorStorage, NVFP4Tensor),
        )

    def test_non_hybrid_passthrough(self):
        plain = torch.randn(4, 4, device="cuda")
        for layout in ("TN", "NN", "NT"):
            assert _unwrap_hybrid_A(plain, layout) is plain
            assert _unwrap_hybrid_B(plain, layout) is plain

    def test_fp8_tensor_passthrough(self):
        quantizer = _make_fp8_quantizer()
        inp = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
        fp8 = quantizer.quantize(inp)
        for layout in ("TN", "NN", "NT"):
            assert _unwrap_hybrid_A(fp8, layout) is fp8
            assert _unwrap_hybrid_B(fp8, layout) is fp8


@requires_fp8
class TestHybridBiasGradient:
    """Verify bias gradients are computed correctly with HybridQuantizer.

    tex.bgrad_quantize doesn't recognize HybridQuantizer, so the unfused
    bgrad path is used instead.
    """

    def _make_uniform_hybrid_factory(self):
        def factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return HybridQuantizer(
                rowwise_quantizer=Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                ),
                columnwise_quantizer=Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                ),
            )

        return factory

    def test_bias_grad_matches_vanilla_fp8(self):
        torch.manual_seed(456)
        in_features, out_features, batch = 64, 64, 16

        model_ref = Linear(in_features, out_features, bias=True, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(
            in_features, out_features, bias=True, params_dtype=torch.bfloat16
        ).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)

        # Reference
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        with autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        # Hybrid
        inp_hyb = base_inp.clone().detach().requires_grad_(True)
        with autocast(
            enabled=True, recipe=recipe.CustomRecipe(qfactory=self._make_uniform_hybrid_factory())
        ):
            out_hyb = model_hybrid(inp_hyb)
        out_hyb.float().sum().backward()

        ref_bias_grad = dict(model_ref.named_parameters())["bias"].grad
        hyb_bias_grad = dict(model_hybrid.named_parameters())["bias"].grad
        assert ref_bias_grad is not None and hyb_bias_grad is not None
        assert torch.equal(
            ref_bias_grad, hyb_bias_grad
        ), f"Bias grad mismatch: max diff = {(ref_bias_grad - hyb_bias_grad).abs().max().item()}"

    def test_no_bias_fwd_bwd(self):
        """Linear with bias=False skips bgrad_quantize entirely."""
        torch.manual_seed(42)
        in_features, out_features, batch = 64, 64, 16

        model = Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        with autocast(
            enabled=True, recipe=recipe.CustomRecipe(qfactory=self._make_uniform_hybrid_factory())
        ):
            out = model(inp)
        out.float().sum().backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"


@requires_fp8_and_nvfp4
class TestHybridScalingModeCompatibility:
    """cuBLAS requires matching scaling modes within a single GEMM.

    For hybrid quantization, this means the columnwise format for
    linear_input/linear_weight must match the columnwise format for
    linear_grad_output — otherwise the wgrad GEMM (NT layout) fails.
    """

    def test_matching_columnwise_formats_succeed(self):
        """Both operands use NVFP4 columnwise → wgrad GEMM succeeds."""
        torch.manual_seed(42)
        # NVFP4 GEMM requires dimensions ≥ 128 for cuBLAS support.
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return _make_nvfp4_quantizer()
            return _make_fp8_quantizer()

        with autocast(enabled=True, recipe=recipe.CustomRecipe(qfactory=factory)):
            out = model(inp)
        out.float().sum().backward()
        assert inp.grad is not None

    def test_mismatched_columnwise_formats_raise(self):
        """NVFP4 input × FP8 grad_output columnwise → cuBLAS rejects."""
        torch.manual_seed(42)
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return _make_fp8_quantizer()

        with autocast(enabled=True, recipe=recipe.CustomRecipe(qfactory=factory)):
            out = model(inp)
        with pytest.raises(RuntimeError, match="scaling_mode"):
            out.float().sum().backward()


@requires_fp8_and_nvfp4
class TestHybridReversedDirection:
    """Reversed hybrid: NVFP4 rowwise (fprop) + FP8 columnwise (backward).

    Exercises NVFP4×NVFP4 in the fprop (TN) GEMM and FP8×FP8 in the
    dgrad (NN) and wgrad (NT) GEMMs — the opposite of the primary
    FP8-row/NVFP4-col configuration.
    """

    def test_nvfp4_row_fp8_col_forward_only(self):
        """Forward (TN) with NVFP4×NVFP4 rowwise succeeds."""
        torch.manual_seed(99)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)

        def factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            return _make_nvfp4_quantizer()

        mixed_recipe = recipe.CustomRecipe(qfactory=factory)
        with torch.no_grad():
            with autocast(enabled=True, recipe=mixed_recipe):
                out = model(inp)

        assert out.shape == (batch, out_features)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_nvfp4_row_fp8_col_full_fwd_bwd(self):
        """Full fwd+bwd with NVFP4 rowwise (fprop) + FP8 columnwise (backward)."""
        torch.manual_seed(99)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("input", "weight")
            ):
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if (
                role is not None
                and role.module_type in ("linear", "grouped_linear")
                and role.tensor_type in ("grad_output", "grad_input")
            ):
                return _make_fp8_quantizer()
            return _make_nvfp4_quantizer()

        mixed_recipe = recipe.CustomRecipe(qfactory=factory)
        with autocast(enabled=True, recipe=mixed_recipe):
            out = model(inp)

        assert out.shape == (batch, out_features)
        assert not torch.isnan(out).any(), "Output contains NaN"

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None, "Input gradient is None"
        assert not torch.isnan(inp.grad).any(), "Input gradient contains NaN"
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"
            assert not torch.isnan(p.grad).any(), f"Gradient for '{name}' contains NaN"


@requires_fp8
class TestHybridMixedWithNonHybrid:
    """Only one operand is hybrid; the other uses a plain TE quantizer.

    Exercises _unwrap_hybrid passthrough for the non-hybrid operand.
    All roles must use compatible scaling modes for each GEMM:
      fprop (TN): all rowwise formats must match
      dgrad (NN): weight rowwise must match grad_output rowwise
      wgrad (NT): input columnwise must match grad_output columnwise
    """

    @staticmethod
    def _assert_native_fp8_parity(hybrid_role, *, seed):
        torch.manual_seed(seed)
        in_features, out_features, batch = 128, 128, 32
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_native = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_native.load_state_dict(model_hybrid.state_dict())
        base_input = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch, out_features, device="cuda", dtype=torch.bfloat16)

        def mixed_factory(role):
            is_linear = role is not None and role.module_type in (
                "linear",
                "grouped_linear",
            )
            if is_linear and role.tensor_type == hybrid_role:
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if is_linear and role.tensor_type in ("grad_output", "grad_input"):
                return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

        hybrid_result = _run_linear_forward_backward(
            model_hybrid,
            base_input,
            grad_output,
            recipe.CustomRecipe(qfactory=mixed_factory),
            seed=seed + 100,
        )
        native_result = _run_linear_forward_backward(
            model_native,
            base_input,
            grad_output,
            recipe.Float8CurrentScaling(),
            seed=seed + 100,
        )
        _assert_linear_results_exact(
            hybrid_result,
            native_result,
            output=True,
            input_grad=True,
            param_grads=True,
        )

    def test_hybrid_input_plain_weight_fwd_bwd(self):
        """Input is hybrid (FP8 row / FP8 col), weight + grad_output plain FP8.

        Wgrad columnwise: FP8 (input.col) × FP8 (grad_output.col) → compatible.
        """
        self._assert_native_fp8_parity("input", seed=77)

    def test_plain_input_hybrid_weight_fwd_bwd(self):
        """Input is plain FP8, weight is hybrid (FP8 row / FP8 col)."""
        self._assert_native_fp8_parity("weight", seed=88)


# ---------------------------------------------------------------------------
# Parametrized cross-format tests (stateless quantizers)
# ---------------------------------------------------------------------------


def _make_mxfp8_quantizer(*, rowwise=True, columnwise=True):
    return MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _make_mxfp8_quantizer_e5m2(*, rowwise=True, columnwise=True):
    return MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E5M2,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _make_block_quantizer(*, rowwise=True, columnwise=True):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _make_block_quantizer_e5m2(*, rowwise=True, columnwise=True):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E5M2,
        rowwise=rowwise,
        columnwise=columnwise,
    )


# (fwd_e4m3_factory, bwd_e5m2_factory, skip_condition, skip_reason)
_QUANTIZER_CONFIGS = {
    "fp8_current": (
        _make_fp8_quantizer,
        lambda **kw: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda", **kw),
        not fp8_available,
        f"FP8: {reason_for_no_fp8}",
    ),
    "mxfp8": (
        _make_mxfp8_quantizer,
        _make_mxfp8_quantizer_e5m2,
        not mxfp8_available,
        f"MXFP8: {reason_for_no_mxfp8}",
    ),
    "block_fp8": (
        _make_block_quantizer,
        _make_block_quantizer_e5m2,
        not fp8_block_scaling_available,
        reason_for_no_fp8_block_scaling,
    ),
    "nvfp4": (
        _make_nvfp4_quantizer,
        None,  # NVFP4 has no E5M2 variant
        not (fp8_available and nvfp4_available),
        f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
    ),
}


def _build_cross_format_params():
    """Build parametrize list for all stateless cross-format hybrid combos."""
    combos = [
        ("fp8_current", "mxfp8"),
        ("fp8_current", "nvfp4"),
        ("fp8_current", "block_fp8"),
        ("mxfp8", "fp8_current"),
        ("mxfp8", "mxfp8"),
        ("mxfp8", "nvfp4"),
        ("mxfp8", "block_fp8"),
        ("block_fp8", "fp8_current"),
        ("block_fp8", "mxfp8"),
        ("block_fp8", "nvfp4"),
        ("block_fp8", "block_fp8"),
        ("nvfp4", "fp8_current"),
        ("nvfp4", "mxfp8"),
        ("nvfp4", "block_fp8"),
    ]
    params = []
    for row, col in combos:
        row_cfg = _QUANTIZER_CONFIGS[row]
        col_cfg = _QUANTIZER_CONFIGS[col]
        hw_skip = row_cfg[2] or col_cfg[2]
        hw_reason = "; ".join(
            filter(None, [row_cfg[3] if row_cfg[2] else "", col_cfg[3] if col_cfg[2] else ""])
        )
        marks = []
        if hw_skip:
            marks.append(pytest.mark.skipif(True, reason=hw_reason or "N/A"))
        params.append(pytest.param(row, col, id=f"{row}_row_x_{col}_col", marks=marks))
    return params


def _set_quantization_test_seed(seed):
    """Reset every RNG that a quantizer or GEMM helper may consume."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_linear_forward(model, base_input, fp8_recipe, *, seed):
    """Run a deterministic forward pass and return a detached result."""
    _set_quantization_test_seed(seed)
    with torch.no_grad():
        with autocast(enabled=True, recipe=fp8_recipe):
            output = model(base_input)
    return output.detach().clone()


def _run_linear_forward_backward(model, base_input, grad_output, fp8_recipe, *, seed):
    """Run Linear with an external gradient and capture every numerical result."""
    model.zero_grad(set_to_none=True)
    run_input = base_input.clone().detach().requires_grad_(True)
    _set_quantization_test_seed(seed)
    with autocast(enabled=True, recipe=fp8_recipe):
        output = model(run_input)
    output.backward(grad_output)
    return (
        output.detach().clone(),
        run_input.grad.detach().clone(),
        {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        },
    )


def _plain_linear_qfactory(operand_factory, grad_factory):
    """Build a non-hybrid factory with role-correct operand and grad dtypes."""

    def factory(role):
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        if is_linear and role.tensor_type in ("grad_output", "grad_input"):
            return grad_factory()
        return operand_factory()

    return factory


def _assert_linear_results_exact(actual, expected, *, output, input_grad, param_grads):
    """Compare selected Linear results with zero tolerance."""
    if output:
        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    if input_grad:
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    if param_grads:
        assert actual[2].keys() == expected[2].keys()
        for name in actual[2]:
            torch.testing.assert_close(
                actual[2][name],
                expected[2][name],
                rtol=0.0,
                atol=0.0,
                msg=f"Parameter gradient {name!r} differs",
            )


class TestHybridCrossFormatParametrized:
    """Parametrized fwd+bwd over all stateless quantizer cross-format pairs."""

    @pytest.mark.parametrize("row_name,col_name", _build_cross_format_params())
    def test_fwd_bwd(self, row_name, col_name):
        torch.manual_seed(42)
        in_features, out_features, batch = 128, 128, 32

        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_fprop_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_bwd_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_fprop_ref.load_state_dict(model_hybrid.state_dict())
        model_bwd_ref.load_state_dict(model_hybrid.state_dict())

        base_input = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch, out_features, device="cuda", dtype=torch.bfloat16)

        row_cfg = _QUANTIZER_CONFIGS[row_name]
        col_cfg = _QUANTIZER_CONFIGS[col_name]
        make_row_operand = row_cfg[0]
        make_row_grad = row_cfg[1] if row_cfg[1] is not None else row_cfg[0]
        make_col_operand = col_cfg[0]
        make_col_grad = col_cfg[1] if col_cfg[1] is not None else col_cfg[0]

        def hybrid_factory(role):
            is_linear = role is not None and role.module_type in (
                "linear",
                "grouped_linear",
            )
            if is_linear and role.tensor_type in ("input", "weight"):
                return HybridQuantizer(
                    rowwise_quantizer=make_row_operand(),
                    columnwise_quantizer=make_col_operand(),
                )
            if is_linear and role.tensor_type in ("grad_output", "grad_input"):
                return make_col_grad()
            return make_row_operand()

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_factory)
        fprop_ref_recipe = recipe.CustomRecipe(
            qfactory=_plain_linear_qfactory(make_row_operand, make_row_grad)
        )
        bwd_ref_recipe = recipe.CustomRecipe(
            qfactory=_plain_linear_qfactory(make_col_operand, make_col_grad)
        )

        hybrid_result = _run_linear_forward_backward(
            model_hybrid,
            base_input,
            grad_output,
            hybrid_recipe,
            seed=1234,
        )
        fprop_ref = _run_linear_forward(
            model_fprop_ref,
            base_input,
            fprop_ref_recipe,
            seed=1234,
        )
        bwd_ref = _run_linear_forward_backward(
            model_bwd_ref,
            base_input,
            grad_output,
            bwd_ref_recipe,
            seed=1234,
        )

        torch.testing.assert_close(hybrid_result[0], fprop_ref, rtol=0.0, atol=0.0)
        _assert_linear_results_exact(
            hybrid_result,
            bwd_ref,
            output=False,
            input_grad=True,
            param_grads=True,
        )


# ---------------------------------------------------------------------------
# CPU offload push/pop protocol (v2 OffloadableLayerState path)
# ---------------------------------------------------------------------------


class TestHybridCpuOffloadPushPop:
    """Exercise the cpu_offload_v2 push/pop protocol on HybridQuantizedTensor.

    Uses :class:`OffloadableLayerState` directly — same pattern as
    ``test_cpu_offloading.py::TestsOffloadableLayerState::test_general``.
    Each test runs the full cycle:

        push → start_offload → release_activation_forward_gpu_memory
        → start_reload → pop → release_all_memory

    The push path decomposes the hybrid via ``prepare_for_saving``
    (HybridQuantizedTensorStorage), recursively pushes each sub-storage
    buffer, then reconstructs on pop via ``restore_from_saved``. Sub-buffers
    below the 256K-element offload threshold (e.g. small block scales) are
    returned unchanged; large data buffers round-trip through CPU.
    """

    # Hybrid tensor shape — each sub-storage primary buffer must exceed the
    # cpu_offload _check_if_offload threshold (256K elements) so the path is
    # actually exercised end-to-end.
    _SHAPE = (1024, 1024)

    def _run_roundtrip(self, hybrid_tensor):
        """Push → offload → release → reload → pop one hybrid tensor.

        Returns the reloaded tensor (a new HybridQuantizedTensor instance
        reconstructed from the gathered-back buffers).
        """
        from transformer_engine.pytorch.cpu_offload import OffloadableLayerState

        stream = torch.cuda.Stream()
        state = OffloadableLayerState(offload_stream=stream)

        tid = state.push_tensor(hybrid_tensor)
        state.start_offload()
        state.release_activation_forward_gpu_memory()
        state.start_reload()
        reloaded = state.pop_tensor(tid)
        torch.cuda.synchronize()

        try:
            return reloaded
        finally:
            state.release_all_memory()

    @pytest.mark.parametrize("row_name,col_name", _build_cross_format_params())
    def test_push_pop_roundtrip(self, row_name, col_name):
        """Dequantize-equivalence round-trip across the full 14-pair matrix."""
        torch.manual_seed(42)
        inp = torch.randn(*self._SHAPE, dtype=torch.bfloat16, device="cuda")

        row_cfg = _QUANTIZER_CONFIGS[row_name]
        col_cfg = _QUANTIZER_CONFIGS[col_name]
        hq = HybridQuantizer(
            rowwise_quantizer=row_cfg[0](),
            columnwise_quantizer=col_cfg[0](),
        )
        hybrid = hq.quantize(inp)
        expected = hybrid.dequantize()

        reloaded = self._run_roundtrip(hybrid)

        _assert_hybrid_tensor_exact(reloaded, hybrid, context="CPU offload roundtrip")
        assert isinstance(reloaded, HybridQuantizedTensor)
        torch.testing.assert_close(reloaded.dequantize(), expected, rtol=0.0, atol=0.0)

    @requires_fp8_and_nvfp4
    def test_push_pop_preserves_sub_storage_types(self):
        """Reconstructed hybrid preserves each sub-storage's concrete type."""
        torch.manual_seed(7)
        inp = torch.randn(*self._SHAPE, dtype=torch.bfloat16, device="cuda")

        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hybrid = hq.quantize(inp)
        row_type = type(hybrid.rowwise_sub_storage)
        col_type = type(hybrid.columnwise_sub_storage)

        reloaded = self._run_roundtrip(hybrid)

        assert isinstance(reloaded.rowwise_sub_storage, row_type)
        _assert_hybrid_tensor_exact(reloaded, hybrid, context="CPU offload storage types")
        assert isinstance(reloaded.columnwise_sub_storage, col_type)

    @requires_fp8_and_nvfp4
    def test_push_pop_with_rowwise_only(self):
        """Columnwise sub-storage dropped pre-push — roundtrip still works."""
        torch.manual_seed(11)
        inp = torch.randn(*self._SHAPE, dtype=torch.bfloat16, device="cuda")

        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hybrid = hq.quantize(inp)
        hybrid.update_usage(columnwise_usage=False)
        assert hybrid.columnwise_sub_storage is None
        expected = hybrid.dequantize()

        reloaded = self._run_roundtrip(hybrid)

        assert isinstance(reloaded, HybridQuantizedTensor)
        assert reloaded.columnwise_sub_storage is None
        _assert_hybrid_tensor_exact(reloaded, hybrid, context="CPU offload rowwise-only")
        assert reloaded.rowwise_sub_storage is not None
        torch.testing.assert_close(reloaded.dequantize(), expected, rtol=0.0, atol=0.0)

    @requires_fp8_and_nvfp4
    def test_push_pop_with_columnwise_only(self):
        """Rowwise sub-storage dropped pre-push — roundtrip still works.

        Uses the reversed hybrid (NVFP4 rowwise + FP8 columnwise) so that
        ``hybrid.dequantize()`` can fall through to the columnwise sub-storage.
        ``HybridQuantizedTensorStorage.dequantize`` prefers rowwise and only
        falls back to columnwise when rowwise is ``None``; NVFP4 does not yet
        support columnwise-only dequantize, but Float8 does.
        """
        torch.manual_seed(13)
        inp = torch.randn(*self._SHAPE, dtype=torch.bfloat16, device="cuda")

        hq = _make_hybrid_quantizer_fp4_row_fp8_col()
        hybrid = hq.quantize(inp)
        hybrid.update_usage(rowwise_usage=False)
        assert hybrid.rowwise_sub_storage is None
        expected = hybrid.dequantize()

        reloaded = self._run_roundtrip(hybrid)

        assert isinstance(reloaded, HybridQuantizedTensor)
        assert reloaded.rowwise_sub_storage is None
        _assert_hybrid_tensor_exact(reloaded, hybrid, context="CPU offload columnwise-only")
        assert reloaded.columnwise_sub_storage is not None
        torch.testing.assert_close(reloaded.dequantize(), expected, rtol=0.0, atol=0.0)

    @requires_fp8_and_nvfp4
    def test_push_pop_roundtrip_does_not_leak_intermediate_buffers(self):
        """After release_all_memory the offloader holds no hybrid buffers.

        Sanity check that the v2 cycle completes cleanly — no dangling CPU
        pinned buffers left behind on a one-shot push/pop.
        """
        from transformer_engine.pytorch.cpu_offload import OffloadableLayerState

        torch.manual_seed(17)
        inp = torch.randn(*self._SHAPE, dtype=torch.bfloat16, device="cuda")

        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        hybrid = hq.quantize(inp)

        stream = torch.cuda.Stream()
        state = OffloadableLayerState(offload_stream=stream)

        tid = state.push_tensor(hybrid)
        state.start_offload()
        state.release_activation_forward_gpu_memory()
        state.start_reload()
        _ = state.pop_tensor(tid)
        torch.cuda.synchronize()
        state.release_all_memory()

        assert len(state.fwd_gpu_tensor_group.tensor_list) == 0
        assert len(state.cpu_tensor_group.tensor_list) == 0
        assert len(state.bwd_gpu_tensor_group.tensor_list) == 0
        assert state.state == "not_offloaded"


# ---------------------------------------------------------------------------
# 3-format hybrid: different quantization for fprop, dgrad, wgrad
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (fp8_available and mxfp8_available and nvfp4_available),
    reason="Requires FP8 + MXFP8 + NVFP4",
)
class TestHybridThreeFormats:
    """Three distinct formats: FormatA (fprop), FormatB (dgrad), FormatC (wgrad).

    Per-operand unwrap selects the correct sub-storage per GEMM:
      fprop TN: weight.row(A) × input.row(A)       → FormatA × FormatA
      dgrad NN: weight.col(B) × grad_output.row(B)  → FormatB × FormatB
      wgrad NT: input.col(C)  × grad_output.col(C)  → FormatC × FormatC

    grad_output is itself hybrid (FormatB row + FormatC col) when B ≠ C.
    """

    @staticmethod
    def _assert_three_format_routing(
        make_fprop, make_dgrad, make_wgrad, *, seed, plain_grad_output=False
    ):
        in_features, out_features, batch = 128, 128, 32
        torch.manual_seed(seed)
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_fprop_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_dgrad_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_wgrad_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        state_dict = model_hybrid.state_dict()
        model_fprop_ref.load_state_dict(state_dict)
        model_dgrad_ref.load_state_dict(state_dict)
        model_wgrad_ref.load_state_dict(state_dict)

        base_input = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch, out_features, device="cuda", dtype=torch.bfloat16)

        def hybrid_factory(role):
            is_linear = role is not None and role.module_type in (
                "linear",
                "grouped_linear",
            )
            if is_linear and role.tensor_type == "weight":
                return HybridQuantizer(
                    rowwise_quantizer=make_fprop(),
                    columnwise_quantizer=make_dgrad(),
                )
            if is_linear and role.tensor_type == "input":
                return HybridQuantizer(
                    rowwise_quantizer=make_fprop(),
                    columnwise_quantizer=make_wgrad(),
                )
            if is_linear and role.tensor_type in ("grad_output", "grad_input"):
                if plain_grad_output:
                    return make_dgrad()
                return HybridQuantizer(
                    rowwise_quantizer=make_dgrad(),
                    columnwise_quantizer=make_wgrad(),
                )
            return make_fprop()

        hybrid_result = _run_linear_forward_backward(
            model_hybrid,
            base_input,
            grad_output,
            recipe.CustomRecipe(qfactory=hybrid_factory),
            seed=seed + 100,
        )
        fprop_ref = _run_linear_forward(
            model_fprop_ref,
            base_input,
            recipe.CustomRecipe(qfactory=_plain_linear_qfactory(make_fprop, make_fprop)),
            seed=seed + 100,
        )
        dgrad_ref = _run_linear_forward_backward(
            model_dgrad_ref,
            base_input,
            grad_output,
            recipe.CustomRecipe(qfactory=_plain_linear_qfactory(make_dgrad, make_dgrad)),
            seed=seed + 100,
        )
        wgrad_ref = _run_linear_forward_backward(
            model_wgrad_ref,
            base_input,
            grad_output,
            recipe.CustomRecipe(qfactory=_plain_linear_qfactory(make_wgrad, make_wgrad)),
            seed=seed + 100,
        )

        torch.testing.assert_close(hybrid_result[0], fprop_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(hybrid_result[1], dgrad_ref[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            hybrid_result[2]["weight"],
            wgrad_ref[2]["weight"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            hybrid_result[2]["bias"],
            dgrad_ref[2]["bias"],
            rtol=0.0,
            atol=0.0,
        )

    def test_fp8_fprop_mxfp8_dgrad_nvfp4_wgrad(self):
        """FP8 current (fprop) + MXFP8 (dgrad) + NVFP4 (wgrad)."""
        self._assert_three_format_routing(
            _make_fp8_quantizer,
            _make_mxfp8_quantizer,
            _make_nvfp4_quantizer,
            seed=300,
        )

    def test_nvfp4_fprop_fp8_dgrad_mxfp8_wgrad(self):
        """NVFP4 (fprop) + FP8 current (dgrad) + MXFP8 (wgrad)."""
        self._assert_three_format_routing(
            _make_nvfp4_quantizer,
            _make_fp8_quantizer,
            _make_mxfp8_quantizer,
            seed=301,
        )

    def test_same_dgrad_wgrad_reduces_to_plain_grad(self):
        """When dgrad format == wgrad format, grad_output can be a plain quantizer."""
        self._assert_three_format_routing(
            _make_nvfp4_quantizer,
            _make_mxfp8_quantizer,
            _make_mxfp8_quantizer,
            plain_grad_output=True,
            seed=302,
        )


# ---------------------------------------------------------------------------
# All-modules test: hybrid quantization through every TE module type
# ---------------------------------------------------------------------------


def _make_hybrid_fp8_factory():
    """Factory returning HybridQuantizer(FP8 row + FP8 col) for fwd roles,
    plain FP8 E5M2 for bwd roles."""

    def factory(role):
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        if is_linear and role.tensor_type in ("input", "weight", "output"):
            return HybridQuantizer(
                rowwise_quantizer=Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                ),
                columnwise_quantizer=Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                ),
            )
        if is_linear and role.tensor_type in ("grad_output", "grad_input"):
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2,
                device="cuda",
            )
        return Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3,
            device="cuda",
        )

    return factory


@requires_fp8
class TestHybridAllModules:
    """Hybrid quantization through all TE module types (not just Linear).

    Uses FP8 in both hybrid directions so the test validates module integration
    without introducing cross-format scaling-mode concerns.
    """

    hidden_size = 128
    ffn_hidden_size = 128
    num_heads = 4
    batch = 16
    seq_len = 8

    def _run_fwd_bwd(self, model, inp, *model_args, output_atol=0.0, param_atols=None):
        """Compare same-format hybrid numerics against the native FP8 recipe."""
        import copy

        model_native = copy.deepcopy(model)
        param_atols = {} if param_atols is None else param_atols
        grad_output = torch.randn_like(inp)

        def run(run_model, run_recipe):
            run_model.zero_grad(set_to_none=True)
            run_input = inp.detach().clone().requires_grad_(True)
            _set_quantization_test_seed(3370)
            with autocast(enabled=True, recipe=run_recipe):
                output = run_model(run_input, *model_args)
            output.backward(grad_output)
            return (
                output.detach().clone(),
                run_input.grad.detach().clone(),
                {
                    name: param.grad.detach().clone()
                    for name, param in run_model.named_parameters()
                    if param.grad is not None
                },
            )

        native_result = run(model_native, recipe.Float8CurrentScaling())
        hybrid_result = run(
            model,
            recipe.CustomRecipe(qfactory=_make_hybrid_fp8_factory()),
        )

        torch.testing.assert_close(hybrid_result[0], native_result[0], rtol=0.0, atol=output_atol)
        torch.testing.assert_close(hybrid_result[1], native_result[1], rtol=0.0, atol=0.0)
        assert hybrid_result[2].keys() == native_result[2].keys()
        for name in hybrid_result[2]:
            torch.testing.assert_close(
                hybrid_result[2][name],
                native_result[2][name],
                rtol=0.0,
                atol=param_atols.get(name, 0.0),
                msg=f"Parameter gradient {name!r} differs",
            )

    def test_linear(self):
        torch.manual_seed(500)
        model = Linear(
            self.hidden_size,
            self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        self._run_fwd_bwd(model, inp)

    def test_layernorm_linear(self):
        torch.manual_seed(501)
        model = LayerNormLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        self._run_fwd_bwd(model, inp)

    def test_layernorm_mlp(self):
        torch.manual_seed(502)
        model = LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        # Native fuses LN+quantize while Hybrid takes the explicit quantize path.
        # The only non-bitwise results are output (1 BF16 quantum here) and
        # fc2_weight grad; all other gradients remain zero-tolerance checks.
        self._run_fwd_bwd(
            model,
            inp,
            output_atol=0.0009765625,
            param_atols={"fc2_weight": 0.0234375},
        )

    def test_grouped_linear(self):
        torch.manual_seed(504)
        num_gemms = 3
        model = GroupedLinear(
            num_gemms,
            self.hidden_size,
            self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        base = self.batch // num_gemms
        rem = self.batch % num_gemms
        m_splits = [base + (1 if i < rem else 0) for i in range(num_gemms)]

        self._run_fwd_bwd(model, inp, m_splits)

    def test_transformer_layer(self):
        torch.manual_seed(503)
        model = TransformerLayer(
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            fuse_qkv_params=True,
            params_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(
            self.seq_len,
            self.batch,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        # The LayerNormMLP submodule has the same fused-vs-explicit ordering.
        # Keep dgrad and every other parameter exact; bound only the observed
        # BF16 output and final projection-weight accumulation roundoff.
        self._run_fwd_bwd(
            model,
            inp,
            output_atol=0.015625,
            param_atols={"layernorm_mlp.fc2_weight": 0.140625},
        )


@requires_fp8
class TestHybridGroupedLinearValidation:
    """GroupedLinear generation-validation and split-dispatch coverage.

    Structural compatibility is validated once per real quantizer generation.
    Steady-state dispatch reads the first expert after that uniformity check."""

    @pytest.mark.parametrize(
        "quantizers",
        [
            pytest.param(
                [_make_hybrid_quantizer_fp8_row_fp4_col() for _ in range(3)],
                id="hybrid",
            ),
            pytest.param([_make_fp8_quantizer() for _ in range(3)], id="plain"),
            pytest.param([None, None, None], id="none"),
        ],
    )
    def test_uniform_lists_validate(self, quantizers):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_mixed_hybrid_and_plain_raises(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [
            _make_hybrid_quantizer_fp8_row_fp4_col(),
            _make_fp8_quantizer(),
            _make_hybrid_quantizer_fp8_row_fp4_col(),
        ]
        with pytest.raises(ValueError, match="mix HybridQuantizer and non-hybrid"):
            _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_none_plus_hybrid_raises(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [
            _make_hybrid_quantizer_fp8_row_fp4_col(),
            None,
            _make_hybrid_quantizer_fp8_row_fp4_col(),
        ]
        with pytest.raises(ValueError, match="mix None and concrete quantizers"):
            _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_mixed_identity_dtype_raises(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [
            IdentityQuantizer(dtype=torch.bfloat16),
            IdentityQuantizer(dtype=torch.float16),
        ]
        with pytest.raises(ValueError, match="incompatible plain backend configurations"):
            _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_distinct_delayed_scaling_state_is_allowed(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [_make_delayed_quantizer(), _make_delayed_quantizer()]
        quantizers[1].scale.fill_(2.0)
        quantizers[1].amax.fill_(3.0)
        _validate_grouped_quantizer_list(quantizers, operand_name="input")

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            pytest.param((True, False), {"rowwise": True, "columnwise": False}, id="rowwise"),
            pytest.param((False, True), {"rowwise": False, "columnwise": True}, id="columnwise"),
            pytest.param((True, True), {"rowwise": True, "columnwise": True}, id="both"),
        ],
    )
    def test_hybrid_split_quantize_respects_parent_usage_flags(self, usage, expected):
        from transformer_engine.pytorch.module.grouped_linear import (
            _hybrid_split_quantize,
        )

        tensor = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
            )
            for _ in range(2)
        ]
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=usage[0], columnwise=usage[1])

        out = _hybrid_split_quantize(tensor, [16, 16], quantizers)

        assert [storage.get_usages() for storage in out] == [expected, expected]

    def test_columnwise_only_rowwise_dequantized_uses_transient_grouped_row(self, monkeypatch):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear

        real_split_quantize = grouped_linear.tex.split_quantize
        calls = []

        def tracked_split_quantize(tensor, m_splits, quantizers, **kwargs):
            result = real_split_quantize(tensor, m_splits, quantizers, **kwargs)
            calls.append((tensor, result))
            return result

        monkeypatch.setattr(grouped_linear.tex, "split_quantize", tracked_split_quantize)
        tensor = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source="rowwise_dequantized",
            )
            for _ in range(2)
        ]
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=False, columnwise=True)

        out = grouped_linear._hybrid_split_quantize(tensor, [16, 16], quantizers)

        assert len(calls) == 2
        expected_columnwise_source = torch.cat(
            [result.dequantize() for result in calls[0][1]], dim=0
        )
        torch.testing.assert_close(calls[1][0], expected_columnwise_source, rtol=0.0, atol=0.0)
        assert calls[1][0] is not tensor
        assert all(storage.rowwise_sub_storage is None for storage in out)
        assert all(storage.columnwise_sub_storage is not None for storage in out)

    def test_rowwise_only_skips_columnwise_quantization(self, monkeypatch):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear

        real_split_quantize = grouped_linear.tex.split_quantize
        calls = []

        def tracked_split_quantize(tensor, m_splits, quantizers, **kwargs):
            calls.append(tensor)
            return real_split_quantize(tensor, m_splits, quantizers, **kwargs)

        monkeypatch.setattr(grouped_linear.tex, "split_quantize", tracked_split_quantize)
        tensor = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source="rowwise_dequantized",
            )
            for _ in range(2)
        ]
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)

        out = grouped_linear._hybrid_split_quantize(tensor, [16, 16], quantizers)

        assert calls == [tensor]
        assert all(storage.rowwise_sub_storage is not None for storage in out)
        assert all(storage.columnwise_sub_storage is None for storage in out)

    def test_original_source_preserves_two_bulk_call_fast_path(self, monkeypatch):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear

        real_split_quantize = grouped_linear.tex.split_quantize
        calls = []

        def tracked_split_quantize(tensor, m_splits, quantizers, **kwargs):
            calls.append(tensor)
            return real_split_quantize(tensor, m_splits, quantizers, **kwargs)

        monkeypatch.setattr(grouped_linear.tex, "split_quantize", tracked_split_quantize)
        tensor = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source="original",
            )
            for _ in range(2)
        ]

        out = grouped_linear._hybrid_split_quantize(tensor, [16, 16], quantizers)

        assert calls == [tensor, tensor]
        assert all(storage.rowwise_sub_storage is not None for storage in out)
        assert all(storage.columnwise_sub_storage is not None for storage in out)

    def test_validation_rejects_mixed_columnwise_source_policies(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source=source,
            )
            for source in ("original", "rowwise_dequantized")
        ]

        with pytest.raises(ValueError, match="mixed columnwise source policies"):
            _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_validation_rejects_same_family_config_mismatch(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _validate_grouped_quantizer_list,
        )

        quantizers = [_make_fp8_quantizer(), _make_fp8_quantizer()]
        quantizers[1].force_pow_2_scales = True

        with pytest.raises(
            ValueError,
            match="incompatible plain backend configurations",
        ):
            _validate_grouped_quantizer_list(quantizers, operand_name="input")

    def test_validation_runs_only_with_quantizer_generation(self, monkeypatch):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear

        def make_qfactory(columnwise_source):
            def qfactory(_role):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                    columnwise_source=columnwise_source,
                )

            return qfactory

        model = GroupedLinear(2, 128, 128, bias=False, params_dtype=torch.bfloat16).cuda()
        tensor = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        m_splits = torch.tensor([64, 64], dtype=torch.int64)
        original_recipe = recipe.CustomRecipe(qfactory=make_qfactory("original"))

        real_validate = grouped_linear._validate_grouped_quantizer_list
        validation_calls = []

        def tracked_validate(quantizers, *, operand_name="operand"):
            validation_calls.append((operand_name, id(quantizers[0])))
            return real_validate(quantizers, operand_name=operand_name)

        monkeypatch.setattr(
            grouped_linear,
            "_validate_grouped_quantizer_list",
            tracked_validate,
        )

        with torch.no_grad(), autocast(enabled=True, recipe=original_recipe):
            model(tensor, m_splits)
        first_call_count = len(validation_calls)
        first_generation = model._validated_quantizer_generations["scaling_fwd"]
        assert first_call_count > 0

        with torch.no_grad(), autocast(enabled=True, recipe=original_recipe):
            model(tensor, m_splits)
        assert len(validation_calls) == first_call_count
        assert model._validated_quantizer_generations["scaling_fwd"] is first_generation

        rebuilt_recipe = recipe.CustomRecipe(qfactory=make_qfactory("rowwise_dequantized"))
        with torch.no_grad(), autocast(enabled=True, recipe=rebuilt_recipe):
            model(tensor, m_splits)
        rebuilt_generation = model._validated_quantizer_generations["scaling_fwd"]
        assert len(validation_calls) > first_call_count
        assert rebuilt_generation is not first_generation
        assert rebuilt_generation[0].columnwise_source == "rowwise_dequantized"

        input_count = 0

        def mixed_source_qfactory(role):
            nonlocal input_count
            source = "original"
            if role is not None and role.tensor_type == "input":
                source = "original" if input_count == 0 else "rowwise_dequantized"
                input_count += 1
            return HybridQuantizer(
                rowwise_quantizer=_make_fp8_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source=source,
            )

        mixed_recipe = recipe.CustomRecipe(qfactory=mixed_source_qfactory)
        # A failed generation is never marked validated. Base metadata can then
        # early-return on retry, so the O(1) guard must validate it again.
        for _ in range(2):
            with pytest.raises(ValueError, match="mixed columnwise source policies"):
                with torch.no_grad(), autocast(enabled=True, recipe=mixed_recipe):
                    model(tensor, m_splits)
            assert model._validated_quantizer_generations["scaling_fwd"] is rebuilt_generation

        # Stale invalid recipe metadata must not affect the non-quantized path.
        with torch.no_grad():
            model(tensor, m_splits)

    @requires_fp8_and_nvfp4
    def test_hybrid_split_quantize_honors_rowwise_dequantized_source(self):
        """NVFP4 column data must derive from the actual grouped row result."""
        from transformer_engine.pytorch.module.grouped_linear import (
            _hybrid_split_quantize,
        )

        torch.manual_seed(3598)
        # NVFP4 grouped split-quantize requires each M split to be a multiple
        # of 64.
        tensor = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")

        def make_quantizer():
            return HybridQuantizer(
                rowwise_quantizer=_make_nvfp4_quantizer(),
                columnwise_quantizer=_make_fp8_quantizer(),
                columnwise_source="rowwise_dequantized",
            )

        quantizers = [make_quantizer(), make_quantizer()]
        actual = _hybrid_split_quantize(
            tensor,
            [64, 64],
            quantizers,
        )

        for index, (actual_part, quantizer) in enumerate(zip(actual, quantizers)):
            expected_columnwise = quantizer.columnwise_quantizer.quantize(
                actual_part.rowwise_sub_storage.dequantize()
            )
            _assert_storage_data_exact(
                actual_part.columnwise_sub_storage,
                expected_columnwise,
                context=f"GroupedLinear split {index} columnwise provenance",
            )


# ===========================================================================
# Quantized Parameters (quantized_model_init) tests for hybrid quantization
# ===========================================================================


def _hybrid_custom_recipe(row_factory, col_factory, grad_factory=None):
    """Build a CustomRecipe where forward roles use HybridQuantizer and
    backward roles use a plain quantizer (or hybrid if grad_factory builds one).

    Parameters
    ----------
    row_factory : callable() -> Quantizer
        Creates the rowwise sub-quantizer for forward roles.
    col_factory : callable() -> Quantizer
        Creates the columnwise sub-quantizer for forward roles.
    grad_factory : callable() -> Quantizer, optional
        Creates the quantizer for grad_output/grad_input roles.
        If None, uses col_factory (matching columnwise format for wgrad compatibility).
    """
    if grad_factory is None:
        grad_factory = col_factory

    def qfactory(role):
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        if is_linear and role.tensor_type in ("input", "weight", "output"):
            return HybridQuantizer(
                rowwise_quantizer=row_factory(),
                columnwise_quantizer=col_factory(),
            )
        if is_linear and role.tensor_type in ("grad_output", "grad_input"):
            return grad_factory()
        return row_factory()

    return recipe.CustomRecipe(qfactory=qfactory)


# ---------------------------------------------------------------------------
# 1. quantized_model_init: model creation and parameter type verification
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridQuantizedModelInit:
    """Verify that quantized_model_init with a hybrid CustomRecipe produces
    HybridQuantizedTensor parameters."""

    def _hybrid_fp8_recipe(self):
        return _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )

    def test_linear_weight_is_hybrid_quantized_tensor(self):
        """model.weight should be a HybridQuantizedTensor after quantized_model_init."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        weight = model.weight
        assert isinstance(
            weight, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(weight).__name__}"
        assert isinstance(
            weight, QuantizedTensor
        ), "HybridQuantizedTensor should be a QuantizedTensor subclass"

    def test_linear_weight_has_both_sub_storages(self):
        """Quantized param should have rowwise and columnwise sub-storages."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        weight = model.weight
        assert weight.rowwise_sub_storage is not None, "Missing rowwise sub-storage"
        assert weight.columnwise_sub_storage is not None, "Missing columnwise sub-storage"

    def test_linear_weight_shape_preserved(self):
        """Quantized param should retain its logical shape."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 256, params_dtype=torch.bfloat16).cuda()

        assert model.weight.shape == torch.Size([256, 128])

    def test_linear_bias_stays_bf16(self):
        """Bias should remain BF16 (not quantized)."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, bias=True, params_dtype=torch.bfloat16).cuda()

        assert not isinstance(model.bias, QuantizedTensor), "Bias should not be a QuantizedTensor"
        assert model.bias.dtype == torch.bfloat16

    def test_layernorm_linear_weight_is_hybrid(self):
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = LayerNormLinear(128, 128, params_dtype=torch.bfloat16).cuda()

        assert isinstance(model.weight, HybridQuantizedTensor)

    def test_dequantize_close_to_original(self):
        """Dequantized hybrid param should be close to the BF16 init values."""
        hybrid_recipe = self._hybrid_fp8_recipe()

        # Create a non-quantized reference
        torch.manual_seed(42)
        ref_model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        ref_weight = ref_model.weight.detach().clone()

        # Create quantized model with the same seed
        torch.manual_seed(42)
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        dq_weight = model.weight.dequantize()
        torch.testing.assert_close(dq_weight.float(), ref_weight.float(), rtol=0.125, atol=0.1)

    def test_preserve_high_precision_init_val(self):
        """preserve_high_precision_init_val should store original BF16 on CPU."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(
            enabled=True,
            recipe=hybrid_recipe,
            preserve_high_precision_init_val=True,
        ):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        weight = model.weight
        assert isinstance(weight, HybridQuantizedTensor)
        assert hasattr(weight, "get_high_precision_init_val")
        hp_val = weight.get_high_precision_init_val()
        assert hp_val is not None, "High-precision init val should be stored"
        assert hp_val.device.type == "cpu"
        assert hp_val.shape == weight.shape


# ---------------------------------------------------------------------------
# 2. get_weight_workspace cache invalidation for hybrid
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridWeightWorkspaceCache:
    """Test that get_weight_workspace handles HybridQuantizedTensorStorage
    correctly for the quantized-params early-return path and the BF16 cache path."""

    def _hybrid_fp8_recipe(self):
        return _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )

    def test_quantized_param_skips_workspace(self):
        """When weight is already a HybridQuantizedTensor (quantized params),
        get_weight_workspace should return it directly without creating a workspace."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp, is_first_microbatch=True)

        assert out.shape == (32, 128)
        assert "weight" not in model._fp8_workspaces

    def test_bf16_weight_creates_hybrid_workspace(self):
        """When weight is BF16 and recipe produces HybridQuantizer, the workspace
        should be a HybridQuantizedTensor."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp, is_first_microbatch=True)

        assert out.shape == (32, 128)
        workspace = model._fp8_workspaces.get("weight")
        assert isinstance(workspace, HybridQuantizedTensorStorage)
        assert workspace.rowwise_sub_storage is not None
        assert workspace.columnwise_sub_storage is not None

    def test_workspace_cache_reuse_across_microbatches(self):
        """Cached hybrid workspace should be reused on 2nd+ microbatches."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with autocast(enabled=True, recipe=hybrid_recipe):
            with torch.no_grad():
                out1 = model(inp, is_first_microbatch=True)
                workspace = model._fp8_workspaces["weight"]
                buffers = _as_data_tensor_tuple(workspace)
                out2 = model(inp, is_first_microbatch=False)

        assert isinstance(workspace, HybridQuantizedTensorStorage)
        assert model._fp8_workspaces["weight"] is workspace
        current_buffers = _as_data_tensor_tuple(workspace)
        assert len(current_buffers) == len(buffers)
        assert all(current is cached for current, cached in zip(current_buffers, buffers))
        torch.testing.assert_close(out1, out2, rtol=0.0, atol=0.0)

    def test_workspace_cache_invalidation_on_usage_change(self):
        """If usage requirements change (e.g. inference→training), the cache
        should be invalidated and a fresh workspace created."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # First pass: inference (no columnwise needed)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                model(inp, is_first_microbatch=True)
        inference_workspace = model._fp8_workspaces["weight"]
        assert isinstance(inference_workspace, HybridQuantizedTensorStorage)
        assert inference_workspace.rowwise_sub_storage is not None
        assert inference_workspace.columnwise_sub_storage is None

        # Second pass: training (columnwise now needed for backward)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_train = model(inp, is_first_microbatch=True)
        training_workspace = model._fp8_workspaces["weight"]

        assert isinstance(training_workspace, HybridQuantizedTensorStorage)
        assert training_workspace is not inference_workspace
        assert training_workspace.rowwise_sub_storage is not None
        assert training_workspace.columnwise_sub_storage is not None

        loss = out_train.float().sum()
        loss.backward()

        assert inp.grad is not None


# ---------------------------------------------------------------------------
# 3. _update_weight_quantizers for hybrid
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridUpdateWeightQuantizers:
    """Test that quantizer refresh propagates correctly to hybrid sub-quantizers."""

    def _hybrid_fp8_recipe(self):
        return _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )

    def test_quantized_param_survives_multiple_forward_passes(self):
        """Weight should remain a HybridQuantizedTensor across multiple forward passes,
        each of which triggers init_fp8_metadata → potential quantizer updates."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for i in range(3):
            inp_i = inp.detach().clone().requires_grad_(True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                out = model(inp_i)
            out.float().sum().backward()
            assert not torch.isnan(out).any(), f"NaN at iteration {i}"
            assert inp_i.grad is not None, f"No input grad at iteration {i}"

        assert isinstance(
            model.weight, HybridQuantizedTensor
        ), "Weight lost HybridQuantizedTensor type after multiple passes"


# ---------------------------------------------------------------------------
# 3b. quantize_master_weights + post_all_gather_processing for hybrid params
#
# Covers the supported (same-format) cases and the rejected (cross-format,
# missing sub-storage, unsupported sub-quantizer) cases. The supported subset
# is the first incremental hybrid integration with the distributed-optimizer
# quantized-param all-gather flow. Cross-format support is deferred to
# follow-up #3158; the tests below pin the NotImplementedError contract so the
# rejection messaging stays clear as the feature evolves.
# ---------------------------------------------------------------------------


def _ensure_single_rank_dp_group():
    """Return a single-rank NCCL process group for hybrid quantize_master_weights
    tests. Mirrors the local-pytest setup in
    `tests/pytorch/distributed/test_cast_master_weights_to_fp8.py` so we can call
    `torch.distributed.all_reduce` against a trivial group from inside the
    per-format helpers. The group is created lazily on first call and reused
    across tests within the same pytest process.
    """
    # pylint: disable=import-outside-toplevel
    import tempfile
    import pathlib

    if not torch.distributed.is_initialized():
        torch.cuda.set_device(0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            rendezvous_file = pathlib.Path(f.name)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=rendezvous_file.resolve().as_uri(),
            rank=0,
            world_size=1,
        )
    return torch.distributed.GroupMember.WORLD


def _hybrid_recipe_fp8_current():
    """Same-format Float8CurrentScaling on both directions (supported)."""
    return _hybrid_custom_recipe(
        row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        grad_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
    )


def _make_delayed_quantizer(fp8_dtype=None):
    """Construct a ``Float8Quantizer`` (delayed scaling) with locally-allocated
    scale/amax buffers for single-shot unit tests.

    The full delayed-scaling lifecycle (``FP8GlobalStateManager`` updating
    ``amax_history`` -> ``scale`` across iterations) is out of scope here; for
    ``quantize_master_weights`` we only need the helper to read/write
    ``quantizer.amax`` / ``quantizer.scale`` / ``model_weight._scale_inv``,
    which works with any pair of 1-element float32 tensors. Initial scale=1.0
    and amax=0.0 mirror the cold-start state ``FP8GlobalStateManager`` would
    initialize for the first iteration.
    """
    if fp8_dtype is None:
        fp8_dtype = tex.DType.kFloat8E4M3
    return Float8Quantizer(
        scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype,
    )


def _hybrid_recipe_fp8_delayed():
    """Same-format Float8 delayed scaling on both directions (supported)."""
    return _hybrid_custom_recipe(
        row_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E4M3),
        col_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E4M3),
        grad_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E5M2),
    )


def _hybrid_recipe_fp8_delayed_row_current_col():
    """Cross-format per-tensor Float8: delayed rowwise + current columnwise.

    Routed per-direction: row sub-storage -> delayed bucket, col sub-storage
    -> current bucket. The two helpers run independently (no shared state),
    so each direction's scale is computed via its own scaling lifecycle.
    """
    return _hybrid_custom_recipe(
        row_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E4M3),
        col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        # grad_factory matches the columnwise direction so the wgrad GEMM's
        # grad_output sub-quantizer pairs with the input/weight col format.
        grad_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
    )


def _hybrid_recipe_fp8_current_row_delayed_col():
    """Cross-format per-tensor Float8: current rowwise + delayed columnwise.

    Reversed variant of ``_hybrid_recipe_fp8_delayed_row_current_col``: row
    sub-storage -> current bucket, col sub-storage -> delayed bucket.
    """
    return _hybrid_custom_recipe(
        row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        col_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E4M3),
        grad_factory=lambda: _make_delayed_quantizer(tex.DType.kFloat8E5M2),
    )


def _hybrid_recipe_mxfp8():
    """Same-format MXFP8 on both directions (rejected today; TODO #3158)."""
    return _hybrid_custom_recipe(
        row_factory=lambda: MXFP8Quantizer(tex.DType.kFloat8E4M3),
        col_factory=lambda: MXFP8Quantizer(tex.DType.kFloat8E4M3),
        grad_factory=lambda: MXFP8Quantizer(tex.DType.kFloat8E5M2),
    )


def _hybrid_recipe_blockwise():
    """Same-format Float8Blockwise on both directions (rejected today; TODO #3158)."""
    return _hybrid_custom_recipe(
        row_factory=lambda: Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
        ),
        col_factory=lambda: Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
        ),
        grad_factory=lambda: Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E5M2, rowwise=True, columnwise=True
        ),
    )


def _build_hybrid_linear_weight(out_features, in_features, hybrid_recipe):
    """Build a `HybridQuantizedTensor` weight via `quantized_model_init`.

    Returns (weight, fp32_high_precision_init_val) where the high-precision
    init val is on GPU so we can use it as the "master" weight in
    quantize_master_weights tests.
    """
    torch.manual_seed(42)
    with quantized_model_init(
        enabled=True,
        recipe=hybrid_recipe,
        preserve_high_precision_init_val=True,
    ):
        model = Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16).cuda()

    weight = model.weight
    assert isinstance(
        weight, HybridQuantizedTensor
    ), f"Expected HybridQuantizedTensor, got {type(weight).__name__}"
    hp_init_cpu = weight.get_high_precision_init_val()
    assert hp_init_cpu is not None, "preserve_high_precision_init_val should populate the cpu val"
    hp_init = hp_init_cpu.to(weight.device).float()
    return weight, hp_init


def _hybrid_param_for(out_features, in_features, hybrid_recipe):
    """Same as `_build_hybrid_linear_weight` but discards the init val."""
    weight, _ = _build_hybrid_linear_weight(out_features, in_features, hybrid_recipe)
    return weight


@requires_fp8
class TestHybridQuantizeMasterWeights:
    """`quantize_master_weights` + `post_all_gather_processing` for hybrid params.

    Dispatch is per-direction: each sub-storage is routed independently into the
    per-format bucket matching its own sub-quantizer type. Currently-supported
    sub-quantizer types can mix freely across directions (e.g. Float8 delayed
    row + Float8 current col), single-direction hybrid (one sub-storage dropped
    via ``update_usage``) routes the live direction(s) only; per-block sub-
    quantizers (MXFP8, NVFP4, Float8Blockwise) raise NotImplementedError
    regardless of which direction they appear in.

    Supported subset (per-tensor Float8) -- positive tests verify the present
    sub-storage(s) dequantize close to the master weight after the cast:

      * Float8CurrentScaling on both directions (same-format, full master)
      * Float8CurrentScaling on both directions (DP-sharded master, non-zero
        start_offset)
      * Float8 delayed scaling on both directions (same-format)
      * Float8 delayed row + Float8 current col (cross-format; row -> delayed
        bucket, col -> current bucket)
      * Float8 current row + Float8 delayed col (cross-format, reversed)
      * Single-direction (rowwise-only) hybrid via ``update_usage``
      * Single-direction (columnwise-only) hybrid via ``update_usage``

    Rejected subset (NotImplementedError / ValueError) -- negative tests pin
    the per-direction rejection contract and the both-None guardrail:
      * MXFP8 as a hybrid sub-quantizer (rowwise OR columnwise)
      * NVFP4 as a hybrid sub-quantizer (rowwise OR columnwise)
      * Float8Blockwise as a hybrid sub-quantizer
      * Both sub-storages dropped (caller bug: nothing left to cast)
    """

    @staticmethod
    def _make_transpose_only_float8_weight(shape, quantizer, *, fill_value=173):
        """Build a Hopper/L40-style columnwise-only Float8Tensor.

        Blackwell keeps ``_data`` populated for columnwise-only Float8, so these
        tests synthesize the older architecture layout directly: logical
        ``[M, K]`` shape with the only live FP8 bytes in ``_transpose[K, M]``.
        """
        rows, cols = shape
        return Float8Tensor(
            shape=shape,
            dtype=torch.bfloat16,
            data=None,
            data_transpose=torch.full((cols, rows), fill_value, dtype=torch.uint8, device="cuda"),
            fp8_scale_inv=torch.ones(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            requires_grad=False,
            device="cuda",
        )

    @staticmethod
    def _scatter_expected_logical_bytes(initial_transpose, fp8_bytes, logical_shape, start_offset):
        rows, cols = logical_shape
        expected = initial_transpose.clone()
        expected_2d = expected.reshape(cols, rows)
        remaining = fp8_bytes.numel()
        logical_offset = start_offset
        src_offset = 0
        while remaining > 0:
            row = logical_offset // cols
            col = logical_offset % cols
            n = min(remaining, cols - col)
            expected_2d[col : col + n, row].copy_(fp8_bytes[src_offset : src_offset + n])
            logical_offset += n
            src_offset += n
            remaining -= n
        return expected

    @staticmethod
    def _reference_fp8_bytes(master, scale, dtype=torch.bfloat16):
        quantizer = Float8Quantizer(
            scale=scale,
            amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        raw = torch.empty((1, master.numel()), dtype=torch.uint8, device="cuda")
        temp = quantizer.create_tensor_from_data(raw, dtype)
        quantizer.update_quantized(master.reshape(1, -1), temp)
        return temp._data.reshape(-1)

    @staticmethod
    def _logical_float8_bytes(storage):
        """Return FP8 payload in the tensor's logical row-major order."""
        if storage._data is not None:
            return storage._data.reshape(-1)
        assert storage._transpose is not None
        return storage._transpose.transpose(-2, -1).contiguous().reshape(-1)

    def test_fp8_current_transpose_only_nonzero_offset(self):
        """Current-scaling distopt update handles Hopper-style columnwise storage.

        Regression for ``model_weight.reshape(-1)`` reaching
        ``Float8Tensor._ReshapeFunc.forward`` and dereferencing ``_data=None``.
        The nonzero offset spans multiple logical rows, so the test also checks
        row-major shard bytes are scattered into transposed storage correctly.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        shape = (4, 8)
        start_offset = 5
        master = torch.linspace(-2.0, 2.0, steps=17, dtype=torch.float32, device="cuda")
        quantizer = Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3, device="cuda", rowwise=False, columnwise=True
        )
        weight = self._make_transpose_only_float8_weight(shape, quantizer)
        initial = weight._transpose.clone()

        quantize_master_weights([weight], [master], [start_offset], group=group)

        scale = torch.reciprocal(weight._scale_inv.detach().clone())
        fp8_bytes = self._reference_fp8_bytes(master.to(weight.dtype), scale, weight.dtype)
        expected = self._scatter_expected_logical_bytes(initial, fp8_bytes, shape, start_offset)
        assert weight._data is None
        assert weight._transpose_invalid is False
        assert torch.equal(weight._transpose, expected)

    def test_fp8_delayed_transpose_only_nonzero_offset(self):
        """Delayed-scaling distopt update handles Hopper-style columnwise storage.

        Regression for the direct ``model_weight._data.view(-1)`` path in the
        delayed-scaling helper.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        shape = (4, 8)
        start_offset = 6
        master = torch.linspace(-3.0, 1.0, steps=15, dtype=torch.float32, device="cuda")
        quantizer = _make_delayed_quantizer(tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=False, columnwise=True)
        weight = self._make_transpose_only_float8_weight(shape, quantizer)
        initial = weight._transpose.clone()

        quantize_master_weights([weight], [master], [start_offset], group=group)

        fp8_bytes = self._reference_fp8_bytes(
            master.to(weight.dtype), weight._get_quantizer().scale, weight.dtype
        )
        expected = self._scatter_expected_logical_bytes(initial, fp8_bytes, shape, start_offset)
        assert weight._data is None
        assert weight._transpose_invalid is False
        assert torch.equal(weight._transpose, expected)

    # ---------- Positive tests (same-format) ----------

    def test_fp8_current_same_format_full_master(self):
        """Full master (start_offset=0) routes both sub-storages through the
        existing per-format current-scaling helper. Verifies both directions
        dequantize close to the master weight after the cast.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        # Distributed-optimizer convention: master weight is the flat FP32 shard
        # owned by the current rank (or the full param for non-distributed cases).
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        assert weight._rowwise_storage is not None
        assert weight._columnwise_storage is not None
        master_bf16 = master_flat.to(weight.dtype).reshape(weight.shape)
        expected_row = weight._quantizer.rowwise_quantizer.copy().quantize(master_bf16)
        expected_column = weight._quantizer.columnwise_quantizer.copy().quantize(master_bf16)
        _assert_storage_data_exact(
            weight._rowwise_storage,
            expected_row,
            context="full-master rowwise independent quantization",
        )
        _assert_storage_data_exact(
            weight._columnwise_storage,
            expected_column,
            context="full-master columnwise independent quantization",
        )
        dq_row = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col = weight._columnwise_storage.dequantize(dtype=torch.float32)
        # FP8 E4M3 round-trip; matches the loose tolerance the equivalent
        # native-FP8-current test uses (e.g. test_dequantize_close_to_original).
        torch.testing.assert_close(dq_row.reshape(-1), master_flat, rtol=0.125, atol=0.1)
        torch.testing.assert_close(dq_col.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    def test_fp8_current_nonzero_start_offset(self):
        """Mimic DP-sharded master: master covers logical elements
        [start_offset, start_offset + master.numel()) of the full model weight.
        Verifies that the shared logical start_offset is honored by both
        sub-storages' per-format routings.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master_full = _build_hybrid_linear_weight(64, 64, hybrid_recipe)

        half = hp_master_full.numel() // 2
        # Negation preserves the shard amax while guaranteeing that this update
        # is observable. The fixed seed puts the full-tensor amax in this shard,
        # so the current-scaling factor must remain bitwise stable.
        hp_master_shard = -hp_master_full.view(-1)[half:].contiguous()
        start_offset = half

        before = {}
        for direction, storage in (
            ("rowwise", weight._rowwise_storage),
            ("columnwise", weight._columnwise_storage),
        ):
            assert storage is not None
            before[direction] = {
                "bytes": self._logical_float8_bytes(storage).clone(),
                "scale_inv": storage._scale_inv.clone(),
                "dequantized": storage.dequantize(dtype=torch.float32).reshape(-1).clone(),
            }

        quantize_master_weights([weight], [hp_master_shard], [start_offset], group=group)
        post_all_gather_processing([weight])

        for direction, storage in (
            ("rowwise", weight._rowwise_storage),
            ("columnwise", weight._columnwise_storage),
        ):
            previous = before[direction]
            actual_bytes = self._logical_float8_bytes(storage)
            torch.testing.assert_close(storage._scale_inv, previous["scale_inv"], rtol=0, atol=0)
            torch.testing.assert_close(
                actual_bytes[:start_offset], previous["bytes"][:start_offset], rtol=0, atol=0
            )
            expected_shard_bytes = self._reference_fp8_bytes(
                hp_master_shard.to(weight.dtype),
                torch.reciprocal(storage._scale_inv),
                weight.dtype,
            )
            torch.testing.assert_close(
                actual_bytes[start_offset:], expected_shard_bytes, rtol=0, atol=0
            )
            assert not torch.equal(
                actual_bytes[start_offset:], previous["bytes"][start_offset:]
            ), f"{direction} updated shard unexpectedly retained every FP8 byte"
            dequantized = storage.dequantize(dtype=torch.float32).reshape(-1)
            torch.testing.assert_close(
                dequantized[:start_offset],
                previous["dequantized"][:start_offset],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                dequantized[start_offset:], hp_master_shard, rtol=0.125, atol=0.1
            )

    def test_fp8_delayed_same_format_full_master(self):
        """Same-format delayed scaling on both directions. Both sub-storages
        route into the delayed-scaling bucket as independent entries; the
        helper processes them with a single bucket-wide amax all-reduce.
        Verifies each direction dequantizes close to the master weight.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_delayed()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        assert weight._rowwise_storage is not None
        assert weight._columnwise_storage is not None
        dq_row = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col = weight._columnwise_storage.dequantize(dtype=torch.float32)
        torch.testing.assert_close(dq_row.reshape(-1), master_flat, rtol=0.125, atol=0.1)
        torch.testing.assert_close(dq_col.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    def test_fp8_delayed_row_current_col_full_master(self):
        """Cross-format per-tensor Float8: delayed row + current col.

        Pins the new per-direction routing: row sub-storage goes to the
        delayed bucket, col sub-storage goes to the current bucket. Each
        helper runs independently on its single-entry bucket, with no
        cross-pollination between the two scaling lifecycles.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_delayed_row_current_col()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        assert weight._rowwise_storage is not None
        assert weight._columnwise_storage is not None
        assert isinstance(weight._quantizer.rowwise_quantizer, Float8Quantizer)
        assert isinstance(weight._quantizer.columnwise_quantizer, Float8CurrentScalingQuantizer)
        dq_row = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col = weight._columnwise_storage.dequantize(dtype=torch.float32)
        torch.testing.assert_close(dq_row.reshape(-1), master_flat, rtol=0.125, atol=0.1)
        torch.testing.assert_close(dq_col.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    def test_fp8_current_row_delayed_col_full_master(self):
        """Cross-format per-tensor Float8: current row + delayed col.

        Reversed variant of the test above — pins that the per-direction
        loop's second iteration (col) reaches the delayed dispatch arm
        independently of what the rowwise iteration did.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current_row_delayed_col()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        assert weight._rowwise_storage is not None
        assert weight._columnwise_storage is not None
        assert isinstance(weight._quantizer.rowwise_quantizer, Float8CurrentScalingQuantizer)
        assert isinstance(weight._quantizer.columnwise_quantizer, Float8Quantizer)
        dq_row = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col = weight._columnwise_storage.dequantize(dtype=torch.float32)
        torch.testing.assert_close(dq_row.reshape(-1), master_flat, rtol=0.125, atol=0.1)
        torch.testing.assert_close(dq_col.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    # NOTE: Per-block sub-quantizers (MXFP8, NVFP4, Float8Blockwise) are not
    # supported as hybrid sub-quantizers by this initial integration, regardless
    # of which direction they appear in. See the per-direction rejection tests
    # below (``test_mxfp8_*_raises`` covers both rowwise and columnwise rejection
    # of MXFP8; ``test_nvfp4_*_raises`` and ``test_blockwise_*_raises`` similarly).
    # The TODO #3158 block above ``_route_hybrid_to_buckets`` in tensor/utils.py
    # documents the upstream constraints (single-direction cast helper / kernel
    # support) whose unblocker drops per-block format support in for free.

    # ---------- Negative tests (per-direction rejection contract) ----------

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_mxfp8_rowwise_raises(self):
        """MXFP8 in the rowwise sub-quantizer is rejected per-direction.

        ``_cast_master_weights_to_fp8_mxfp8_scaling`` assumes each entry's
        ``model_weight`` has BOTH ``_rowwise_*`` and ``_columnwise_*`` populated
        (the underlying partial-cast kernel is bidirectional), while a hybrid
        sub-storage is single-direction by construction. See
        ``TODO(#3158, hybrid-mxfp8-distopt)`` in tensor/utils.py for the unblocker shape.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_mxfp8()
        # Shape must be a multiple of MXFP8 block size (32) on both axes.
        weight, hp_master = _build_hybrid_linear_weight(64, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="MXFP8Quantizer rowwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_mxfp8_columnwise_raises(self):
        """MXFP8 in the columnwise sub-quantizer is rejected per-direction.

        Pairs FP8 current scaling in the rowwise slot (supported) with MXFP8
        in the columnwise slot (rejected). The rowwise iteration of
        ``_route_hybrid_to_buckets`` routes the FP8 sub-storage into the
        current-scaling bucket cleanly; the columnwise iteration then hits
        MXFP8 and raises. Pins that per-direction dispatch visits and rejects
        the columnwise sub-quantizer too — not just the rowwise one.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: MXFP8Quantizer(tex.DType.kFloat8E4M3),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )
        # Shape must be a multiple of MXFP8 block size (32) on both axes.
        weight, hp_master = _build_hybrid_linear_weight(64, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="MXFP8Quantizer columnwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    @pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
    def test_nvfp4_rowwise_raises(self):
        """NVFP4 in the rowwise sub-quantizer is rejected per-direction.

        The NVFP4 cast path is blocked on a pair of upstream constraints
        documented in the TODO #3158 block above ``_route_hybrid_to_buckets`` in
        tensor/utils.py.

        NOTE: after PR #3027, single-direction 2D NVFP4 construction works,
        so this test now reaches the intended ``quantize_master_weights``
        rejection while using the base weight scaling mode.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: NVFP4Quantizer(
                fp4_dtype=tex.DType.kFloat4E2M1, with_2d_quantization=True
            ),
            col_factory=lambda: NVFP4Quantizer(
                fp4_dtype=tex.DType.kFloat4E2M1, with_2d_quantization=True
            ),
            grad_factory=lambda: NVFP4Quantizer(
                fp4_dtype=tex.DType.kFloat4E2M1, with_2d_quantization=False
            ),
        )
        weight, hp_master = _build_hybrid_linear_weight(64, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="NVFP4Quantizer rowwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    @pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
    def test_nvfp4_columnwise_raises(self):
        """NVFP4 in only the columnwise slot is rejected per-direction."""
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: NVFP4Quantizer(
                fp4_dtype=tex.DType.kFloat4E2M1, with_2d_quantization=True
            ),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )
        weight, hp_master = _build_hybrid_linear_weight(64, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="NVFP4Quantizer columnwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    @pytest.mark.skipif(
        not fp8_block_scaling_available,
        reason=f"Float8 block scaling: {reason_for_no_fp8_block_scaling}",
    )
    def test_blockwise_rowwise_raises(self):
        """Float8BlockQuantizer in the rowwise sub-quantizer is rejected
        per-direction (no e2e factory uses it; TODO #3158 marker in tensor/utils.py).
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_blockwise()
        weight, hp_master = _build_hybrid_linear_weight(128, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="Float8BlockQuantizer rowwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    @pytest.mark.skipif(
        not fp8_block_scaling_available,
        reason=f"Float8 block scaling: {reason_for_no_fp8_block_scaling}",
    )
    def test_blockwise_columnwise_raises(self):
        """Float8BlockQuantizer in only the columnwise slot is rejected."""
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=2,
            ),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )
        weight, hp_master = _build_hybrid_linear_weight(128, 128, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(NotImplementedError, match="Float8BlockQuantizer columnwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)

    def test_rowwise_only_fp8_current_full_master(self):
        """Single-direction hybrid: columnwise dropped via update_usage.

        Pins that the per-direction loop in `_route_hybrid_to_buckets` skips
        the dropped direction silently and routes only the present (rowwise)
        sub-storage. Useful for inference / memory-saving paths that
        deliberately keep only the fprop-side direction.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        weight.update_usage(rowwise_usage=True, columnwise_usage=False)
        assert weight._rowwise_storage is not None
        assert weight._columnwise_storage is None
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        # Columnwise stays dropped (the cast must not silently revive it).
        assert weight._columnwise_storage is None
        expected_row = weight._quantizer.rowwise_quantizer.copy().quantize(
            master_flat.to(weight.dtype).reshape(weight.shape)
        )
        _assert_storage_data_exact(
            weight._rowwise_storage,
            expected_row,
            context="rowwise-only full-master independent quantization",
        )
        # Rowwise is populated and dequantizes close to the master.
        dq_row = weight._rowwise_storage.dequantize(dtype=torch.float32)
        torch.testing.assert_close(dq_row.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    def test_columnwise_only_fp8_current_full_master(self):
        """Single-direction hybrid: rowwise dropped via update_usage.

        Reversed variant — verifies the column-only iteration of the per-
        direction loop reaches the dispatch and routes correctly.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        weight.update_usage(rowwise_usage=False, columnwise_usage=True)
        assert weight._rowwise_storage is None
        assert weight._columnwise_storage is not None
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])

        # Rowwise stays dropped (the cast must not silently revive it).
        assert weight._rowwise_storage is None
        expected_column = weight._quantizer.columnwise_quantizer.copy().quantize(
            master_flat.to(weight.dtype).reshape(weight.shape)
        )
        _assert_storage_data_exact(
            weight._columnwise_storage,
            expected_column,
            context="columnwise-only full-master independent quantization",
        )
        # Columnwise is populated and dequantizes close to the master.
        dq_col = weight._columnwise_storage.dequantize(dtype=torch.float32)
        torch.testing.assert_close(dq_col.reshape(-1), master_flat, rtol=0.125, atol=0.1)

    def test_both_sub_storages_none_raises(self):
        """Both sub-storages dropped via update_usage — nothing left to cast.

        This is the only remaining sub-storage-presence guardrail after the
        single-direction enablement: a fully-dropped hybrid weight reaching
        `quantize_master_weights` is a caller bug, not a deferred feature,
        so we surface it as a ValueError.
        """
        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        weight.update_usage(rowwise_usage=False, columnwise_usage=False)
        assert weight._rowwise_storage is None
        assert weight._columnwise_storage is None
        master_flat = hp_master.view(-1).contiguous()

        with pytest.raises(ValueError, match="both rowwise and columnwise"):
            quantize_master_weights([weight], [master_flat], [0], group=group)


@requires_fp8
class TestHybridPostAllGatherProcessing:
    """Hybrid branch of `post_all_gather_processing` is exercised indirectly by
    the positive `TestHybridQuantizeMasterWeights` tests; the case below pins
    an additional invariant that the routing logic must preserve.
    """

    def test_post_ag_idempotent_for_fp8_current_hybrid(self):
        """Calling post_all_gather_processing twice on a same-format Float8
        hybrid must not corrupt the sub-storages.
        """
        from transformer_engine.pytorch.tensor.utils import (
            quantize_master_weights,
            post_all_gather_processing,
        )

        group = _ensure_single_rank_dp_group()
        hybrid_recipe = _hybrid_recipe_fp8_current()
        weight, hp_master = _build_hybrid_linear_weight(64, 64, hybrid_recipe)
        master_flat = hp_master.view(-1).contiguous()

        quantize_master_weights([weight], [master_flat], [0], group=group)
        post_all_gather_processing([weight])
        dq_row_first = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col_first = weight._columnwise_storage.dequantize(dtype=torch.float32)

        post_all_gather_processing([weight])
        dq_row_second = weight._rowwise_storage.dequantize(dtype=torch.float32)
        dq_col_second = weight._columnwise_storage.dequantize(dtype=torch.float32)

        torch.testing.assert_close(dq_row_first, dq_row_second, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dq_col_first, dq_col_second, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Recipe correspondence validation
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridRecipeCorrespondence:
    """Test _check_weight_tensor_recipe_correspondence with hybrid params."""

    def _hybrid_fp8_recipe(self):
        return _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )

    def test_hybrid_param_with_matching_recipe_does_not_raise(self):
        """Forward pass with matching recipe should not raise."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out = model(inp)
        assert not torch.isnan(out).any()

    def test_hybrid_param_with_mismatched_recipe_raises(self):
        """Forward pass with a non-CustomRecipe on a hybrid param should raise."""
        hybrid_recipe = self._hybrid_fp8_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        mismatch_recipe = recipe.Float8CurrentScaling()
        with pytest.raises(RuntimeError, match="Recipe mismatch"):
            with torch.no_grad():
                with autocast(enabled=True, recipe=mismatch_recipe):
                    model(inp)


# ---------------------------------------------------------------------------
# 5. quantize_ in-place update for hybrid
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridQuantizeInPlace:
    """Test in-place re-quantization (quantize_) for HybridQuantizedTensor.

    This is needed for the optimizer writeback path (param.quantize_(master_weight))
    and the workspace cache update path (out.quantize_(new_bf16_weight)).
    """

    def test_quantize_inplace_updates_data(self):
        """quantize_() should re-quantize both sub-storages from new BF16 data."""
        torch.manual_seed(42)
        hq = HybridQuantizer(
            rowwise_quantizer=Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            ),
        )
        original = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        tensor = hq.quantize(original)

        # Update with different data
        new_data = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        expected = hq.quantize(new_data)
        result = tensor.quantize_(new_data)

        assert result is tensor
        _assert_hybrid_tensor_exact(tensor, expected, context="quantize_")

    def test_quantize_inplace_preserves_tensor_identity(self):
        """quantize_() should update in-place, not create a new tensor."""
        hq = HybridQuantizer(
            rowwise_quantizer=Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            ),
        )
        original = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        tensor = hq.quantize(original)

        new_data = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        result = tensor.quantize_(new_data)

        assert result is tensor, "quantize_() must return the object it updated"

    # noop_flag is a delayed-scaling feature; not tested here since
    # delayed scaling is out of scope for hybrid quantization.


# ---------------------------------------------------------------------------
# 6. FusedAdam with hybrid quantized params
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridFusedAdam:
    """Test FusedAdam optimizer with HybridQuantizedTensor parameters."""

    def _build_hybrid_model(self):
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        return model, hybrid_recipe

    def test_fused_adam_accepts_hybrid_params(self):
        """FusedAdam should not crash when given HybridQuantizedTensor params."""
        model, _ = self._build_hybrid_model()
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        assert optimizer is not None

    def test_fused_adam_master_weights_track_reference(self):
        """FP32 master weights should closely track a reference Adam optimizer.

        Small divergence is expected because HybridQuantizedTensor.float()
        may take a slightly different dequantization path than
        detach().clone().float() through __torch_dispatch__.
        """
        model, _ = self._build_hybrid_model()

        ref_params = [p.detach().clone().float() for p in model.parameters()]

        options = {"lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            list(model.parameters()),
            master_weights=True,
            master_weight_dtype=torch.float32,
            use_decoupled_grad=True,
            **options,
        )

        for _ in range(5):
            for p_ref, p in zip(ref_params, model.parameters()):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()

        master_params = [
            tst_optim.get_unscaled_state(p, "master_param") for p in model.parameters()
        ]
        torch.testing.assert_close(ref_params, master_params, rtol=1e-3, atol=1e-3)

    def test_fused_adam_param_remains_hybrid_after_step(self):
        """Weight params should still be HybridQuantizedTensors after optimizer step."""
        model, _ = self._build_hybrid_model()
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
            use_decoupled_grad=True,
        )

        for _ in range(3):
            for p in model.parameters():
                p.decoupled_grad = torch.rand_like(p.float())
            optimizer.step()

        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, HybridQuantizedTensor
                ), f"{name} lost HybridQuantizedTensor type: {type(p).__name__}"

    def test_fused_adam_requires_master_weights(self):
        """FusedAdam without master_weights should raise for hybrid quantized params."""
        model, _ = self._build_hybrid_model()

        with pytest.raises(RuntimeError, match="master_weights"):
            optimizer = te.optimizers.FusedAdam(
                model.parameters(),
                lr=1e-3,
                master_weights=False,
            )
            for p in model.parameters():
                p.grad = torch.rand_like(p.float()).to(p.dtype)
            optimizer.step()


# ---------------------------------------------------------------------------
# 7. End-to-end training loop: fwd + bwd + optimizer step
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridQuantizedParamsEndToEnd:
    """Full training loop: quantized_model_init + autocast fwd + bwd + FusedAdam.step()."""

    def _build_model_and_recipe(self):
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad_factory=lambda: Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            ),
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        return model, hybrid_recipe

    def test_training_loop_loss_decreases(self):
        """Loss should decrease over a few training steps."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_model_and_recipe()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses = []
        for i in range(7):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()

            for name, p in model.named_parameters():
                assert p.grad is not None, f"Step {i}: {name} has no gradient"
                assert torch.isfinite(p.grad).all(), f"Step {i}: {name} has non-finite grad"

            optimizer.step()

        # Strictly monotonic decrease
        assert all(
            losses[i + 1] < losses[i] for i in range(len(losses) - 1)
        ), f"Loss not strictly decreasing each step: {losses}"

    def test_training_loop_params_remain_quantized(self):
        """Params should remain HybridQuantizedTensors after training."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_model_and_recipe()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, HybridQuantizedTensor
                ), f"{name} is {type(p).__name__}, not HybridQuantizedTensor"

    def test_training_loop_optimizer_states_are_fp32(self):
        """Optimizer states should be FP32."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_model_and_recipe()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device="cuda")
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            output.float().sum().backward()
            optimizer.step()

        for name, p in model.named_parameters():
            state = optimizer.state[p]
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32
            if "bias" not in name:
                assert state["master_param"].dtype == torch.float32


# ---------------------------------------------------------------------------
# 8. Mixed-format quantized params (e.g. MXFP8 row + NVFP4 col)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (mxfp8_available and nvfp4_available),
    reason=f"MXFP8: {reason_for_no_mxfp8}; NVFP4: {reason_for_no_nvfp4}",
)
class TestHybridMixedFormatQuantizedParams:
    """Quantized params with genuinely different formats per direction."""

    def _build_mixed_model(self, in_features=256, out_features=256):
        """MXFP8 rowwise (fprop) + role-aware NVFP4 columnwise (bwd)."""

        def qfactory(role):
            is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
            if is_linear and role.tensor_type in ("input", "weight", "output"):
                return HybridQuantizer(
                    rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
                    columnwise_quantizer=nvfp4_quantizer_factory(role),
                )
            if is_linear and role.tensor_type == "grad_output":
                return nvfp4_quantizer_factory(role)
            return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)

        hybrid_recipe = recipe.CustomRecipe(qfactory=qfactory)
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        return model, hybrid_recipe

    def test_mixed_format_param_creation(self):
        """Model init with mixed MXFP8/NVFP4 hybrid should produce a
        HybridQuantizedTensor parameter."""
        model, _ = self._build_mixed_model()
        assert isinstance(model.weight, HybridQuantizedTensor)

    def test_mixed_format_forward_only(self):
        """Forward pass with mixed-format quantized params."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_mixed_model()
        inp = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out = model(inp)

        assert out.shape == (32, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_mixed_format_forward_backward(self):
        """Full fwd+bwd with mixed-format quantized params."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_mixed_model()
        inp = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp)
        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_mixed_format_training_loop(self):
        """End-to-end training loop with mixed-format hybrid quantized params."""
        torch.manual_seed(42)
        model, hybrid_recipe = self._build_mixed_model()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses = []
        for i in range(5):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Strictly monotonic decrease
        assert all(
            losses[i + 1] < losses[i] for i in range(len(losses) - 1)
        ), f"Loss not strictly decreasing each step: {losses}"
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(p, HybridQuantizedTensor), f"{name} is {type(p).__name__}"

    def test_mixed_format_sub_storage_types(self):
        """Verify that sub-storages have the correct types (MXFP8 vs NVFP4)."""
        model, _ = self._build_mixed_model()
        weight = model.weight
        from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
            MXFP8TensorStorage,
        )

        row = weight.rowwise_sub_storage
        col = weight.columnwise_sub_storage
        assert isinstance(row, MXFP8TensorStorage) or hasattr(
            row, "_rowwise_data"
        ), f"Expected MXFP8 rowwise sub-storage, got {type(row).__name__}"
        assert isinstance(col, NVFP4TensorStorage) or hasattr(
            col, "_rowwise_data"
        ), f"Expected NVFP4 columnwise sub-storage, got {type(col).__name__}"


# ---------------------------------------------------------------------------
# 9. Quantized params equivalence: vanilla vs hybrid (same format both dirs)
# ---------------------------------------------------------------------------


def _hybrid_fp8_current_qfactory(role):
    """Hybrid FP8 current scaling (E4M3 both dirs, E5M2 for grad)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            ),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")


def _hybrid_mxfp8_qfactory(role):
    """Hybrid MXFP8 (E4M3 both dirs)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    return HybridQuantizer(
        rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
        columnwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
    )


def _hybrid_block_fp8_qfactory(role):
    """Hybrid block FP8 (E4M3 both dirs)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    is_weight = is_linear and role.tensor_type == "weight"
    dim = 2 if is_weight else 1
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=dim,
        )
    return HybridQuantizer(
        rowwise_quantizer=Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=dim,
        ),
        columnwise_quantizer=Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=dim,
        ),
    )


def _hybrid_nvfp4_qfactory(role):
    """Hybrid NVFP4 (E2M1 both dirs, base role behavior)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type == "grad_output":
        return nvfp4_quantizer_factory(role)
    return HybridQuantizer(
        rowwise_quantizer=nvfp4_quantizer_factory(role),
        columnwise_quantizer=nvfp4_quantizer_factory(role),
    )


class _QuantizedParamsEquivalenceBase:
    """Base for comparing vanilla vs hybrid quantized params training.

    When the hybrid quantizer uses the same format in both directions,
    the full quantized_model_init + training loop should produce
    equivalent results to the vanilla (non-hybrid) quantized params path.
    """

    hidden_size = 256
    num_steps = 5

    def _vanilla_recipe(self):
        raise NotImplementedError

    def _hybrid_recipe(self):
        raise NotImplementedError

    def _build_models(self):
        """Create two models with identical init: one vanilla, one hybrid."""
        torch.manual_seed(42)
        with quantized_model_init(enabled=True, recipe=self._vanilla_recipe()):
            model_ref = Linear(
                self.hidden_size,
                self.hidden_size,
                params_dtype=torch.bfloat16,
            ).cuda()

        torch.manual_seed(42)
        with quantized_model_init(enabled=True, recipe=self._hybrid_recipe()):
            model_hyb = Linear(
                self.hidden_size,
                self.hidden_size,
                params_dtype=torch.bfloat16,
            ).cuda()

        return model_ref, model_hyb

    def _run_training_loop(self, model, train_recipe, x, target, num_steps):
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        trajectory = []
        hybrid_metadata_trajectory = []
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            step_input = x.detach().clone().requires_grad_(True)
            _set_quantization_test_seed(199 + step)
            with autocast(enabled=True, recipe=train_recipe):
                output = model(step_input)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            gradients = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            pre_step_cuda_rng_state = torch.cuda.get_rng_state()
            optimizer.step()
            post_step_cuda_rng_state = torch.cuda.get_rng_state()

            hybrid_parameters = [
                (name, param)
                for name, param in model.named_parameters()
                if isinstance(param, HybridQuantizedTensor)
            ]
            expected_storages = {}
            torch.cuda.set_rng_state(pre_step_cuda_rng_state)
            try:
                for name, param in hybrid_parameters:
                    master = optimizer.get_unscaled_state(param, "master_param")
                    expected_storages[name] = (
                        param._quantizer.rowwise_quantizer.copy().quantize(master),
                        param._quantizer.columnwise_quantizer.copy().quantize(master),
                    )
            finally:
                # The oracle must be observational only. Restore the state left
                # by the real optimizer step after replaying NVFP4 stochastic
                # rounding from its pre-step RNG state.
                torch.cuda.set_rng_state(post_step_cuda_rng_state)

            hybrid_metadata = {}
            for name, param in hybrid_parameters:
                expected_row, expected_column = expected_storages[name]
                _assert_storage_data_exact(
                    param.rowwise_sub_storage,
                    expected_row,
                    context=f"step {step} {name} rowwise writeback",
                )
                _assert_storage_data_exact(
                    param.columnwise_sub_storage,
                    expected_column,
                    context=f"step {step} {name} columnwise writeback",
                )
                hybrid_metadata[name] = {
                    "rowwise": _snapshot_storage_tensor_metadata(
                        param.rowwise_sub_storage, clone=True
                    ),
                    "columnwise": _snapshot_storage_tensor_metadata(
                        param.columnwise_sub_storage, clone=True
                    ),
                }
            hybrid_metadata_trajectory.append(hybrid_metadata)
            logical_parameters = {
                name: (
                    param.dequantize(dtype=torch.float32).detach().clone()
                    if isinstance(param, QuantizedTensor)
                    else param.detach().float().clone()
                )
                for name, param in model.named_parameters()
            }
            trajectory.append(
                {
                    "output": output.detach().clone(),
                    "loss": loss.detach().clone(),
                    "input_gradient": step_input.grad.detach().clone(),
                    "parameter_gradients": gradients,
                    "logical_parameters": logical_parameters,
                    "optimizer": _clone_nested_state(optimizer.state_dict()),
                }
            )
        return trajectory, hybrid_metadata_trajectory

    def _test_equivalence(self):
        model_ref, model_hyb = self._build_models()

        torch.manual_seed(99)
        x = torch.randn(4, 32, self.hidden_size, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        ref_trajectory, ref_hybrid_metadata = self._run_training_loop(
            model_ref,
            self._vanilla_recipe(),
            x,
            target,
            self.num_steps,
        )
        hybrid_trajectory, hybrid_metadata_trajectory = self._run_training_loop(
            model_hyb,
            self._hybrid_recipe(),
            x,
            target,
            self.num_steps,
        )
        _assert_nested_state_exact(
            hybrid_trajectory,
            ref_trajectory,
            path="same-format quantized-parameter trajectory",
        )
        assert all(not step_metadata for step_metadata in ref_hybrid_metadata)
        assert len(hybrid_metadata_trajectory) == self.num_steps
        for step_metadata in hybrid_metadata_trajectory:
            assert step_metadata.keys() == {"weight"}
            assert step_metadata["weight"]["rowwise"] is not None
            assert step_metadata["weight"]["columnwise"] is not None


@requires_fp8
class TestQuantizedParamsEquivalenceFP8CurrentScaling(_QuantizedParamsEquivalenceBase):
    """Vanilla versus same-format hybrid FP8-current training parity."""

    def _vanilla_recipe(self):
        return recipe.Float8CurrentScaling()

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_fp8_current_qfactory)

    def test_equivalence(self):
        self._test_equivalence()


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
class TestQuantizedParamsEquivalenceMXFP8(_QuantizedParamsEquivalenceBase):
    """Vanilla MXFP8BlockScaling vs hybrid MXFP8 (same format both dirs)."""

    def _vanilla_recipe(self):
        return recipe.MXFP8BlockScaling()

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_mxfp8_qfactory)

    def test_equivalence(self):
        self._test_equivalence()


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
class TestQuantizedParamsEquivalenceBlockFP8(_QuantizedParamsEquivalenceBase):
    """Vanilla Float8BlockScaling vs hybrid block FP8 (same format both dirs)."""

    def _vanilla_recipe(self):
        return recipe.Float8BlockScaling()

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_block_fp8_qfactory)

    def test_equivalence(self):
        self._test_equivalence()


@pytest.mark.skipif(
    not (fp8_available and nvfp4_available),
    reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
)
class TestQuantizedParamsEquivalenceNVFP4(_QuantizedParamsEquivalenceBase):
    """Vanilla NVFP4BlockScaling vs same-format hybrid NVFP4."""

    def _vanilla_recipe(self):
        return recipe.NVFP4BlockScaling()

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_nvfp4_qfactory)

    def test_equivalence(self):
        self._test_equivalence()


# ---------------------------------------------------------------------------
# 10. State dict save/load (checkpointing) for hybrid quantized params
# ---------------------------------------------------------------------------


# Module-level qfactories give TE-to-TE quantized-param checkpoints a stable
# importable reference for any pickled quantizer/recipe metadata. Portable BF16
# checkpoint loading should not depend on importing these factories.


@requires_mxfp8_and_nvfp4
class TestHybridCheckpoint:
    """Test state_dict save/load round-trips for models with hybrid quantized params."""

    def _hybrid_checkpoint_recipe(self):
        return recipe.CustomRecipe(qfactory=mxfp8_fwd_nvfp4_bwd_quantizer_factory)

    def test_state_dict_save_load_roundtrip(self):
        """state_dict → save → load → same model should produce identical outputs."""
        torch.manual_seed(42)
        hybrid_recipe = self._hybrid_checkpoint_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out_before = model(inp)

        state_dict = model.state_dict()

        # Create a fresh model and load
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model2 = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        model2.load_state_dict(state_dict)

        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out_after = model2(inp)

        torch.testing.assert_close(out_before, out_after, rtol=0.0, atol=0.0)

    def test_state_dict_contains_weight(self):
        """state_dict should contain the weight key."""
        hybrid_recipe = self._hybrid_checkpoint_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        sd = model.state_dict()
        assert "weight" in sd, f"state_dict keys: {list(sd.keys())}"

    def test_load_bf16_state_dict_into_hybrid_model(self):
        """Loading a BF16 state_dict into a hybrid quantized model should work.

        This is the common scenario: pretrained BF16 weights loaded into a
        model initialized with quantized_model_init.
        """
        torch.manual_seed(42)
        hybrid_recipe = self._hybrid_checkpoint_recipe()

        # Create BF16 model and get its state_dict
        ref_model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        bf16_state_dict = ref_model.state_dict()

        # Create hybrid quantized model
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        # Load BF16 weights into hybrid model
        model.load_state_dict(bf16_state_dict)

        # Verify model produces valid output
        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out = model(inp)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_state_dict_torch_save_load(self):
        """Full round-trip through torch.save/torch.load (file-based)."""
        import tempfile
        import os

        torch.manual_seed(42)
        hybrid_recipe = self._hybrid_checkpoint_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out_before = model(inp)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name

        try:
            with quantized_model_init(enabled=True, recipe=hybrid_recipe):
                model2 = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
            state_dict = torch.load(tmp_path, weights_only=False)
            model2.load_state_dict(state_dict)

            with torch.no_grad():
                with autocast(enabled=True, recipe=hybrid_recipe):
                    out_after = model2(inp)

            torch.testing.assert_close(out_before, out_after, rtol=0.0, atol=0.0)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _checkpoint_training_step(model, optimizer, x, target, hybrid_recipe):
        optimizer.zero_grad(set_to_none=True)
        step_input = x.detach().clone().requires_grad_(True)
        with autocast(enabled=True, recipe=hybrid_recipe):
            output = model(step_input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        gradients = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        optimizer.step()
        return {
            "output": output.detach().clone(),
            "loss": loss.detach().clone(),
            "input_gradient": step_input.grad.detach().clone(),
            "gradients": gradients,
            "parameters": _snapshot_model_parameters(model),
            "optimizer": _clone_nested_state(optimizer.state_dict()),
        }

    def test_checkpoint_resume_training(self):
        """Save mid-training, load into new model+optimizer, verify training continues."""
        import os
        import tempfile

        _set_quantization_test_seed(42)
        hybrid_recipe = self._hybrid_checkpoint_recipe()
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        for _ in range(3):
            self._checkpoint_training_step(model, optimizer, x, target, hybrid_recipe)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as checkpoint_file:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                },
                checkpoint_file.name,
            )
            checkpoint_path = checkpoint_file.name

        try:
            uninterrupted = self._checkpoint_training_step(
                model,
                optimizer,
                x,
                target,
                hybrid_recipe,
            )

            with quantized_model_init(enabled=True, recipe=hybrid_recipe):
                resumed_model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
            resumed_optimizer = te.optimizers.FusedAdam(
                resumed_model.parameters(),
                lr=1e-3,
                master_weights=True,
                master_weight_dtype=torch.float32,
            )
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            resumed_model.load_state_dict(checkpoint["model"])
            resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["cpu_rng_state"])
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

            resumed = self._checkpoint_training_step(
                resumed_model,
                resumed_optimizer,
                x,
                target,
                hybrid_recipe,
            )

            _assert_nested_state_exact(
                resumed,
                uninterrupted,
                path="checkpoint continuation",
            )
        finally:
            os.unlink(checkpoint_path)


# ---------------------------------------------------------------------------
# 11. FSDP2 prerequisites: __torch_dispatch__ ops that FSDP2 relies on
# ---------------------------------------------------------------------------

aten = torch.ops.aten


def _make_hybrid_param_for_dispatch(
    row_factory, col_factory, grad_factory=None, in_features=256, out_features=256
):
    """Create a HybridQuantizedTensor weight via quantized_model_init for dispatch tests."""
    hybrid_recipe = _hybrid_custom_recipe(row_factory, col_factory, grad_factory)
    with quantized_model_init(enabled=True, recipe=hybrid_recipe):
        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    return model.weight


def _fp8_row_factory():
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")


def _fp8_col_factory():
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")


def _fp8_grad_factory():
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")


def _mxfp8_factory():
    return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)


_dispatch_configs = [
    pytest.param("fp8_fp8", id="same-format-fp8"),
]
if mxfp8_available:
    _dispatch_configs.append(pytest.param("mxfp8_mxfp8", id="same-format-mxfp8"))


def _get_dispatch_hybrid_param(config_name):
    """Return a HybridQuantizedTensor weight for the given config."""
    if config_name == "fp8_fp8":
        return _make_hybrid_param_for_dispatch(
            _fp8_row_factory,
            _fp8_col_factory,
            _fp8_grad_factory,
        )
    elif config_name == "mxfp8_mxfp8":
        return _make_hybrid_param_for_dispatch(
            _mxfp8_factory,
            _mxfp8_factory,
            grad_factory=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E5M2),
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")


@requires_fp8
class TestFloat8TransposeOnlySplit:
    """Regression coverage for columnwise-only Float8 split metadata.

    A columnwise-only per-tensor Float8 sub-storage may have ``_data=None`` and
    store its bytes in ``_transpose`` with physical shape ``[K, M]``. Splitting
    that tensor must still produce pieces whose wrapper shape is the logical
    row-major shape ``[M_i, K]``; otherwise HybridQuantizedTensor uses the
    transposed shape when rowwise storage is absent.
    """

    @staticmethod
    def _make_transpose_only_float8_tensor(shape=(12, 16)):
        m, k = shape
        data_transpose = torch.empty((k, m), dtype=torch.uint8, device="cuda")
        return Float8Tensor(
            shape=shape,
            dtype=torch.bfloat16,
            data=None,
            data_transpose=data_transpose,
            fp8_scale_inv=torch.ones(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
            requires_grad=False,
            device="cuda",
        )

    @pytest.mark.parametrize(
        "split_size,dim,expected_shapes,expected_transpose_shapes",
        [
            (5, 0, [(5, 16), (5, 16), (2, 16)], [(16, 5), (16, 5), (16, 2)]),
            (6, 1, [(12, 6), (12, 6), (12, 4)], [(6, 12), (6, 12), (4, 12)]),
        ],
    )
    def test_float8_split_uses_logical_shape_for_transpose_only_storage(
        self, split_size, dim, expected_shapes, expected_transpose_shapes
    ):
        tensor = self._make_transpose_only_float8_tensor()

        pieces = torch.split(tensor, split_size, dim=dim)

        assert [tuple(piece.shape) for piece in pieces] == expected_shapes
        assert [tuple(piece._transpose.shape) for piece in pieces] == expected_transpose_shapes
        assert all(piece._data is None for piece in pieces)
        assert all(piece._transpose_invalid is False for piece in pieces)

    def test_float8_view_preserves_transpose_only_storage(self):
        tensor = self._make_transpose_only_float8_tensor()

        viewed = tensor.view(12, 16)

        assert tuple(viewed.shape) == (12, 16)
        assert viewed._data is None
        assert viewed._transpose is not None
        assert tuple(viewed._transpose.shape) == (16, 12)
        assert viewed._transpose_invalid is False

    def test_float8_aten_view_preserves_transpose_only_storage(self):
        tensor = self._make_transpose_only_float8_tensor()

        viewed = torch.ops.aten.view.default(tensor, [12, 16])

        assert tuple(viewed.shape) == (12, 16)
        assert viewed._data is None
        assert viewed._transpose is not None
        assert tuple(viewed._transpose.shape) == (16, 12)
        assert viewed._transpose_invalid is False

    def test_float8_storage_view_preserves_transpose_only_storage(self):
        data_transpose = torch.empty((16, 12), dtype=torch.uint8, device="cuda")
        storage = Float8TensorStorage(
            data=None,
            data_transpose=data_transpose,
            fp8_scale_inv=torch.ones(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
            fake_dtype=torch.bfloat16,
        )

        viewed = storage.view(torch.Size((12, 16)))

        assert viewed._data is None
        assert viewed._transpose is not None
        assert tuple(viewed._transpose.shape) == (16, 12)
        assert viewed._transpose_invalid is False

    def test_float8_shape_changing_view_raises_for_transpose_only_storage(self):
        tensor = self._make_transpose_only_float8_tensor()

        with pytest.raises(NotImplementedError, match="columnwise-only data"):
            tensor.view(6, 32)

    def test_hybrid_split_uses_columnwise_logical_shape_when_rowwise_is_absent(self):
        columnwise = self._make_transpose_only_float8_tensor()
        quantizer = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        quantizer.set_usage(rowwise=False, columnwise=True)
        hybrid = HybridQuantizedTensor(
            shape=columnwise.shape,
            dtype=columnwise.dtype,
            rowwise_storage=None,
            columnwise_storage=columnwise,
            quantizer=quantizer,
            device="cuda",
        )

        pieces = torch.split(hybrid, 5, dim=0)

        assert [tuple(piece.shape) for piece in pieces] == [
            (5, 16),
            (5, 16),
            (2, 16),
        ]
        assert all(piece.rowwise_sub_storage is None for piece in pieces)
        assert [tuple(piece.columnwise_sub_storage.shape) for piece in pieces] == [
            (5, 16),
            (5, 16),
            (2, 16),
        ]
        assert [tuple(piece.columnwise_sub_storage._transpose.shape) for piece in pieces] == [
            (16, 5),
            (16, 5),
            (16, 2),
        ]


class TestHybridNewZeros:
    """Public new_zeros semantics independent of FSDP-specific allocation."""

    @staticmethod
    def _make_identity_hybrid():
        quantizer = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        source = quantizer(torch.ones((4, 8), dtype=torch.bfloat16, device="cuda"))
        return quantizer, source

    @pytest.mark.parametrize(
        "keep_rowwise,keep_columnwise",
        [(True, True), (True, False), (False, True)],
    )
    def test_initializes_every_present_direction_and_preserves_kwargs(
        self,
        keep_rowwise,
        keep_columnwise,
    ):
        quantizer, source = self._make_identity_hybrid()
        if not keep_rowwise:
            source.update_usage(rowwise_usage=False)
        if not keep_columnwise:
            source.update_usage(columnwise_usage=False)

        # Direction selection must come from source storage, not mutable parent
        # usage flags, and new_zeros must not change those flags.
        quantizer.set_usage(rowwise=False, columnwise=False)
        result = torch.ops.aten.new_zeros.default(
            source,
            [3, 5],
            dtype=torch.float32,
            device=source.device,
        )

        assert isinstance(result, HybridQuantizedTensor)
        assert result.shape == torch.Size((3, 5))
        assert result.dtype == torch.float32
        assert result.device == source.device
        assert (result.rowwise_sub_storage is not None) is keep_rowwise
        assert (result.columnwise_sub_storage is not None) is keep_columnwise
        assert quantizer.get_usages() == {"rowwise": False, "columnwise": False}
        assert result._quantizer is not quantizer
        assert result._quantizer.get_usages() == {
            "rowwise": keep_rowwise,
            "columnwise": keep_columnwise,
        }

        for sub_storage in (
            result.rowwise_sub_storage,
            result.columnwise_sub_storage,
        ):
            if sub_storage is not None:
                torch.testing.assert_close(
                    sub_storage.dequantize(),
                    torch.zeros((3, 5), dtype=torch.float32, device=source.device),
                    rtol=0.0,
                    atol=0.0,
                )

        # A plain-source copy routes through the result's parent quantizer. It
        # must update every allocated direction even though the source parent
        # had both of its mutable usage flags disabled before new_zeros.
        plain_source = torch.full((3, 5), 7.0, dtype=torch.float32, device=source.device)
        result.copy_(plain_source)
        for sub_storage in (
            result.rowwise_sub_storage,
            result.columnwise_sub_storage,
        ):
            if sub_storage is not None:
                torch.testing.assert_close(
                    sub_storage.dequantize(), plain_source, rtol=0.0, atol=0.0
                )
        assert quantizer.get_usages() == {"rowwise": False, "columnwise": False}

    def test_identity_substorages_allow_integer_dtype(self):
        _, source = self._make_identity_hybrid()
        result = source.new_zeros((2, 3), dtype=torch.int32)

        assert result.dtype == torch.int32
        for sub_storage in (
            result.rowwise_sub_storage,
            result.columnwise_sub_storage,
        ):
            assert sub_storage.dequantize().dtype == torch.int32
            assert torch.count_nonzero(sub_storage.dequantize()).item() == 0

    @requires_fp8
    @pytest.mark.parametrize("unsupported_dtype", (torch.int32, torch.float64, torch.bool))
    def test_non_identity_substorage_rejects_unsupported_dtype(self, unsupported_dtype):
        quantizer = HybridQuantizer(
            rowwise_quantizer=_make_fp8_quantizer(),
            columnwise_quantizer=_make_fp8_quantizer(),
        )
        source = quantizer(torch.ones((4, 8), dtype=torch.bfloat16, device="cuda"))

        with pytest.raises(TypeError, match="new_zeros only supports"):
            source.new_zeros((2, 3), dtype=unsupported_dtype)

    def test_does_not_invoke_live_quantizers_or_consume_rng(self, monkeypatch):
        quantizer, source = self._make_identity_hybrid()
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(source.device).clone()

        def fail_live_make_empty(*args, **kwargs):
            raise AssertionError("new_zeros invoked a live quantizer")

        monkeypatch.setattr(quantizer, "make_empty", fail_live_make_empty)
        monkeypatch.setattr(
            source.rowwise_sub_storage._quantizer,
            "make_empty",
            fail_live_make_empty,
        )
        monkeypatch.setattr(
            source.columnwise_sub_storage._quantizer,
            "make_empty",
            fail_live_make_empty,
        )

        result = source.new_zeros((2, 6))

        assert isinstance(result, HybridQuantizedTensor)
        assert torch.equal(torch.get_rng_state(), cpu_rng_before)
        assert torch.equal(torch.cuda.get_rng_state(source.device), cuda_rng_before)

    def test_rejects_empty_hybrid(self):
        _, source = self._make_identity_hybrid()
        source.update_usage(rowwise_usage=False, columnwise_usage=False)

        with pytest.raises(RuntimeError, match="at least one present sub-storage"):
            source.new_zeros((2, 6))

    @requires_fp8_and_nvfp4
    def test_initializes_nvfp4_data_scale_and_amax_buffers(self):
        quantizer = _make_hybrid_quantizer_fp8_row_fp4_col()
        source = quantizer(torch.randn((128, 256), dtype=torch.bfloat16, device="cuda"))

        result = source.new_zeros(source.shape)

        assert type(result.rowwise_sub_storage) is type(source.rowwise_sub_storage)
        assert type(result.columnwise_sub_storage) is type(source.columnwise_sub_storage)
        torch.testing.assert_close(
            result.dequantize(),
            torch.zeros_like(result.dequantize()),
            rtol=0.0,
            atol=0.0,
        )
        for sub_storage in (
            result.rowwise_sub_storage,
            result.columnwise_sub_storage,
        ):
            buffers, storage = sub_storage.prepare_for_saving()
            try:
                assert all(
                    buffer is None or torch.count_nonzero(buffer).item() == 0 for buffer in buffers
                )
            finally:
                assert storage.restore_from_saved(buffers) == []

    @pytest.mark.skipif(
        not fp8_block_scaling_available,
        reason=f"Float8Blockwise: {reason_for_no_fp8_block_scaling}",
    )
    def test_initializes_float8_block_data_and_scale_buffers(self):
        quantizer = HybridQuantizer(
            rowwise_quantizer=Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=2,
            ),
            columnwise_quantizer=Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=2,
            ),
        )
        source = quantizer(torch.randn((128, 256), dtype=torch.bfloat16, device="cuda"))

        result = source.new_zeros(source.shape)

        assert type(result.rowwise_sub_storage) is type(source.rowwise_sub_storage)
        assert type(result.columnwise_sub_storage) is type(source.columnwise_sub_storage)
        torch.testing.assert_close(
            result.dequantize(),
            torch.zeros_like(result.dequantize()),
            rtol=0.0,
            atol=0.0,
        )
        for sub_storage in (
            result.rowwise_sub_storage,
            result.columnwise_sub_storage,
        ):
            buffers, storage = sub_storage.prepare_for_saving()
            try:
                assert all(
                    buffer is None or torch.count_nonzero(buffer).item() == 0 for buffer in buffers
                )
            finally:
                assert storage.restore_from_saved(buffers) == []


@requires_fp8
class TestHybridTorchDispatchFSDP2Ops:
    """Test aten ops that FSDP2 relies on to preserve the HybridQuantizedTensor type.

    Each op is called directly via torch.ops.aten and the result is verified to
    still be HybridQuantizedTensor with valid sub-storages.
    """

    @pytest.fixture(params=_dispatch_configs)
    def hybrid_param(self, request):
        torch.manual_seed(42)
        return _get_dispatch_hybrid_param(request.param)

    def test_split_preserves_hybrid_type(self, hybrid_param):
        """torch.split must return a list of HybridQuantizedTensor pieces."""
        dim0 = hybrid_param.shape[0]
        chunk_size = dim0 // 2
        pieces = torch.split(hybrid_param, chunk_size, dim=0)
        expected_rowwise = torch.split(hybrid_param.rowwise_sub_storage, chunk_size, dim=0)
        expected_columnwise = torch.split(hybrid_param.columnwise_sub_storage, chunk_size, dim=0)
        assert len(pieces) == len(expected_rowwise) == len(expected_columnwise)

        assert len(pieces) >= 2
        for piece in pieces:
            assert isinstance(
                piece, HybridQuantizedTensor
            ), f"Expected HybridQuantizedTensor, got {type(piece).__name__}"
            assert piece.rowwise_sub_storage is not None
            assert piece.columnwise_sub_storage is not None
        for index, (piece, expected_row, expected_column) in enumerate(
            zip(pieces, expected_rowwise, expected_columnwise)
        ):
            _assert_storage_data_exact(
                piece.rowwise_sub_storage,
                expected_row,
                context=f"split {index} rowwise",
            )
            _assert_storage_data_exact(
                piece.columnwise_sub_storage,
                expected_column,
                context=f"split {index} columnwise",
            )

        total_rows = sum(p.shape[0] for p in pieces)
        assert total_rows == dim0

        orig_deq = hybrid_param.dequantize()
        reassembled = torch.cat([p.dequantize() for p in pieces], dim=0)
        torch.testing.assert_close(orig_deq, reassembled, rtol=0.0, atol=0.0)

    def test_split_sub_storage_types_preserved(self, hybrid_param):
        """After split, sub-storage types must match the original."""
        orig_row_type = type(hybrid_param.rowwise_sub_storage)
        orig_col_type = type(hybrid_param.columnwise_sub_storage)

        chunk_size = hybrid_param.shape[0] // 2
        pieces = torch.split(hybrid_param, chunk_size, dim=0)
        for piece in pieces:
            assert type(piece.rowwise_sub_storage) is orig_row_type
            assert type(piece.columnwise_sub_storage) is orig_col_type

    def test_view_preserves_hybrid_type(self, hybrid_param):
        """view must return a HybridQuantizedTensor (used by FSDP2 reset_sharded_param)."""
        shape_2d = hybrid_param.shape
        result = aten.view.default(hybrid_param, list(shape_2d))
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.rowwise_sub_storage is not None
        assert result.columnwise_sub_storage is not None

    def test_view_same_shape_preserves_hybrid(self, hybrid_param):
        """view with same shape must return HybridQuantizedTensor."""
        shape_2d = list(hybrid_param.shape)
        result = aten.view.default(hybrid_param, shape_2d)
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"

    def test_as_strided_noop_preserves_hybrid(self, hybrid_param):
        """as_strided with matching shape/strides is a no-op that preserves type."""
        shape = tuple(hybrid_param.size())
        strides = (shape[-1], 1)
        result = aten.as_strided.default(hybrid_param, list(shape), list(strides))
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.rowwise_sub_storage is not None
        assert result.columnwise_sub_storage is not None

    def test_slice_noop_preserves_hybrid(self, hybrid_param):
        """slice with full range is a no-op that preserves type."""
        result = aten.slice.Tensor(hybrid_param, 0, 0, hybrid_param.size(0))
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.rowwise_sub_storage is not None

    def test_copy_between_hybrid_tensors(self, hybrid_param):
        """copy_ between compatible HybridQuantizedTensors copies quantized data directly."""
        src_deq = hybrid_param.dequantize().clone()
        dst = hybrid_param._quantizer.make_empty(
            shape=hybrid_param.shape,
            dtype=hybrid_param.dtype,
            device=hybrid_param.device,
        )
        assert isinstance(dst, HybridQuantizedTensor)

        aten.copy_.default(dst, hybrid_param)
        dst_deq = dst.dequantize()
        torch.testing.assert_close(src_deq, dst_deq, rtol=0.0, atol=0.0)
        _assert_hybrid_tensor_exact(dst, hybrid_param, context="hybrid copy_")

    def test_copy_from_bf16_to_hybrid(self, hybrid_param):
        """copy_ from BF16 into HybridQuantizedTensor triggers quantize_."""
        param = hybrid_param.detach()
        bf16_data = torch.randn_like(param.dequantize())
        expected = param._quantizer.quantize(bf16_data)
        aten.copy_.default(param, bf16_data)
        assert isinstance(param, HybridQuantizedTensor)
        _assert_hybrid_tensor_exact(param, expected, context="BF16 copy_")
        torch.testing.assert_close(param.dequantize(), expected.dequantize(), rtol=0.0, atol=0.0)

    def test_new_zeros_returns_hybrid(self, hybrid_param):
        """new_zeros returns initialized storage that remains FSDP-copyable."""
        new_shape = list(hybrid_param.shape)
        result = aten.new_zeros.default(hybrid_param, new_shape)

        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.shape == hybrid_param.shape
        assert result.rowwise_sub_storage is not None
        assert result.columnwise_sub_storage is not None
        assert type(result.rowwise_sub_storage) is type(hybrid_param.rowwise_sub_storage)
        assert type(result.columnwise_sub_storage) is type(hybrid_param.columnwise_sub_storage)
        torch.testing.assert_close(
            result.dequantize(),
            torch.zeros_like(result.dequantize()),
            rtol=0.0,
            atol=0.0,
        )

        # FSDP2 overwrites the initialized destination via copy_ after gather.
        aten.copy_.default(result, hybrid_param)
        torch.testing.assert_close(
            result.dequantize(), hybrid_param.dequantize(), rtol=0.0, atol=0.0
        )
        _assert_hybrid_tensor_exact(result, hybrid_param, context="new_zeros then copy_")

    def test_empty_like_returns_hybrid(self, hybrid_param):
        """empty_like must return a HybridQuantizedTensor."""
        result = aten.empty_like.default(hybrid_param)
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.shape == hybrid_param.shape
        assert result.rowwise_sub_storage is not None

    def test_clone_returns_hybrid(self, hybrid_param):
        """clone must return an independent HybridQuantizedTensor with same data."""
        result = aten.clone.default(hybrid_param)
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result is not hybrid_param
        torch.testing.assert_close(
            result.dequantize(), hybrid_param.dequantize(), rtol=0.0, atol=0.0
        )
        _assert_hybrid_tensor_exact(result, hybrid_param, context="clone")


# ---------------------------------------------------------------------------
# 12. FSDP2 prerequisites: fsdp_pre_all_gather protocol
# ---------------------------------------------------------------------------


def _make_fsdp_protocol_param(config_name):
    """Create a HybridQuantizedTensor weight for FSDP protocol tests."""
    if config_name == "fp8_fp8":
        r = _hybrid_custom_recipe(_fp8_row_factory, _fp8_col_factory, _fp8_grad_factory)
    elif config_name == "mxfp8_fp8":
        r = _hybrid_custom_recipe(_mxfp8_factory, _fp8_col_factory, _fp8_grad_factory)
    elif config_name == "block_fp8":
        r = recipe.CustomRecipe(qfactory=_hybrid_block_fp8_qfactory)
    else:
        raise ValueError(f"Unknown config: {config_name}")
    with quantized_model_init(enabled=True, recipe=r):
        model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    return model.weight


_fsdp_protocol_configs = [pytest.param("fp8_fp8", id="same-format")]
if mxfp8_available:
    _fsdp_protocol_configs.append(pytest.param("mxfp8_fp8", id="mixed-mxfp8-fp8"))
if fp8_block_scaling_available:
    _fsdp_protocol_configs.append(pytest.param("block_fp8", id="same-format-block-fp8"))


@requires_fp8
class TestHybridFsdpPreAllGatherProtocol:
    """Test the fsdp_pre_all_gather method on HybridQuantizedTensor.

    These tests call the method directly (no actual all-gather communication)
    to verify the protocol contract: returns (sharded_tensors, metadata) where
    sharded_tensors is a tuple of plain torch.Tensor.
    """

    @pytest.fixture(params=_fsdp_protocol_configs)
    def hybrid_param(self, request):
        torch.manual_seed(42)
        return _make_fsdp_protocol_param(request.param)

    def test_pre_all_gather_returns_tuple_pair(self, hybrid_param):
        """fsdp_pre_all_gather returns (sharded_tensors, metadata)."""
        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        assert isinstance(
            sharded_tensors, tuple
        ), f"sharded_tensors should be tuple, got {type(sharded_tensors).__name__}"
        assert len(sharded_tensors) > 0, "sharded_tensors should not be empty"
        assert isinstance(
            metadata, tuple
        ), f"metadata should be tuple, got {type(metadata).__name__}"

    def test_pre_all_gather_buffers_are_plain_tensors(self, hybrid_param):
        """Every element in sharded_tensors must be a plain torch.Tensor."""
        sharded_tensors, _ = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        for i, t in enumerate(sharded_tensors):
            assert isinstance(
                t, torch.Tensor
            ), f"sharded_tensors[{i}] should be torch.Tensor, got {type(t).__name__}"
            assert not isinstance(
                t, QuantizedTensor
            ), f"sharded_tensors[{i}] should NOT be QuantizedTensor subclass"

    def test_pre_all_gather_buffer_count_consistent(self, hybrid_param):
        """Buffer count must be the same across repeated calls (FSDP2 buffer reuse)."""
        sharded_1, _ = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        sharded_2, _ = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        assert len(sharded_1) == len(
            sharded_2
        ), f"Buffer count changed: {len(sharded_1)} vs {len(sharded_2)}"

    def test_pre_all_gather_metadata_sufficient_for_reconstruction(self, hybrid_param):
        """Metadata must contain enough info to reconstruct the tensor."""
        _, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        assert metadata is not None
        assert len(metadata) > 0, "metadata should not be empty"


# ---------------------------------------------------------------------------
# 13. FSDP2 prerequisites: fsdp_post_all_gather protocol
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridFsdpPostAllGatherProtocol:
    """Test the fsdp_post_all_gather method on HybridQuantizedTensor.

    Simulates the post-all-gather phase by passing the sharded_tensors
    from pre_all_gather directly (mimicking a single-rank all-gather).
    """

    @pytest.fixture(params=_fsdp_protocol_configs)
    def hybrid_param(self, request):
        torch.manual_seed(42)
        return _make_fsdp_protocol_param(request.param)

    def test_post_all_gather_first_call_returns_hybrid_tensor(self, hybrid_param):
        """With out=None, post_all_gather returns (HybridQuantizedTensor, outputs)."""
        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        result, ag_outputs = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.shape == hybrid_param.shape
        assert result.rowwise_sub_storage is not None
        assert result.columnwise_sub_storage is not None

    def test_post_all_gather_buffer_reuse(self, hybrid_param):
        """On second call with out=previous, the same object is returned (buffer reuse)."""
        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        first_result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )

        second_result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=first_result,
        )
        assert (
            second_result is first_result
        ), "Buffer reuse: post_all_gather(out=prev) should return the same object"
        _assert_hybrid_tensor_exact(second_result, hybrid_param, context="post-all-gather reuse")

    def test_post_all_gather_dequantize_matches_original(self, hybrid_param):
        """Reconstructed tensor should dequantize close to the original."""
        orig_deq = hybrid_param.dequantize()

        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )
        result_deq = result.dequantize()
        _assert_hybrid_tensor_exact(result, hybrid_param, context="post-all-gather")
        torch.testing.assert_close(orig_deq, result_deq, rtol=0.0, atol=0.0)

    def test_post_all_gather_sub_storage_types_correct(self, hybrid_param):
        """Reconstructed tensor's sub-storages match the original types."""
        orig_row_type = type(hybrid_param.rowwise_sub_storage)
        orig_col_type = type(hybrid_param.columnwise_sub_storage)

        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )
        assert type(result.rowwise_sub_storage) is orig_row_type
        _assert_hybrid_tensor_exact(result, hybrid_param, context="post-all-gather storage types")
        assert type(result.columnwise_sub_storage) is orig_col_type


# ---------------------------------------------------------------------------
# 14. FSDP2 prerequisites: pre/post roundtrip
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridFsdpRoundtrip:
    """End-to-end single-process roundtrip (pre -> post) without communication."""

    @pytest.fixture(params=_fsdp_protocol_configs)
    def hybrid_param(self, request):
        torch.manual_seed(42)
        return _make_fsdp_protocol_param(request.param)

    def test_pre_post_roundtrip_preserves_data(self, hybrid_param):
        """pre_all_gather -> post_all_gather(out=None) -> dequantize matches original."""
        orig_deq = hybrid_param.dequantize()

        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )
        _assert_hybrid_tensor_exact(result, hybrid_param, context="FSDP roundtrip")
        torch.testing.assert_close(orig_deq, result.dequantize(), rtol=0.0, atol=0.0)

    def test_pre_post_roundtrip_buffer_reuse_preserves_data(self, hybrid_param):
        """Second roundtrip with out=previous preserves data (iteration 2+ simulation)."""
        sharded_tensors, metadata = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        first_result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors,
            metadata,
            hybrid_param.dtype,
            out=None,
        )

        sharded_tensors_2, metadata_2 = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        second_result, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors_2,
            metadata_2,
            hybrid_param.dtype,
            out=first_result,
        )
        assert second_result is first_result
        torch.testing.assert_close(
            hybrid_param.dequantize(), second_result.dequantize(), rtol=0.0, atol=0.0
        )
        _assert_hybrid_tensor_exact(second_result, hybrid_param, context="FSDP roundtrip reuse")

    def test_scale_refresh_across_iterations(self):
        """After a sharded optimizer-style requantize, iter-2+ gathers see the new scale.

        Per-tensor FP8 does NOT include ``_scale_inv`` in ``fsdp_buffer_fields``
        (only ``_data`` is gathered; the scalar scale travels via iter-1
        metadata). This relies on the invariant that the sharded and gathered
        ``Float8Tensor`` s share the same ``_scale_inv`` tensor object, and
        that ``Float8CurrentScalingQuantizer.update_quantized`` writes the new
        scale in place rather than replacing the tensor reference. If either
        invariant broke, the gathered copy would carry a stale scale on
        iter-2+ and silently apply the wrong dequantization.

        This test locks the invariant down by forcing a radically different
        scale between iterations and asserting the gathered tensor's
        dequantization tracks the sharded one.
        """
        torch.manual_seed(42)
        hybrid_recipe = _hybrid_custom_recipe(
            _fp8_row_factory,
            _fp8_col_factory,
            _fp8_grad_factory,
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        hybrid_param = model.weight

        # Iter-1 gather with the initial (small-magnitude) weights
        sharded_tensors_1, metadata_1 = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        gathered, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors_1,
            metadata_1,
            hybrid_param.dtype,
            out=None,
        )

        # Simulate an optimizer writeback that produces a much larger weight;
        # Float8CurrentScalingQuantizer.update_quantized must recompute
        # _scale_inv for this range. If the gathered copy didn't see the new
        # scale, the dequantize below would disagree with the sharded copy.
        huge_master = torch.randn_like(hybrid_param.dequantize()) * 100.0
        hybrid_param._quantizer.update_quantized(huge_master, hybrid_param)

        # Iter-2+ path: reuse the gathered buffer
        sharded_tensors_2, metadata_2 = hybrid_param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=hybrid_param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        gathered_refreshed, _ = hybrid_param.fsdp_post_all_gather(
            sharded_tensors_2,
            metadata_2,
            hybrid_param.dtype,
            out=gathered,
        )
        assert gathered_refreshed is gathered

        # The gathered copy must now reflect the new sharded scale, not the
        # tiny original scale.
        torch.testing.assert_close(
            hybrid_param.dequantize(),
            gathered_refreshed.dequantize(),
            rtol=0.0,
            atol=0.0,
        )
        _assert_hybrid_tensor_exact(gathered_refreshed, hybrid_param, context="FSDP scale refresh")
        # And the magnitude really did change (sanity: this test would pass
        # vacuously if update_quantized didn't actually change anything).
        assert gathered_refreshed.dequantize().abs().max() > 10.0, (
            "update_quantized did not produce a sufficiently different "
            "weight; the scale-refresh invariant is not being exercised"
        )

    def test_nvfp4_sub_storage_raises_on_pre_all_gather(self):
        """Hybrid FSDP2 with an NVFP4 sub-storage must raise a clear error.

        NVIDIA/TransformerEngine#3158 tracks the missing NVFP4 FSDP2 support:
        packed FP4 dim-0 alignment, columnwise dequantization, and RHT-cache
        handling are not implemented. Until that lands, hybrid pre-all-gather must
        refuse an NVFP4 sub-storage cleanly via the ``fsdp_buffer_fields``
        protocol rather than silently producing wrong data.

        This test pins that contract: any hybrid whose sub-storage does not
        implement ``fsdp_buffer_fields`` raises ``NotImplementedError`` at
        ``fsdp_pre_all_gather`` time. The prior version of this test
        inadvertently asserted the opposite when buffer extraction used
        implicit ``get_metadata()``-based tensor scanning.
        """
        if not (fp8_available and nvfp4_available):
            pytest.skip("Requires FP8 + NVFP4 support")

        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: NVFP4Quantizer(),
            col_factory=lambda: NVFP4Quantizer(),
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        param = model.weight

        # Clean refusal: hybrid's pre_all_gather raises an NVFP4-specific
        # message identifying the unsupported sub-storage protocol, not a generic
        # "NVFP4Tensor does not implement fsdp_buffer_fields" from deep inside
        # the base class.
        with pytest.raises(NotImplementedError) as exc_info:
            param.fsdp_pre_all_gather(
                mesh=None,
                orig_size=param.shape,
                contiguous_orig_stride=None,
                module=None,
                mp_policy=None,
            )
        msg = str(exc_info.value)
        assert "NVFP4Tensor" in msg
        assert "rowwise sub-storage" in msg
        assert "Use a supported sub-quantizer" in msg
        assert "fsdp_buffer_fields" in msg


# ---------------------------------------------------------------------------
# 15. FSDP2 prerequisites: make_like correctness
# ---------------------------------------------------------------------------


@requires_fp8
class TestHybridMakeLike:
    """Test that make_like produces correct copies for __torch_dispatch__ usage."""

    def _make_hybrid_param(self):
        hybrid_recipe = _hybrid_custom_recipe(
            _fp8_row_factory,
            _fp8_col_factory,
            _fp8_grad_factory,
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        return model.weight

    def test_make_like_preserves_sub_storages(self):
        """make_like result has the same sub-storage types, quantizers, and dtype."""
        param = self._make_hybrid_param()
        copy = HybridQuantizedTensor.make_like(param)

        assert isinstance(copy, HybridQuantizedTensor)
        assert copy.dtype == param.dtype
        assert copy.shape == param.shape
        assert type(copy.rowwise_sub_storage) is type(param.rowwise_sub_storage)
        assert type(copy.columnwise_sub_storage) is type(param.columnwise_sub_storage)
        _assert_hybrid_tensor_exact(copy, param, context="make_like")
        torch.testing.assert_close(copy.dequantize(), param.dequantize(), rtol=0.0, atol=0.0)

    def test_make_like_is_independent(self):
        """make_like result should not share the same tensor identity."""
        param = self._make_hybrid_param()
        copy = HybridQuantizedTensor.make_like(param)
        assert copy is not param


# ---------------------------------------------------------------------------
# 15b. Hopper-only paths: columnwise-only Float8 sub-storage
# ---------------------------------------------------------------------------
#
# On architectures where ``is_non_tn_fp8_gemm_supported()`` returns False
# (Hopper sm_90, L40 sm_89), per-tensor FP8 GEMM only supports the TN
# layout — non-TN layouts are simulated by feeding pre-transposed data.
# So a columnwise-only ``Float8TensorStorage`` (used as a hybrid sub-
# storage) holds its quantized data in ``_transpose`` instead of
# ``_data``, with ``_data = None``.
#
# This is the exact layout the FSDP2 buffer protocol must recognize
# when the sub-storage is part of a ``HybridQuantizedTensor`` parameter.
# These tests pin the contracts that would break if the buffer
# protocol regressed to the unconditional ``("_data",)`` field name
# (which would all-gather a ``None`` tensor on Hopper).
#
# Skip on Blackwell where the C++ kernel always populates ``_data`` and
# the columnwise-only Float8 path doesn't exercise ``_transpose``.

from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported  # noqa: E402

requires_hopper_fp8 = pytest.mark.skipif(
    is_non_tn_fp8_gemm_supported() or not fp8_available,
    reason=(
        "Hopper-only: requires per-tensor FP8 with non-TN GEMM unsupported "
        "(Hopper sm_90 / L40 sm_89). On Blackwell the C++ kernel populates "
        "_data even for columnwise-only mode, so the _transpose-only path "
        "is not exercised."
    ),
)


@requires_hopper_fp8
class TestHybridFloat8ColumnwiseOnlyHopperPath:
    """Float8TensorStorage columnwise-only sub-storage exercises the
    ``_transpose`` field on Hopper. The FSDP2 buffer protocol must
    recognize this layout.
    """

    def _make_columnwise_only_float8_storage(self):
        """Build a Float8TensorStorage in the layout a columnwise-only
        hybrid sub-storage would have on Hopper: ``_data=None`` and
        the actual quantized bytes in ``_transpose``.
        """
        q = Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3,
            device="cuda",
            rowwise=False,
            columnwise=True,
        )
        src = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        out = q(src)
        # Columnwise-only Float8 on Hopper: _data is None, _transpose holds data
        assert (
            out._data is None
        ), f"Test precondition failed: expected _data is None on Hopper, got {out._data}"
        assert out._transpose is not None, "Test precondition failed: _transpose is None"
        return out

    def test_fsdp_buffer_fields_returns_transpose(self):
        """``fsdp_buffer_fields`` must return ``("_transpose",)`` when
        ``_data`` is ``None`` and ``_transpose`` is populated. The
        unconditional ``("_data",)`` would have FSDP2 all-gather a
        ``None`` tensor on Hopper hybrid + FSDP2.
        """
        storage = self._make_columnwise_only_float8_storage()
        assert storage.fsdp_buffer_fields() == ("_transpose",)

    def test_fsdp_extract_buffers_returns_transpose_data(self):
        """``fsdp_extract_buffers`` (default impl, reads named fields)
        must return the actual ``_transpose`` tensor, not ``None``.
        """
        storage = self._make_columnwise_only_float8_storage()
        buffers, meta = storage.fsdp_extract_buffers()
        assert len(buffers) == 1
        assert buffers[0] is not None
        assert buffers[0] is storage._transpose
        assert meta["field_names"] == ("_transpose",)

    def test_fsdp_assign_gathered_resets_transpose_invalid(self):
        """After the gathered transpose buffer is written back via
        ``fsdp_assign_gathered``, ``_transpose_invalid`` must be False
        — otherwise ``update_usage`` / ``get_usages`` would treat the
        freshly gathered transpose as stale on first use.
        """
        storage = self._make_columnwise_only_float8_storage()
        # Simulate stale state pre-gather
        storage._transpose_invalid = True
        new_transpose = torch.zeros_like(storage._transpose)
        storage.fsdp_assign_gathered((new_transpose,), {"field_names": ("_transpose",)})
        assert storage._transpose is new_transpose
        assert storage._transpose_invalid is False
        # And ``get_usages`` correctly reports columnwise-available
        assert storage.get_usages()["columnwise"] is True

    def test_fsdp_buffer_fields_falls_back_to_data_when_both_present(self):
        """A normally-constructed Float8TensorStorage has ``_data``
        populated; ``fsdp_buffer_fields`` should still prefer ``_data``
        — direction-aware logic must not regress the vanilla path.
        """
        q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        src = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        out = q(src)
        assert out._data is not None
        assert out.fsdp_buffer_fields() == ("_data",)


@requires_hopper_fp8
class TestHybridFsdpPostAllGatherUpdateUsage:
    """``HybridQuantizedTensor.fsdp_post_all_gather`` must call
    ``update_usage`` on each sub-storage after writing gathered data
    (mirroring vanilla ``Float8Tensor.fsdp_post_all_gather:888``).
    Without it, on Hopper a previously-cached ``_transpose`` from the
    prior iteration is silently reused with the new ``_data``, producing
    incorrect dgrad / wgrad GEMMs.
    """

    def _make_param(self):
        hybrid_recipe = _hybrid_custom_recipe(
            _fp8_row_factory,
            _fp8_col_factory,
            _fp8_grad_factory,
        )
        with quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = Linear(64, 64, params_dtype=torch.bfloat16).cuda()
        return model.weight

    def test_iter2_invalidates_stale_transpose_on_rowwise_substorage(self):
        """Simulates iter-2+ buffer reuse: pre-existing ``out`` with a
        possibly-stale ``_transpose`` cache; after ``fsdp_post_all_gather``
        the rowwise sub-storage's ``_transpose`` must be invalidated /
        regenerated to match the freshly gathered ``_data``.
        """
        param = self._make_param()
        # Build a plausible iter-2+ "out" with stale state.
        out = HybridQuantizedTensor.make_like(param)
        # Rowwise sub-storage on Hopper has _data populated. Force a stale
        # _transpose and invalidate flag to mimic the regression scenario.
        if out._rowwise_storage._transpose is None:
            # Set up a fake stale transpose (non-None, marked invalid by
            # the mismatching shape would catch nothing, so just plant
            # a tensor and clear the invalid flag to "valid").
            out._rowwise_storage._transpose = torch.zeros_like(out._rowwise_storage._data).t()
            out._rowwise_storage._transpose_invalid = False
        stale_transpose_id = id(out._rowwise_storage._transpose)

        # Drive a real all-gather round trip via the protocol
        sharded_tensors, metadata = param.fsdp_pre_all_gather(
            mesh=None,
            orig_size=param.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        out2, _ = param.fsdp_post_all_gather(sharded_tensors, metadata, param.dtype, out=out)

        # After fsdp_post_all_gather, the rowwise sub-quantizer is pinned
        # columnwise=False, so update_usage(rowwise=True, columnwise=False)
        # must clear the stale _transpose (preventing the silent
        # stale-cache regression on Hopper).
        assert out2._rowwise_storage._transpose is None or (
            out2._rowwise_storage._transpose_invalid
            and id(out2._rowwise_storage._transpose) != stale_transpose_id
        ), "Stale _transpose was not invalidated after fsdp_post_all_gather"


# ---------------------------------------------------------------------------
# 16. Activation recomputation (torch.utils.checkpoint / te.checkpoint)
# ---------------------------------------------------------------------------


def _reset_rng(seed: int = 1234):
    """Reset deterministic RNG for reproducible forward/backward comparisons.

    Activation recompute relies on RNG equality between the first forward
    and the recomputed forward. These tests use dropout-free modules, so
    RNG advancement doesn't affect numerics, but we still reset between
    runs so the reference (no-recompute) and checkpointed paths see
    identical weight init, input, and grad_output seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect_outputs(out, inp, model):
    """Gather forward output, input grad, and parameter grads into a flat list.

    Mirrors ``test_numerics.py::_test_e2e_*_recompute`` conventions so the
    comparison against a non-recomputed baseline is a simple zip.
    """
    results = [out.detach().clone()]
    if inp.grad is not None:
        results.append(inp.grad.detach().clone())
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            results.append(p.grad.detach().clone())
    return results


def _assert_outputs_bitwise_equal(ref, test, label):
    """All stateless same-format hybrid recipes should be bitwise-identical
    under activation recompute: same input bytes → same quantized bytes →
    same GEMM result. Any drift means the recompute path silently diverged
    (e.g. fell back to a different quantization path)."""
    assert len(ref) == len(test), f"{label}: output count mismatch"
    for i, (r, t) in enumerate(zip(ref, test)):
        torch.testing.assert_close(
            t, r, rtol=0, atol=0, msg=f"{label}: bitwise mismatch at output {i}"
        )


@requires_fp8
class TestHybridActivationRecompute:
    """Activation recomputation around TE modules with a hybrid CustomRecipe.

    Probes the interaction between ``HybridQuantizedTensor`` /
    ``HybridQuantizedTensorStorage`` and the three activation-checkpoint
    paths in use today:

    * ``te.checkpoint(fn, ..., use_reentrant=True)`` — reentrant path; wraps
      ``torch.autograd.Function`` that re-runs the forward under
      ``activation_recompute_forward(recompute_phase=True)``. This is the
      Megatron-style path.
    * ``te.checkpoint(fn, ..., use_reentrant=False)`` — non-reentrant path;
      uses ``_checkpoint_hook`` (torch saved-tensors hooks) to discard
      saved tensors on the first forward and recompute them on unpack.
    * ``torch.utils.checkpoint.checkpoint(fn, ..., use_reentrant=False)``
      — vanilla PyTorch path without TE wrapper. Exercised because users
      (and some Megatron configs) invoke it directly around TE modules.

    Failure modes it catches:

    * Silent BF16 fallback during recompute (would break bitwise parity
      but pass loose tolerance — hence the bitwise assertion for
      same-format stateless recipes).
    * ``HybridQuantizedTensorStorage.prepare_for_saving`` /
      ``restore_from_saved`` chain losing a sub-storage across the
      save-for-backward boundary.
    * ``HybridQuantizedTensor`` subclass being stripped by the autograd
      engine (would manifest as ``AttributeError`` on the recomputed
      tensor).
    """

    in_features = 128
    out_features = 128
    batch = 32

    # ----- helpers ---------------------------------------------------

    def _same_format_fp8_recipe(self):
        """Same-format FP8 current scaling both directions → bitwise-safe
        baseline. Matches
        :class:`TestHybridGemmBitwiseIdentical` construction so
        recompute parity can be asserted bitwise-equal."""
        return _hybrid_custom_recipe(
            row_factory=_fp8_row_factory,
            col_factory=_fp8_col_factory,
            grad_factory=_fp8_grad_factory,
        )

    def _same_format_mxfp8_recipe(self):
        """Same-format MXFP8 both directions — stateless, per-block scales
        computed from the tensor content; bitwise-stable under recompute."""
        return _hybrid_custom_recipe(
            row_factory=_mxfp8_factory,
            col_factory=_mxfp8_factory,
            grad_factory=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E5M2),
        )

    def _cross_format_fp8_mxfp8_recipe(self):
        """Cross-format FP8 row + MXFP8 col — the canonical hybrid
        scenario. Numerical parity is not bitwise because the wgrad GEMM
        uses MXFP8 scaling modes on both operands (so grad_output must be
        MXFP8 columnwise), pairing differently from the fprop path."""
        return _hybrid_custom_recipe(
            row_factory=_fp8_row_factory,
            col_factory=_mxfp8_factory,
            grad_factory=_mxfp8_factory,
        )

    def _run_linear(self, recipe_obj, *, checkpoint_fn=None):
        """Build a fresh Linear, run forward+backward, return collected
        outputs. ``checkpoint_fn`` is an optional callable of the form
        ``fn(model, inp) -> output`` that wraps the forward in an
        activation-checkpoint implementation; ``None`` is the reference
        (non-recompute) baseline.
        """
        _reset_rng(seed=4242)
        model = Linear(self.in_features, self.out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            self.batch,
            self.in_features,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inp.retain_grad()

        with autocast(enabled=True, recipe=recipe_obj):
            out = checkpoint_fn(model, inp) if checkpoint_fn is not None else model(inp)
        out.float().sum().backward()
        return _collect_outputs(out, inp, model)

    def _run_transformer_layer(self, recipe_obj, *, checkpoint_fn=None):
        """Small TransformerLayer (no dropout, fuse_qkv) with optional
        activation checkpointing around the whole block."""
        _reset_rng(seed=5151)
        hidden = 128
        ffn = 128
        nheads = 4
        seq = 8
        bs = 4

        model = TransformerLayer(
            hidden,
            ffn,
            nheads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            fuse_qkv_params=True,
            params_dtype=torch.bfloat16,
        ).cuda()

        inp = torch.randn(seq, bs, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        inp.retain_grad()

        with autocast(enabled=True, recipe=recipe_obj):
            out = checkpoint_fn(model, inp) if checkpoint_fn is not None else model(inp)
        out.float().sum().backward()
        return _collect_outputs(out, inp, model)

    # ----- te.checkpoint, reentrant ---------------------------------

    def test_te_checkpoint_reentrant_linear_fp8_bitwise(self):
        """te.checkpoint(use_reentrant=True) around te.Linear with
        same-format FP8 hybrid → bitwise parity with non-recompute.

        This is the Megatron-style activation-recompute path. Bitwise
        parity catches silent BF16 fallback (would pass loose tolerance).
        """
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=True)

        ref = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(reentrant) FP8")

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_te_checkpoint_reentrant_linear_mxfp8_bitwise(self):
        """Same as FP8 but MXFP8 hybrid — per-block scales must recompute
        identically. Asserts that the MXFP8 path does not get disabled
        during recompute."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=True)

        ref = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(reentrant) MXFP8")

    # ----- te.checkpoint, non-reentrant -----------------------------

    def test_te_checkpoint_non_reentrant_linear_fp8_bitwise(self):
        """te.checkpoint(use_reentrant=False) — the saved-tensors-hooks
        path. Different recompute infra (``_checkpoint_hook``) than the
        reentrant path; validates the hybrid activation survives the
        pack/unpack transport."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=False)

        ref = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(non-reentrant) FP8")

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_te_checkpoint_non_reentrant_linear_mxfp8_bitwise(self):
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=False)

        ref = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(non-reentrant) MXFP8")

    # ----- torch.utils.checkpoint (vanilla, non-reentrant) ----------
    #
    # These tests document a *known* TE-level incompatibility between
    # vanilla ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
    # and TE's weight-workspace cache (``_linear_forward_impl`` in
    # ``module/linear.py``). The mechanism:
    #
    #   * First forward: ``quantize_weight`` takes the cache-miss path,
    #     creating a fresh hybrid workspace and threading it into
    #     ``prepare_for_saving`` → ``ctx.save_for_backward``.
    #   * Recompute forward: the workspace is already populated on the
    #     module, so ``quantize_weight`` takes the cache-hit path and
    #     saves a different tensor-count.
    #
    # Vanilla ``torch.utils.checkpoint`` (``use_reentrant=False``)
    # enforces a strict count match between original-forward and
    # recompute-forward ``save_for_backward`` calls, and rejects the
    # discrepancy with ``CheckpointError: A different number of tensors
    # was saved``. The 2:1 count ratio (``8`` forward vs ``4`` recompute)
    # is a hybrid signature — both sub-storages are saved on cache-miss
    # and only the remaining one on cache-hit.
    #
    # ``te.checkpoint`` avoids this by threading ``is_first_microbatch``
    # / ``skip_fp8_weight_update`` correctly across the recompute phase,
    # which is why the ``te.checkpoint`` tests above pass bitwise.
    #
    # Keeping the xfail'd tests here (tracked by #3158):
    #   1. pins the boundary — users hitting this failure get a clear
    #      diagnosis and pointer to ``te.checkpoint``;
    #   2. becomes a regression signal if the underlying cache-vs-
    #      checkpoint interaction is ever resolved (the xfail flips to
    #      an unexpected pass).
    #
    # Not hybrid-specific *in nature* (any quantized TE module with
    # weight-workspace caching hits it under vanilla torch checkpoint),
    # but hybrid amplifies and surfaces it via the 2x sub-storage count.

    _TORCH_CHECKPOINT_CACHE_XFAIL = pytest.mark.xfail(
        raises=torch.utils.checkpoint.CheckpointError,
        strict=True,
        reason=(
            "Vanilla torch.utils.checkpoint(use_reentrant=False) is"
            " incompatible with TE's weight-workspace cache: cache-miss"
            " on the first forward saves a different tensor count than"
            " cache-hit on recompute. Use te.checkpoint instead (tested"
            " above). Tracked by #3158."
        ),
    )

    @_TORCH_CHECKPOINT_CACHE_XFAIL
    def test_torch_checkpoint_non_reentrant_linear_fp8_bitwise(self):
        """Vanilla ``torch.utils.checkpoint.checkpoint`` without TE wrapper
        around a hybrid-quantized te.Linear.

        Users invoke ``torch.utils.checkpoint`` directly in many Megatron
        branches and custom recomputation schemes. Currently fails due to
        the weight-workspace cache interaction documented above; pins the
        boundary so a future fix would flip this to an unexpected pass.
        """

        def fn(model, inp):
            return torch.utils.checkpoint.checkpoint(model, inp, use_reentrant=False)

        ref = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "torch.utils.checkpoint FP8")

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    @_TORCH_CHECKPOINT_CACHE_XFAIL
    def test_torch_checkpoint_non_reentrant_linear_mxfp8_bitwise(self):
        def fn(model, inp):
            return torch.utils.checkpoint.checkpoint(model, inp, use_reentrant=False)

        ref = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._same_format_mxfp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "torch.utils.checkpoint MXFP8")

    # ----- cross-format + recompute (functional, loose tolerance) ---

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_te_checkpoint_reentrant_linear_cross_format(self):
        """Cross-format hybrid (FP8 row + MXFP8 col) under activation
        recompute. Numerics are allowed to drift from non-recompute only
        through paths recompute is allowed to affect; in practice they
        should still match tightly because the recipe is stateless. Loose
        tolerance catches only catastrophic silent fallbacks."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=True)

        ref = self._run_linear(self._cross_format_fp8_mxfp8_recipe(), checkpoint_fn=None)
        test = self._run_linear(self._cross_format_fp8_mxfp8_recipe(), checkpoint_fn=fn)
        # Expected to match bitwise since both quantizers are stateless
        # and the input bytes are identical between the two runs. Use a
        # strict tolerance; if this ever drifts it's a real bug.
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(reentrant) FP8xMXFP8 cross-format")

    # ----- TransformerLayer -----------------------------------------

    def test_te_checkpoint_reentrant_transformer_layer_fp8(self):
        """te.checkpoint(reentrant) around a full TransformerLayer under
        hybrid FP8. Exercises LayerNormLinear + DPA + LayerNormMLP in one
        shot — the ``with_quantized_norm=False`` unfused path for hybrid
        in ``layernorm_linear.py`` / ``layernorm_mlp.py`` must produce
        the same result when recomputed.

        Asserted bitwise: the module uses ``hidden_dropout=0.0``,
        ``attention_dropout=0.0``, and ``te.checkpoint`` restores RNG
        state before recompute, so every kernel sees identical inputs and
        there are no stochastic ops. Non-determinism at this level would
        indicate a real regression (e.g. a kernel quietly taking a
        non-deterministic code path) — not measurement noise."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=True)

        ref = self._run_transformer_layer(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_transformer_layer(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(reentrant) TransformerLayer FP8")

    def test_te_checkpoint_non_reentrant_transformer_layer_fp8(self):
        """Same TransformerLayer setup but through the non-reentrant
        saved-tensors-hooks recompute path. Same bitwise-equality
        rationale as the reentrant variant above."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=False)

        ref = self._run_transformer_layer(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_transformer_layer(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(
            ref, test, "te.checkpoint(non-reentrant) TransformerLayer FP8"
        )

    # ----- quantized_model_init + recompute -------------------------

    def test_te_checkpoint_reentrant_quantized_model_init_fp8_bitwise(self):
        """Combine ``quantized_model_init`` (persistent
        HybridQuantizedTensor weights) with activation recompute —
        verifies the recompute path doesn't try to re-quantize an already-
        quantized weight incorrectly, and the HybridQuantizer workspace
        caching stays consistent across first-forward + recomputed-forward."""
        import transformer_engine.pytorch as te_pytorch

        hybrid_recipe = self._same_format_fp8_recipe()

        def _build_and_run(use_checkpoint):
            _reset_rng(seed=7777)
            with quantized_model_init(enabled=True, recipe=hybrid_recipe):
                model = Linear(
                    self.in_features, self.out_features, params_dtype=torch.bfloat16
                ).cuda()
            inp = torch.randn(
                self.batch,
                self.in_features,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            inp.retain_grad()
            with autocast(enabled=True, recipe=hybrid_recipe):
                if use_checkpoint:
                    out = te_pytorch.checkpoint(model, inp, use_reentrant=True)
                else:
                    out = model(inp)
            out.float().sum().backward()
            return _collect_outputs(out, inp, model)

        ref = _build_and_run(use_checkpoint=False)
        test = _build_and_run(use_checkpoint=True)
        _assert_outputs_bitwise_equal(
            ref, test, "quantized_model_init + te.checkpoint(reentrant) FP8"
        )

    # ----- GroupedLinear + recompute --------------------------------

    def _run_grouped_linear(self, recipe_obj, *, checkpoint_fn=None):
        """Build a GroupedLinear, run forward+backward with optional
        activation checkpointing around the module. Exercises the
        ``_hybrid_split_quantize`` code path under recompute.

        GroupedLinear is the MoE token-dispatch kernel: a single batch
        is split along dim-0 into ``num_gemms`` chunks and each chunk
        goes through its own weight matrix. Under hybrid quantization,
        ``_hybrid_split_quantize`` (``module/grouped_linear.py``) runs
        ``tex.split_quantize`` twice (once per sub-quantizer direction)
        and zips the results into a list of ``HybridQuantizedTensor``
        chunks — save-for-backward then receives a *list* of hybrid
        tensors, not a single one, so the ``prepare_for_saving`` chain
        has to handle an extended tensor-object list.
        """
        _reset_rng(seed=9090)
        num_gemms = 3
        hidden = 128
        ffn = 128
        bs = 24

        model = GroupedLinear(num_gemms, hidden, ffn, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(bs, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        inp.retain_grad()
        base = bs // num_gemms
        rem = bs % num_gemms
        m_splits = [base + (1 if i < rem else 0) for i in range(num_gemms)]

        with autocast(enabled=True, recipe=recipe_obj):
            if checkpoint_fn is not None:
                out = checkpoint_fn(model, inp, m_splits)
            else:
                out = model(inp, m_splits)
        out.float().sum().backward()
        return _collect_outputs(out, inp, model)

    def test_te_checkpoint_reentrant_grouped_linear_fp8_bitwise(self):
        """GroupedLinear + te.checkpoint(reentrant) under same-format FP8
        hybrid. Exercises the MoE ``_hybrid_split_quantize`` + list-of-
        hybrid-tensors save-for-backward path under recompute."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp, m_splits):
            return te_pytorch.checkpoint(model, inp, m_splits, use_reentrant=True)

        ref = self._run_grouped_linear(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_grouped_linear(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(reentrant) GroupedLinear FP8")

    def test_te_checkpoint_non_reentrant_grouped_linear_fp8_bitwise(self):
        """Same GroupedLinear recompute setup but through the non-
        reentrant saved-tensors-hooks path — verifies that the list of
        hybrid activations survives the pack/unpack transport (one hook
        invocation per split × per sub-storage buffer, not just one)."""
        import transformer_engine.pytorch as te_pytorch

        def fn(model, inp, m_splits):
            return te_pytorch.checkpoint(model, inp, m_splits, use_reentrant=False)

        ref = self._run_grouped_linear(self._same_format_fp8_recipe(), checkpoint_fn=None)
        test = self._run_grouped_linear(self._same_format_fp8_recipe(), checkpoint_fn=fn)
        _assert_outputs_bitwise_equal(ref, test, "te.checkpoint(non-reentrant) GroupedLinear FP8")

    # ----- Selective attention recompute ----------------------------

    def test_selective_attention_recompute_transformer_layer_fp8_bitwise(self):
        """``TransformerLayer(..., checkpoint_core_attention=True)`` —
        the Megatron default memory-savings pattern.

        Unlike full-layer recompute (``te.checkpoint(layer, inp)``),
        selective attention recompute is a TransformerLayer-internal
        option: only the DPA (dot-product attention) block is wrapped
        in a checkpoint, everything else runs normally. This is a
        *different* code path in ``transformer.py`` from the
        ``te.checkpoint(...)`` tests above — DPA internally invokes its
        own checkpoint context around the attention kernel.

        For hybrid, the question is whether a hybrid activation produced
        by LayerNormLinear (QKV projection) survives the DPA-internal
        recompute boundary (which saves it for backward) and is
        consumable by the backward GEMM unchanged.

        Bitwise because the model uses ``hidden_dropout=0.0``,
        ``attention_dropout=0.0``, and the DPA checkpoint restores RNG
        state — so reference and recomputed paths should be identical
        to the last bit."""
        _reset_rng(seed=5151)
        hidden = 128
        ffn = 128
        nheads = 4
        seq = 8
        bs = 4

        def _run(checkpoint_core_attention):
            _reset_rng(seed=5151)
            model = TransformerLayer(
                hidden,
                ffn,
                nheads,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                fuse_qkv_params=True,
                params_dtype=torch.bfloat16,
            ).cuda()
            inp = torch.randn(
                seq, bs, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            inp.retain_grad()
            with autocast(enabled=True, recipe=self._same_format_fp8_recipe()):
                out = model(inp, checkpoint_core_attention=checkpoint_core_attention)
            out.float().sum().backward()
            return _collect_outputs(out, inp, model)

        ref = _run(checkpoint_core_attention=False)
        test = _run(checkpoint_core_attention=True)
        _assert_outputs_bitwise_equal(ref, test, "checkpoint_core_attention TransformerLayer FP8")

    # ----- Linear bitwise parametrized across all 4 stateless formats -----

    @pytest.mark.parametrize(
        "format_name,reentrant",
        [
            pytest.param("fp8_current", True, id="fp8_current-reentrant"),
            pytest.param("fp8_current", False, id="fp8_current-nonreentrant"),
            pytest.param(
                "mxfp8",
                True,
                id="mxfp8-reentrant",
                marks=pytest.mark.skipif(
                    not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}"
                ),
            ),
            pytest.param(
                "mxfp8",
                False,
                id="mxfp8-nonreentrant",
                marks=pytest.mark.skipif(
                    not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}"
                ),
            ),
            pytest.param(
                "block_fp8",
                True,
                id="block_fp8-reentrant",
                marks=pytest.mark.skipif(
                    not fp8_block_scaling_available,
                    reason=f"BlockFP8: {reason_for_no_fp8_block_scaling}",
                ),
            ),
            pytest.param(
                "block_fp8",
                False,
                id="block_fp8-nonreentrant",
                marks=pytest.mark.skipif(
                    not fp8_block_scaling_available,
                    reason=f"BlockFP8: {reason_for_no_fp8_block_scaling}",
                ),
            ),
            pytest.param(
                "nvfp4",
                True,
                id="nvfp4-reentrant",
                marks=pytest.mark.skipif(
                    not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}"
                ),
            ),
            pytest.param(
                "nvfp4",
                False,
                id="nvfp4-nonreentrant",
                marks=pytest.mark.skipif(
                    not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}"
                ),
            ),
        ],
    )
    def test_te_checkpoint_linear_all_stateless_formats_bitwise(self, format_name, reentrant):
        """Bitwise parity of Linear + te.checkpoint across all four
        stateless hybrid formats (FP8 current, MXFP8, BlockFP8, NVFP4),
        both reentrant and non-reentrant.

        Each format has a distinct history of columnwise-only kernel
        support — BlockFP8 required C++ null-check patches before
        columnwise-only mode worked, NVFP4 has packed FP4 layout plus
        optional RHT cache, MXFP8 has [128,4]/[4,128] scale padding.
        The recompute path exercises columnwise-only sub-quantizers
        (rowwise is freed after fprop and only recreated on backward),
        so format-specific columnwise-only handling is on the critical
        path.

        A regression in any of these would silently fall back to BF16
        during recompute; bitwise equality catches that immediately."""
        import transformer_engine.pytorch as te_pytorch

        row_factory, col_factory_for_grad, hw_skip, hw_reason = _QUANTIZER_CONFIGS[format_name]
        # Most formats have a distinct E5M2 variant for grad; NVFP4 has
        # only one format (col_factory_for_grad is None → reuse
        # row_factory, which is what the existing hybrid NVFP4 tests do).
        grad_factory = col_factory_for_grad if col_factory_for_grad is not None else row_factory

        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=row_factory,
            col_factory=row_factory,
            grad_factory=grad_factory,
        )

        def fn(model, inp):
            return te_pytorch.checkpoint(model, inp, use_reentrant=reentrant)

        ref = self._run_linear(hybrid_recipe, checkpoint_fn=None)
        test = self._run_linear(hybrid_recipe, checkpoint_fn=fn)
        label = (
            f"te.checkpoint({'reentrant' if reentrant else 'non-reentrant'}) Linear {format_name}"
        )
        _assert_outputs_bitwise_equal(ref, test, label)

    # ----- save_for_backward round-trip (unit-level) ----------------

    def test_prepare_restore_roundtrip_is_identity(self):
        """Unit-level guarantee: the
        ``prepare_for_saving`` / ``restore_from_saved`` chain used by
        activation-recompute ``ctx.save_for_backward`` preserves both
        sub-storages bitwise.

        This is the primitive the recompute path is built on; pinning it
        here gives a focused failure signal independent of the module-
        level recompute tests above."""
        torch.manual_seed(0)
        inp = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        hq = HybridQuantizer(
            rowwise_quantizer=_fp8_row_factory(),
            columnwise_quantizer=_fp8_col_factory(),
        )
        hybrid = hq.quantize(inp)
        expected = hybrid.dequantize()

        saved_tensors, saved_obj = hybrid.prepare_for_saving()
        # Mimic the autograd ctx round-trip: all saved tensors pass
        # through ``ctx.save_for_backward`` (a no-op for semantics).
        leftover = saved_obj.restore_from_saved(list(saved_tensors))
        assert leftover == [], "restore_from_saved should consume every element"
        torch.testing.assert_close(saved_obj.dequantize(), expected, rtol=0, atol=0)
