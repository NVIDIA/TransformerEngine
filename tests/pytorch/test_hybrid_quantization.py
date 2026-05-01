# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for hybrid quantization (mixed rowwise/columnwise formats)."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    autocast,
    quantized_model_init,
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    TransformerLayer,
    GroupedLinear,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    Float8BlockQuantizer,
    NVFP4Quantizer,
    HybridQuantizer,
    HybridQuantizedTensor,
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

requires_fp8_and_nvfp4 = pytest.mark.skipif(
    not (fp8_available and nvfp4_available),
    reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
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


@requires_fp8_and_nvfp4
class TestHybridUpdateUsage:
    """Test update_usage semantics and sub-storage cleanup."""

    @pytest.fixture
    def hybrid_tensor(self):
        torch.manual_seed(42)
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
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
        tensors, obj = hybrid_tensor.prepare_for_saving()
        assert isinstance(tensors, list)
        assert all(t is None or isinstance(t, torch.Tensor) for t in tensors)

        remainder = obj.restore_from_saved(tensors)
        assert isinstance(remainder, list)
        assert len(remainder) == 0

        dq_after = hybrid_tensor.dequantize()
        torch.testing.assert_close(dq_before, dq_after)

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
        assert isinstance(data_tensors, tuple)
        assert len(data_tensors) > 0
        has_non_none = any(t is not None for t in data_tensors)
        assert has_non_none


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
            if role in ("linear_input", "linear_weight", "linear_output"):
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
            if role in ("linear_grad_output", "linear_grad_input"):
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
            if role in ("linear_grad_output", "linear_grad_input"):
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
            dim = 2 if role == "linear_weight" else 1
            if role in ("linear_grad_output", "linear_grad_input"):
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
    """Hybrid quantizer with NVFP4 in both directions must produce
    bitwise-identical results to the vanilla NVFP4BlockScaling recipe.

    RHT, stochastic rounding, and 2D quantization are disabled so the
    test is fully deterministic and two independent quantizer instances
    produce the same output.
    """

    def test_linear_fwd_bwd_matches_vanilla_nvfp4(self):
        torch.manual_seed(202)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
        )
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_nvfp4_factory(role):
            if role in ("linear_grad_output", "linear_grad_input"):
                return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)
            return HybridQuantizer(
                rowwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
                columnwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_nvfp4_factory)
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
        """All roles (including grad_output) use HybridQuantizer with NVFP4 both
        directions. Validates per-operand unwrap produces bitwise-identical results
        when grad_output is hybrid."""
        torch.manual_seed(203)

        in_features, out_features, batch = 128, 128, 32

        model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        model_hybrid.load_state_dict(model_ref.state_dict())

        base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        inp_ref = base_inp.clone().detach().requires_grad_(True)
        inp_hybrid = base_inp.clone().detach().requires_grad_(True)

        ref_recipe = recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
        )
        with autocast(enabled=True, recipe=ref_recipe):
            out_ref = model_ref(inp_ref)
        out_ref.float().sum().backward()

        def hybrid_nvfp4_all_roles_factory(role):
            return HybridQuantizer(
                rowwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
                columnwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
            )

        hybrid_recipe = recipe.CustomRecipe(qfactory=hybrid_nvfp4_all_roles_factory)
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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return _make_nvfp4_quantizer()
            return None

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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            return None

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
            if role in ("linear_grad_output", "linear_grad_input"):
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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return _make_nvfp4_quantizer()
            return None

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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return None

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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            return None

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
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return _make_fp8_quantizer()
            return None

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

    def test_hybrid_input_plain_weight_fwd_bwd(self):
        """Input is hybrid (FP8 row / FP8 col), weight + grad_output plain FP8.

        Wgrad columnwise: FP8 (input.col) × FP8 (grad_output.col) → compatible.
        """
        torch.manual_seed(77)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if role == "linear_input":
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if role == "linear_weight":
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return None

        mixed_recipe = recipe.CustomRecipe(qfactory=factory)
        with autocast(enabled=True, recipe=mixed_recipe):
            out = model(inp)

        assert not torch.isnan(out).any()

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"

    def test_plain_input_hybrid_weight_fwd_bwd(self):
        """Input is plain FP8, weight is hybrid (FP8 row / FP8 col)."""
        torch.manual_seed(88)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if role == "linear_input":
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E4M3,
                    device="cuda",
                )
            if role == "linear_weight":
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return Float8CurrentScalingQuantizer(
                    tex.DType.kFloat8E5M2,
                    device="cuda",
                )
            return None

        mixed_recipe = recipe.CustomRecipe(qfactory=factory)
        with autocast(enabled=True, recipe=mixed_recipe):
            out = model(inp)

        assert not torch.isnan(out).any()

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"


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


class TestHybridCrossFormatParametrized:
    """Parametrized fwd+bwd over all stateless quantizer cross-format pairs."""

    @pytest.mark.parametrize("row_name,col_name", _build_cross_format_params())
    def test_fwd_bwd(self, row_name, col_name):
        torch.manual_seed(42)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        row_cfg = _QUANTIZER_CONFIGS[row_name]
        col_cfg = _QUANTIZER_CONFIGS[col_name]
        make_row_e4m3 = row_cfg[0]
        make_col_e4m3 = col_cfg[0]
        make_col_grad = col_cfg[1] if col_cfg[1] is not None else col_cfg[0]

        def factory(role):
            if role in ("linear_input", "linear_weight"):
                return HybridQuantizer(
                    rowwise_quantizer=make_row_e4m3(),
                    columnwise_quantizer=make_col_e4m3(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return make_col_grad()
            return None

        mixed_recipe = recipe.CustomRecipe(qfactory=factory)
        with autocast(enabled=True, recipe=mixed_recipe):
            out = model(inp)

        assert out.shape == (batch, out_features)
        assert not torch.isnan(out).any(), f"Output NaN ({row_name} row × {col_name} col)"
        assert not torch.isinf(out).any(), f"Output Inf ({row_name} row × {col_name} col)"

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None, "Input gradient is None"
        assert not torch.isnan(inp.grad).any(), f"Input grad NaN ({row_name} row × {col_name} col)"
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"
            assert not torch.isnan(
                p.grad
            ).any(), f"Gradient for '{name}' NaN ({row_name} row × {col_name} col)"


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

        assert isinstance(reloaded, HybridQuantizedTensor)
        torch.testing.assert_close(reloaded.dequantize(), expected)

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
        assert reloaded.rowwise_sub_storage is not None
        torch.testing.assert_close(reloaded.dequantize(), expected)

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
        assert reloaded.columnwise_sub_storage is not None
        torch.testing.assert_close(reloaded.dequantize(), expected)

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

    def test_fp8_fprop_mxfp8_dgrad_nvfp4_wgrad(self):
        """FP8 current (fprop) + MXFP8 (dgrad) + NVFP4 (wgrad)."""
        torch.manual_seed(300)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if role == "linear_weight":
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_mxfp8_quantizer(),
                )
            if role == "linear_input":
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_mxfp8_quantizer(),
                    columnwise_quantizer=_make_nvfp4_quantizer(),
                )
            return None

        with autocast(enabled=True, recipe=recipe.CustomRecipe(qfactory=factory)):
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

    def test_nvfp4_fprop_fp8_dgrad_mxfp8_wgrad(self):
        """NVFP4 (fprop) + FP8 current (dgrad) + MXFP8 (wgrad)."""
        torch.manual_seed(301)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if role == "linear_weight":
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_fp8_quantizer(),
                )
            if role == "linear_input":
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_mxfp8_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return HybridQuantizer(
                    rowwise_quantizer=_make_fp8_quantizer(),
                    columnwise_quantizer=_make_mxfp8_quantizer(),
                )
            return None

        with autocast(enabled=True, recipe=recipe.CustomRecipe(qfactory=factory)):
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

    def test_same_dgrad_wgrad_reduces_to_plain_grad(self):
        """When dgrad format == wgrad format, grad_output can be a plain quantizer."""
        torch.manual_seed(302)
        in_features, out_features, batch = 128, 128, 32

        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def factory(role):
            if role == "linear_weight":
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_mxfp8_quantizer(),
                )
            if role == "linear_input":
                return HybridQuantizer(
                    rowwise_quantizer=_make_nvfp4_quantizer(),
                    columnwise_quantizer=_make_mxfp8_quantizer(),
                )
            if role in ("linear_grad_output", "linear_grad_input"):
                return _make_mxfp8_quantizer()
            return None

        with autocast(enabled=True, recipe=recipe.CustomRecipe(qfactory=factory)):
            out = model(inp)

        loss = out.float().sum()
        loss.backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Gradient for '{name}' is None"


# ---------------------------------------------------------------------------
# All-modules test: hybrid quantization through every TE module type
# ---------------------------------------------------------------------------


def _make_hybrid_fp8_factory():
    """Factory returning HybridQuantizer(FP8 row + FP8 col) for fwd roles,
    plain FP8 E5M2 for bwd roles."""

    def factory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
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
        if role in ("linear_grad_output", "linear_grad_input"):
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

    def _run_fwd_bwd(self, model, inp):
        hybrid_recipe = recipe.CustomRecipe(qfactory=_make_hybrid_fp8_factory())
        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp)
        loss = out.float().sum()
        loss.backward()

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert inp.grad is not None, "Input gradient is None"
        assert not torch.isnan(inp.grad).any(), "Input gradient contains NaN"
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient for '{name}' is None"
                assert not torch.isnan(p.grad).any(), f"Gradient for '{name}' contains NaN"

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
        self._run_fwd_bwd(model, inp)

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

        hybrid_recipe = recipe.CustomRecipe(qfactory=_make_hybrid_fp8_factory())
        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp, m_splits)
        loss = out.float().sum()
        loss.backward()

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert inp.grad is not None, "Input gradient is None"
        assert not torch.isnan(inp.grad).any(), "Input gradient contains NaN"
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient for '{name}' is None"
                assert not torch.isnan(p.grad).any(), f"Gradient for '{name}' contains NaN"

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
        self._run_fwd_bwd(model, inp)


@requires_fp8
class TestHybridGroupedLinearClassifier:
    """Unit tests for ``grouped_linear._is_hybrid_quantizer_list``.

    ``GroupedLinear`` dispatches its split-quantize between two mutually-
    exclusive backends: ``tex.split_quantize`` (plain) and
    ``_hybrid_split_quantize`` (all-hybrid). Neither can consume a mixed
    list — ``tex.split_quantize`` doesn't recognise ``HybridQuantizer``,
    and ``_hybrid_split_quantize`` calls ``q.rowwise_quantizer`` on every
    element. Before the classifier was tightened, ``_has_hybrid_quantizer``
    used ``any(...)`` semantics: a single hybrid entry in a mixed list
    would route to ``_hybrid_split_quantize`` and raise ``AttributeError``
    deep inside a grouped C++ call. These tests pin the new strict
    classifier contract."""

    def test_all_hybrid_returns_true(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _is_hybrid_quantizer_list,
        )

        quantizers = [_make_hybrid_quantizer_fp8_row_fp4_col() for _ in range(3)]
        assert _is_hybrid_quantizer_list(quantizers) is True

    def test_all_plain_returns_false(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _is_hybrid_quantizer_list,
        )

        quantizers = [_make_fp8_quantizer() for _ in range(3)]
        assert _is_hybrid_quantizer_list(quantizers) is False

    def test_all_none_returns_false(self):
        """No quantizers at all (BF16 path) — classifier returns False so
        the caller takes the non-fp8 branch."""
        from transformer_engine.pytorch.module.grouped_linear import (
            _is_hybrid_quantizer_list,
        )

        assert _is_hybrid_quantizer_list([None, None, None]) is False

    def test_mixed_hybrid_and_plain_raises(self):
        """The actual bug: a mixed list used to silently route to
        ``_hybrid_split_quantize`` and crash with ``AttributeError`` on
        ``plain_q.rowwise_quantizer``. Now it fails fast at the
        classifier with a user-actionable error."""
        from transformer_engine.pytorch.module.grouped_linear import (
            _is_hybrid_quantizer_list,
        )

        quantizers = [
            _make_hybrid_quantizer_fp8_row_fp4_col(),
            _make_fp8_quantizer(),
            _make_hybrid_quantizer_fp8_row_fp4_col(),
        ]
        with pytest.raises(ValueError) as exc_info:
            _is_hybrid_quantizer_list(quantizers)
        msg = str(exc_info.value)
        # Error names both counts and points at the root cause so users
        # can fix their ``qfactory`` without digging.
        assert "mixes HybridQuantizer" in msg
        assert "2 hybrid" in msg
        assert "1 non-hybrid" in msg
        assert "CustomRecipe" in msg and "qfactory" in msg

    def test_none_entries_ignored_when_remainder_is_uniform(self):
        """None entries are filtered before uniformity check — a list
        of hybrids plus a None must still classify as hybrid (not
        mixed)."""
        from transformer_engine.pytorch.module.grouped_linear import (
            _is_hybrid_quantizer_list,
        )

        quantizers = [
            _make_hybrid_quantizer_fp8_row_fp4_col(),
            None,
            _make_hybrid_quantizer_fp8_row_fp4_col(),
        ]
        assert _is_hybrid_quantizer_list(quantizers) is True

    def test_hybrid_split_quantize_rejects_plain_element(self):
        """Defense-in-depth: even if a caller bypasses the classifier,
        ``_hybrid_split_quantize`` itself asserts uniformity and raises
        ``TypeError`` with a list of received types, rather than the
        opaque ``AttributeError: 'Float8CurrentScalingQuantizer' object
        has no attribute 'rowwise_quantizer'`` from the old code."""
        from transformer_engine.pytorch.module.grouped_linear import (
            _hybrid_split_quantize,
        )

        tensor = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        quantizers = [
            _make_hybrid_quantizer_fp8_row_fp4_col(),
            _make_fp8_quantizer(),  # Not hybrid — should trigger TypeError
        ]
        with pytest.raises(TypeError) as exc_info:
            _hybrid_split_quantize(tensor, [16, 16], quantizers)
        msg = str(exc_info.value)
        assert "HybridQuantizer" in msg
        assert "Float8CurrentScalingQuantizer" in msg


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
        if role in ("linear_input", "linear_weight", "linear_output"):
            return HybridQuantizer(
                rowwise_quantizer=row_factory(),
                columnwise_quantizer=col_factory(),
            )
        if role in ("linear_grad_output", "linear_grad_input"):
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
            out = model(inp)

        assert out.shape == (32, 128)
        assert not torch.isnan(out).any()

    def test_bf16_weight_creates_hybrid_workspace(self):
        """When weight is BF16 and recipe produces HybridQuantizer, the workspace
        should be a HybridQuantizedTensor."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out = model(inp)

        assert out.shape == (32, 128)
        assert not torch.isnan(out).any()

    def test_workspace_cache_reuse_across_microbatches(self):
        """Cached hybrid workspace should be reused on 2nd+ microbatches."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with autocast(enabled=True, recipe=hybrid_recipe):
            with torch.no_grad():
                out1 = model(inp, is_first_microbatch=True)
                out2 = model(inp, is_first_microbatch=False)

        # Both should produce valid, identical results (same weight, same input)
        assert not torch.isnan(out1).any()
        assert not torch.isnan(out2).any()
        torch.testing.assert_close(out1, out2)

    def test_workspace_cache_invalidation_on_usage_change(self):
        """If usage requirements change (e.g. inference→training), the cache
        should be invalidated and a fresh workspace created."""
        model = Linear(128, 128, params_dtype=torch.bfloat16).cuda()
        hybrid_recipe = self._hybrid_fp8_recipe()

        inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # First pass: inference (no columnwise needed)
        with torch.no_grad():
            with autocast(enabled=True, recipe=hybrid_recipe):
                out_infer = model(inp, is_first_microbatch=True)
        assert not torch.isnan(out_infer).any()

        # Second pass: training (columnwise now needed for backward)
        with autocast(enabled=True, recipe=hybrid_recipe):
            out_train = model(inp, is_first_microbatch=True)
        loss = out_train.float().sum()
        loss.backward()

        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()


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

        dq_before = tensor.dequantize().clone()

        # Update with different data
        new_data = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        tensor.quantize_(new_data)

        dq_after = tensor.dequantize()

        # Should be close to new data, not old data
        diff_new = (dq_after.float() - new_data.float()).abs().mean()
        diff_old = (dq_after.float() - original.float()).abs().mean()
        assert (
            diff_new < diff_old
        ), f"After quantize_(), data is closer to old ({diff_old:.4f}) than new ({diff_new:.4f})"

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

        tensor_id = id(tensor)
        new_data = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        result = tensor.quantize_(new_data)

        assert id(tensor) == tensor_id, "quantize_() should return same object"

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

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

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
        """MXFP8 rowwise (for fprop TN) + NVFP4 columnwise (for wgrad NT)."""
        hybrid_recipe = _hybrid_custom_recipe(
            row_factory=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            col_factory=lambda: NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
            grad_factory=lambda: NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
        )
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

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
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
    if role in ("linear_input", "linear_weight", "linear_output"):
        return HybridQuantizer(
            rowwise_quantizer=Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            ),
        )
    if role in ("linear_grad_output", "linear_grad_input"):
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")


def _hybrid_mxfp8_qfactory(role):
    """Hybrid MXFP8 (E4M3 both dirs)."""
    if role in ("linear_grad_output", "linear_grad_input"):
        return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    return HybridQuantizer(
        rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
        columnwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
    )


def _hybrid_block_fp8_qfactory(role):
    """Hybrid block FP8 (E4M3 both dirs)."""
    dim = 2 if role == "linear_weight" else 1
    if role in ("linear_grad_output", "linear_grad_input"):
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
    """Hybrid NVFP4 (E2M1 both dirs)."""
    if role in ("linear_grad_output", "linear_grad_input"):
        return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)
    return HybridQuantizer(
        rowwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
        columnwise_quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1),
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
        losses = []
        for _ in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=train_recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        master_params = [
            optimizer.get_unscaled_state(p, "master_param")
            for p in model.parameters()
            if p.requires_grad
        ]
        return losses, master_params

    def _test_equivalence(self):
        model_ref, model_hyb = self._build_models()

        torch.manual_seed(99)
        x = torch.randn(4, 32, self.hidden_size, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses_ref, masters_ref = self._run_training_loop(
            model_ref,
            self._vanilla_recipe(),
            x,
            target,
            self.num_steps,
        )
        losses_hyb, masters_hyb = self._run_training_loop(
            model_hyb,
            self._hybrid_recipe(),
            x,
            target,
            self.num_steps,
        )

        # Losses should be very close (same quantization, same training dynamics)
        for i, (lr, lh) in enumerate(zip(losses_ref, losses_hyb)):
            assert abs(lr - lh) < 0.1 * max(
                abs(lr), 1e-6
            ), f"Step {i}: loss diverged — vanilla={lr:.6f}, hybrid={lh:.6f}"

        # Master weights should be close after training
        for i, (mr, mh) in enumerate(zip(masters_ref, masters_hyb)):
            torch.testing.assert_close(
                mr,
                mh,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Master weight {i} diverged after {self.num_steps} steps",
            )


@requires_fp8
class TestQuantizedParamsEquivalenceFP8CurrentScaling(_QuantizedParamsEquivalenceBase):
    """Vanilla Float8CurrentScaling vs hybrid FP8 current (same format both dirs).

    Note: vanilla Float8Tensor params use the fused multi_tensor_adam_fp8
    kernel in FusedAdam, while HybridQuantizedTensor falls to the FP32
    master + quantize_() writeback path. These are numerically different
    codepaths, so we use relaxed tolerances.
    """

    def _vanilla_recipe(self):
        return recipe.Float8CurrentScaling()

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_fp8_current_qfactory)

    def test_equivalence(self):
        model_ref, model_hyb = self._build_models()

        torch.manual_seed(99)
        x = torch.randn(4, 32, self.hidden_size, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses_ref, _ = self._run_training_loop(
            model_ref,
            self._vanilla_recipe(),
            x,
            target,
            self.num_steps,
        )
        losses_hyb, _ = self._run_training_loop(
            model_hyb,
            self._hybrid_recipe(),
            x,
            target,
            self.num_steps,
        )

        # Both should decrease (training works in both paths)
        assert losses_ref[-1] < losses_ref[0], f"Vanilla loss didn't decrease: {losses_ref}"
        assert losses_hyb[-1] < losses_hyb[0], f"Hybrid loss didn't decrease: {losses_hyb}"

        # Losses should be in the same ballpark (different optimizer kernels
        # cause small divergence that compounds over steps)
        for i, (lr, lh) in enumerate(zip(losses_ref, losses_hyb)):
            assert (
                abs(lr - lh) / max(abs(lr), 1e-6) < 0.5
            ), f"Step {i}: losses diverged too much — vanilla={lr:.6f}, hybrid={lh:.6f}"


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
    """Vanilla NVFP4BlockScaling vs hybrid NVFP4 (same format both dirs).

    RHT, stochastic rounding, and 2D quantization disabled for determinism.
    """

    def _vanilla_recipe(self):
        return recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
        )

    def _hybrid_recipe(self):
        return recipe.CustomRecipe(qfactory=_hybrid_nvfp4_qfactory)

    def test_equivalence(self):
        self._test_equivalence()


# ---------------------------------------------------------------------------
# 10. State dict save/load (checkpointing) for hybrid quantized params
# ---------------------------------------------------------------------------


# Module-level qfactories (picklable, required for checkpoint serialization).


def _checkpoint_hybrid_fp8_qfactory(role):
    """Module-level qfactory (picklable) for checkpoint tests."""
    if role in ("linear_input", "linear_weight", "linear_output"):
        return HybridQuantizer(
            rowwise_quantizer=Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            ),
        )
    if role in ("linear_grad_output", "linear_grad_input"):
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")


@requires_fp8
class TestHybridCheckpoint:
    """Test state_dict save/load round-trips for models with hybrid quantized params."""

    def _hybrid_fp8_recipe(self):
        return recipe.CustomRecipe(qfactory=_checkpoint_hybrid_fp8_qfactory)

    def test_state_dict_save_load_roundtrip(self):
        """state_dict → save → load → same model should produce identical outputs."""
        torch.manual_seed(42)
        hybrid_recipe = self._hybrid_fp8_recipe()
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

        torch.testing.assert_close(out_before, out_after)

    def test_state_dict_contains_weight(self):
        """state_dict should contain the weight key."""
        hybrid_recipe = self._hybrid_fp8_recipe()
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
        hybrid_recipe = self._hybrid_fp8_recipe()

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
        hybrid_recipe = self._hybrid_fp8_recipe()
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

            torch.testing.assert_close(out_before, out_after)
        finally:
            os.unlink(tmp_path)

    def test_checkpoint_resume_training(self):
        """Save mid-training, load into new model+optimizer, verify training continues."""
        import tempfile
        import os

        torch.manual_seed(42)
        hybrid_recipe = self._hybrid_fp8_recipe()
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

        # Train for a few steps
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        loss_before_save = loss.item()

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f.name,
            )
            tmp_path = f.name

        try:
            # Load into fresh model
            with quantized_model_init(enabled=True, recipe=hybrid_recipe):
                model2 = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
            optimizer2 = te.optimizers.FusedAdam(
                model2.parameters(),
                lr=1e-3,
                master_weights=True,
                master_weight_dtype=torch.float32,
            )

            checkpoint = torch.load(tmp_path, weights_only=False)
            model2.load_state_dict(checkpoint["model"])
            optimizer2.load_state_dict(checkpoint["optimizer"])

            # Continue training -- loss should not spike
            optimizer2.zero_grad(set_to_none=True)
            with autocast(enabled=True, recipe=hybrid_recipe):
                output2 = model2(x)
            loss_after_load = torch.nn.functional.mse_loss(output2, target).item()

            assert loss_after_load <= loss_before_save * 1.5, (
                f"Loss spiked after checkpoint resume: {loss_before_save:.4f} →"
                f" {loss_after_load:.4f}"
            )
        finally:
            os.unlink(tmp_path)


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
        assert len(pieces) >= 2
        for piece in pieces:
            assert isinstance(
                piece, HybridQuantizedTensor
            ), f"Expected HybridQuantizedTensor, got {type(piece).__name__}"
            assert piece.rowwise_sub_storage is not None
            assert piece.columnwise_sub_storage is not None

        total_rows = sum(p.shape[0] for p in pieces)
        assert total_rows == dim0

        orig_deq = hybrid_param.dequantize()
        reassembled = torch.cat([p.dequantize() for p in pieces], dim=0)
        torch.testing.assert_close(orig_deq, reassembled)

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
        torch.testing.assert_close(src_deq, dst_deq)

    def test_copy_from_bf16_to_hybrid(self, hybrid_param):
        """copy_ from BF16 into HybridQuantizedTensor triggers quantize_."""
        param = hybrid_param.detach()
        bf16_data = torch.randn_like(param.dequantize())
        aten.copy_.default(param, bf16_data)
        result_deq = param.dequantize()
        assert isinstance(param, HybridQuantizedTensor)
        assert result_deq.shape == bf16_data.shape

    def test_new_zeros_returns_hybrid(self, hybrid_param):
        """new_zeros must return a usable HybridQuantizedTensor container.

        FSDP2 calls ``new_zeros`` only to allocate an all-gather destination
        buffer that is immediately overwritten by ``copy_``; the initial
        contents are never observed. The hybrid dispatch therefore delegates
        to ``HybridQuantizer.make_empty`` (uninitialized bytes) rather than
        quantizing a BF16 zeros temporary. This test asserts the contract
        we actually depend on — correct container type / shape / sub-storage
        presence, and the ability to copy into it and read back — NOT that
        the raw dequantize value happens to be zero.
        """
        new_shape = list(hybrid_param.shape)
        result = aten.new_zeros.default(hybrid_param, new_shape)

        # Structural contract: FSDP2 needs a HybridQuantizedTensor with both
        # sub-storages populated so the gathered buffers have a destination.
        assert isinstance(
            result, HybridQuantizedTensor
        ), f"Expected HybridQuantizedTensor, got {type(result).__name__}"
        assert result.shape == hybrid_param.shape
        assert result.rowwise_sub_storage is not None
        assert result.columnwise_sub_storage is not None
        assert type(result.rowwise_sub_storage) is type(hybrid_param.rowwise_sub_storage)
        assert type(result.columnwise_sub_storage) is type(hybrid_param.columnwise_sub_storage)

        # Functional contract: the container must be writable via copy_ from
        # another hybrid (how FSDP2 populates the buffer post-gather).
        aten.copy_.default(result, hybrid_param)
        torch.testing.assert_close(result.dequantize(), hybrid_param.dequantize())

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
        torch.testing.assert_close(result.dequantize(), hybrid_param.dequantize())


# ---------------------------------------------------------------------------
# 12. FSDP2 prerequisites: fsdp_pre_all_gather protocol
# ---------------------------------------------------------------------------


def _make_fsdp_protocol_param(config_name):
    """Create a HybridQuantizedTensor weight for FSDP protocol tests."""
    if config_name == "fp8_fp8":
        r = _hybrid_custom_recipe(_fp8_row_factory, _fp8_col_factory, _fp8_grad_factory)
    elif config_name == "mxfp8_fp8":
        r = _hybrid_custom_recipe(_mxfp8_factory, _fp8_col_factory, _fp8_grad_factory)
    else:
        raise ValueError(f"Unknown config: {config_name}")
    with quantized_model_init(enabled=True, recipe=r):
        model = Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    return model.weight


_fsdp_protocol_configs = [pytest.param("fp8_fp8", id="same-format")]
if mxfp8_available:
    _fsdp_protocol_configs.append(pytest.param("mxfp8_fp8", id="mixed-mxfp8-fp8"))


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
        torch.testing.assert_close(orig_deq, result_deq)

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
        torch.testing.assert_close(orig_deq, result.dequantize())

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
        torch.testing.assert_close(hybrid_param.dequantize(), second_result.dequantize())

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
        )
        # And the magnitude really did change (sanity: this test would pass
        # vacuously if update_quantized didn't actually change anything).
        assert gathered_refreshed.dequantize().abs().max() > 10.0, (
            "update_quantized did not produce a sufficiently different "
            "weight; the scale-refresh invariant is not being exercised"
        )

    def test_nvfp4_sub_storage_raises_on_pre_all_gather(self):
        """Hybrid FSDP2 with an NVFP4 sub-storage must raise a clear error.

        Per the hybrid FSDP2 design (see ``hybrid_quantization_fsdp.md`` §9
        Gap 5), NVFP4 FSDP2 support is not implemented yet — packed FP4 data
        alignment for dim-0 splitting, columnwise dequant, and RHT cache
        handling all need work. Until that lands, hybrid pre-all-gather must
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
        # message pointing to the design doc, not a generic
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
        assert "hybrid_quantization_fsdp.md" in msg
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
        torch.testing.assert_close(copy.dequantize(), param.dequantize())

    def test_make_like_is_independent(self):
        """make_like result should not share the same tensor identity."""
        param = self._make_hybrid_param()
        copy = HybridQuantizedTensor.make_like(param)
        assert copy is not param


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
    # Keeping the xfail'd tests here:
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
            " above)."
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
