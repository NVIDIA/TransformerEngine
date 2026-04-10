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

    def test_compatible_recipe_is_none(self):
        hq = _make_hybrid_quantizer_fp8_row_fp4_col()
        assert hq._get_compatible_recipe() is None


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
