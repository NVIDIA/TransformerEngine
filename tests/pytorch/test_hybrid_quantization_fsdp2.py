# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for HybridQuantizedTensor behavior required by PyTorch FSDP2."""

import io

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from hybrid_quantization_utils import (
    as_data_tensor_tuple as _as_data_tensor_tuple,
    assert_hybrid_tensor_exact as _assert_hybrid_tensor_exact,
    assert_storage_data_exact as _assert_storage_data_exact,
    fp8_e4m3_factory as _fp8_row_factory,
    fp8_e5m2_factory as _fp8_grad_factory,
    hybrid_block_fp8_e4m3_qfactory as _hybrid_block_fp8_qfactory,
    hybrid_custom_recipe as _hybrid_custom_recipe,
    make_fp8_quantizer as _make_fp8_quantizer,
    make_hybrid_quantizer_fp8_row_fp4_col as _make_hybrid_quantizer_fp8_row_fp4_col,
    mxfp8_e4m3_factory as _mxfp8_factory,
)
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
    Float8TensorStorage,
    HybridQuantizedTensor,
    HybridQuantizer,
    IdentityQuantizer,
    Linear,
    MXFP8Quantizer,
    NVFP4Quantizer,
    QuantizedTensor,
    quantized_model_init,
)
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported

_fp8_col_factory = _fp8_row_factory


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    te.is_fp8_block_scaling_available(return_reason=True)
)

_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8 = pytest.mark.xfail(
    condition=not is_non_tn_fp8_gemm_supported(),
    raises=NotImplementedError,
    strict=True,
    reason=(
        "Hopper does not yet support columnwise-only per-tensor FP8 quantization; "
        "tracked by NVIDIA/TransformerEngine#3158"
    ),
)

requires_fp8 = pytest.mark.skipif(
    not fp8_available,
    reason=f"FP8: {reason_for_no_fp8}",
)

requires_fp8_and_nvfp4 = pytest.mark.skipif(
    not (fp8_available and nvfp4_available),
    reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
)

requires_mxfp8 = pytest.mark.skipif(
    not mxfp8_available,
    reason=f"MXFP8: {reason_for_no_mxfp8}",
)


# ---------------------------------------------------------------------------
# 1. __torch_dispatch__ operations that FSDP2 relies on
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


_dispatch_configs = [
    pytest.param(
        "fp8_fp8",
        id="same-format-fp8",
        marks=_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8,
    ),
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
        assert [
            tuple(piece._transpose.shape) for piece in pieces
        ] == expected_transpose_shapes
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
        assert [
            tuple(piece.columnwise_sub_storage._transpose.shape) for piece in pieces
        ] == [
            (16, 5),
            (16, 5),
            (16, 2),
        ]

    @staticmethod
    def _make_valid_transpose_only_float8_tensor(shape=(12, 16)):
        """Build transpose-only storage with known, numerically valid FP8 bytes."""
        source = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        rowwise_quantizer = Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3,
            device="cuda",
            rowwise=True,
            columnwise=False,
        )
        rowwise = rowwise_quantizer(source)
        columnwise_quantizer = Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3,
            device="cuda",
            rowwise=False,
            columnwise=True,
        )
        tensor = Float8Tensor(
            shape=shape,
            dtype=torch.bfloat16,
            data=None,
            data_transpose=rowwise._data.movedim(-1, 0).contiguous(),
            fp8_scale_inv=rowwise._scale_inv.detach().clone(),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=columnwise_quantizer,
            requires_grad=False,
            device="cuda",
        )
        return tensor, rowwise.dequantize()

    @pytest.mark.parametrize("weights_only", (False, True))
    def test_serialization_preserves_transpose_only_payload(self, weights_only):
        tensor, expected = self._make_valid_transpose_only_float8_tensor()
        buffer = io.BytesIO()

        torch.save(tensor, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=weights_only)

        assert loaded._data is None
        assert loaded._transpose_invalid is False
        assert torch.equal(loaded._transpose, tensor._transpose)
        torch.testing.assert_close(loaded.dequantize(), expected, rtol=0.0, atol=0.0)

    def test_dequantize_from_transpose_only_payload(self):
        tensor, expected = self._make_valid_transpose_only_float8_tensor()

        torch.testing.assert_close(tensor.dequantize(), expected, rtol=0.0, atol=0.0)

    def test_slice_and_select_preserve_transpose_only_payload(self):
        tensor, expected = self._make_valid_transpose_only_float8_tensor()

        sliced = tensor[2:8]
        selected = torch.select(tensor, 1, 3)

        assert isinstance(sliced, Float8Tensor)
        assert sliced._data is None
        assert sliced._transpose.shape == torch.Size((16, 6))
        torch.testing.assert_close(
            sliced.dequantize(), expected[2:8], rtol=0.0, atol=0.0
        )
        assert isinstance(selected, Float8Tensor)
        assert selected._data is None
        assert selected._transpose.shape == torch.Size((12,))
        torch.testing.assert_close(
            selected.dequantize(), expected[:, 3], rtol=0.0, atol=0.0
        )

    def test_as_strided_row_shard_preserves_transpose_only_payload(self):
        tensor, expected = self._make_valid_transpose_only_float8_tensor()

        shard = torch.as_strided(tensor, (5, 16), (16, 1), 16)

        assert isinstance(shard, Float8Tensor)
        assert shard._data is None
        assert shard._transpose.shape == torch.Size((16, 5))
        torch.testing.assert_close(
            shard.dequantize(), expected[1:6], rtol=0.0, atol=0.0
        )

    def test_as_strided_falls_back_for_nonrepresentable_layout(self):
        tensor, expected = self._make_valid_transpose_only_float8_tensor()

        output = torch.as_strided(tensor, (16, 12), (1, 16), 0)

        assert type(output) is torch.Tensor
        reference = torch.as_strided(expected, (16, 12), (1, 16), 0)
        torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)

    def test_new_zeros_preserves_transpose_only_layout(self):
        tensor, _ = self._make_valid_transpose_only_float8_tensor()

        output = tensor.new_zeros((5, 16))

        assert isinstance(output, Float8Tensor)
        assert output.shape == torch.Size((5, 16))
        assert output._data is None
        assert output._transpose.shape == torch.Size((16, 5))
        assert output._transpose.dtype == torch.uint8
        assert torch.count_nonzero(output._transpose).item() == 0
        torch.testing.assert_close(
            output.dequantize(),
            torch.zeros((5, 16), dtype=output.dtype, device=output.device),
            rtol=0.0,
            atol=0.0,
        )


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
        plain_source = torch.full(
            (3, 5), 7.0, dtype=torch.float32, device=source.device
        )
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
    @_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8
    @pytest.mark.parametrize(
        "unsupported_dtype", (torch.int32, torch.float64, torch.bool)
    )
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
        assert type(result.columnwise_sub_storage) is type(
            source.columnwise_sub_storage
        )
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
                    buffer is None or torch.count_nonzero(buffer).item() == 0
                    for buffer in buffers
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
        assert type(result.columnwise_sub_storage) is type(
            source.columnwise_sub_storage
        )
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
                    buffer is None or torch.count_nonzero(buffer).item() == 0
                    for buffer in buffers
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
        expected_rowwise = torch.split(
            hybrid_param.rowwise_sub_storage, chunk_size, dim=0
        )
        expected_columnwise = torch.split(
            hybrid_param.columnwise_sub_storage, chunk_size, dim=0
        )
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

    @requires_mxfp8
    def test_split_rejects_mxfp8_high_precision_fallback(self):
        """Fail before wrapping unquantizable MXFP8 shards for Hybrid FSDP2."""
        quantizer = HybridQuantizer(
            rowwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            columnwise_quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
        )
        tensor = quantizer(torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"))

        with pytest.raises(NotImplementedError, match="local shape.*divisible by 32"):
            torch.split(tensor, 16, dim=0)

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

    def test_copy_between_mismatched_usages_raises_atomically(self, hybrid_param):
        quantizer = hybrid_param._quantizer.copy()
        src = quantizer.quantize(torch.ones_like(hybrid_param.dequantize()))
        dst = quantizer.quantize(torch.zeros_like(hybrid_param.dequantize()))
        src.update_usage(columnwise_usage=False)
        dst_before = tuple(
            None if tensor is None else tensor.clone()
            for tensor in _as_data_tensor_tuple(dst)
        )

        with pytest.raises(
            NotImplementedError,
            match="requires matching rowwise/columnwise usages",
        ):
            aten.copy_.default(dst, src)

        dst_after = _as_data_tensor_tuple(dst)
        assert len(dst_after) == len(dst_before)
        for before, after in zip(dst_before, dst_after):
            if before is None:
                assert after is None
            else:
                torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)

    def test_copy_from_bf16_to_hybrid(self, hybrid_param):
        """copy_ from BF16 into HybridQuantizedTensor triggers quantize_."""
        param = hybrid_param.detach()
        bf16_data = torch.randn_like(param.dequantize())
        expected = param._quantizer.quantize(bf16_data)
        aten.copy_.default(param, bf16_data)
        assert isinstance(param, HybridQuantizedTensor)
        _assert_hybrid_tensor_exact(param, expected, context="BF16 copy_")
        torch.testing.assert_close(
            param.dequantize(), expected.dequantize(), rtol=0.0, atol=0.0
        )

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
        assert type(result.rowwise_sub_storage) is type(
            hybrid_param.rowwise_sub_storage
        )
        assert type(result.columnwise_sub_storage) is type(
            hybrid_param.columnwise_sub_storage
        )
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
        _assert_hybrid_tensor_exact(
            result, hybrid_param, context="new_zeros then copy_"
        )

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
# 2. fsdp_pre_all_gather protocol
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


_fsdp_protocol_configs = [
    pytest.param(
        "fp8_fp8",
        id="same-format",
        marks=_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8,
    )
]
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
# 3. fsdp_post_all_gather protocol
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
        _assert_hybrid_tensor_exact(
            second_result, hybrid_param, context="post-all-gather reuse"
        )

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
        _assert_hybrid_tensor_exact(
            result, hybrid_param, context="post-all-gather storage types"
        )
        assert type(result.columnwise_sub_storage) is orig_col_type


# ---------------------------------------------------------------------------
# 4. Pre/post all-gather roundtrip
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
        _assert_hybrid_tensor_exact(
            second_result, hybrid_param, context="FSDP roundtrip reuse"
        )

    @_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8
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
        _assert_hybrid_tensor_exact(
            gathered_refreshed, hybrid_param, context="FSDP scale refresh"
        )
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
# 5. make_like correctness
# ---------------------------------------------------------------------------


@requires_fp8
@_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8
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
        torch.testing.assert_close(
            copy.dequantize(), param.dequantize(), rtol=0.0, atol=0.0
        )

    def test_make_like_is_independent(self):
        """make_like result should not share the same tensor identity."""
        param = self._make_hybrid_param()
        copy = HybridQuantizedTensor.make_like(param)
        assert copy is not param


# ---------------------------------------------------------------------------
# 5b. Hopper-only paths: columnwise-only Float8 sub-storage
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
    """Hopper transpose-only Float8 uses an M-major FSDP transport layout."""

    @staticmethod
    def _make_columnwise_only_float8_storage(shape=(32, 64), value=1.0, quantizer=None):
        source = torch.full(shape, value, device="cuda", dtype=torch.bfloat16)
        byte_quantizer = Float8Quantizer(
            scale=torch.ones(1, device="cuda", dtype=torch.float32),
            amax=torch.zeros(1, device="cuda", dtype=torch.float32),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        rowwise = byte_quantizer(source)
        if quantizer is None:
            quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
                rowwise=False,
                columnwise=True,
            )
        storage = Float8Tensor(
            shape=shape,
            dtype=source.dtype,
            data=None,
            data_transpose=rowwise._data.movedim(-1, 0).contiguous(),
            fp8_scale_inv=rowwise._scale_inv.clone(),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            requires_grad=False,
            device="cuda",
        )
        return storage, rowwise.dequantize()

    @classmethod
    def _make_hybrid_shard(cls, value):
        quantizer = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
            ),
        )
        source = torch.full((32, 64), value, device="cuda", dtype=torch.bfloat16)
        rowwise = quantizer.rowwise_quantizer(source)
        columnwise, expected = cls._make_columnwise_only_float8_storage(
            value=value, quantizer=quantizer.columnwise_quantizer
        )
        hybrid = HybridQuantizedTensor(
            shape=source.shape,
            dtype=source.dtype,
            rowwise_storage=rowwise,
            columnwise_storage=columnwise,
            quantizer=quantizer,
            requires_grad=False,
            device="cuda",
        )
        return hybrid, expected

    def test_columnwise_only_float8_fsdp_buffer_fields_returns_transpose(self):
        out, _ = self._make_columnwise_only_float8_storage()

        assert out._data is None
        assert out._transpose is not None
        assert not out._transpose_invalid
        assert out.fsdp_buffer_fields() == ("_transpose",)

    def test_columnwise_only_float8_fsdp_extract_uses_m_major_transport(self):
        out, _ = self._make_columnwise_only_float8_storage()

        buffers, metadata = out.fsdp_extract_buffers()

        assert len(buffers) == 1
        assert buffers[0].shape == out.shape
        assert buffers[0].is_contiguous()
        torch.testing.assert_close(
            buffers[0], out._transpose.movedim(0, -1).contiguous()
        )
        assert metadata == {
            "field_names": ("_transpose",),
            "transport_layout": "columnwise_m_major",
        }

    def test_columnwise_only_float8_fsdp_assign_restores_columnwise_layout(self):
        out, expected = self._make_columnwise_only_float8_storage()
        buffers, metadata = out.fsdp_extract_buffers()
        gathered = torch.cat((buffers[0], buffers[0]), dim=0)
        rebuilt = Float8Tensor.make_like(out, shape=gathered.shape)

        rebuilt.fsdp_assign_gathered((gathered,), metadata)

        assert rebuilt.shape == torch.Size((64, 64))
        assert rebuilt._data is None
        assert rebuilt._transpose.shape == torch.Size((64, 64))
        assert not rebuilt._transpose_invalid
        torch.testing.assert_close(
            rebuilt._transpose, gathered.movedim(-1, 0).contiguous()
        )
        torch.testing.assert_close(
            rebuilt.dequantize(), torch.cat((expected, expected), dim=0)
        )

    def test_hybrid_fsdp_two_rank_gather_and_buffer_reuse(self):
        shards = [self._make_hybrid_shard(value) for value in (1.0, 2.0)]
        extracted = [
            shard.fsdp_pre_all_gather(
                mesh=None,
                orig_size=shard.shape,
                contiguous_orig_stride=None,
                module=None,
                mp_policy=None,
            )
            for shard, _ in shards
        ]
        gathered = tuple(
            torch.cat((extracted[0][0][i], extracted[1][0][i]), dim=0)
            for i in range(len(extracted[0][0]))
        )
        expected = torch.cat((shards[0][1], shards[1][1]), dim=0)

        rebuilt, _ = shards[0][0].fsdp_post_all_gather(
            gathered, extracted[0][1], torch.bfloat16
        )

        assert rebuilt.shape == torch.Size((64, 64))
        assert rebuilt._columnwise_storage.shape == torch.Size((64, 64))
        assert rebuilt._columnwise_storage._data is None
        assert rebuilt._columnwise_storage._transpose.shape == torch.Size((64, 64))
        torch.testing.assert_close(rebuilt._columnwise_storage.dequantize(), expected)

        reused, _ = shards[0][0].fsdp_post_all_gather(
            gathered, extracted[0][1], torch.bfloat16, out=rebuilt
        )
        assert reused is rebuilt
        assert reused._columnwise_storage._transpose.shape == torch.Size((64, 64))
        torch.testing.assert_close(reused._columnwise_storage.dequantize(), expected)

    def test_fsdp_buffer_fields_falls_back_to_data_when_both_present(self):
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
        )
        out = quantizer(torch.randn(32, 64, device="cuda", dtype=torch.bfloat16))

        assert out._data is not None
        assert out._transpose is not None
        assert out.fsdp_buffer_fields() == ("_data",)
        buffers, metadata = out.fsdp_extract_buffers()
        assert len(buffers) == 1
        assert buffers[0] is out._data
        assert metadata == {
            "field_names": ("_data",),
            "transport_layout": "native",
        }


@requires_hopper_fp8
class TestPerTensorFloat8ColumnwiseOnlyGuard:
    """Per-tensor FP8 cannot yet emit only the physical transpose."""

    @staticmethod
    def _make_quantizer(scaling):
        if scaling == "current":
            return Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
                rowwise=False,
                columnwise=True,
            )
        return Float8Quantizer(
            scale=torch.ones(1, device="cuda", dtype=torch.float32),
            amax=torch.zeros(1, device="cuda", dtype=torch.float32),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=False,
            columnwise=True,
        )

    @pytest.mark.parametrize("scaling", ("current", "delayed"))
    def test_initial_quantize_raises_not_implemented(self, scaling):
        quantizer = self._make_quantizer(scaling)
        source = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(
            NotImplementedError,
            match="Columnwise-only per-tensor FP8 quantization is not implemented",
        ):
            quantizer(source)

    @pytest.mark.parametrize("scaling", ("current", "delayed"))
    def test_update_quantized_raises_not_implemented(self, scaling):
        quantizer = self._make_quantizer(scaling)
        source = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        output = quantizer.make_empty(
            source.shape,
            dtype=source.dtype,
            device=source.device,
        )
        assert output._data is None
        assert output._transpose is not None

        with pytest.raises(
            NotImplementedError,
            match="Columnwise-only per-tensor FP8 quantization is not implemented",
        ):
            quantizer.update_quantized(source, output)


@requires_hopper_fp8
@_XFAIL_HOPPER_COLUMNWISE_PER_TENSOR_FP8
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
            out._rowwise_storage._transpose = torch.zeros_like(
                out._rowwise_storage._data
            ).t()
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
        out2, _ = param.fsdp_post_all_gather(
            sharded_tensors, metadata, param.dtype, out=out
        )

        # After fsdp_post_all_gather, the rowwise sub-quantizer is pinned
        # columnwise=False, so update_usage(rowwise=True, columnwise=False)
        # must clear the stale _transpose (preventing the silent
        # stale-cache regression on Hopper).
        assert out2._rowwise_storage._transpose is None or (
            out2._rowwise_storage._transpose_invalid
            and id(out2._rowwise_storage._transpose) != stale_transpose_id
        ), "Stale _transpose was not invalidated after fsdp_post_all_gather"
