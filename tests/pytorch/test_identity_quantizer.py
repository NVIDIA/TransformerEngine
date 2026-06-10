# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for IdentityQuantizer (high-precision passthrough) and its use as a
per-direction component of HybridQuantizer to express mixed forward/backward
precision via the CustomRecipe + qfactory machinery. Scoped to te.Linear,
single GPU.
"""

import io

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    HybridQuantizer,
    HybridQuantizedTensor,
    IdentityQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
from transformer_engine.pytorch.tensor.identity_tensor import IdentityTensor
from transformer_engine.pytorch.tensor.storage.identity_tensor_storage import (
    IdentityTensorStorage,
)

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


# ── Module-level qfactories (picklable / autocast-friendly) ──────────


def identity_all_factory(role):  # pylint: disable=unused-argument
    """Whole layer in high precision: Identity for every slot."""
    return IdentityQuantizer()


def _fp8_cs(fp8_dtype=tex.DType.kFloat8E4M3):
    return Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device="cuda")


def _mxfp8(fp8_dtype=tex.DType.kFloat8E4M3):
    return MXFP8Quantizer(fp8_dtype=fp8_dtype)


def _float8_blockwise(fp8_dtype=tex.DType.kFloat8E4M3):
    return Float8BlockQuantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)


def _nvfp4():
    return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)


_HYBRID_IDENTITY_FORMATS = [
    pytest.param(
        "fp8_current",
        marks=pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}"),
    ),
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}"),
    ),
    pytest.param(
        "float8_blockwise",
        marks=pytest.mark.skipif(
            not fp8_block_scaling_available,
            reason=f"Float8Blockwise: {reason_for_no_fp8_block_scaling}",
        ),
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(
            not (fp8_available and nvfp4_available),
            reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
        ),
    ),
]

_HYBRID_IDENTITY_RECOMPUTE_FORMATS = [
    pytest.param(
        "fp8_current",
        marks=pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}"),
    ),
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}"),
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(
            not (fp8_available and nvfp4_available),
            reason=f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}",
        ),
    ),
]


def _format_quantizer(format_name):
    if format_name == "fp8_current":
        return _fp8_cs(tex.DType.kFloat8E4M3)
    if format_name == "mxfp8":
        return _mxfp8(tex.DType.kFloat8E4M3)
    if format_name == "float8_blockwise":
        return _float8_blockwise(tex.DType.kFloat8E4M3)
    if format_name == "nvfp4":
        return _nvfp4()
    raise ValueError(format_name)


def _hybrid_quantized_fwd_identity_bwd_factory(format_name):
    def qfactory(role):
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        if is_linear and role.tensor_type in ("grad_output", "grad_input"):
            return IdentityQuantizer()
        return HybridQuantizer(
            rowwise_quantizer=_format_quantizer(format_name),
            columnwise_quantizer=IdentityQuantizer(),
        )

    return qfactory


def fwd_hp_bwd_fp8_factory(role):
    """High-precision forward, FP8 backward (per-direction via hybrid)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return _fp8_cs(tex.DType.kFloat8E5M2)
    return HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=_fp8_cs(tex.DType.kFloat8E4M3),
    )


def fwd_fp8_bwd_hp_factory(role):
    """FP8 forward, high-precision backward (per-direction via hybrid)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return IdentityQuantizer()
    return HybridQuantizer(
        rowwise_quantizer=_fp8_cs(tex.DType.kFloat8E4M3),
        columnwise_quantizer=IdentityQuantizer(),
    )


def hybrid_all_identity_factory(role):
    """All directions high precision, expressed through the hybrid container.

    weight / input / output -> Hybrid(Identity, Identity); grad -> Identity.
    Exercises the HybridQuantizedTensor path with Identity sub-storages in both
    directions (distinct from the non-hybrid whole-layer-HP path).
    """
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return IdentityQuantizer()
    return HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=IdentityQuantizer(),
    )


def fp8_fwd_factory(role):
    """Plain FP8 current scaling for every slot (E4M3 fwd, E5M2 grad).

    Used with ``backward_override="high_precision"`` as the reference that the
    per-direction Identity machinery (``fwd_fp8_bwd_hp_factory``) must reproduce
    bitwise: same FP8 forward, high-precision backward.
    """
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return _fp8_cs(tex.DType.kFloat8E5M2)
    return _fp8_cs(tex.DType.kFloat8E4M3)


def _offload_roundtrip(tensor):
    from transformer_engine.pytorch.cpu_offload import OffloadableLayerState

    stream = torch.cuda.Stream()
    state = OffloadableLayerState(offload_stream=stream)
    tid = state.push_tensor(tensor)
    state.start_offload()
    state.release_activation_forward_gpu_memory()
    state.start_reload()
    reloaded = state.pop_tensor(tid)
    torch.cuda.synchronize()
    try:
        return reloaded
    finally:
        state.release_all_memory()


# ── Unit tests ───────────────────────────────────────────────────────


class TestIdentityQuantizerUnit:
    """IdentityQuantizer / IdentityTensorStorage basic behavior."""

    def test_quantize_returns_identity_tensor(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        out = IdentityQuantizer()(x)
        assert isinstance(out, IdentityTensor)

    def test_internal_returns_storage(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        q = IdentityQuantizer()
        q.internal = True
        out = q(x)
        assert isinstance(out, IdentityTensorStorage)
        assert not isinstance(out, IdentityTensor)

    def test_grouped_split_all_identity_uses_plain_tensor_views(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _split_quantize_with_identity_fallback,
        )

        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        m_splits = [3, 5]
        quantizers = [IdentityQuantizer(), IdentityQuantizer()]

        out = _split_quantize_with_identity_fallback(
            x, m_splits, quantizers, activation_dtype=torch.bfloat16
        )

        assert all(isinstance(t, torch.Tensor) for t in out)
        assert not any(isinstance(t, IdentityTensorStorage) for t in out)
        for actual, expected in zip(out, torch.split(x, m_splits)):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        cast_quantizers = [
            IdentityQuantizer(dtype=torch.float32),
            IdentityQuantizer(dtype=torch.float32),
        ]
        cast_out = _split_quantize_with_identity_fallback(
            x, m_splits, cast_quantizers, activation_dtype=torch.bfloat16
        )
        assert all(isinstance(t, IdentityTensorStorage) for t in cast_out)
        assert all(t.dequantize().dtype == torch.float32 for t in cast_out)

    def test_grouped_split_rejects_mixed_identity_and_quantized_operands(self):
        from transformer_engine.pytorch.module.grouped_linear import (
            _split_quantize_with_identity_fallback,
        )

        x = torch.empty(64, 64, dtype=torch.bfloat16)
        m_splits = [32, 32]
        cases = [
            [IdentityQuantizer(), _mxfp8(tex.DType.kFloat8E4M3)],
            [
                HybridQuantizer(
                    rowwise_quantizer=IdentityQuantizer(),
                    columnwise_quantizer=IdentityQuantizer(),
                ),
                HybridQuantizer(
                    rowwise_quantizer=_mxfp8(tex.DType.kFloat8E4M3),
                    columnwise_quantizer=IdentityQuantizer(),
                ),
            ],
        ]

        for quantizers in cases:
            with pytest.raises(ValueError, match="mixes Identity-backed and non-Identity-backed"):
                _split_quantize_with_identity_fallback(
                    x,
                    m_splits,
                    quantizers,
                    activation_dtype=torch.bfloat16,
                )

    def test_hybrid_split_forwards_disable_bulk_allocation_to_both_directions(self, monkeypatch):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear
        from transformer_engine.pytorch.module.grouped_linear import _hybrid_split_quantize

        calls = []

        def fake_split_quantize(tensor, m_splits, quantizers, *, disable_bulk_allocation=False):
            calls.append(disable_bulk_allocation)
            return [
                quantizer(tensor_part)
                for tensor_part, quantizer in zip(torch.split(tensor, m_splits), quantizers)
            ]

        monkeypatch.setattr(grouped_linear.tex, "split_quantize", fake_split_quantize)
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        m_splits = [3, 5]
        quantizers = [
            HybridQuantizer(
                rowwise_quantizer=IdentityQuantizer(),
                columnwise_quantizer=IdentityQuantizer(),
            )
            for _ in m_splits
        ]

        out = _hybrid_split_quantize(
            x,
            m_splits,
            quantizers,
            disable_bulk_allocation=True,
        )

        assert calls == [True, True]
        assert len(out) == len(m_splits)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_grouped_linear_cpu_offload_disables_bulk_allocation_for_hybrid_input(
        self, monkeypatch
    ):
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear

        class StopAfterFlagCapture(RuntimeError):
            pass

        def qfactory(role):
            if role is not None and role.module_type == "grouped_linear":
                return HybridQuantizer(
                    rowwise_quantizer=_fp8_cs(tex.DType.kFloat8E4M3),
                    columnwise_quantizer=_fp8_cs(tex.DType.kFloat8E4M3),
                )
            return _fp8_cs(tex.DType.kFloat8E4M3)

        calls = []

        def fake_hybrid_split_quantize(
            tensor, m_splits, quantizers, *, disable_bulk_allocation=False
        ):
            del tensor, m_splits, quantizers
            calls.append(disable_bulk_allocation)
            raise StopAfterFlagCapture("captured hybrid split kwargs")

        monkeypatch.setattr(grouped_linear, "is_cpu_offload_enabled", lambda: True)
        monkeypatch.setattr(grouped_linear, "_hybrid_split_quantize", fake_hybrid_split_quantize)

        model = te.GroupedLinear(2, 64, 64, params_dtype=torch.bfloat16).cuda()
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        m_splits = torch.tensor([32, 32], device="cuda", dtype=torch.int32)

        with pytest.raises(StopAfterFlagCapture):
            with te.autocast(enabled=True, recipe=CustomRecipe(qfactory=qfactory)):
                model(x, m_splits=m_splits)

        assert calls == [True]

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_grouped_linear_rejects_mixed_identity_weight_quantizers(self):
        weight_count = 0

        def qfactory(role):
            nonlocal weight_count
            if role is not None and role.module_type == "grouped_linear":
                if role.tensor_type == "weight":
                    weight_count += 1
                    if weight_count == 1:
                        return IdentityQuantizer()
                return _mxfp8(tex.DType.kFloat8E4M3)
            return _mxfp8(tex.DType.kFloat8E4M3)

        model = te.GroupedLinear(2, 64, 64, params_dtype=torch.bfloat16).cuda()
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        m_splits = torch.tensor([32, 32], device="cuda", dtype=torch.int32)

        with pytest.raises(ValueError, match="mixes Identity-backed and non-Identity-backed"):
            with te.autocast(enabled=True, recipe=CustomRecipe(qfactory=qfactory)):
                model(x, m_splits=m_splits)

    def test_identity_contiguous_preserves_wrapper_and_values(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16).t()
        t = IdentityQuantizer()(x)

        out = t.contiguous()

        assert isinstance(out, IdentityTensor)
        assert out.is_contiguous()
        torch.testing.assert_close(out.dequantize(), x.contiguous(), rtol=0.0, atol=0.0)

    def test_hybrid_identity_contiguous_preserves_wrapper_and_values(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        t = q(x)

        out = t.contiguous()

        assert out is t
        assert isinstance(out, HybridQuantizedTensor)
        assert isinstance(out.rowwise_sub_storage, IdentityTensor)
        assert isinstance(out.columnwise_sub_storage, IdentityTensor)
        torch.testing.assert_close(out.dequantize(), x, rtol=0.0, atol=0.0)

    def test_hybrid_identity_cpu_preserves_nested_storage_types(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        t = q(x)

        out = t.cpu()

        assert isinstance(out, HybridQuantizedTensor)
        assert out.device.type == "cpu"
        assert isinstance(out.rowwise_sub_storage, IdentityTensor)
        assert isinstance(out.columnwise_sub_storage, IdentityTensor)
        assert out.rowwise_sub_storage.device.type == "cpu"
        assert out.columnwise_sub_storage.device.type == "cpu"
        torch.testing.assert_close(out.dequantize(), x.cpu(), rtol=0.0, atol=0.0)
        assert len(out.get_data_tensors()) == 4
        out.copy_(torch.ones_like(x, device="cpu"))
        torch.testing.assert_close(
            out.dequantize(), torch.ones_like(x, device="cpu"), rtol=0.0, atol=0.0
        )

    def test_hybrid_quantizer_copy_preserves_parent_flags(self):
        q = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        q.set_usage(rowwise=True, columnwise=False)
        q.internal = True
        q.optimize_for_gemm = True

        out = q.copy()

        assert isinstance(out, HybridQuantizer)
        assert out is not q
        assert out.rowwise_quantizer is not q.rowwise_quantizer
        assert out.columnwise_quantizer is not q.columnwise_quantizer
        assert out.rowwise_usage is True
        assert out.columnwise_usage is False
        assert out.internal is True
        assert out.optimize_for_gemm is True
        assert out.rowwise_quantizer.rowwise_usage is True
        assert out.rowwise_quantizer.columnwise_usage is False
        assert out.columnwise_quantizer.rowwise_usage is False
        assert out.columnwise_quantizer.columnwise_usage is True

    def test_te_ops_basic_linear_accepts_hybrid_identity_quantized_weight(self):
        import transformer_engine.pytorch.ops as te_ops

        def qfactory(role):  # pylint: disable=unused-argument
            return HybridQuantizer(
                rowwise_quantizer=IdentityQuantizer(),
                columnwise_quantizer=IdentityQuantizer(),
            )

        custom_recipe = CustomRecipe(qfactory=qfactory)
        with te.quantized_model_init(enabled=True, recipe=custom_recipe):
            op = te_ops.BasicLinear(16, 16, device="cuda", dtype=torch.bfloat16)

        x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with te.autocast(enabled=True, recipe=custom_recipe):
            y = op(x)
        y.sum().backward()

        assert isinstance(op.weight, HybridQuantizedTensor)
        assert x.grad is not None

    @pytest.mark.parametrize(
        "qfactory",
        [
            pytest.param(lambda role: IdentityQuantizer(), id="identity"),
            pytest.param(
                lambda role: HybridQuantizer(
                    rowwise_quantizer=IdentityQuantizer(),
                    columnwise_quantizer=IdentityQuantizer(),
                ),
                id="hybrid_identity",
            ),
        ],
    )
    def test_te_ops_quantize_then_gelu_accepts_identity_backed_tensors(self, qfactory):
        import transformer_engine.pytorch.ops as te_ops

        model = te_ops.Sequential(te_ops.Quantize(forward=True), te_ops.GELU())
        x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        with te.autocast(enabled=True, recipe=CustomRecipe(qfactory=qfactory)):
            y = model(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape

    def test_hybrid_fsdp_rejects_storage_only_sub_storages(self):
        row_quantizer = IdentityQuantizer()
        col_quantizer = IdentityQuantizer()
        row_quantizer.internal = True
        col_quantizer.internal = True
        q = HybridQuantizer(
            rowwise_quantizer=row_quantizer,
            columnwise_quantizer=col_quantizer,
        )
        t = q(torch.randn(8, 16, device="cuda", dtype=torch.bfloat16))

        with pytest.raises(NotImplementedError, match="storage-only rowwise sub-storage"):
            t.fsdp_pre_all_gather(
                mesh=None,
                orig_size=t.shape,
                contiguous_orig_stride=t.stride(),
                module=None,
                mp_policy=None,
            )

    def test_hybrid_quantizer_rejects_nested_quantizer_requests(self):
        from transformer_engine.pytorch.quantization import DelayedScalingRequest

        with pytest.raises(TypeError, match="does not support nested QuantizerRequest"):
            HybridQuantizer(
                rowwise_quantizer=DelayedScalingRequest(),
                columnwise_quantizer=IdentityQuantizer(),
            )

    def test_fp8_dpa_rejects_identity_quantizer_with_type_error(self):
        from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            META_DO,
            META_DP,
            META_DQKV,
            META_O,
            META_QKV,
            META_S,
        )

        n_fwd = max(META_QKV, META_S, META_O) + 1
        n_bwd = max(META_DO, META_DP, META_DQKV) + 1
        quantizers = {
            "scaling_fwd": [IdentityQuantizer() for _ in range(n_fwd)],
            "scaling_bwd": [IdentityQuantizer() for _ in range(n_bwd)],
        }

        with pytest.raises(TypeError, match="FP8 attention requires FP8-compatible quantizers"):
            dpa_utils.get_attention_quantizers(True, quantizers)

    def test_dequantize_bitwise_identical(self):
        x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        out = IdentityQuantizer()(x)
        assert torch.equal(out.dequantize(), x)

    def test_dtype_cast(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        out = IdentityQuantizer(dtype=torch.bfloat16)(x)
        assert out.dequantize().dtype == torch.bfloat16

    def test_update_usage_is_noop(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        q = IdentityQuantizer()
        q.internal = True
        st = q(x)
        st.update_usage(rowwise_usage=False, columnwise_usage=True)
        assert torch.equal(st.dequantize(), x)
        assert st.get_usages() == {"rowwise": True, "columnwise": True}

    def test_save_restore_roundtrip(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        q = IdentityQuantizer()
        q.internal = True
        st = q(x)
        tensors, _ = st.prepare_for_saving()
        assert st._hp_data is None
        leftover = st.restore_from_saved(tensors)
        assert leftover == []
        assert torch.equal(st.dequantize(), x)

    def test_update_quantized_inplace(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        q = IdentityQuantizer()
        st = q.make_empty((4, 8), dtype=torch.bfloat16, device="cuda")
        q.update_quantized(x, st)
        assert torch.equal(st.dequantize(), x)

    def test_tensor_ops_preserve_identity_and_values(self):
        x = torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(6, 4)
        t = IdentityQuantizer()(x)

        view = t.view(3, 8)
        assert isinstance(view, IdentityTensor)
        torch.testing.assert_close(view.dequantize(), x.view(3, 8), rtol=0.0, atol=0.0)

        pieces = torch.split(t, 2, dim=0)
        assert all(isinstance(piece, IdentityTensor) for piece in pieces)
        for piece, ref in zip(pieces, torch.split(x, 2, dim=0)):
            torch.testing.assert_close(piece.dequantize(), ref, rtol=0.0, atol=0.0)

        sliced = t[1:5:2]
        assert isinstance(sliced, IdentityTensor)
        torch.testing.assert_close(sliced.dequantize(), x[1:5:2], rtol=0.0, atol=0.0)

        strided = torch.as_strided(t, (3, 4), (4, 1), 4)
        assert isinstance(strided, IdentityTensor)
        torch.testing.assert_close(strided.dequantize(), x[1:4], rtol=0.0, atol=0.0)

        cloned = torch.clone(t)
        assert isinstance(cloned, IdentityTensor)
        torch.testing.assert_close(cloned.dequantize(), x, rtol=0.0, atol=0.0)

        zeros = t.new_zeros((2, 3))
        assert isinstance(zeros, IdentityTensor)
        torch.testing.assert_close(
            zeros.dequantize(),
            torch.zeros((2, 3), device="cuda", dtype=x.dtype),
            rtol=0.0,
            atol=0.0,
        )

        dst = IdentityQuantizer().make_empty(x.shape, dtype=x.dtype, device="cuda")
        dst.copy_(t)
        torch.testing.assert_close(dst.dequantize(), x, rtol=0.0, atol=0.0)

    def test_fsdp_pre_post_all_gather_roundtrip(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        t = IdentityQuantizer()(x)
        sharded_tensors, metadata = t.fsdp_pre_all_gather(
            mesh=None, orig_size=t.shape, contiguous_orig_stride=None, module=None, mp_policy=None
        )
        gathered, outputs = t.fsdp_post_all_gather(sharded_tensors, metadata, t.dtype, out=None)
        assert isinstance(gathered, IdentityTensor)
        assert outputs is sharded_tensors
        torch.testing.assert_close(gathered.dequantize(), x, rtol=0.0, atol=0.0)

        reuse, _ = t.fsdp_post_all_gather(sharded_tensors, metadata, t.dtype, out=gathered)
        assert reuse is gathered
        torch.testing.assert_close(reuse.dequantize(), x, rtol=0.0, atol=0.0)

    def test_torch_weights_only_load_preserves_identity_tensor(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        t = IdentityQuantizer()(x)
        buffer = io.BytesIO()
        torch.save(t, buffer)
        buffer.seek(0)

        loaded = torch.load(buffer, weights_only=True)

        assert isinstance(loaded, IdentityTensor)
        assert isinstance(loaded._quantizer, IdentityQuantizer)
        torch.testing.assert_close(loaded.dequantize(), x, rtol=0.0, atol=0.0)

    def test_torch_weights_only_load_preserves_hybrid_identity_tensor(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=IdentityQuantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
        t = q(x)
        buffer = io.BytesIO()
        torch.save(t, buffer)
        buffer.seek(0)

        loaded = torch.load(buffer, weights_only=True)

        assert isinstance(loaded, HybridQuantizedTensor)
        assert isinstance(loaded._quantizer, HybridQuantizer)
        assert isinstance(loaded._rowwise_storage, IdentityTensor)
        assert isinstance(loaded._columnwise_storage, IdentityTensor)
        torch.testing.assert_close(loaded.dequantize(), x, rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_torch_weights_only_load_preserves_hybrid_mxfp8_identity_tensor(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=_mxfp8(tex.DType.kFloat8E4M3),
            columnwise_quantizer=IdentityQuantizer(),
        )
        t = q(x)
        expected = t.dequantize()
        buffer = io.BytesIO()
        torch.save(t, buffer)
        buffer.seek(0)

        loaded = torch.load(buffer, weights_only=True)

        assert isinstance(loaded, HybridQuantizedTensor)
        assert isinstance(loaded._quantizer, HybridQuantizer)
        assert isinstance(loaded._columnwise_storage, IdentityTensor)
        torch.testing.assert_close(loaded.dequantize(), expected, rtol=0.0, atol=0.0)

    def test_cpu_offload_roundtrip_identity_exact(self):
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        t = IdentityQuantizer()(x)

        reloaded = _offload_roundtrip(t)

        assert isinstance(reloaded, IdentityTensor)
        torch.testing.assert_close(reloaded.dequantize(), x, rtol=0.0, atol=0.0)

    def test_replace_raw_data_preserves_identity_values(self):
        from transformer_engine.pytorch.tensor.utils import replace_raw_data

        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        t = IdentityQuantizer()(x)
        new_raw = torch.empty_like(x)
        replace_raw_data(t, new_raw)
        assert t._hp_data is new_raw
        torch.testing.assert_close(t.dequantize(), x, rtol=0.0, atol=0.0)

    def test_quantize_master_weights_identity_exact_nonzero_offset(self):
        from transformer_engine.pytorch.tensor.utils import (
            post_all_gather_processing,
            quantize_master_weights,
        )

        group = _ensure_single_rank_dp_group()
        q = IdentityQuantizer()
        weight = q.make_empty((4, 8), dtype=torch.bfloat16, device="cuda")
        original = torch.randn_like(weight.dequantize())
        q.update_quantized(original, weight)

        master_full = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        start_offset = master_full.numel() // 2
        master_shard = master_full.reshape(-1)[start_offset:].contiguous()

        quantize_master_weights([weight], [master_shard], [start_offset], group=group)
        post_all_gather_processing([weight])

        expected = original.clone()
        expected.reshape(-1)[start_offset:] = master_shard.to(torch.bfloat16)
        torch.testing.assert_close(weight.dequantize(), expected, rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_quantize_master_weights_hybrid_identity_fp8_current(self):
        from transformer_engine.pytorch.tensor.utils import (
            post_all_gather_processing,
            quantize_master_weights,
        )

        group = _ensure_single_rank_dp_group()
        recipe = CustomRecipe(qfactory=fwd_hp_bwd_fp8_factory)
        torch.manual_seed(123)
        with te.quantized_model_init(enabled=True, recipe=recipe):
            model = te.Linear(32, 32, bias=False, params_dtype=torch.bfloat16).cuda()
        weight = model.weight
        assert isinstance(weight, HybridQuantizedTensor)
        assert isinstance(weight._rowwise_storage, IdentityTensorStorage)

        master = torch.randn_like(weight.dequantize(dtype=torch.float32)).reshape(-1).contiguous()
        quantize_master_weights([weight], [master], [0], group=group)
        post_all_gather_processing([weight])

        expected = master.to(torch.bfloat16)
        row_deq = weight._rowwise_storage.dequantize().reshape(-1)
        col_deq = weight._columnwise_storage.dequantize(dtype=torch.float32).reshape(-1)
        torch.testing.assert_close(row_deq, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(col_deq, master, rtol=0.125, atol=0.1)


# ── te.Linear integration ────────────────────────────────────────────


def _make_linears(in_f, out_f, seed=1234, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ref = te.Linear(in_f, out_f, params_dtype=dtype).cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    test = te.Linear(in_f, out_f, params_dtype=dtype).cuda()
    with torch.no_grad():
        for p_test, p_ref in zip(test.parameters(), ref.parameters()):
            p_test.copy_(p_ref)
    return ref, test


def _rel_l2_error(actual, reference):
    """Relative L2-norm error ``||actual - reference|| / ||reference||``.

    The right metric for comparing a quantized result to a high-precision
    reference: element-wise ``rtol`` is meaningless here because reference grads
    contain near-zero entries (relative error on ~1e-12 values explodes), while
    the aggregate norm error reflects the true quantization noise.
    """
    a = actual.float()
    b = reference.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def _ensure_single_rank_dp_group():
    import pathlib
    import tempfile

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


def _fwd_bwd(model, x, recipe=None):
    x = x.clone().detach().requires_grad_(True)
    if recipe is not None:
        with te.autocast(enabled=True, recipe=recipe):
            y = model(x)
    else:
        y = model(x)
    torch.manual_seed(99)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    wgrads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return y.detach().clone(), x.grad.detach().clone(), wgrads


def _fwd_bwd_checkpoint(model, x, recipe, use_reentrant):
    x = x.clone().detach().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe):
        if use_reentrant is None:
            y = model(x)
        else:
            y = te.checkpoint(model, x, use_reentrant=use_reentrant)
    torch.manual_seed(99)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    wgrads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return y.detach().clone(), x.grad.detach().clone(), wgrads


_IDENTITY_MODULE_NAMES = (
    "Linear",
    "LayerNormLinear",
    "LayerNormMLP",
    "GroupedLinear",
    "TransformerLayer",
)


def _make_identity_module(module_name, seed=1234, dtype=torch.bfloat16):
    hidden_size = 64
    ffn_hidden_size = 128
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if module_name == "Linear":
        return te.Linear(hidden_size, hidden_size, params_dtype=dtype).cuda()
    if module_name == "LayerNormLinear":
        return te.LayerNormLinear(hidden_size, hidden_size, params_dtype=dtype).cuda()
    if module_name == "LayerNormMLP":
        return te.LayerNormMLP(hidden_size, ffn_hidden_size, params_dtype=dtype).cuda()
    if module_name == "GroupedLinear":
        return te.GroupedLinear(2, hidden_size, hidden_size, params_dtype=dtype).cuda()
    if module_name == "TransformerLayer":
        return te.TransformerLayer(
            hidden_size,
            ffn_hidden_size,
            4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=dtype,
        ).cuda()
    raise ValueError(module_name)


def _make_identity_module_pair(module_name, seed=1234, dtype=torch.bfloat16):
    ref = _make_identity_module(module_name, seed=seed, dtype=dtype)
    test = _make_identity_module(module_name, seed=seed + 1, dtype=dtype)
    with torch.no_grad():
        for p_test, p_ref in zip(test.parameters(), ref.parameters()):
            p_test.copy_(p_ref)
    return ref, test


def _identity_module_input(module_name):
    torch.manual_seed(7)
    if module_name == "TransformerLayer":
        return torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    return torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)


def _identity_module_forward(module_name, module, x):
    if module_name == "GroupedLinear":
        m_splits = torch.tensor([8, 8], device="cuda", dtype=torch.int32)
        return module(x, m_splits=m_splits)
    return module(x)


def _fwd_bwd_module(module_name, model, x, recipe=None):
    x = x.clone().detach().requires_grad_(True)
    if recipe is not None:
        with te.autocast(enabled=True, recipe=recipe):
            y = _identity_module_forward(module_name, model, x)
    else:
        y = _identity_module_forward(module_name, model, x)
    torch.manual_seed(99)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    wgrads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return y.detach().clone(), x.grad.detach().clone(), wgrads


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestIdentityTEModuleCoverage:
    """All-Identity recipes should route every TE module through HP-compatible paths."""

    @pytest.mark.parametrize("module_name", _IDENTITY_MODULE_NAMES)
    @pytest.mark.parametrize(
        "qfactory",
        [
            pytest.param(identity_all_factory, id="plain_identity"),
            pytest.param(hybrid_all_identity_factory, id="hybrid_identity"),
        ],
    )
    def test_identity_recipe_matches_bf16_bitwise(self, module_name, qfactory):
        ref, test = _make_identity_module_pair(module_name, seed=7300)
        x = _identity_module_input(module_name)
        recipe = CustomRecipe(qfactory=qfactory)

        y_ref, dx_ref, wg_ref = _fwd_bwd_module(module_name, ref, x, recipe=None)
        y_id, dx_id, wg_id = _fwd_bwd_module(module_name, test, x, recipe=recipe)

        # Linear / LayerNormLinear / GroupedLinear route through the same HP
        # math with Identity and should stay bitwise exact. Composite modules
        # can select different fused/unfused BF16 kernel paths after prior FP8
        # tests have warmed TE/CUDA state, so require tight BF16 numerical
        # parity instead of order-dependent bitwise identity.
        kwargs = (
            {"rtol": 0.0, "atol": 0.0}
            if module_name in ("Linear", "LayerNormLinear", "GroupedLinear")
            else {"rtol": 2.0e-2, "atol": 8.0e-3}
        )
        torch.testing.assert_close(y_id, y_ref, **kwargs)
        torch.testing.assert_close(dx_id, dx_ref, **kwargs)
        assert len(wg_id) == len(wg_ref)
        for g_id, g_ref in zip(wg_id, wg_ref):
            torch.testing.assert_close(g_id, g_ref, **kwargs)

    @pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
    def test_grouped_linear_mxfp8_forward_identity_backward_matches_override(self):
        def mxfp8_all_factory(role):  # pylint: disable=unused-argument
            return _mxfp8(tex.DType.kFloat8E4M3)

        def run(model, x, recipe):
            x = x.detach().clone().requires_grad_(True)
            m_splits = torch.tensor([32, 32], device="cuda", dtype=torch.int32)
            with te.autocast(enabled=True, recipe=recipe):
                y = model(x, m_splits=m_splits)
            torch.manual_seed(9001)
            target = torch.randn_like(y)
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            wgrads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
            return y.detach().clone(), x.grad.detach().clone(), wgrads

        torch.manual_seed(8300)
        ref = te.GroupedLinear(2, 64, 64, params_dtype=torch.bfloat16).cuda()
        torch.manual_seed(8301)
        test = te.GroupedLinear(2, 64, 64, params_dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            for p_test, p_ref in zip(test.parameters(), ref.parameters()):
                p_test.copy_(p_ref)

        torch.manual_seed(8302)
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        y_bo, dx_bo, wg_bo = run(
            ref,
            x,
            CustomRecipe(qfactory=mxfp8_all_factory, backward_override="high_precision"),
        )
        y_id, dx_id, wg_id = run(
            test,
            x,
            CustomRecipe(qfactory=_hybrid_quantized_fwd_identity_bwd_factory("mxfp8")),
        )

        torch.testing.assert_close(y_id, y_bo, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_bo, rtol=0.0, atol=0.0)
        assert len(wg_id) == len(wg_bo)
        for g_id, g_bo in zip(wg_id, wg_bo):
            torch.testing.assert_close(g_id, g_bo, rtol=0.0, atol=0.0)


class TestIdentityHybridFormatProtocols:
    SHAPE = (256, 256)
    OFFLOAD_SHAPE = (1024, 1024)

    @pytest.mark.parametrize("format_name", _HYBRID_IDENTITY_FORMATS)
    def test_save_restore_keeps_identity_direction_exact(self, format_name):
        torch.manual_seed(401)
        x = torch.randn(*self.SHAPE, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=_format_quantizer(format_name),
            columnwise_quantizer=IdentityQuantizer(),
        )
        hybrid = q.quantize(x)
        expected_row = hybrid._rowwise_storage.dequantize().clone()
        expected_col = x.clone()

        tensors, obj = hybrid.prepare_for_saving()
        leftover = obj.restore_from_saved(tensors)

        assert leftover == []
        assert isinstance(hybrid._columnwise_storage, IdentityTensorStorage)
        torch.testing.assert_close(
            hybrid._columnwise_storage.dequantize(), expected_col, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            hybrid._rowwise_storage.dequantize(), expected_row, rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("format_name", _HYBRID_IDENTITY_FORMATS)
    def test_cpu_offload_keeps_identity_direction_exact(self, format_name):
        torch.manual_seed(402)
        x = torch.randn(*self.OFFLOAD_SHAPE, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=_format_quantizer(format_name),
            columnwise_quantizer=IdentityQuantizer(),
        )
        hybrid = q.quantize(x)
        expected_row = hybrid._rowwise_storage.dequantize().clone()

        reloaded = _offload_roundtrip(hybrid)

        assert isinstance(reloaded, HybridQuantizedTensor)
        assert isinstance(reloaded._columnwise_storage, IdentityTensorStorage)
        torch.testing.assert_close(reloaded._columnwise_storage.dequantize(), x, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            reloaded._rowwise_storage.dequantize(), expected_row, rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("format_name", ["mxfp8", "float8_blockwise", "nvfp4"])
    def test_quantize_master_weights_per_block_hybrid_identity_rejected(self, format_name):
        if format_name == "mxfp8" and not mxfp8_available:
            pytest.skip(f"MXFP8: {reason_for_no_mxfp8}")
        if format_name == "float8_blockwise" and not fp8_block_scaling_available:
            pytest.skip(f"Float8Blockwise: {reason_for_no_fp8_block_scaling}")
        if format_name == "nvfp4" and not (fp8_available and nvfp4_available):
            pytest.skip(f"FP8: {reason_for_no_fp8}; NVFP4: {reason_for_no_nvfp4}")

        from transformer_engine.pytorch.tensor.utils import quantize_master_weights

        group = _ensure_single_rank_dp_group()
        x = torch.randn(*self.SHAPE, device="cuda", dtype=torch.bfloat16)
        q = HybridQuantizer(
            rowwise_quantizer=_format_quantizer(format_name),
            columnwise_quantizer=IdentityQuantizer(),
        )
        weight = q.quantize(x)
        master = torch.randn_like(x, dtype=torch.float32).reshape(-1).contiguous()

        with pytest.raises(NotImplementedError, match="HybridQuantizer"):
            quantize_master_weights([weight], [master], [0], group=group)

    @pytest.mark.parametrize("format_name", _HYBRID_IDENTITY_RECOMPUTE_FORMATS)
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_activation_recompute_matches_no_checkpoint(self, format_name, use_reentrant):
        recipe = CustomRecipe(qfactory=_hybrid_quantized_fwd_identity_bwd_factory(format_name))
        ref, test = _make_linears(128, 128, seed=440)
        torch.manual_seed(441)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        y_ref, dx_ref, wg_ref = _fwd_bwd_checkpoint(ref, x, recipe, use_reentrant=None)
        y_test, dx_test, wg_test = _fwd_bwd_checkpoint(test, x, recipe, use_reentrant=use_reentrant)

        torch.testing.assert_close(y_test, y_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_test, dx_ref, rtol=0.0, atol=0.0)
        assert len(wg_test) == len(wg_ref)
        for g_test, g_ref in zip(wg_test, wg_ref):
            torch.testing.assert_close(g_test, g_ref, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestIdentityLinear:
    """End-to-end te.Linear with Identity-based recipes."""

    IN_F = 128
    OUT_F = 128
    BATCH = 64

    def _input(self):
        torch.manual_seed(7)
        return torch.randn(self.BATCH, self.IN_F, device="cuda", dtype=torch.bfloat16)

    def test_whole_layer_hp_matches_bf16_bitwise(self):
        """Identity for every slot => all GEMMs high precision => bitwise-equal
        to a plain BF16 te.Linear (no autocast)."""
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        x = self._input()

        y_ref, dx_ref, wg_ref = _fwd_bwd(ref, x, recipe=None)
        y_id, dx_id, wg_id = _fwd_bwd(test, x, recipe=CustomRecipe(qfactory=identity_all_factory))

        torch.testing.assert_close(y_id, y_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_ref, rtol=0.0, atol=0.0)
        assert len(wg_id) == len(wg_ref)
        for g_id, g_ref in zip(wg_id, wg_ref):
            torch.testing.assert_close(g_id, g_ref, rtol=0.0, atol=0.0)

    def test_fwd_hp_bwd_fp8_forward_bitwise(self):
        """High-precision forward must be bitwise-equal to BF16 forward; the
        backward runs in FP8 (finite, close to BF16 within a loose tolerance)."""
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        x = self._input()

        y_ref, dx_ref, wg_ref = _fwd_bwd(ref, x, recipe=None)
        y_h, dx_h, wg_h = _fwd_bwd(test, x, recipe=CustomRecipe(qfactory=fwd_hp_bwd_fp8_factory))

        # Forward is high precision -> bitwise equal.
        torch.testing.assert_close(y_h, y_ref, rtol=0.0, atol=0.0)
        # Backward is FP8 (E4M3 weight col, E5M2 grad) -> relative L2 error vs the
        # BF16 reference reflects pure FP8 quant noise. Measured: dgrad ~5.7e-2,
        # weight-grad ~5.8e-2 (E4M3 ~6% step). Bound 7e-2 keeps a small margin.
        assert torch.isfinite(dx_h).all()
        assert _rel_l2_error(dx_h, dx_ref) < 7e-2
        for g, g_ref in zip(wg_h, wg_ref):
            assert torch.isfinite(g).all()
            if g.dim() == 1:
                # Bias grad = sum(dY) is computed in high precision (dY is bitwise
                # identical since the forward is bitwise), so it must match exactly.
                torch.testing.assert_close(g, g_ref, rtol=0.0, atol=0.0)
            else:
                assert _rel_l2_error(g, g_ref) < 7e-2

    def test_fwd_fp8_bwd_hp_runs_and_backward_high_precision(self):
        """FP8 forward + high-precision backward. Forward differs from BF16
        (quantized), backward GEMMs run in high precision."""
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        x = self._input()

        y_ref, dx_ref, _ = _fwd_bwd(ref, x, recipe=None)
        y_q, dx_q, wg_q = _fwd_bwd(test, x, recipe=CustomRecipe(qfactory=fwd_fp8_bwd_hp_factory))

        # Forward is FP8 (E4M3) -> relative L2 error vs BF16 is the quant noise.
        # Measured ~3.7e-2; bound 5e-2.
        assert torch.isfinite(y_q).all()
        assert _rel_l2_error(y_q, y_ref) < 5e-2
        # Backward GEMMs run in high precision. dgrad differs from the BF16
        # reference only because the FP8 forward perturbs dY; measured ~1.2e-2,
        # bound 3e-2 (the bitwise HP-backward guarantee is locked by the
        # backward_override equivalence test below).
        assert torch.isfinite(dx_q).all()
        assert _rel_l2_error(dx_q, dx_ref) < 3e-2
        for g in wg_q:
            assert torch.isfinite(g).all()

    def test_hybrid_all_identity_matches_bf16_bitwise(self):
        """All-Identity through the *hybrid* container must be bitwise-equal to a
        plain BF16 te.Linear. Complements the non-hybrid whole-layer-HP test: this
        exercises HybridQuantizedTensor with Identity sub-storages in both
        directions and the per-operand unwrap of every GEMM."""
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        x = self._input()

        y_ref, dx_ref, wg_ref = _fwd_bwd(ref, x, recipe=None)
        y_id, dx_id, wg_id = _fwd_bwd(
            test, x, recipe=CustomRecipe(qfactory=hybrid_all_identity_factory)
        )

        torch.testing.assert_close(y_id, y_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_ref, rtol=0.0, atol=0.0)
        assert len(wg_id) == len(wg_ref)
        for g_id, g_ref in zip(wg_id, wg_ref):
            torch.testing.assert_close(g_id, g_ref, rtol=0.0, atol=0.0)

    def test_identity_reproduces_backward_override_high_precision_bitwise(self):
        """The per-direction Identity machinery must reproduce
        ``backward_override="high_precision"`` **bitwise**.

        Both runs quantize the forward to the same FP8 (current scaling) and run
        the backward in high precision against the original operands. The Identity
        path expresses this per-tensor (weight/input = Hybrid(row=FP8, col=Identity),
        grad = Identity); the reference uses the global ``backward_override`` knob.
        Identical forward FP8 + identical original HP backward operands => bitwise.
        """
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        x = self._input()

        y_bo, dx_bo, wg_bo = _fwd_bwd(
            ref,
            x,
            recipe=CustomRecipe(qfactory=fp8_fwd_factory, backward_override="high_precision"),
        )
        y_id, dx_id, wg_id = _fwd_bwd(test, x, recipe=CustomRecipe(qfactory=fwd_fp8_bwd_hp_factory))

        torch.testing.assert_close(y_id, y_bo, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_bo, rtol=0.0, atol=0.0)
        assert len(wg_id) == len(wg_bo)
        for g_id, g_bo in zip(wg_id, wg_bo):
            torch.testing.assert_close(g_id, g_bo, rtol=0.0, atol=0.0)

    def test_identity_matches_bf16_multistep_training_bitwise(self):
        """Multi-step SGD: an all-Identity recipe must track a plain BF16
        te.Linear bitwise across optimizer steps (no drift from workspace caching
        or any hidden state)."""
        ref, test = _make_linears(self.IN_F, self.OUT_F)
        opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
        opt_test = torch.optim.SGD(test.parameters(), lr=0.1)
        recipe = CustomRecipe(qfactory=identity_all_factory)

        for step in range(4):
            torch.manual_seed(1000 + step)
            x = torch.randn(self.BATCH, self.IN_F, device="cuda", dtype=torch.bfloat16)
            torch.manual_seed(2000 + step)
            target = torch.randn(self.BATCH, self.OUT_F, device="cuda", dtype=torch.bfloat16)

            opt_ref.zero_grad()
            y_ref = ref(x)
            loss_ref = torch.nn.functional.mse_loss(y_ref, target)
            loss_ref.backward()
            opt_ref.step()

            opt_test.zero_grad()
            with te.autocast(enabled=True, recipe=recipe):
                y_test = test(x)
            loss_test = torch.nn.functional.mse_loss(y_test, target)
            loss_test.backward()
            opt_test.step()

            torch.testing.assert_close(y_test, y_ref, rtol=0.0, atol=0.0)
            torch.testing.assert_close(loss_test, loss_ref, rtol=0.0, atol=0.0)
            for p_test, p_ref in zip(test.parameters(), ref.parameters()):
                torch.testing.assert_close(p_test, p_ref, rtol=0.0, atol=0.0, msg=f"step {step}")

    def test_quantized_model_init_identity_matches_bf16_bitwise(self):
        """Persistent Identity params from quantized_model_init should match BF16 exactly."""
        torch.manual_seed(314)
        ref = te.Linear(self.IN_F, self.OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()
        recipe = CustomRecipe(qfactory=identity_all_factory)
        torch.manual_seed(2718)
        with te.quantized_model_init(enabled=True, recipe=recipe):
            test = te.Linear(self.IN_F, self.OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            for p_test, p_ref in zip(test.parameters(), ref.parameters()):
                assert isinstance(p_test, IdentityTensor)
                p_test.copy_(p_ref)

        x = self._input()
        y_ref, dx_ref, wg_ref = _fwd_bwd(ref, x, recipe=None)
        y_id, dx_id, wg_id = _fwd_bwd(test, x, recipe=recipe)

        torch.testing.assert_close(y_id, y_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_ref, rtol=0.0, atol=0.0)
        for g_id, g_ref in zip(wg_id, wg_ref):
            torch.testing.assert_close(g_id, g_ref, rtol=0.0, atol=0.0)

    def test_quantized_model_init_identity_training_loss_decreases_bitwise(self):
        """All-Identity quantized params train like BF16 and loss decreases."""
        torch.manual_seed(777)
        ref = te.Linear(self.IN_F, self.OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()
        recipe = CustomRecipe(qfactory=identity_all_factory)
        torch.manual_seed(888)
        with te.quantized_model_init(enabled=True, recipe=recipe):
            test = te.Linear(self.IN_F, self.OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            for p_test, p_ref in zip(test.parameters(), ref.parameters()):
                assert isinstance(p_test, IdentityTensor)
                p_test.copy_(p_ref)

        torch.manual_seed(909)
        x = torch.randn(self.BATCH, self.IN_F, device="cuda", dtype=torch.bfloat16)
        target = torch.zeros(self.BATCH, self.OUT_F, device="cuda", dtype=torch.bfloat16)
        opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
        opt_test = torch.optim.SGD(test.parameters(), lr=0.1)
        losses_ref = []
        losses_test = []

        for _ in range(5):
            opt_ref.zero_grad()
            y_ref = ref(x)
            loss_ref = torch.nn.functional.mse_loss(y_ref, target)
            loss_ref.backward()
            opt_ref.step()
            losses_ref.append(loss_ref.detach().clone())

            opt_test.zero_grad()
            with te.autocast(enabled=True, recipe=recipe):
                y_test = test(x)
            loss_test = torch.nn.functional.mse_loss(y_test, target)
            loss_test.backward()
            opt_test.step()
            losses_test.append(loss_test.detach().clone())

            torch.testing.assert_close(y_test, y_ref, rtol=0.0, atol=0.0)
            torch.testing.assert_close(loss_test, loss_ref, rtol=0.0, atol=0.0)
            for p_test, p_ref in zip(test.parameters(), ref.parameters()):
                torch.testing.assert_close(p_test.dequantize(), p_ref, rtol=0.0, atol=0.0)

        assert all(
            losses_ref[i + 1].item() < losses_ref[i].item() for i in range(len(losses_ref) - 1)
        ), f"BF16 loss did not strictly decrease: {[x.item() for x in losses_ref]}"
        for loss_test, loss_ref in zip(losses_test, losses_ref):
            torch.testing.assert_close(loss_test, loss_ref, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_identity_activation_recompute_matches_bf16_bitwise(self, use_reentrant):
        """All-Identity recompute should be exactly the BF16 no-checkpoint path."""
        ref, test = _make_linears(self.IN_F, self.OUT_F, seed=4242)
        recipe = CustomRecipe(qfactory=identity_all_factory)
        x = self._input()

        y_ref, dx_ref, wg_ref = _fwd_bwd(ref, x, recipe=None)
        y_id, dx_id, wg_id = _fwd_bwd_checkpoint(test, x, recipe, use_reentrant=use_reentrant)

        torch.testing.assert_close(y_id, y_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dx_id, dx_ref, rtol=0.0, atol=0.0)
        for g_id, g_ref in zip(wg_id, wg_ref):
            torch.testing.assert_close(g_id, g_ref, rtol=0.0, atol=0.0)

    def test_quantized_model_init_identity_state_dict_save_load_exact(self):
        recipe = CustomRecipe(qfactory=identity_all_factory)
        torch.manual_seed(5151)
        with te.quantized_model_init(enabled=True, recipe=recipe):
            model = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16).cuda()

        torch.manual_seed(5152)
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
            out_before = model(x)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        with te.quantized_model_init(enabled=True, recipe=recipe):
            model2 = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16).cuda()
        model2.load_state_dict(torch.load(buffer, weights_only=True))

        with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
            out_after = model2(x)

        assert isinstance(model2.weight, IdentityTensor)
        torch.testing.assert_close(out_after, out_before, rtol=0.0, atol=0.0)

    def test_load_bf16_state_dict_into_identity_model_exact(self):
        recipe = CustomRecipe(qfactory=identity_all_factory)
        torch.manual_seed(6161)
        ref = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16).cuda()
        with te.quantized_model_init(enabled=True, recipe=recipe):
            model = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16).cuda()

        model.load_state_dict(ref.state_dict())
        assert isinstance(model.weight, IdentityTensor)
        torch.testing.assert_close(model.weight.dequantize(), ref.weight, rtol=0.0, atol=0.0)

        torch.manual_seed(6162)
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out_ref = ref(x)
            with te.autocast(enabled=True, recipe=recipe):
                out_id = model(x)

        torch.testing.assert_close(out_id, out_ref, rtol=0.0, atol=0.0)
