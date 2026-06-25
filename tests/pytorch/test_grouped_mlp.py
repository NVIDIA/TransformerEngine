# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable
import os
import math
import random
from typing import Optional

import pytest

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops.fused.grouped_mlp import (
    _cudnn_frontend_supports_grouped_gemm_srelu,
    _cudnn_frontend_version_supported,
)
from transformer_engine.pytorch import (
    QuantizedTensor,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
    QuantizerRole,
    is_bf16_available,
)
import transformer_engine_torch as tex

# Import utility functions
from utils import (
    assert_close,
    assert_close_grads,
    dtype_tols,
    make_recipe,
    MegatronTrainingHelper,
    quantization_tols,
    reset_rng_states,
)

# Check for supported quantization schemes
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# Supported data types
_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_available():  # bf16 requires sm_80 or higher
    _dtypes.append(torch.bfloat16)

# Supported quantization recipes
_quantization_list: list[Optional[str]] = [None]
if fp8_available:
    _quantization_list.extend(("fp8_delayed_scaling", "fp8_current_scaling"))
if mxfp8_available:
    _quantization_list.append("mxfp8")
if nvfp4_available:
    _quantization_list.append("nvfp4")
    _quantization_list.append("nvfp4_4over6")

# Quantization recipes supported by grouped MLP fused op
_grouped_mlp_quantization_list: list[Optional[str]] = [None]
if mxfp8_available:
    _grouped_mlp_quantization_list.append("mxfp8")
if nvfp4_available:
    _grouped_mlp_quantization_list.append("nvfp4_rht")


@pytest.fixture(autouse=True, scope="function")
def _reset_rng_states_per_test():
    """Restore torch, CUDA, and Python ``random`` before each test in this module."""
    reset_rng_states()
    yield


def maybe_skip_quantization(
    quantization: Optional[str],
    *,
    dims: Optional[Iterable[int] | int] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Skip test case if a quantization scheme is not supported"""

    # Don't skip if there is no quantization
    if quantization is None:
        return

    # Check if quantization scheme is supported on device
    if device is not None and torch.device(device).type != "cuda":
        pytest.skip("Quantization is only supported on CUDA devices")
    if quantization in ("fp8", "fp8_delayed_scaling", "fp8_current_scaling") and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if (
        quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht")
        and not nvfp4_available
    ):
        pytest.skip(reason_for_no_nvfp4)

    # Check dims
    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = (dims,)
        if quantization in ("fp8", "fp8_delayed_scaling", "fp8_current_scaling"):
            if math.prod(dims[:-1]) % 16 != 0 or dims[-1] % 16 != 0:
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")
        elif quantization == "mxfp8":
            if math.prod(dims[:-1]) % 32 != 0 or dims[-1] % 32 != 0:
                pytest.skip("MXFP8 GEMMs require dims that are divisible by 32")
        elif quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht"):
            if math.prod(dims[:-1]) % 16 != 0 or dims[-1] % 16 != 0:
                pytest.skip("NVFP4 GEMMs require dims that are divisible by 16")

    # Check dtype
    if dtype is not None:
        if (
            quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht")
            and dtype != torch.bfloat16
        ):
            pytest.skip("NVFP4 quantization is only supported with BF16 data")


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    *,
    min: float = 0.0,
    max: float = 1.0,
    quantization: Optional[str] = None,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_quantized: bool = False,
    quantizer_role: Optional[QuantizerRole] = None,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    If a quantization scheme is provided, the tensor values are
    quantized so that they are representable.

    """

    # Random reference tensor
    ref = torch.empty(shape, dtype=ref_dtype, device=ref_device)
    ref.uniform_(min, max)

    # Construct test tensor from reference tensor
    test = ref.to(device=test_device, dtype=test_dtype)
    if quantization is None:
        if test_is_quantized:
            raise ValueError("Quantization scheme not provided")
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    elif quantization in ("fp8", "fp8_delayed_scaling"):
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
            amax=torch.zeros(1, dtype=torch.float32, device=test_device),
            fp8_dtype=te.DType.kFloat8E4M3,
        )
        test = quantizer(test)
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            device=test_device,
        )
        test = quantizer(test)
    elif quantization == "mxfp8":
        test = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)(test)
    elif quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_rht"):
        tensor_type = "input"
        if quantizer_role is not None:
            tensor_type = quantizer_role.tensor_type
        with_rht = quantization == "nvfp4_rht" and tensor_type != "weight"
        test = NVFP4Quantizer(
            with_rht=with_rht,
            with_post_rht_amax=with_rht,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )(test)
    elif quantization == "nvfp4_4over6":
        tensor_type = "input"
        if quantizer_role is not None:
            tensor_type = quantizer_role.tensor_type

        nvfp4_use_4over6 = False
        with_2d_quantization = False
        nvfp4_e4m3_max = 448
        if tensor_type not in ("grad_output", "grad_input"):
            nvfp4_use_4over6 = True
            nvfp4_e4m3_max = 256
            if tensor_type == "weight":
                with_2d_quantization = True

        test = NVFP4Quantizer(
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=with_2d_quantization,
            stochastic_rounding=False,
            with_random_sign_mask=False,
            nvfp4_use_4over6=nvfp4_use_4over6,
            nvfp4_e4m3_max=nvfp4_e4m3_max,
        )(test)
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")
    if isinstance(test, QuantizedTensor) and not test_is_quantized:
        test = test.dequantize()

    # Make sure reference and test tensors match each other
    ref.copy_(test.to(dtype=ref.dtype))

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


class TestGroupedLinearOp:
    """Tests for advanced features with grouped linear basic op"""

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("quantized_compute", (False, True))
    @pytest.mark.parametrize("quantized_weight", (False, True))
    @pytest.mark.parametrize("input_requires_grad", (False, True))
    @pytest.mark.parametrize("weight_requires_grad", (False, True))
    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("single_grouped_bias", (False, True))
    def test_grouped_linear(
        self,
        *,
        group_size: int = 4,
        bias: bool,
        weight_shape: tuple[int, int] = (128, 128),
        split_alignment: int = 128,
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_compute: bool,
        quantized_weight: bool,
        input_requires_grad: bool,
        weight_requires_grad: bool,
        delay_wgrad_compute: bool,
        single_grouped_weight: bool,
        single_grouped_bias: bool,
    ) -> None:
        """Grouped GEMM"""
        if os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0") == "0" and (
            single_grouped_weight or single_grouped_bias
        ):
            pytest.skip(
                "single_grouped_weight/single_grouped_bias requires"
                " NVTE_GROUPED_LINEAR_SINGLE_PARAM=1"
            )
        # Split sizes
        split_sizes = [split_alignment * i for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int, device=device)

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = (split_sizes.sum().item(), in_features)
        out_shape = (in_shape[0], out_features)

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device, dtype=dtype)
        maybe_skip_quantization(quantization, dims=out_shape)
        if quantization is None and (quantized_compute or quantized_weight):
            pytest.skip("Quantization scheme is not specified")
        if quantization is not None and not (quantized_compute or quantized_weight):
            pytest.skip("Quantization scheme is not used")
        if quantization is not None and dtype not in (torch.bfloat16, torch.float16):
            pytest.skip("Quantized group GEMM is only supported with BF16/FP16")
        if quantization == "nvfp4_4over6":
            pytest.skip("NVFP4 4over6 grouped quantization is not supported")

        if single_grouped_bias and not bias:
            pytest.skip("single_grouped_bias requires bias=True")
        if (
            single_grouped_weight
            and quantized_weight
            and quantization in ("fp8_delayed_scaling", "fp8_current_scaling")
        ):
            pytest.skip(
                "single_grouped_weight does not support FP8 delayed/current scaling "
                "with quantized_model_init"
            )

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
            requires_grad=input_requires_grad,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        ws_ref, ws_test = [], []
        bs_ref, bs_test = [], []
        for _ in range(group_size):
            w_ref, w_test = make_reference_and_test_tensors(
                (out_features, in_features),
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
                requires_grad=weight_requires_grad,
            )
            b_ref, b_test = None, None
            if bias:
                b_ref, b_test = make_reference_and_test_tensors(
                    out_features,
                    test_dtype=dtype,
                    test_device=device,
                    requires_grad=weight_requires_grad,
                )
            ws_ref.append(w_ref)
            ws_test.append(w_test)
            bs_ref.append(b_ref)
            bs_test.append(b_test)

        # Plain PyTorch implementation
        xs_ref = torch.split(x_ref, split_sizes.tolist())
        ys_ref = []
        for x, w, b in zip(xs_ref, ws_ref, bs_ref):
            ys_ref.append(torch.nn.functional.linear(x, w, bias=b))
        y_ref = torch.cat(ys_ref)
        if input_requires_grad or weight_requires_grad:
            y_ref.backward(dy_ref)

        # Construct fusible operation
        recipe = make_recipe(quantization)
        with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
            op = te.ops.GroupedLinear(
                group_size,
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                delay_wgrad_compute=delay_wgrad_compute,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
        with torch.no_grad():
            if single_grouped_weight:
                op_weights = op.weight.quantized_tensors
                if op_weights is None:
                    op_weights = op.weight.split_into_quantized_tensors()
            if single_grouped_bias:
                op_bias_parts = op.bias.split_into_quantized_tensors()
            for group_idx in range(group_size):
                if single_grouped_weight:
                    op_weights[group_idx].copy_(ws_test[group_idx])
                else:
                    getattr(op, f"weight{group_idx}").copy_(ws_test[group_idx])
                if bias:
                    if single_grouped_bias:
                        op_bias_parts[group_idx].reshape(-1).copy_(bs_test[group_idx])
                    else:
                        getattr(op, f"bias{group_idx}").copy_(bs_test[group_idx])
            del ws_test, bs_test
            for param in op.parameters():
                param.requires_grad_(requires_grad=weight_requires_grad)

        # Forward and backward pass with op
        with te.autocast(enabled=quantized_compute, recipe=recipe):
            y_test = op(x_test, split_sizes)
        if input_requires_grad or weight_requires_grad:
            y_test.backward(dy_test)
            if delay_wgrad_compute and weight_requires_grad:
                op.backward_dw()

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute:
            tols = quantization_tols(quantization)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        if input_requires_grad:
            dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        else:
            assert x_test.grad is None
        if single_grouped_weight:
            if weight_requires_grad:
                dw_test_all = op.weight.grad.to(dtype=torch.float64, device="cpu")
                w_ref_grad = torch.stack([w.grad for w in ws_ref], dim=0)
                torch.testing.assert_close(dw_test_all, w_ref_grad, **tols)
            else:
                assert op.weight.grad is None
        else:
            for group_idx in range(group_size):
                w_test = getattr(op, f"weight{group_idx}")
                if weight_requires_grad:
                    dw_test = w_test.grad.to(dtype=torch.float64, device="cpu")
                    torch.testing.assert_close(dw_test, ws_ref[group_idx].grad, **tols)
                else:
                    assert w_test.grad is None
        if bias:
            if single_grouped_bias:
                if weight_requires_grad:
                    db_test_all = op.bias.grad.to(dtype=torch.float64, device="cpu")
                    b_ref_grad = torch.stack([b.grad for b in bs_ref], dim=0)
                    torch.testing.assert_close(db_test_all, b_ref_grad, **tols)
                else:
                    assert op.bias.grad is None
            else:
                for group_idx in range(group_size):
                    b_test = getattr(op, f"bias{group_idx}")
                    if weight_requires_grad:
                        db_test = b_test.grad.to(dtype=torch.float64, device="cpu")
                        torch.testing.assert_close(db_test, bs_ref[group_idx].grad, **tols)
                    else:
                        assert b_test.grad is None

    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    @pytest.mark.parametrize(
        "quantization",
        [None]
        + (["fp8_current_scaling"] if fp8_available else [])
        + (["mxfp8"] if mxfp8_available else [])
        + (["nvfp4_rht"] if nvfp4_available else []),
    )
    @pytest.mark.parametrize("quantized_weight", (False, True))
    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    def test_grouped_linear_cuda_graph_safe(
        self,
        *,
        dtype: torch.dtype,
        quantization: Optional[str],
        quantized_weight: bool,
        bias: bool,
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool,
        device: torch.device = "cuda",
        group_size: int = 4,
        in_features: int = 128,
        out_features: int = 128,
        split_alignment: int = 128,
        token_padding: int = 256,
    ) -> None:
        """GroupedLinear forward+backward should be CUDA graph capturable.

        Exercises the grouped-tensor / cublas-grouped-gemm path which uses
        GPU-resident split offsets and is the only flow safe to capture.
        """

        # Skip invalid configurations
        if os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0") == "0" and (
            single_grouped_weight
        ):
            pytest.skip(
                "single_grouped_weight/single_grouped_bias requires"
                " NVTE_GROUPED_LINEAR_SINGLE_PARAM=1"
            )
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("Grouped GEMM CUDA-graph-safe path requires SM100+ (Blackwell)")
        if quantization is None and quantized_weight:
            pytest.skip("quantized_weight requires a quantization recipe")
        if quantization is not None and quantization.startswith("nvfp4") and dtype != torch.bfloat16:
            pytest.skip("NVFP4 grouped GEMM only supports BF16 output")

        single_grouped_bias = bias and single_grouped_weight

        # Split sizes (statically pinned for graph capture)
        split_sizes = [split_alignment * (i + 1) for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int, device=device)

        # Pad input tokens to validate the sync-free flow
        in_shape = (split_sizes.sum().item() + token_padding, in_features)
        out_shape = (in_shape[0], out_features)

        recipe = make_recipe(quantization)
        with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
            op = te.ops.GroupedLinear(
                group_size,
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                accumulate_into_main_grad=accumulate_into_main_grad,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )

        def _weight_params() -> list[torch.nn.Parameter]:
            if single_grouped_weight:
                return [op.weight]
            return [getattr(op, f"weight{i}") for i in range(group_size)]

        def _bias_params() -> list[torch.nn.Parameter]:
            if not bias:
                return []
            if single_grouped_bias:
                return [op.bias]
            return [getattr(op, f"bias{i}") for i in range(group_size)]

        def _init_main_grads(value: float = 0.0) -> None:
            if not accumulate_into_main_grad:
                return
            with torch.no_grad():
                for w in _weight_params():
                    if getattr(w, "main_grad", None) is None:
                        w.main_grad = torch.empty(w.size(), device=device, dtype=torch.float32)
                    w.main_grad.fill_(value)

        def _collect_main_grads() -> list[torch.Tensor]:
            return [w.main_grad.detach().clone() for w in _weight_params()]

        def _zero_param_grads() -> None:
            for param in op.parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad.zero_()

        static_split_sizes = split_sizes.clone()

        def train_step(
            x: torch.Tensor,
            dy: torch.Tensor,
            out_buf: torch.Tensor,
            *,
            use_graphed: bool,
        ) -> torch.Tensor:
            with te.autocast(enabled=quantization is not None, recipe=recipe):
                out = (
                    graphed_module(x, static_split_sizes)
                    if use_graphed
                    else op(x, static_split_sizes)
                )
            out.backward(dy)
            out_buf.copy_(out)
            return out_buf

        _init_main_grads(0.0)

        static_x = torch.randn(in_shape, device=device, dtype=dtype, requires_grad=True)
        static_dy = torch.randn(out_shape, device=device, dtype=dtype)
        static_out_buf = torch.empty(out_shape, device=device, dtype=dtype)

        graphed_module = te.make_graphed_callables(
            op,
            (static_x, static_split_sizes),
            num_warmup_iters=3,
            enabled=quantization is not None,
            recipe=recipe,
        )

        # Replace static buffers with fresh data (graph captures must replay
        # against new inputs without re-recording).
        fresh_x = torch.randn_like(static_x)
        fresh_dy = torch.randn_like(static_dy)
        with torch.no_grad():
            static_x.copy_(fresh_x)
            static_dy.copy_(fresh_dy)

        # Reset grads & main_grads so the captured iteration starts fresh.
        _zero_param_grads()
        _init_main_grads(0.5)
        if static_x.grad is not None:
            static_x.grad.zero_()

        # Replay the graph
        graph_out = (
            train_step(static_x, static_dy, static_out_buf, use_graphed=True).detach().clone()
        )
        torch.cuda.synchronize()
        graph_dx = static_x.grad.detach().clone()
        if accumulate_into_main_grad:
            graph_main_grads = _collect_main_grads()
            graph_param_grads: list[torch.Tensor] = []
        else:
            graph_main_grads = []
            graph_param_grads = [param.grad.detach().clone() for param in op.parameters()]

        # Reference: same op invoked eagerly with the same fresh inputs and
        # the same starting grad/main_grad state.
        _zero_param_grads()
        _init_main_grads(0.5)
        static_x.grad.zero_()

        expected_x = fresh_x.detach().clone().requires_grad_(True)
        expected_dy = fresh_dy.detach().clone()
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            expected_out = op(expected_x, static_split_sizes)
        expected_out.backward(expected_dy)

        tols = dtype_tols(dtype)
        if quantization is not None:
            tols = quantization_tols(quantization)

        assert_close(graph_out, expected_out, **tols)
        assert_close(graph_dx, expected_x.grad, **tols)
        if accumulate_into_main_grad:
            for g, w in zip(graph_main_grads, _weight_params()):
                assert_close(g, w.main_grad, **tols)
        else:
            for g, param in zip(graph_param_grads, op.parameters()):
                assert_close(g, param.grad, **tols)


class TestGroupedMLPFusedOp:
    """Tests for grouped MLP fused op"""

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("quantization", _grouped_mlp_quantization_list)
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("hidden_size", (128, 256))
    @pytest.mark.parametrize(
        "activation",
        (
            "scaled_swiglu",
            "scaled_clamped_qgeglu",
            "scaled_clamped_qgeglu_custom",
            "scaled_srelu",
        ),
    )
    def test_grouped_mlp(
        self,
        *,
        group_size: int = 4,
        bias: bool,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        quantization: Optional[str],
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool = False,
        device: torch.device = "cuda",
        split_alignment: int = 256,
        delay_wgrad_compute: bool = False,
        activation: str,
    ) -> None:
        """GroupedLinear + scaled activation + GroupedLinear"""

        # Split sizes
        split_sizes = [split_alignment * (i) for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int, device=device)

        # Make input shape
        in_shape = (split_sizes.sum().item(), hidden_size)
        out_shape = in_shape

        with_quantization = quantization is not None

        activation_is_glu = activation in (
            "scaled_swiglu",
            "scaled_clamped_qgeglu",
            "scaled_clamped_qgeglu_custom",
        )
        glu_interleave_size = 32 if activation_is_glu else None

        single_grouped_bias = bias and single_grouped_weight

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device, dtype=dtype)
        if dtype == torch.bfloat16 and not is_bf16_available():
            pytest.skip("BF16 requires SM 8.0+")
        if single_grouped_weight and quantization != "mxfp8":
            pytest.skip("single_grouped_weight is only supported for MXFP8 quantization")
        if single_grouped_bias and not bias:
            pytest.skip("single_grouped_bias requires bias=True")
        if with_quantization and dtype not in (torch.bfloat16, torch.float16):
            pytest.skip("Quantized group GEMM is only supported with BF16/FP16")
        if not activation_is_glu and quantization not in ("mxfp8", "nvfp4", "nvfp4_rht"):
            pytest.skip("Scaled unary grouped MLP is only supported with MXFP8 or NVFP4")
        if not activation_is_glu and glu_interleave_size is not None:
            pytest.skip("Unary activations do not use GLU interleaving")
        if quantization == "nvfp4_4over6":
            pytest.skip("NVFP4 4over6 grouped quantization is not supported")
        if activation == "scaled_srelu" and quantization in ("nvfp4", "nvfp4_rht") and bias:
            pytest.skip("NVFP4 SReLU grouped MLP coverage is limited to no-bias")
        if quantization == "nvfp4_rht":
            if activation == "scaled_swiglu" and (bias or glu_interleave_size != 32):
                pytest.skip("NVFP4 RHT SwiGLU grouped MLP coverage is limited to no-bias")
            if activation not in ("scaled_swiglu", "scaled_srelu"):
                pytest.skip("NVFP4 RHT grouped MLP coverage is limited to SwiGLU and SReLU")
        if (
            with_quantization
            and quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht")
            and activation.startswith("scaled_clamped_qgeglu")
            and bias
        ):
            # TODO: ksivaman: Need to debug numerics for this case.
            pytest.skip("Bias/dbias not yet supported in NVFP4 fused grouped MLP with GeGLU")
        fc1_out_features = 2 * hidden_size if activation_is_glu else hidden_size
        # Activation parameters for clamped QGeGLU variants
        if activation == "scaled_clamped_qgeglu_custom":
            geglu_limit = 5.0
            geglu_alpha = 1.5
            geglu_offset = 0.5
        else:
            geglu_limit = 7.0
            geglu_alpha = 1.702
            geglu_offset = 1.0

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            min=-0.25,
            max=0.25,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            min=-0.25,
            max=0.25,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        probs_ref, probs_test = make_reference_and_test_tensors(
            (in_shape[0],),
            test_dtype=dtype,
            test_device=device,
        )
        fc1_ws_ref, fc1_ws_test = [], []
        fc1_bs_ref, fc1_bs_test = [], []
        fc2_ws_ref, fc2_ws_test = [], []
        fc2_bs_ref, fc2_bs_test = [], []
        for _ in range(group_size):
            fc1_w_ref, fc1_w_test = make_reference_and_test_tensors(
                (fc1_out_features, hidden_size),
                min=-0.25,
                max=0.25,
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
            )
            fc2_w_ref, fc2_w_test = make_reference_and_test_tensors(
                (hidden_size, hidden_size),
                min=-0.25,
                max=0.25,
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
            )
            fc1_b_ref, fc1_b_test = None, None
            fc2_b_ref, fc2_b_test = None, None
            if bias:
                fc1_b_ref, fc1_b_test = make_reference_and_test_tensors(
                    (fc1_out_features,),
                    min=-0.5,
                    max=0.5,
                    test_dtype=dtype,
                    test_device=device,
                )
                fc2_b_ref, fc2_b_test = make_reference_and_test_tensors(
                    (hidden_size,),
                    min=-0.5,
                    max=0.5,
                    test_dtype=dtype,
                    test_device=device,
                )
            fc1_ws_ref.append(fc1_w_ref)
            fc1_bs_ref.append(fc1_b_ref)
            fc1_ws_test.append(fc1_w_test)
            fc1_bs_test.append(fc1_b_test)
            fc2_ws_ref.append(fc2_w_ref)
            fc2_bs_ref.append(fc2_b_ref)
            fc2_ws_test.append(fc2_w_test)
            fc2_bs_test.append(fc2_b_test)

        def _apply_activation(x: torch.Tensor) -> torch.Tensor:
            if activation_is_glu and glu_interleave_size is not None:
                x = x.reshape(
                    -1,
                    2 * hidden_size // (2 * glu_interleave_size),
                    2,
                    glu_interleave_size,
                )
                x = x.transpose(1, 2)
                x = x.reshape(-1, 2 * hidden_size)
            if activation == "scaled_swiglu":
                x1, x2 = x.chunk(2, dim=-1)
                return torch.nn.functional.silu(x1) * x2
            if activation.startswith("scaled_clamped_qgeglu"):
                x1, x2 = x.chunk(2, dim=-1)
                lim = torch.tensor(geglu_limit, device=x1.device, dtype=x1.dtype)
                x1c = torch.minimum(x1, lim)
                x2c = torch.clamp(x2, -lim, lim)
                return (x2c + geglu_offset) * (x1c * torch.sigmoid(geglu_alpha * x1c))
            if activation == "scaled_srelu":
                return torch.nn.functional.relu(x).square()
            raise ValueError(f"Unexpected grouped MLP activation ({activation})")

        # Reference implementation
        xs = torch.split(x_ref, split_sizes.tolist())
        probs = torch.split(probs_ref, split_sizes.tolist())
        ys = []
        for group_idx in range(group_size):
            x = xs[group_idx]
            fc1_out = torch.nn.functional.linear(
                x, fc1_ws_ref[group_idx], bias=fc1_bs_ref[group_idx]
            )
            fc2_in = _apply_activation(fc1_out)
            fc2_in = fc2_in * probs[group_idx].unsqueeze(-1)
            y = torch.nn.functional.linear(fc2_in, fc2_ws_ref[group_idx])
            if bias:
                y = y + fc2_bs_ref[group_idx] * probs[group_idx].unsqueeze(-1)
            ys.append(y)
        y_ref = torch.cat(ys)
        y_ref.backward(dy_ref)

        # Construct operations
        recipe = make_recipe(quantization)

        def _make_scaled_act():
            if activation == "scaled_swiglu":
                return te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
            if activation == "scaled_clamped_qgeglu_custom":
                return te.ops.ScaledClampedQGeGLU(
                    glu_interleave_size=glu_interleave_size,
                    limit=geglu_limit,
                    alpha=geglu_alpha,
                    glu_linear_offset=geglu_offset,
                )
            if activation.startswith("scaled_clamped_qgeglu"):
                return te.ops.ScaledClampedQGeGLU(glu_interleave_size=glu_interleave_size)
            if activation == "scaled_srelu":
                return te.ops.ScaledSReLU()
            raise ValueError(f"Unexpected grouped MLP activation ({activation})")

        def _make_module():
            with te.quantized_model_init(enabled=with_quantization, recipe=recipe):
                fc1_op = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    fc1_out_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    single_grouped_bias=single_grouped_bias,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )

                fc2_op = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    single_grouped_bias=single_grouped_bias,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                    scale_bias=bias,
                )
                return te.ops.Sequential(fc1_op, _make_scaled_act(), fc2_op), fc1_op, fc2_op

        module, fc1, fc2 = _make_module()

        # Copy weights
        with torch.no_grad():
            if single_grouped_weight:
                fc1_weights = fc1.weight.quantized_tensors
                if fc1_weights is None:
                    fc1_weights = fc1.weight.split_into_quantized_tensors()
                fc2_weights = fc2.weight.quantized_tensors
                if fc2_weights is None:
                    fc2_weights = fc2.weight.split_into_quantized_tensors()
            for group_idx in range(group_size):
                if single_grouped_weight:
                    fc1_weights[group_idx].copy_(fc1_ws_test[group_idx])
                    fc2_weights[group_idx].copy_(fc2_ws_test[group_idx])
                else:
                    getattr(fc1, f"weight{group_idx}").copy_(fc1_ws_test[group_idx])
                    getattr(fc2, f"weight{group_idx}").copy_(fc2_ws_test[group_idx])
                if bias:
                    if single_grouped_bias:
                        fc1_bparts = fc1.bias.split_into_quantized_tensors()
                        fc2_bparts = fc2.bias.split_into_quantized_tensors()
                        fc1_bparts[group_idx].reshape(-1).copy_(fc1_bs_test[group_idx])
                        fc2_bparts[group_idx].reshape(-1).copy_(fc2_bs_test[group_idx])
                    else:
                        getattr(fc1, f"bias{group_idx}").copy_(fc1_bs_test[group_idx])
                        getattr(fc2, f"bias{group_idx}").copy_(fc2_bs_test[group_idx])
            if accumulate_into_main_grad:
                # 0.5 sentinel lets us reconstruct ``expected = ref_grad + 0.5``
                # below and detect a missed accumulation.
                main_grad_sentinel = 0.5
                if single_grouped_weight:
                    weight_params_for_main_grad = [fc1.weight, fc2.weight]
                else:
                    weight_params_for_main_grad = [
                        getattr(fc, f"weight{i}") for fc in (fc1, fc2) for i in range(group_size)
                    ]
                MegatronTrainingHelper.init_main_grad_buffers(
                    weight_params_for_main_grad,
                    fill_value=main_grad_sentinel,
                    overwrite_main_grad=False,
                )
        del fc1_ws_test, fc1_bs_test, fc2_ws_test, fc2_bs_test

        # Fuse ops and perform forward and backward pass
        with te.autocast(enabled=with_quantization, recipe=recipe):
            fc2_extra = (split_sizes, probs_test) if bias else (split_sizes,)
            y_test = module(x_test, split_sizes, probs_test, *fc2_extra)
        y_test.backward(dy_test)
        if delay_wgrad_compute:
            fc1.backward_dw()
            fc2.backward_dw()

        # Check for expected fusions
        cudnn_frontend_supports_grouped_mlp = (
            _cudnn_frontend_supports_grouped_gemm_srelu()
            if activation == "scaled_srelu"
            else _cudnn_frontend_version_supported()
        )
        expected_grouped_mlp_fusion = cudnn_frontend_supports_grouped_mlp and (
            (
                quantization == "mxfp8"
                and dtype in (torch.bfloat16, torch.float16)
                and (
                    (not activation_is_glu and glu_interleave_size is None)
                    or (activation_is_glu and glu_interleave_size == 32)
                )
            )
            or (
                quantization == "nvfp4_rht"
                and dtype == torch.bfloat16
                and activation == "scaled_srelu"
                and glu_interleave_size is None
            )
        )
        if expected_grouped_mlp_fusion:
            if activation_is_glu:
                fused_cls = te.ops.fused.GroupedMLP_CuTeGEMMGLU
            else:
                fused_cls = te.ops.fused.GroupedMLP_CuTeGEMMUnary
            if fused_cls.is_supported():
                forward_ops = module._module_groups[0]._forward_ops
                backward_ops = module._module_groups[0]._backward_ops
                assert len(forward_ops) == 1
                assert len(backward_ops) == 1
                assert isinstance(
                    forward_ops[0][0],
                    fused_cls,
                )
                assert backward_ops[0][0] is forward_ops[0][0]

        # Loose tols for sanity checking
        tols = {"rtol": 0.125, "atol": 0.25}
        if quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht"):
            tols = {"rtol": 0.25, "atol": 0.5}

        # Check values
        assert_close(y_test, y_ref, **tols)
        assert_close_grads(x_test, x_ref, **tols)
        assert_close_grads(probs_test, probs_ref, **tols)
        for group_idx in range(group_size):
            if bias:
                if single_grouped_bias:
                    assert_close(
                        fc2.bias.grad[group_idx],
                        fc2_bs_ref[group_idx].grad,
                        **tols,
                    )
                    assert_close(
                        fc1.bias.grad[group_idx],
                        fc1_bs_ref[group_idx].grad,
                        **tols,
                    )
                else:
                    assert_close_grads(
                        getattr(fc2, f"bias{group_idx}"), fc2_bs_ref[group_idx], **tols
                    )
                    assert_close_grads(
                        getattr(fc1, f"bias{group_idx}"), fc1_bs_ref[group_idx], **tols
                    )
            if not single_grouped_weight and not accumulate_into_main_grad:
                assert_close_grads(
                    getattr(fc2, f"weight{group_idx}"), fc2_ws_ref[group_idx], **tols
                )
                assert_close_grads(
                    getattr(fc1, f"weight{group_idx}"), fc1_ws_ref[group_idx], **tols
                )
        fc1_w_ref_grad = torch.stack([w.grad for w in fc1_ws_ref], dim=0)
        fc2_w_ref_grad = torch.stack([w.grad for w in fc2_ws_ref], dim=0)
        if accumulate_into_main_grad:
            # main_grad should accumulate the ref wgrad onto the 0.5 sentinel.
            # Per-param expected views must line up with
            # ``weight_params_for_main_grad`` registered above.
            fc1_expected = (
                [fc1_w_ref_grad + main_grad_sentinel]
                if single_grouped_weight
                else [g + main_grad_sentinel for g in fc1_w_ref_grad]
            )
            fc2_expected = (
                [fc2_w_ref_grad + main_grad_sentinel]
                if single_grouped_weight
                else [g + main_grad_sentinel for g in fc2_w_ref_grad]
            )
            MegatronTrainingHelper.verify_main_grad_accumulation(
                weight_params_for_main_grad,
                expected_main_grads=fc1_expected + fc2_expected,
                **tols,
            )
        elif single_grouped_weight:
            assert_close(fc1.weight.grad, fc1_w_ref_grad, **tols)
            assert_close(fc2.weight.grad, fc2_w_ref_grad, **tols)

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("quantization", _grouped_mlp_quantization_list)
    @pytest.mark.parametrize(
        "activation",
        (
            "scaled_swiglu",
            "scaled_clamped_qgeglu",
            "scaled_clamped_qgeglu_custom",
            "scaled_srelu",
        ),
    )
    def test_grouped_mlp_fp16(
        self,
        *,
        bias: bool,
        quantization: Optional[str],
        activation: str,
    ) -> None:
        """Grouped MLP with high-precision tensors in FP16"""
        self.test_grouped_mlp(
            bias=bias,
            hidden_size=128,
            dtype=torch.float16,
            quantization=quantization,
            single_grouped_weight=True,
            activation=activation,
        )

    @pytest.mark.parametrize("quantization", _grouped_mlp_quantization_list)
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_srelu"))
    def test_grouped_mlp_mcore_integrations(
        self,
        *,
        quantization: Optional[str],
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool,
        delay_wgrad_compute: bool,
        activation: str,
    ) -> None:
        """Grouped MLP with main_grad accumulation and delayed wgrad"""
        if not accumulate_into_main_grad and not delay_wgrad_compute:
            pytest.skip("Configuration is already tests in test_grouped_mlp")
        self.test_grouped_mlp(
            bias=False,
            hidden_size=128,
            quantization=quantization,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
            delay_wgrad_compute=delay_wgrad_compute,
            activation=activation,
        )

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_clamped_qgeglu"))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_single_weight_numerics(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool,
        activation: str,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
    ) -> None:
        """single_grouped_weight=True/False should match exactly for fused MXFP8 grouped MLP."""

        if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP is not supported on this system")

        split_sizes = [split_alignment * (i + 1) for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device=device)
        in_shape = (split_sizes.sum().item(), hidden_size)
        recipe = make_recipe("mxfp8")

        x_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        probs_base = torch.empty((in_shape[0],), device=device, dtype=dtype).uniform_(-0.25, 0.25)
        dy_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        fc1_ws_base = [
            torch.empty((2 * hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc2_ws_base = [
            torch.empty((hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc1_bs_base = (
            [
                torch.empty((2 * hidden_size,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
                for _ in range(group_size)
            ]
            if bias
            else None
        )
        fc2_bs_base = (
            [
                torch.empty((hidden_size,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
                for _ in range(group_size)
            ]
            if bias
            else None
        )

        def _run_case(single_grouped_weight: bool) -> tuple[torch.Tensor, ...]:
            with te.quantized_model_init(enabled=True, recipe=recipe):
                scaled_act = (
                    te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
                    if activation == "scaled_swiglu"
                    else te.ops.ScaledClampedQGeGLU(glu_interleave_size=glu_interleave_size)
                )
                fc1 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    2 * hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                )
                fc2 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    scale_bias=bias,
                )
                module = te.ops.Sequential(fc1, scaled_act, fc2)

            with torch.no_grad():
                if single_grouped_weight:
                    fc1_weights = fc1.weight.quantized_tensors
                    if fc1_weights is None:
                        fc1_weights = fc1.weight.split_into_quantized_tensors()
                    fc2_weights = fc2.weight.quantized_tensors
                    if fc2_weights is None:
                        fc2_weights = fc2.weight.split_into_quantized_tensors()
                for group_idx in range(group_size):
                    if single_grouped_weight:
                        fc1_weights[group_idx].copy_(fc1_ws_base[group_idx])
                        fc2_weights[group_idx].copy_(fc2_ws_base[group_idx])
                    else:
                        getattr(fc1, f"weight{group_idx}").copy_(fc1_ws_base[group_idx])
                        getattr(fc2, f"weight{group_idx}").copy_(fc2_ws_base[group_idx])
                    if bias:
                        getattr(fc1, f"bias{group_idx}").copy_(fc1_bs_base[group_idx])
                        getattr(fc2, f"bias{group_idx}").copy_(fc2_bs_base[group_idx])

            x = x_base.detach().clone().requires_grad_(True)
            probs = probs_base.detach().clone().requires_grad_(True)
            dy = dy_base.detach().clone()

            with te.autocast(enabled=True, recipe=recipe):
                fc2_extra = (split_sizes, probs) if bias else (split_sizes,)
                y = module(x, split_sizes, probs, *fc2_extra)
            y.backward(dy)

            forward_ops = module._module_groups[0]._forward_ops
            backward_ops = module._module_groups[0]._backward_ops
            assert len(forward_ops) == 1
            assert isinstance(
                forward_ops[0][0],
                te.ops.fused.GroupedMLP_CuTeGEMMGLU,
            )
            assert len(backward_ops) == 1
            assert isinstance(
                backward_ops[0][0],
                te.ops.fused.GroupedMLP_CuTeGEMMGLU,
            )
            assert backward_ops[0][0] is forward_ops[0][0]

            if single_grouped_weight:
                fc1_dw = fc1.weight.grad.detach().clone()
                fc2_dw = fc2.weight.grad.detach().clone()
            else:
                fc1_dw = torch.stack(
                    [
                        getattr(fc1, f"weight{group_idx}").grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )
                fc2_dw = torch.stack(
                    [
                        getattr(fc2, f"weight{group_idx}").grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )

            fc1_db = None
            fc2_db = None
            if bias:
                fc1_db = torch.stack(
                    [
                        getattr(fc1, f"bias{group_idx}").grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )
                fc2_db = torch.stack(
                    [
                        getattr(fc2, f"bias{group_idx}").grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )

            return (
                y.detach().clone(),
                x.grad.detach().clone(),
                probs.grad.detach().clone(),
                fc1_dw,
                fc2_dw,
                fc1_db,
                fc2_db,
            )

        (
            y_false,
            dx_false,
            dprobs_false,
            fc1_dw_false,
            fc2_dw_false,
            fc1_db_false,
            fc2_db_false,
        ) = _run_case(False)
        (
            y_true,
            dx_true,
            dprobs_true,
            fc1_dw_true,
            fc2_dw_true,
            fc1_db_true,
            fc2_db_true,
        ) = _run_case(True)

        torch.testing.assert_close(y_false, y_true, rtol=0, atol=0)
        torch.testing.assert_close(dx_false, dx_true, rtol=0, atol=0)
        torch.testing.assert_close(dprobs_false, dprobs_true, rtol=0, atol=0)
        torch.testing.assert_close(fc1_dw_false, fc1_dw_true, rtol=0, atol=0)
        torch.testing.assert_close(fc2_dw_false, fc2_dw_true, rtol=0, atol=0)
        if bias:
            bias_tols = {"rtol": 0.05, "atol": 0.015625}
            torch.testing.assert_close(fc1_db_false, fc1_db_true, **bias_tols)
            torch.testing.assert_close(fc2_db_false, fc2_db_true, **bias_tols)

    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("zero_out_wgrad", (False, True))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_overwrite_main_grad(
        self,
        *,
        single_grouped_weight: bool,
        delay_wgrad_compute: bool,
        zero_out_wgrad: bool,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
    ) -> None:
        """End-to-end check that the fused grouped-MLP backward writes the
        wgrad into ``weight.main_grad`` correctly under the MegatronFSDP
        ``overwrite_main_grad=True`` convention.
        ``test_grouped_mlp`` already covers the standard Megatron-LM
        ``fuse_wgrad_accumulation`` (DDP) path where the wgrad GEMM
        *accumulates* into ``main_grad``. This test focuses exclusively on
        the MegatronFSDP variant where the wgrad GEMM must *overwrite*
        ``main_grad`` (because FSDP has already ReduceScattered the previous
        accumulation), so ``main_grad`` after backward equals ``wgrad``
        regardless of the prior contents.

        Also exercises the MegatronFSDP ``zero_out_wgrad`` flag, which is
        independent of ``main_grad`` and only controls whether the dummy
        ``param.grad`` returned to autograd is zeroed (so downstream hooks
        that read ``.grad`` don't see stale bytes from the cached dummy).
        """

        if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP is not supported on this system")

        recipe = make_recipe("mxfp8")
        split_sizes = [split_alignment * (i + 1) for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device=device)
        in_shape = (split_sizes.sum().item(), hidden_size)
        x_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        probs_base = torch.empty((in_shape[0],), device=device, dtype=dtype).uniform_(-0.25, 0.25)
        dy_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        fc1_ws_base = [
            torch.empty((2 * hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc2_ws_base = [
            torch.empty((hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]

        def _build_module(*, accumulate_into_main_grad: bool):
            with te.quantized_model_init(enabled=True, recipe=recipe):
                fc1 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    2 * hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
                fc2 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
                scaled_act = te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
                module = te.ops.Sequential(fc1, scaled_act, fc2)

            with torch.no_grad():
                if single_grouped_weight:
                    fc1_weights = (
                        fc1.weight.quantized_tensors or fc1.weight.split_into_quantized_tensors()
                    )
                    fc2_weights = (
                        fc2.weight.quantized_tensors or fc2.weight.split_into_quantized_tensors()
                    )
                    for group_idx in range(group_size):
                        fc1_weights[group_idx].copy_(fc1_ws_base[group_idx])
                        fc2_weights[group_idx].copy_(fc2_ws_base[group_idx])
                else:
                    for group_idx in range(group_size):
                        getattr(fc1, f"weight{group_idx}").copy_(fc1_ws_base[group_idx])
                        getattr(fc2, f"weight{group_idx}").copy_(fc2_ws_base[group_idx])
            return module, fc1, fc2

        def _weight_params(fc):
            if single_grouped_weight:
                return [fc.weight]
            return [getattr(fc, f"weight{i}") for i in range(group_size)]

        def _run_backward(module, fc1, fc2):
            x = x_base.detach().clone().requires_grad_(True)
            probs = probs_base.detach().clone().requires_grad_(True)
            with te.autocast(enabled=True, recipe=recipe):
                y = module(x, split_sizes, probs, split_sizes)
            y.backward(dy_base)
            if delay_wgrad_compute:
                fc1.backward_dw()
                fc2.backward_dw()

        # Reference run: vanilla autograd, no Megatron protocol.
        ref_module, ref_fc1, ref_fc2 = _build_module(accumulate_into_main_grad=False)
        _run_backward(ref_module, ref_fc1, ref_fc2)
        ref_fc1_grads = [wp.grad.detach().clone() for wp in _weight_params(ref_fc1)]
        ref_fc2_grads = [wp.grad.detach().clone() for wp in _weight_params(ref_fc2)]

        # Test run: main_grad fusion with overwrite_main_grad=True (MegatronFSDP).
        # NaN sentinel makes a missed write loud (would surface as NaN diff).
        test_module, test_fc1, test_fc2 = _build_module(accumulate_into_main_grad=True)
        for fc in (test_fc1, test_fc2):
            MegatronTrainingHelper.init_main_grad_buffers(
                _weight_params(fc),
                fill_value=float("nan"),
                overwrite_main_grad=True,
                zero_out_wgrad=zero_out_wgrad,
            )
        _run_backward(test_module, test_fc1, test_fc2)

        # main_grad must be overwritten to exactly the ref wgrad (bitwise:
        # the wgrad GEMM is deterministic across the two runs because the
        # quantized weights and inputs are identical).
        MegatronTrainingHelper.verify_main_grad_accumulation(
            _weight_params(test_fc1), expected_main_grads=ref_fc1_grads
        )
        MegatronTrainingHelper.verify_main_grad_accumulation(
            _weight_params(test_fc2), expected_main_grads=ref_fc2_grads
        )

    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_clamped_qgeglu"))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_cuda_graph_safe_mxfp8(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool,
        activation: str,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
        token_padding: int = 2048,
    ) -> None:
        """Grouped MLP forward+backward should be CUDA graph capturable (MXFP8)."""

        if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP is not supported on this system")
        if dtype not in (torch.bfloat16, torch.float16):
            pytest.skip("MXFP8 fused grouped MLP is only supported with BF16/FP16")

        split_sizes = [split_alignment * (i + 1) for i in range(group_size)]
        random.shuffle(split_sizes)
        split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device=device)
        # Pad the input tokens to validate the sync-free MOE
        in_shape = (split_sizes.sum().item() + token_padding, hidden_size)
        recipe = make_recipe("mxfp8")
        with te.quantized_model_init(enabled=True, recipe=recipe):
            fc1 = te.ops.GroupedLinear(
                group_size,
                hidden_size,
                2 * hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            fc2 = te.ops.GroupedLinear(
                group_size,
                hidden_size,
                hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            scaled_act = (
                te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
                if activation == "scaled_swiglu"
                else te.ops.ScaledClampedQGeGLU(glu_interleave_size=glu_interleave_size)
            )
            module = te.ops.Sequential(
                fc1,
                scaled_act,
                fc2,
            )

        def _init_main_grads(value: float = 0.0) -> None:
            if not accumulate_into_main_grad:
                return
            with torch.no_grad():
                if single_grouped_weight:
                    if getattr(fc1.weight, "main_grad", None) is None:
                        fc1.weight.main_grad = torch.empty(
                            fc1.weight.size(),
                            device=device,
                            dtype=torch.float32,
                        )
                    if getattr(fc2.weight, "main_grad", None) is None:
                        fc2.weight.main_grad = torch.empty(
                            fc2.weight.size(),
                            device=device,
                            dtype=torch.float32,
                        )
                    fc1.weight.main_grad.fill_(value)
                    fc2.weight.main_grad.fill_(value)
                else:
                    for group_idx in range(group_size):
                        fc1_weight = getattr(fc1, f"weight{group_idx}")
                        fc2_weight = getattr(fc2, f"weight{group_idx}")
                        if getattr(fc1_weight, "main_grad", None) is None:
                            fc1_weight.main_grad = torch.empty(
                                fc1_weight.size(),
                                device=device,
                                dtype=torch.float32,
                            )
                        if getattr(fc2_weight, "main_grad", None) is None:
                            fc2_weight.main_grad = torch.empty(
                                fc2_weight.size(),
                                device=device,
                                dtype=torch.float32,
                            )
                        fc1_weight.main_grad.fill_(value)
                        fc2_weight.main_grad.fill_(value)

        def _collect_main_grads() -> tuple[torch.Tensor, torch.Tensor]:
            if single_grouped_weight:
                fc1_main_grad = fc1.weight.main_grad.detach().clone()
                fc2_main_grad = fc2.weight.main_grad.detach().clone()
            else:
                fc1_main_grad = torch.stack(
                    [
                        getattr(fc1, f"weight{group_idx}").main_grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )
                fc2_main_grad = torch.stack(
                    [
                        getattr(fc2, f"weight{group_idx}").main_grad.detach().clone()
                        for group_idx in range(group_size)
                    ],
                    dim=0,
                )
            return fc1_main_grad, fc2_main_grad

        static_split_sizes = split_sizes.clone()

        def train_step(
            x: torch.Tensor,
            probs: torch.Tensor,
            dy: torch.Tensor,
            out_buf: torch.Tensor,
            *,
            use_graphed: bool,
        ) -> torch.Tensor:
            with te.autocast(enabled=True, recipe=recipe):
                out = (
                    graphed_module(x, static_split_sizes, probs, static_split_sizes)
                    if use_graphed
                    else module(x, static_split_sizes, probs, static_split_sizes)
                )
            out.backward(dy)
            out_buf.copy_(out)
            return out_buf

        _init_main_grads(0.0)

        static_x = torch.randn(in_shape, device=device, dtype=dtype, requires_grad=True)
        static_probs = torch.randn((in_shape[0],), device=device, dtype=dtype, requires_grad=True)
        static_dy = torch.randn(in_shape, device=device, dtype=dtype)
        static_out_buf = torch.empty((in_shape[0], hidden_size), device=device, dtype=dtype)

        graphed_module = te.make_graphed_callables(
            module,
            (static_x, static_split_sizes, static_probs, static_split_sizes),
            num_warmup_iters=3,
            enabled=True,
            recipe=recipe,
        )

        forward_ops = module._module_groups[0]._forward_ops
        backward_ops = module._module_groups[0]._backward_ops
        assert len(forward_ops) == 1
        assert isinstance(
            forward_ops[0][0],
            te.ops.fused.GroupedMLP_CuTeGEMMGLU,
        )
        assert len(backward_ops) == 1
        assert isinstance(
            backward_ops[0][0],
            te.ops.fused.GroupedMLP_CuTeGEMMGLU,
        )
        assert backward_ops[0][0] is forward_ops[0][0]

        fresh_x = torch.randn_like(static_x)
        fresh_probs = torch.randn_like(static_probs)
        fresh_dy = torch.randn_like(static_dy)
        with torch.no_grad():
            static_x.copy_(fresh_x)
            static_probs.copy_(fresh_probs)
            static_dy.copy_(fresh_dy)

        for param in module.parameters():
            param.grad = torch.zeros_like(param)
        _init_main_grads(0.5)
        if static_x.grad is not None:
            static_x.grad.zero_()
        if static_probs.grad is not None:
            static_probs.grad.zero_()

        graph_out = (
            train_step(static_x, static_probs, static_dy, static_out_buf, use_graphed=True)
            .detach()
            .clone()
        )
        torch.cuda.synchronize()
        graph_dx = static_x.grad.detach().clone()
        graph_dprobs = static_probs.grad.detach().clone()
        if accumulate_into_main_grad:
            graph_fc1_main_grad, graph_fc2_main_grad = _collect_main_grads()
        else:
            graph_param_grads = [param.grad.detach().clone() for param in module.parameters()]

        for param in module.parameters():
            param.grad.zero_()
        _init_main_grads(0.5)
        static_x.grad.zero_()
        static_probs.grad.zero_()

        expected_x = fresh_x.detach().clone().requires_grad_(True)
        expected_probs = fresh_probs.detach().clone().requires_grad_(True)
        expected_dy = fresh_dy.detach().clone()
        with te.autocast(enabled=True, recipe=recipe):
            expected_out = module(
                expected_x,
                static_split_sizes,
                expected_probs,
                static_split_sizes,
            )
        expected_out.backward(expected_dy)

        tols = dtype_tols(dtype)
        assert_close(graph_out, expected_out, **tols)
        assert_close(graph_dx, expected_x.grad, **tols)
        assert_close(graph_dprobs, expected_probs.grad, **tols)
        if accumulate_into_main_grad:
            expected_fc1_main_grad, expected_fc2_main_grad = _collect_main_grads()
            assert_close(graph_fc1_main_grad, expected_fc1_main_grad, **tols)
            assert_close(graph_fc2_main_grad, expected_fc2_main_grad, **tols)
        else:
            for graph_grad, param in zip(graph_param_grads, module.parameters()):
                assert_close(graph_grad, param.grad, **tols)


def test_grouped_gemm_quant_cute_matches_mxfp8_quantized() -> None:
    if not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Requires SM100+ for grouped GEMM quant kernel.")

    try:
        from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module
    except ImportError as exc:
        pytest.skip(f"grouped_gemm_quant_wrapper_sm100 unavailable: {exc}")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if is_bf16_available() else torch.float16
    num_groups = 4
    m = 256
    n = 512
    k = 512
    total_m = num_groups * m
    split_sizes = torch.full((num_groups,), m, device=device, dtype=torch.int64)

    q = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.optimize_for_gemm = False

    torch.manual_seed(0)
    a_full = torch.randn(total_m, k, device=device, dtype=dtype)
    weights = [torch.randn(n, k, device=device, dtype=dtype) for _ in range(num_groups)]

    grouped_a = tex.group_quantize(a_full, q, num_groups, split_sizes)
    a_groups = grouped_a.split_into_quantized_tensors()
    b_groups = [q(w) for w in weights]

    # Reference GEMM on dequantized tensors.
    ref = torch.empty((total_m, n), device=device, dtype=torch.float32)
    start = 0
    for group_idx in range(num_groups):
        end = start + m
        a_deq = a_groups[group_idx].dequantize(dtype=torch.float32)
        b_deq = b_groups[group_idx].dequantize(dtype=torch.float32)
        ref[start:end, :] = a_deq @ b_deq.t()
        start = end
    ref = ref.to(dtype=torch.bfloat16).to(torch.float32)

    # Allocate empty input tensors needed for cuTE DSL kernel
    padded_offsets = torch.tensor(
        [m * (i + 1) for i in range(num_groups)],
        dtype=torch.int32,
        device=device,
    )
    inputs = {
        "a_tensor": torch.empty(1, total_m, k, dtype=torch.float8_e4m3fn, device=device).permute(
            1, 2, 0
        ),
        "b_tensor": torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=device).permute(
            1, 2, 0
        ),
        "sfa_tensor": torch.empty(
            1,
            total_m // 128,
            k // 128,
            32,
            4,
            4,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ).permute(3, 4, 1, 5, 2, 0),
        "sfb_tensor": torch.empty(
            num_groups,
            n // 128,
            k // 128,
            32,
            4,
            4,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ).permute(3, 4, 1, 5, 2, 0),
        "alpha_tensor": torch.empty(num_groups, dtype=torch.float32, device=device),
        "prob_tensor": torch.empty(total_m, 1, 1, dtype=torch.float32, device=device),
        "padded_offsets_tensor": padded_offsets,
    }
    # Overwrite inputs with quantized data/scales from MXFP8 quantizer.
    a_data = grouped_a.rowwise_data.view(total_m, k).view(dtype=torch.float8_e4m3fn)
    a_data = a_data.unsqueeze(0).permute(1, 2, 0).contiguous()
    inputs["a_tensor"].copy_(a_data)

    a_scales = grouped_a.scale_inv.view(dtype=torch.float8_e8m0fnu)
    a_scales = a_scales.view(1, total_m // 128, 4, 32, k // 128, 4)
    a_scales = a_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
    a_scales = a_scales.permute(3, 4, 1, 5, 2, 0).contiguous()
    inputs["sfa_tensor"].copy_(a_scales)

    b_data = torch.cat([w._rowwise_data.reshape(-1) for w in b_groups])
    b_data = b_data.view(dtype=torch.float8_e4m3fn)
    b_data = b_data.view(num_groups, n, k).permute(1, 2, 0).contiguous()
    inputs["b_tensor"].copy_(b_data)

    b_scales = torch.cat([w._rowwise_scale_inv for w in b_groups])
    b_scales = b_scales.view(dtype=torch.float8_e8m0fnu)
    b_scales = b_scales.view(num_groups, n // 128, 4, 32, k // 128, 4)
    b_scales = b_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
    b_scales = b_scales.permute(3, 4, 1, 5, 2, 0).contiguous()
    inputs["sfb_tensor"].copy_(b_scales)

    inputs["alpha_tensor"].fill_(1.0)
    inputs["prob_tensor"].fill_(1.0)

    cute_out = grouped_gemm_quant_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        norm_const_tensor=None,
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=torch.float32,
        d_dtype=torch.bfloat16,
        cd_major="n",
        sf_vec_size=32,
        discrete_col_sfd=True,
        current_stream=None,
    )

    if isinstance(cute_out, dict):
        outputs = cute_out
    else:
        d_tensor, d_col_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor = cute_out
        outputs = {
            "d_tensor": d_tensor,
            "d_col_tensor": d_col_tensor,
            "amax_tensor": amax_tensor,
            "sfd_row_tensor": sfd_row_tensor,
            "sfd_col_tensor": sfd_col_tensor,
        }

    d_cute = outputs["d_tensor"]
    if d_cute.dim() == 3:
        d_cute = d_cute.squeeze(-1)
    tols = dtype_tols(torch.bfloat16)
    assert_close(d_cute[:total_m].float(), ref, **tols)
