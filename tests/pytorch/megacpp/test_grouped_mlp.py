# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops


_HIDDEN_SIZE = 512
_FFN_HIDDEN_SIZE = 256


def _megacpp_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if not te.is_bf16_available():
        return False, "BF16 is required"
    if torch.cuda.get_device_capability() < (10, 0):
        return False, "megacpp grouped MLP uses SM100 grouped GEMM"
    if not te_ops.fused.ForwardGroupedMLP_MegaCpp.is_supported():
        return False, "ForwardGroupedMLP_MegaCpp is not supported"
    if not te_ops.fused.BackwardGroupedMLP_MegaCpp.is_supported():
        return False, "BackwardGroupedMLP_MegaCpp is not supported"
    return True, ""


_AVAILABLE, _SKIP_REASON = _megacpp_available()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=_SKIP_REASON)


def _make_grouped_mlp(
    *,
    num_groups: int,
    hidden_size: int,
    ffn_hidden_size: int,
    activation_kind: str,
    bias: bool,
    delay_wgrad_compute: bool,
    accumulate_into_main_grad: bool,
    glu_interleave_size: int | None,
    single_grouped_param: bool,
) -> te_ops.Sequential:
    gated_activation = activation_kind in ("scaled_swiglu", "scaled_clamped_qgeglu")
    fc1_out_features = 2 * ffn_hidden_size if gated_activation else ffn_hidden_size
    fc1 = te_ops.GroupedLinear(
        num_groups,
        hidden_size,
        fc1_out_features,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        single_grouped_weight=single_grouped_param,
        single_grouped_bias=single_grouped_param and bias,
    )
    if activation_kind == "scaled_swiglu":
        act = te_ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
    elif activation_kind == "scaled_clamped_qgeglu":
        act = te_ops.ScaledClampedQGeGLU(glu_interleave_size=glu_interleave_size)
    elif activation_kind == "scaled_srelu":
        act = te_ops.ScaledSReLU()
    else:
        raise ValueError(f"Unsupported test activation_kind={activation_kind}.")
    fc2 = te_ops.GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        single_grouped_weight=single_grouped_param,
        single_grouped_bias=single_grouped_param and bias,
    )
    return te_ops.Sequential(fc1, act, fc2)


def _copy_grouped_mlp_params(dst: te_ops.Sequential, src: te_ops.Sequential) -> None:
    with torch.no_grad():
        for dst_linear, src_linear in ((dst[0], src[0]), (dst[2], src[2])):
            if dst_linear.single_grouped_weight:
                dst_linear.weight.rowwise_data.copy_(src_linear.weight.rowwise_data)
                if dst_linear.has_bias:
                    dst_linear.bias.rowwise_data.copy_(src_linear.bias.rowwise_data)
            else:
                for group_idx in range(dst_linear.num_groups):
                    getattr(dst_linear, f"weight{group_idx}").copy_(
                        getattr(src_linear, f"weight{group_idx}")
                    )
                    if dst_linear.has_bias:
                        getattr(dst_linear, f"bias{group_idx}").copy_(
                            getattr(src_linear, f"bias{group_idx}")
                        )


def _init_main_grads(module: te_ops.Sequential, dtype: torch.dtype) -> None:
    for linear in (module[0], module[2]):
        if linear.single_grouped_weight:
            linear.weight.main_grad = torch.zeros(
                linear.num_groups,
                linear.out_features,
                linear.in_features,
                device="cuda",
                dtype=dtype,
            )
        else:
            for group_idx in range(linear.num_groups):
                weight = getattr(linear, f"weight{group_idx}")
                weight.main_grad = torch.zeros_like(weight, dtype=dtype)


def _run_grouped_mlp(
    module: te_ops.Sequential,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    act_scales: torch.Tensor,
    dy: torch.Tensor,
    *,
    delay_wgrad_compute: bool,
) -> torch.Tensor:
    y = module(x, split_sizes, act_scales, split_sizes)
    y.backward(dy)
    if delay_wgrad_compute:
        module[0].backward_dw()
        module[2].backward_dw()
    return y


def _assert_grouped_mlp_close(
    test: te_ops.Sequential,
    ref: te_ops.Sequential,
    *,
    accumulate_into_main_grad: bool,
) -> None:
    for test_linear, ref_linear in ((test[0], ref[0]), (test[2], ref[2])):
        if test_linear.single_grouped_weight:
            if accumulate_into_main_grad:
                torch.testing.assert_close(
                    test_linear.weight.main_grad,
                    ref_linear.weight.main_grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    test_linear.weight.grad,
                    ref_linear.weight.grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    test_linear.bias.grad,
                    ref_linear.bias.grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            continue
        for group_idx in range(test_linear.num_groups):
            if accumulate_into_main_grad:
                torch.testing.assert_close(
                    getattr(test_linear, f"weight{group_idx}").main_grad,
                    getattr(ref_linear, f"weight{group_idx}").main_grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    getattr(test_linear, f"weight{group_idx}").grad,
                    getattr(ref_linear, f"weight{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    getattr(test_linear, f"bias{group_idx}").grad,
                    getattr(ref_linear, f"bias{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )


def _assert_grouped_mlp_nonzero_expert_grads_close(
    test: te_ops.Sequential,
    ref: te_ops.Sequential,
    split_sizes: list[int],
) -> None:
    """Compare only non-empty experts; zero-token expert grads may be unwritten."""
    for test_linear, ref_linear in ((test[0], ref[0]), (test[2], ref[2])):
        for group_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            torch.testing.assert_close(
                getattr(test_linear, f"weight{group_idx}").grad,
                getattr(ref_linear, f"weight{group_idx}").grad,
                rtol=2e-2,
                atol=2e-2,
            )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    getattr(test_linear, f"bias{group_idx}").grad,
                    getattr(ref_linear, f"bias{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )


def _assert_valid_prefix_close(
    test: torch.Tensor,
    ref: torch.Tensor,
    valid_tokens: int,
) -> None:
    """Paged-stashed buffers only guarantee correctness in the valid token prefix."""
    if valid_tokens == 0:
        return
    torch.testing.assert_close(test[:valid_tokens], ref[:valid_tokens], rtol=2e-2, atol=2e-2)


def _make_split_tensor(
    split_sizes: list[int],
    *,
    dtype: torch.dtype = torch.int64,
    device: str = "cuda",
) -> torch.Tensor:
    return torch.tensor(split_sizes, dtype=dtype, device=device)


def _run_megacpp_against_python(
    *,
    split_sizes_list: list[int],
    physical_tokens: int,
    split_dtype: torch.dtype,
    split_device: str,
    bias: bool = True,
    glu_interleave_size: int | None = None,
    activation_kind: str = "scaled_swiglu",
    single_grouped_param: bool = False,
    accumulate_into_main_grad: bool = False,
    main_grad_dtype: torch.dtype | None = None,
    compare_zero_expert_grads: bool = True,
    monkeypatch,
) -> None:
    num_groups = len(split_sizes_list)
    valid_tokens = sum(split_sizes_list)
    assert physical_tokens >= valid_tokens
    if single_grouped_param:
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    split_sizes = _make_split_tensor(split_sizes_list, dtype=split_dtype, device=split_device)
    ref = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        activation_kind=activation_kind,
        bias=bias,
        delay_wgrad_compute=False,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    test = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        activation_kind=activation_kind,
        bias=bias,
        delay_wgrad_compute=False,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    _copy_grouped_mlp_params(test, ref)
    if accumulate_into_main_grad:
        if main_grad_dtype is None:
            raise ValueError("main_grad_dtype must be set when using Megatron-owned main_grad.")
        _init_main_grads(ref, main_grad_dtype)
        _init_main_grads(test, main_grad_dtype)

    # Paged stashing passes a static physical buffer to the op while m_splits
    # describe only the valid prefix. Rows after sum(m_splits) are garbage and
    # must not affect outputs/gradients for the valid prefix.
    x_ref = torch.randn(
        physical_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    x_test = x_ref.detach().clone().requires_grad_()
    act_scales_ref = torch.rand(
        physical_tokens, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    act_scales_test = act_scales_ref.detach().clone().requires_grad_()
    dy = torch.randn(physical_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "0")
    y_ref = _run_grouped_mlp(
        ref,
        x_ref,
        split_sizes,
        act_scales_ref,
        dy,
        delay_wgrad_compute=False,
    )
    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "1")
    y_test = _run_grouped_mlp(
        test,
        x_test,
        split_sizes,
        act_scales_test,
        dy,
        delay_wgrad_compute=False,
    )

    fuser = test._module_groups[0]
    assert isinstance(fuser._forward_ops[0][0], te_ops.fused.ForwardGroupedMLP_MegaCpp)
    assert isinstance(fuser._backward_ops[0][0], te_ops.fused.BackwardGroupedMLP_MegaCpp)

    _assert_valid_prefix_close(y_test, y_ref, valid_tokens)
    _assert_valid_prefix_close(x_test.grad, x_ref.grad, valid_tokens)
    _assert_valid_prefix_close(
        act_scales_test.grad,
        act_scales_ref.grad,
        valid_tokens,
    )
    if valid_tokens == physical_tokens and compare_zero_expert_grads:
        _assert_grouped_mlp_close(test, ref, accumulate_into_main_grad=accumulate_into_main_grad)
    elif valid_tokens > 0 and not single_grouped_param and not accumulate_into_main_grad:
        _assert_grouped_mlp_nonzero_expert_grads_close(test, ref, split_sizes_list)


@pytest.mark.parametrize(
    "single_grouped_param",
    [False, True],
    ids=["discrete_weight", "packed_weight"],
)
@pytest.mark.parametrize(
    "accumulate_into_main_grad,main_grad_dtype",
    [
        pytest.param(False, None, id="cpp_allocated_wgrad"),
        pytest.param(True, torch.bfloat16, id="megatron_main_grad_bf16"),
        pytest.param(True, torch.float32, id="megatron_main_grad_fp32"),
    ],
)
def test_megacpp_grouped_mlp_wgrad_storage_matches_python(
    single_grouped_param,
    accumulate_into_main_grad,
    main_grad_dtype,
    monkeypatch,
):
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=[256, 256, 512],
        physical_tokens=1024,
        split_dtype=torch.int64,
        split_device="cuda",
        single_grouped_param=single_grouped_param,
        accumulate_into_main_grad=accumulate_into_main_grad,
        main_grad_dtype=main_grad_dtype,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "split_dtype,split_device",
    [
        pytest.param(torch.int64, "cuda", id="i64_cuda"),
        pytest.param(torch.int32, "cuda", id="i32_cuda"),
        pytest.param(torch.int64, "cpu", id="i64_cpu"),
    ],
)
def test_megacpp_grouped_mlp_split_source_matches_python(
    split_dtype,
    split_device,
    monkeypatch,
):
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=[256, 256, 512],
        physical_tokens=1024,
        split_dtype=split_dtype,
        split_device=split_device,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "activation_kind",
    ["scaled_swiglu", "scaled_srelu", "scaled_clamped_qgeglu"],
    ids=["swiglu", "srelu", "clamped_qgeglu"],
)
@pytest.mark.parametrize(
    "glu_interleave_size",
    [None, 32],
    ids=["no_interleave", "interleave_32"],
)
def test_megacpp_grouped_mlp_activation_matches_python(
    activation_kind,
    glu_interleave_size,
    monkeypatch,
):
    if activation_kind == "scaled_srelu" and glu_interleave_size is not None:
        pytest.skip("ScaledSReLU is not a GLU activation.")
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=[256, 256, 512],
        physical_tokens=1024,
        split_dtype=torch.int64,
        split_device="cuda",
        activation_kind=activation_kind,
        glu_interleave_size=glu_interleave_size,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_megacpp_grouped_mlp_bias_matches_python(bias, monkeypatch):
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=[256, 256, 512],
        physical_tokens=1024,
        split_dtype=torch.int64,
        split_device="cuda",
        bias=bias,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "split_sizes_list,physical_tokens",
    [
        pytest.param([256, 256, 256, 256], 1024, id="even"),
        pytest.param([0, 256, 256, 512], 1024, id="zero_front"),
        pytest.param([256, 0, 256, 512], 1024, id="zero_middle"),
        pytest.param([256, 256, 512, 0], 1024, id="zero_end"),
        pytest.param([256, 256], 1024, id="paged_stashing_even_with_garbage"),
        pytest.param([0, 256, 256], 1024, id="paged_stashing_zero_front_with_garbage"),
        pytest.param([256, 0, 256], 1024, id="paged_stashing_zero_middle_with_garbage"),
        pytest.param([256, 256, 0], 1024, id="paged_stashing_zero_end_with_garbage"),
        pytest.param([0, 0, 0, 0], 1024, id="paged_stashing_zero_tokens_all_nonempty_input"),
    ],
)
def test_megacpp_grouped_mlp_split_edge_cases(
    split_sizes_list,
    physical_tokens,
    monkeypatch,
):
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=split_sizes_list,
        physical_tokens=physical_tokens,
        split_dtype=torch.int64,
        split_device="cuda",
        compare_zero_expert_grads=False,
        monkeypatch=monkeypatch,
    )


def test_megacpp_grouped_mlp_delay_wgrad_raises(monkeypatch):
    torch.manual_seed(1234)
    num_groups = 3
    split_sizes = torch.tensor([256, 256, 512], dtype=torch.int64, device="cuda")
    total_tokens = int(split_sizes.sum().item())
    module = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        activation_kind="scaled_swiglu",
        bias=True,
        delay_wgrad_compute=True,
        accumulate_into_main_grad=False,
        glu_interleave_size=None,
        single_grouped_param=False,
    )
    x = torch.randn(
        total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    act_scales = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn(total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "1")
    with pytest.raises(ValueError, match="delay_wgrad_compute"):
        y = module(x, split_sizes, act_scales, split_sizes)
        y.backward(dy)
