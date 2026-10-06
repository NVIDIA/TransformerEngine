# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""cuDNN MoE API compatibility and ordered dispatch without kernel compilation."""

import importlib
import inspect
import warnings
from types import SimpleNamespace

import pytest
import transformer_engine_jax

adapter = importlib.import_module("transformer_engine.jax.cpp_extensions.grouped_gemm_swiglu")
moe = importlib.import_module("transformer_engine.jax.moe")


def api(arguments, change=None):
    def function(**kwargs):
        raise AssertionError("Probes must not execute kernels")

    parameters = [
        inspect.Parameter(
            name,
            (
                inspect.Parameter.POSITIONAL_ONLY
                if change == "positional" and name == "a_tensor"
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
        )
        for name in arguments
        if not (change == "renamed" and name == "discrete_col_sfd")
    ]
    if change in ("optional", "mandatory"):
        parameters.append(
            inspect.Parameter(
                "new_arg",
                inspect.Parameter.KEYWORD_ONLY,
                default=None if change == "optional" else inspect.Parameter.empty,
            )
        )
    function.__signature__ = inspect.Signature(parameters)
    return function


@pytest.fixture
def frontend(monkeypatch):
    module = SimpleNamespace(
        grouped_gemm_glu=api(adapter._FORWARD_ARGS),
        grouped_gemm_swiglu=api(adapter._FORWARD_ARGS),
        grouped_gemm_dswiglu=api(adapter._BACKWARD_ARGS),
    )
    original = importlib.import_module

    def import_module(name, package=None):
        if name == "cudnn.jax":
            return module
        if name == "cutlass.jax":
            return SimpleNamespace(is_available=lambda: True)
        return original(name, package)

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    return module


@pytest.mark.parametrize("rubin", [False, True])
@pytest.mark.parametrize("operation", ["forward", "backward"])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "not_callable",
        "renamed",
        "mandatory",
        "optional",
        "positional",
        "opaque",
    ],
)
def test_api_signature(frontend, rubin, operation, change):
    name = (
        ("grouped_gemm_glu" if rubin else "grouped_gemm_swiglu")
        if operation == "forward"
        else "grouped_gemm_dswiglu"
    )
    args = adapter._FORWARD_ARGS if operation == "forward" else adapter._BACKWARD_ARGS
    if change == "missing":
        delattr(frontend, name)
    elif change == "not_callable":
        setattr(frontend, name, 42)
    elif change == "opaque":
        setattr(frontend, name, lambda **kwargs: None)
    else:
        setattr(frontend, name, api(args, change))
    available, reason = adapter.grouped_gemm_swiglu_dependencies_available(rubin)
    assert available == (change == "optional")
    assert bool(reason) == (change != "optional")


@pytest.mark.parametrize("capability", [90, 100, 103, 107, 120])
@pytest.mark.parametrize(
    "missing",
    [None, "grouped_gemm_glu", "grouped_gemm_swiglu", "grouped_gemm_dswiglu", "all"],
)
def test_ordered_fallback(frontend, monkeypatch, capability, missing):
    if missing == "all":
        vars(frontend).clear()
    elif missing:
        delattr(frontend, missing)
    monkeypatch.setattr(
        transformer_engine_jax, "get_device_compute_capability", lambda _: capability
    )
    expected = False
    if capability >= 100 and missing not in ("grouped_gemm_dswiglu", "all"):
        if capability == 107 and missing != "grouped_gemm_glu":
            expected = "rubin"
        elif missing != "grouped_gemm_swiglu":
            expected = "blackwell"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert moe._select_cudnn_jax_fusion([]) == expected
    if expected == "rubin":
        assert not caught
    else:
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "falling back" in message and "1.31.0" in message
        assert ("generic Blackwell+ fused" if expected else "unfused TE") in message


@pytest.mark.parametrize("name", ["cudnn.jax", "cutlass.jax"])
def test_missing_dependency(monkeypatch, name):
    original = importlib.import_module

    def import_module(module, package=None):
        if module == name:
            raise ModuleNotFoundError(f"No module named {module}")
        return original(module, package)

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    available, reason = adapter.grouped_gemm_swiglu_dependencies_available()
    assert not available and name in reason


def test_ineligible_call_and_device_query_failure(frontend, monkeypatch):
    monkeypatch.setattr(transformer_engine_jax, "get_device_compute_capability", lambda _: 107)
    with pytest.warns(UserWarning, match="bias unsupported"):
        assert moe._select_cudnn_jax_fusion(["bias unsupported"]) is False

    def fail(_):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(transformer_engine_jax, "get_device_compute_capability", fail)
    with pytest.warns(UserWarning, match="device query failed"):
        assert moe._select_cudnn_jax_fusion([]) is False


def test_installed_frontend_contract():
    for rubin in (False, True):
        available, reason = adapter.grouped_gemm_swiglu_dependencies_available(rubin)
        assert available or reason


@pytest.mark.parametrize("request_fusion", [None, True, False])
@pytest.mark.parametrize("native_layout", [False, True])
@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize(
    "missing,expected",
    [
        (None, "rubin"),
        ("grouped_gemm_glu", "blackwell"),
        ("grouped_gemm_dswiglu", False),
    ],
)
def test_public_moe_passes_selected_path(
    frontend, monkeypatch, native_layout, square, missing, expected, request_fusion
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from transformer_engine.jax.sharding import MeshResource

    if missing:
        delattr(frontend, missing)
    monkeypatch.setattr(transformer_engine_jax, "get_device_compute_capability", lambda _: 107)
    monkeypatch.setattr(moe, "_cudnn_jax_fusion_rejection_reasons", lambda *args, **kwargs: [])
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("ep",))
    monkeypatch.setattr(moe, "_get_mesh", lambda: mesh)
    monkeypatch.setattr(moe, "_with_sharding_constraint_cast_bwd", lambda x, _: x)
    received = []
    signature = inspect.signature(moe._moe)

    def execute(*args):
        bound = signature.bind(*args).arguments
        assert bound["use_cudnn_fusion"] is (request_fusion is not False)
        assert bound["cudnn_native_weight_layout"] is native_layout
        received.append(bound["use_cudnn_jax_fusion"])
        return args[0], None, jnp.asarray(0)

    monkeypatch.setattr(moe, "_moe", execute)
    x = jnp.ones((1, 1, 128), jnp.bfloat16)
    wi = jnp.ones((2, 256, 128) if native_layout else (2, 128, 256), jnp.bfloat16)
    kwargs = {} if request_fusion is None else {"use_cudnn_fusion": request_fusion}
    if square:
        wi = jnp.ones((2, 128, 128), jnp.bfloat16)
        kwargs["cudnn_native_weight_layout"] = native_layout
    if request_fusion is False:
        expected = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output, _, _ = moe.moe(
            x,
            jnp.ones((128, 2)),
            wi,
            jnp.ones((2, 128, 128)),
            num_experts=2,
            num_experts_per_tok=1,
            mesh_resource=MeshResource(ep_resource="ep"),
            **kwargs,
        )
    assert received == [expected]
    assert output is x
    assert len(caught) == (0 if request_fusion is False or expected == "rubin" else 1)


@pytest.mark.parametrize("path", ["rubin", "blackwell"])
@pytest.mark.parametrize("native_layout", [False, True])
@pytest.mark.parametrize("gather_gated_dimension", [False, True])
def test_forward_uses_selected_kernel(monkeypatch, path, native_layout, gather_gated_dimension):
    import jax.numpy as jnp

    class SelectedKernel(Exception):
        pass

    def selected(*args, **kwargs):
        assert args[1].shape == (1, 256, 128)
        raise SelectedKernel

    def rejected(*args, **kwargs):
        raise AssertionError("Forward ignored the selected fallback")

    monkeypatch.setattr(moe.tex, "grouped_gemm_glu", selected if path == "rubin" else rejected)
    monkeypatch.setattr(
        moe.tex, "grouped_gemm_swiglu", selected if path == "blackwell" else rejected
    )

    def quantize(data, *args, **kwargs):
        return SimpleNamespace(
            get_tensor=lambda **kwargs: SimpleNamespace(
                data=data, scale_inv=jnp.ones((1,), dtype=jnp.uint8)
            )
        )

    monkeypatch.setattr(moe.tex, "grouped_quantize", quantize)

    def gather(tensor, fsdp_axis, fsdp_size, sharded_axis):
        data = tensor.get_tensor().data
        assert sharded_axis == (1 if native_layout else 2)
        return quantize(jnp.concatenate([data, data], axis=sharded_axis))

    monkeypatch.setattr(moe, "_gather_quantized_weight", gather)

    def reorder(tensor, *, interleave):
        assert interleave and gather_gated_dimension and not native_layout
        gate, up = jnp.split(tensor.get_tensor().data, 2, axis=-1)
        return quantize(moe.tex.pack_swiglu_pair(gate, up))

    monkeypatch.setattr(moe, "_reorder_quantized_swiglu_weight", reorder)
    quantizers = SimpleNamespace(x=SimpleNamespace(q_dtype=jnp.float8_e4m3fn), kernel=None)
    kwargs = dict.fromkeys(inspect.signature(moe._ffn_fwd_per_shard).parameters)
    combined = 128 if gather_gated_dimension else 256
    kwargs.update(
        recv_tokens_local=jnp.ones((1, 256, 128), jnp.bfloat16),
        recv_topk_weights_local=jnp.ones((1, 256)),
        token_counts_local=jnp.asarray([256]),
        wi=jnp.ones((1, combined, 128) if native_layout else (1, 128, combined), jnp.bfloat16),
        wo=jnp.ones((1, 128, 128), jnp.bfloat16),
        quantizer_sets=(quantizers, quantizers),
        num_local_experts=1,
        use_cudnn_jax_fusion=path,
        cudnn_native_weight_layout=native_layout,
        quant_before_fsdp_ag=gather_gated_dimension,
        wi_fsdp_axis=(1 if native_layout else 2),
        fsdp_axis="fsdp",
        fsdp_size=2,
    )
    with pytest.raises(SelectedKernel):
        moe._ffn_fwd_per_shard(**kwargs)


@pytest.mark.parametrize("request_fusion", [None, True, False])
def test_flax_forwards_fusion_bool(monkeypatch, request_fusion):
    import jax
    import jax.numpy as jnp
    from transformer_engine.jax.flax import _MoEBlock
    from transformer_engine.jax.sharding import MeshResource

    flax_moe = importlib.import_module("transformer_engine.jax.flax.moe")
    received = []

    def execute(inputs, *args, **kwargs):
        received.append(kwargs["use_cudnn_fusion"])
        return inputs, None, jnp.asarray(0)

    monkeypatch.setattr(flax_moe, "moe", execute)
    kwargs = {} if request_fusion is None else {"use_cudnn_fusion": request_fusion}
    block = _MoEBlock(
        num_experts=2,
        intermediate_size=32,
        mesh_resource=MeshResource(ep_resource="ep"),
        **kwargs,
    )
    block.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 32)))
    assert received == [request_fusion is not False]


@pytest.mark.parametrize("request_fusion", [True, False])
def test_capacity_follows_explicit_fusion_bool(request_fusion):
    kwargs = dict(num_experts=8, num_experts_per_tok=2, max_tokens_per_rank=64, ep_size=2)
    alignment = moe._CUDNN_JAX_ALIGN_SIZE if request_fusion else moe._ALIGN_SIZE
    expected = moe.get_moe_recv_capacity_per_rank(**kwargs, alignment=alignment)
    assert moe.get_moe_recv_capacity_per_rank(**kwargs, use_cudnn_fusion=request_fusion) == expected
    if request_fusion:
        assert moe.get_moe_recv_capacity_per_rank(**kwargs) == expected


@pytest.mark.parametrize("invalid", [0, 1, None, "true"])
def test_fusion_argument_requires_bool(invalid):
    with pytest.raises(TypeError, match="use_cudnn_fusion must be a bool"):
        moe.moe(
            None,
            None,
            None,
            None,
            num_experts=2,
            num_experts_per_tok=1,
            use_cudnn_fusion=invalid,
        )
    with pytest.raises(TypeError, match="use_cudnn_fusion must be a bool"):
        moe.get_moe_recv_capacity_per_rank(
            num_experts=2,
            num_experts_per_tok=1,
            max_tokens_per_rank=16,
            ep_size=1,
            use_cudnn_fusion=invalid,
        )


def test_vjp_bool_is_static_and_defaults_to_true():
    assert inspect.signature(moe._moe).parameters["use_cudnn_fusion"].default is True
    assert inspect.signature(moe.moe).parameters["use_cudnn_fusion"].default is True
    assert 32 in moe._moe.nondiff_argnums
