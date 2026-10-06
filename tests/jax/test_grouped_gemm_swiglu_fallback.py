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


@pytest.mark.parametrize("native_layout", [False, True])
@pytest.mark.parametrize(
    "missing,expected",
    [(None, "rubin"), ("grouped_gemm_glu", "blackwell"), ("grouped_gemm_dswiglu", False)],
)
def test_public_moe_passes_selected_path(frontend, monkeypatch, native_layout, missing, expected):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from transformer_engine.jax.sharding import MeshResource

    if missing:
        delattr(frontend, missing)
    monkeypatch.setenv(moe._CUDNN_JAX_ENV, "1")
    monkeypatch.setattr(transformer_engine_jax, "get_device_compute_capability", lambda _: 107)
    monkeypatch.setattr(moe, "_cudnn_jax_fusion_rejection_reasons", lambda *args, **kwargs: [])
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("ep",))
    monkeypatch.setattr(moe, "_get_mesh", lambda: mesh)
    monkeypatch.setattr(moe, "_with_sharding_constraint_cast_bwd", lambda x, _: x)
    received = []
    signature = inspect.signature(moe._moe)

    def execute(*args):
        received.append(signature.bind(*args).arguments["use_cudnn_jax_fusion"])
        return args[0], None, jnp.asarray(0)

    monkeypatch.setattr(moe, "_moe", execute)
    x = jnp.ones((1, 1, 128), jnp.bfloat16)
    wi = jnp.ones((2, 256, 128) if native_layout else (2, 128, 256), jnp.bfloat16)
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
        )
    assert received == [expected]
    assert output is x
    assert len(caught) == (0 if expected == "rubin" else 1)


@pytest.mark.parametrize("path", ["rubin", "blackwell"])
def test_forward_uses_selected_kernel(monkeypatch, path):
    import jax.numpy as jnp

    class SelectedKernel(Exception):
        pass

    def selected(*args, **kwargs):
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
    quantizers = SimpleNamespace(x=SimpleNamespace(q_dtype=jnp.float8_e4m3fn), kernel=None)
    kwargs = dict.fromkeys(inspect.signature(moe._ffn_fwd_per_shard).parameters)
    kwargs.update(
        recv_tokens_local=jnp.ones((1, 256, 128), jnp.bfloat16),
        recv_topk_weights_local=jnp.ones((1, 256)),
        token_counts_local=jnp.asarray([256]),
        wi=jnp.ones((1, 128, 256), jnp.bfloat16),
        wo=jnp.ones((1, 128, 128), jnp.bfloat16),
        quantizer_sets=(quantizers, quantizers),
        num_local_experts=1,
        use_cudnn_jax_fusion=path,
        cudnn_native_weight_layout=False,
        quant_before_fsdp_ag=False,
    )
    with pytest.raises(SelectedKernel):
        moe._ffn_fwd_per_shard(**kwargs)
