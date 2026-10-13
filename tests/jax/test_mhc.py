# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the mHC (manifold Hyper-Connection) JAX API, mirroring tests/pytorch/test_mhc.py."""

import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from utils import assert_allclose, pytest_parametrize_wrapper


@pytest.fixture(autouse=True, scope="function")
def _inject_mhc(request):
    """Lazy-load the mHC API only for tests marked 'triton'."""
    if not request.node.get_closest_marker("triton"):
        yield
        return
    from transformer_engine.jax import mhc as te_mhc

    mod = sys.modules[__name__]
    for name in te_mhc.__all__:
        setattr(mod, name, getattr(te_mhc, name))
    yield


ENFORCE_DETERMINISTIC = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "0"
HIGHEST = jax.lax.Precision.HIGHEST
FP32_EPS = float(jnp.finfo(jnp.float32).eps)
N_STREAMS = 4
N_MIX = 2 * N_STREAMS + N_STREAMS * N_STREAMS
H_PADDED_DIM = 32

# (b, s, C), as in tests/pytorch/test_mhc.py.
ALL_CONFIGS = [
    (8, 32, 32),
    (8, 128, 16 * 64),
    (4, 128, 16 * 64),
    (2, 2048, 24 * 128),
    (1, 2048, 24 * 128),
    (13, 1, 16 * 128),
    (7, 1, 16 * 256),
    (8, 1, 16 * 192),
    (8, 128, 5129),
    (8, 512, 8000),
    (4, 1024, 8192),
    (2, 4096, 8192),
    (8, 128, 16384),
]
CONFIGS = {"L0": ALL_CONFIGS[:3], "L1": ALL_CONFIGS[:6], "L2": ALL_CONFIGS}
DTYPES = [jnp.float32, jnp.bfloat16]
# (x dtype, phi dtype)
PROJECTION_DTYPES = [
    pytest.param((jnp.float32, jnp.float32), id="x_fp32_phi_fp32"),
    pytest.param((jnp.bfloat16, jnp.bfloat16), id="x_bf16_phi_bf16"),
    pytest.param((jnp.bfloat16, jnp.float32), id="x_bf16_phi_fp32"),
]


def get_tols(dtype):
    if dtype == jnp.bfloat16:
        return {"atol": 2.5e-2, "rtol": 2.5e-2}
    return {"atol": 5e-3, "rtol": 5e-3}


def _normal(key, shape, dtype, scale=1.0):
    return (scale * jax.random.normal(key, shape, jnp.float32)).astype(dtype)


def _sum_loss(*outputs):
    return sum(jnp.sum(out.astype(jnp.float32)) for out in outputs)


# Reference operators, following the numerics of the Triton kernels.


def mhc_projection_ref(x, phi, norm_weight):
    """H = x @ (phi * norm_weight)^T and ms = mean(x^2), computed in fp32."""
    x = x.astype(jnp.float32)
    phi = phi.astype(jnp.float32)
    if norm_weight is not None:
        phi = phi * norm_weight.astype(jnp.float32)[None, :]
    h = jnp.einsum("mk,nk->mn", x, phi, precision=HIGHEST)
    ms = jnp.mean(x * x, axis=-1)
    return h, ms


def mhc_scale_ref(h, alpha, beta, ms, n, eps=FP32_EPS):
    """H_pre, H_post and H_res from the projection, computed in fp32."""
    dtype = h.dtype
    h = h.astype(jnp.float32)
    alpha = alpha.astype(jnp.float32)
    beta = beta.astype(jnp.float32).reshape(-1)
    rms = jnp.sqrt(ms.astype(jnp.float32) + eps)[..., None]
    h_pre = jax.nn.sigmoid(h[..., :n] * alpha[0] / rms + beta[:n])
    h_post = 2 * jax.nn.sigmoid(h[..., n : 2 * n] * alpha[1] / rms + beta[n : 2 * n])
    h_res = h[..., 2 * n : 2 * n + n * n] * alpha[2] / rms + beta[2 * n : 2 * n + n * n]
    return h_pre.astype(dtype), h_post.astype(dtype), h_res.astype(dtype)


def mhc_sinkhorn_ref(h_res, iters=20):
    """Log-space Sinkhorn of (..., n, n) in fp32."""
    dtype = h_res.dtype
    h = h_res.astype(jnp.float32)
    f = jnp.zeros(h.shape[:-1], jnp.float32)
    g = jnp.zeros(h.shape[:-1], jnp.float32)
    for _ in range(iters):
        f = -jax.nn.logsumexp(h + g[..., None, :], axis=-1)
        g = -jax.nn.logsumexp(h + f[..., :, None], axis=-2)
    return jnp.exp(f[..., :, None] + h + g[..., None, :]).astype(dtype)


def mhc_aggregate_ref(x, h_pre):
    """(..., C, n) @ (..., n) -> (..., C) with fp32 accumulation."""
    out = jnp.einsum(
        "...cn,...n->...c", x.astype(jnp.float32), h_pre.astype(jnp.float32), precision=HIGHEST
    )
    return out.astype(x.dtype)


def mhc_expand_combine_ref(f, bias, h_post, x, h_res):
    """(f [+ bias]) @ H_post + x @ H_res with fp32 accumulation."""
    dtype = f.dtype
    f = f.astype(jnp.float32)
    if bias is not None:
        f = f + bias.astype(jnp.float32)
    out = f[..., :, None] * h_post.astype(jnp.float32)[..., None, :] + jnp.einsum(
        "...ck,...km->...cm", x.astype(jnp.float32), h_res.astype(jnp.float32), precision=HIGHEST
    )
    return out.astype(dtype)


def mhc_ref(x, params, branch_fn, norm_eps, pre_eps, iters):
    """
    `mhc` on x of shape (..., C, n), following the kernels' numerics: RMSNorm folded into the
    projection, fp32 mappings and a log-space Sinkhorn.
    """
    *tokens, C, n = x.shape
    phi, alpha, beta, norm_weight = jax.tree.map(lambda p: p.astype(jnp.float32), params)
    x32 = x.astype(jnp.float32)

    x_flat = x32.reshape(*tokens, C * n)
    ms = jnp.mean(x_flat * x_flat, axis=-1, keepdims=True)
    h = jnp.einsum("...k,nk->...n", x_flat, phi * norm_weight[None, :], precision=HIGHEST)
    h = h / jnp.sqrt(ms + norm_eps)

    pre = jax.nn.sigmoid(alpha[0] * h[..., :n] + beta[:n]) + pre_eps
    post = 2 * jax.nn.sigmoid(alpha[1] * h[..., n : 2 * n] + beta[n : 2 * n])
    res = alpha[2] * h[..., 2 * n :] + beta[2 * n :]
    res = mhc_sinkhorn_ref(res.reshape(*tokens, n, n), iters)

    layer_input = jnp.einsum("...cn,...n->...c", x32, pre, precision=HIGHEST)
    layer_input = _kernel_boundary(layer_input, x.dtype).astype(x.dtype)
    layer_out = branch_fn(layer_input)
    layer_out = _kernel_boundary(layer_out.astype(jnp.float32), layer_out.dtype)
    out = layer_out[..., :, None] * post[..., None, :] + jnp.einsum(
        "...ck,...km->...cm", x32, res, precision=HIGHEST
    )
    return _kernel_boundary(out, x.dtype).astype(x.dtype)


def _reduce_precision(y, dtype):
    info = jnp.finfo(dtype)
    return jax.lax.reduce_precision(y, exponent_bits=info.nexp, mantissa_bits=info.nmant)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _kernel_boundary(y, dtype):
    """
    Round fp32 y and its cotangent to `dtype`, as where the kernels read or write an array.
    XLA may otherwise keep fused low-precision intermediates in fp32 (xla_allow_excess_precision).
    """
    return _reduce_precision(y, dtype)


_kernel_boundary.defvjp(
    lambda y, dtype: (_reduce_precision(y, dtype), None),
    lambda dtype, _, g: (_reduce_precision(g, dtype),),
)


def _branch_fn(y):
    """A pre-norm followed by a nonlinear sub-layer."""
    y32 = y.astype(jnp.float32)
    y32 = y32 * jax.lax.rsqrt(jnp.mean(y32 * y32, axis=-1, keepdims=True) + 1e-6)
    return jnp.tanh(2 * y32).astype(y.dtype)


def _make_params(key, C, n, dtype):
    """(phi, alpha, beta, norm_weight) for x of shape (..., C, n)."""
    keys = jax.random.split(key, 4)
    nC = n * C
    return (
        _normal(keys[0], (N_MIX, nC), jnp.float32, nC**-0.5).astype(dtype),
        (1.0 + _normal(keys[1], (3,), jnp.float32, 0.1)).astype(dtype),
        _normal(keys[2], (N_MIX,), dtype, 0.5),
        (1.0 + _normal(keys[3], (nC,), jnp.float32, 0.1)).astype(dtype),
    )


@pytest.mark.triton
class TestMHCOps:
    """Each mHC op against its reference, as in tests/pytorch/test_mhc.py."""

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtypes", PROJECTION_DTYPES)
    @pytest_parametrize_wrapper("has_norm_weight", [False, True])
    @pytest_parametrize_wrapper("use_split_k", [True, False])
    def test_mhc_projection(self, b, s, C, dtypes, has_norm_weight, use_split_k):
        if ENFORCE_DETERMINISTIC and use_split_k:
            pytest.skip("Split-K is not deterministic, skip the test under deterministic mode")
        x_dtype, phi_dtype = dtypes
        tols = get_tols(x_dtype)
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        nC = N_STREAMS * C
        x = _normal(keys[0], (s * b, nC), x_dtype)
        phi = _normal(keys[1], (N_MIX, nC), phi_dtype)
        norm_weight = _normal(keys[2], (nC,), x_dtype) if has_norm_weight else None

        def fused(x, phi, norm_weight):
            h, ms = mhc_projection(x, phi, norm_weight, use_tf32=False, use_split_k=use_split_k)
            return h[:, :N_MIX], ms

        fused_h, fused_ms = jax.jit(fused)(x, phi, norm_weight)
        ref_h, ref_ms = jax.jit(mhc_projection_ref)(x, phi, norm_weight)
        assert_allclose(fused_h, ref_h, **tols)
        assert_allclose(fused_ms, ref_ms, **tols)

        argnums = (0, 1, 2) if has_norm_weight else (0, 1)
        grads = jax.jit(jax.grad(lambda *a: _sum_loss(*fused(*a)), argnums))(x, phi, norm_weight)
        ref_grads = jax.jit(jax.grad(lambda *a: _sum_loss(*mhc_projection_ref(*a)), argnums))(
            x, phi, norm_weight
        )
        for name, grad, ref_grad in zip(("x", "phi", "norm_weight"), grads, ref_grads):
            assert grad.dtype == ref_grad.dtype, name
            assert_allclose(grad, ref_grad, err_msg=name, **tols)

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_mhc_scale(self, b, s, C, dtype):
        del C
        tols = get_tols(dtype)
        n = N_STREAMS
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        h = _normal(keys[0], (s * b, H_PADDED_DIM), dtype)
        alpha = _normal(keys[1], (3,), dtype)
        beta = _normal(keys[2], (1, N_MIX), dtype)
        ms = (jnp.abs(_normal(keys[3], (s * b,), jnp.float32)) + 1.0).astype(dtype)

        fused = lambda h, alpha, beta, ms: mhc_scale(h, alpha, beta, ms, n)
        ref = lambda h, alpha, beta, ms: mhc_scale_ref(h[:, :N_MIX], alpha, beta, ms, n)

        args = (h, alpha, beta, ms)
        for out, ref_out in zip(jax.jit(fused)(*args), jax.jit(ref)(*args)):
            assert_allclose(out, ref_out, **tols)

        argnums = (0, 1, 2, 3)
        grads = jax.jit(jax.grad(lambda *a: _sum_loss(*fused(*a)), argnums))(*args)
        ref_grads = jax.jit(jax.grad(lambda *a: _sum_loss(*ref(*a)), argnums))(*args)
        grads = (grads[0][:, :N_MIX],) + grads[1:]
        ref_grads = (ref_grads[0][:, :N_MIX],) + ref_grads[1:]
        for name, grad, ref_grad in zip(("h", "alpha", "beta", "ms"), grads, ref_grads):
            assert_allclose(grad, ref_grad, err_msg=name, **tols)

    @pytest_parametrize_wrapper("b,s,C", [(8, 128, 1024)])
    def test_mhc_scale_mixed_alpha_beta_dtypes(self, b, s, C):
        """A fp32 beta gets a fp32 gradient even when alpha is bf16."""
        del C
        n = N_STREAMS
        keys = jax.random.split(jax.random.PRNGKey(10), 4)
        h = _normal(keys[0], (s * b, H_PADDED_DIM), jnp.float32)
        alpha = _normal(keys[1], (3,), jnp.bfloat16)
        beta = _normal(keys[2], (1, N_MIX), jnp.float32)
        ms = jnp.abs(_normal(keys[3], (s * b,), jnp.float32)) + 1.0

        def grad_beta(fn):
            loss = lambda beta: _sum_loss(*fn(h, alpha, beta, ms))
            return jax.jit(jax.grad(loss))(beta)

        grad = grad_beta(lambda *a: mhc_scale(*a, n))
        ref_grad = grad_beta(lambda h, *a: mhc_scale_ref(h[:, :N_MIX], *a, n))
        assert grad.dtype == jnp.float32
        # Tighter than bf16 resolution, so rounding grad_beta through alpha's dtype would fail.
        assert_allclose(grad, ref_grad, rtol=1e-4, atol=1e-4)

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtypes", PROJECTION_DTYPES)
    @pytest_parametrize_wrapper("has_norm_weight", [False, True])
    @pytest_parametrize_wrapper("use_split_k", [True, False])
    def test_mhc_rmsnorm(self, b, s, C, dtypes, has_norm_weight, use_split_k):
        """Projection + scale (RMSNorm split around the matmul) vs RMSNorm applied first."""
        if ENFORCE_DETERMINISTIC and use_split_k:
            pytest.skip("Split-K is not deterministic, skip the test under deterministic mode")
        x_dtype, phi_dtype = dtypes
        tols = get_tols(x_dtype)
        n = N_STREAMS
        nC = n * C
        keys = jax.random.split(jax.random.PRNGKey(2), 5)
        x = _normal(keys[0], (s * b, nC), x_dtype)
        phi = _normal(keys[1], (N_MIX, nC), phi_dtype)
        alpha = _normal(keys[2], (3,), phi_dtype)
        beta = _normal(keys[3], (1, N_MIX), phi_dtype)
        norm_weight = _normal(keys[4], (nC,), x_dtype) if has_norm_weight else None

        def fused(x, phi, alpha, beta, norm_weight):
            h, ms = mhc_projection(x, phi, norm_weight, use_tf32=False, use_split_k=use_split_k)
            return mhc_scale(h, alpha, beta, ms, n)

        def combined(x, phi, alpha, beta, norm_weight):
            x = x.astype(jnp.float32)
            x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + FP32_EPS)
            if norm_weight is not None:
                x = x * norm_weight.astype(jnp.float32)
            h = jnp.einsum("mk,nk->mn", x, phi.astype(jnp.float32), precision=HIGHEST)
            alpha = alpha.astype(jnp.float32)
            beta = beta.astype(jnp.float32).reshape(-1)
            return (
                jax.nn.sigmoid(h[:, :n] * alpha[0] + beta[:n]),
                2 * jax.nn.sigmoid(h[:, n : 2 * n] * alpha[1] + beta[n : 2 * n]),
                h[:, 2 * n :] * alpha[2] + beta[2 * n :],
            )

        args = (x, phi, alpha, beta, norm_weight)
        for out, ref_out in zip(jax.jit(fused)(*args), jax.jit(combined)(*args)):
            assert_allclose(out, ref_out, **tols)

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("recompute", [False, True])
    @pytest_parametrize_wrapper("iters", [1, 20])
    def test_mhc_sinkhorn(self, b, s, C, dtype, recompute, iters):
        del C
        tols = get_tols(dtype)
        n = N_STREAMS
        keys = jax.random.split(jax.random.PRNGKey(3), 2)
        h_res = _normal(keys[0], (s, b, n, n), dtype)
        # Rows and columns of the output each sum to 1, so a sum loss has a ~zero gradient.
        cotangent = _normal(keys[1], (s, b, n, n), jnp.float32)

        fused = lambda h: mhc_sinkhorn(h, n, recompute_hist=recompute, iters=iters)
        ref = lambda h: mhc_sinkhorn_ref(h, iters)
        assert_allclose(jax.jit(fused)(h_res), jax.jit(ref)(h_res), **tols)

        def grad_of(fn):
            return jax.jit(jax.grad(lambda h: jnp.sum(fn(h).astype(jnp.float32) * cotangent)))

        assert_allclose(grad_of(fused)(h_res), grad_of(ref)(h_res), **tols)

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_mhc_aggregate(self, b, s, C, dtype):
        tols = get_tols(dtype)
        n = N_STREAMS
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        x = _normal(keys[0], (s, b, C, n), dtype)
        h_pre = _normal(keys[1], (s, b, n), dtype)

        fused = lambda x, h_pre: mhc_aggregate(x, h_pre, use_tf32=False)
        assert_allclose(jax.jit(fused)(x, h_pre), jax.jit(mhc_aggregate_ref)(x, h_pre), **tols)

        grads = jax.jit(jax.grad(lambda *a: _sum_loss(fused(*a)), (0, 1)))(x, h_pre)
        ref_grads = jax.jit(jax.grad(lambda *a: _sum_loss(mhc_aggregate_ref(*a)), (0, 1)))(x, h_pre)
        for name, grad, ref_grad in zip(("x", "h_pre"), grads, ref_grads):
            assert_allclose(grad, ref_grad, err_msg=name, **tols)

    @pytest_parametrize_wrapper("b,s,C", CONFIGS)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("with_bias", [True, False])
    def test_mhc_expand_combine(self, b, s, C, dtype, with_bias):
        tols = get_tols(dtype)
        n = N_STREAMS
        keys = jax.random.split(jax.random.PRNGKey(5), 5)
        f = _normal(keys[0], (s, b, C), dtype)
        bias = _normal(keys[1], (C,), dtype) if with_bias else None
        h_post = _normal(keys[2], (s, b, n), dtype)
        x = _normal(keys[3], (s, b, C, n), dtype)
        h_res = _normal(keys[4], (s, b, n, n), dtype)

        fused = lambda f, bias, h_post, x, h_res: mhc_expand_combine(
            f, bias, h_post, x, h_res, use_tf32=False
        )
        args = (f, bias, h_post, x, h_res)
        assert_allclose(jax.jit(fused)(*args), jax.jit(mhc_expand_combine_ref)(*args), **tols)

        argnums = (0, 1, 2, 3, 4) if with_bias else (0, 2, 3, 4)
        names = [("f", "bias", "h_post", "x", "h_res")[i] for i in argnums]
        grads = jax.jit(jax.grad(lambda *a: _sum_loss(fused(*a)), argnums))(*args)
        ref_grads = jax.jit(jax.grad(lambda *a: _sum_loss(mhc_expand_combine_ref(*a)), argnums))(
            *args
        )
        for name, grad, ref_grad in zip(names, grads, ref_grads):
            assert_allclose(grad, ref_grad, err_msg=name, **tols)


NORM_EPS = 1e-5
PRE_EPS = 1e-6
PARAM_NAMES = ("phi", "alpha", "beta", "norm_weight")
# (b, s, C)
E2E_SHAPES = {
    "L0": [(8, 32, 32), (8, 128, 1024)],
    "L1": [(8, 32, 32), (8, 128, 1024)],
    "L2": [(8, 32, 32), (8, 128, 1024), (2, 2048, 3072)],
}


@pytest.mark.triton
class TestMHC:
    """`mhc` around a sub-layer against `mhc_ref`."""

    @staticmethod
    def _te_mhc(x, params, sinkhorn_iters):
        phi, alpha, beta, norm_weight = params
        return mhc(
            x,
            phi,
            alpha,
            beta,
            _branch_fn,
            norm_weight,
            use_tf32=False,
            norm_eps=NORM_EPS,
            pre_eps=PRE_EPS,
            sinkhorn_iters=sinkhorn_iters,
        )

    @staticmethod
    def _ref_mhc(x, params, sinkhorn_iters):
        return mhc_ref(x, params, _branch_fn, NORM_EPS, PRE_EPS, sinkhorn_iters)

    @staticmethod
    def _make_inputs(b, s, C, dtype, key):
        keys = jax.random.split(key, 3)
        x = _normal(keys[0], (b, s, C, N_STREAMS), dtype)
        params = _make_params(keys[1], C, N_STREAMS, dtype)
        cotangent = _normal(keys[2], (b, s, C, N_STREAMS), jnp.float32)
        return x, params, cotangent

    @pytest_parametrize_wrapper("b,s,C", E2E_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("sinkhorn_iters", [1, 20])
    def test_forward(self, b, s, C, dtype, sinkhorn_iters):
        x, params, _ = self._make_inputs(b, s, C, dtype, jax.random.PRNGKey(6))
        out = jax.jit(self._te_mhc, static_argnums=2)(x, params, sinkhorn_iters)
        ref = jax.jit(self._ref_mhc, static_argnums=2)(x, params, sinkhorn_iters)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert_allclose(out, ref, **get_tols(dtype))

    @pytest_parametrize_wrapper("b,s,C", E2E_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("sinkhorn_iters", [1, 20])
    def test_backward(self, b, s, C, dtype, sinkhorn_iters):
        x, params, cotangent = self._make_inputs(b, s, C, dtype, jax.random.PRNGKey(7))

        def grad_of(fn):
            def loss(x, params):
                out = fn(x, params, sinkhorn_iters)
                return jnp.sum(out.astype(jnp.float32) * cotangent)

            return jax.jit(jax.grad(loss, argnums=(0, 1)))

        dx, dparams = grad_of(self._te_mhc)(x, params)
        ref_dx, ref_dparams = grad_of(self._ref_mhc)(x, params)
        assert dx.dtype == x.dtype
        names = ("x",) + PARAM_NAMES

        if dtype == jnp.float32:
            for name, grad, ref_grad in zip(names, (dx, *dparams), (ref_dx, *ref_dparams)):
                assert grad.shape == ref_grad.shape, name
                assert_allclose(grad, ref_grad, err_msg=name, **get_tols(dtype))
            return

        # In bf16, the branch's backward and the sum of the bf16 grad_x partials amplify
        # rounding differences on a few elements, so element-wise checks are flaky. Instead,
        # TE must be about as close as the bf16 reference to an fp32 ground truth. That error
        # is too noisy on the few-element alpha and beta gradients, which are checked
        # element-wise instead.
        to_fp32 = lambda tree: jax.tree.map(lambda a: a.astype(jnp.float32), tree)
        true_dx, true_dparams = grad_of(self._ref_mhc)(to_fp32(x), to_fp32(params))

        def rel_err(grad, truth):
            grad = grad.astype(jnp.float32)
            return float(jnp.linalg.norm(grad - truth) / (jnp.linalg.norm(truth) + FP32_EPS))

        for name, grad, ref_grad, truth in zip(
            names, (dx, *dparams), (ref_dx, *ref_dparams), (true_dx, *true_dparams)
        ):
            assert grad.shape == truth.shape, name
            if grad.size <= N_MIX:
                assert_allclose(grad, ref_grad, err_msg=name, **get_tols(dtype))
                continue
            te_err, ref_err = rel_err(grad, truth), rel_err(ref_grad, truth)
            assert te_err <= 2 * ref_err + 1e-3, f"{name}: TE error {te_err}, reference {ref_err}"

    @pytest_parametrize_wrapper("b,s,C", [(8, 32, 32)])
    def test_has_aux(self, b, s, C):
        """has_aux=True returns the branch's auxiliary output unchanged."""
        x, (phi, alpha, beta, norm_weight), _ = self._make_inputs(
            b, s, C, jnp.float32, jax.random.PRNGKey(9)
        )

        def branch_with_aux(y):
            out = _branch_fn(y)
            return out, {"mean": jnp.mean(out)}

        def run(branch_fn, has_aux):
            return jax.jit(
                lambda *a: mhc(*a, branch_fn, norm_weight, has_aux=has_aux, use_tf32=False)
            )(x, phi, alpha, beta)

        out, aux = run(branch_with_aux, has_aux=True)
        ref = run(_branch_fn, has_aux=False)

        assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
        assert set(aux) == {"mean"}
