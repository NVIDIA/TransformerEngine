# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""mHC (manifold Hyper-Connection) API for JAX.

This module ties the forward and backward Triton kernels of
``transformer_engine.jax.triton_extensions.mhc`` together with ``jax.custom_vjp``.

The hyper-connection input ``x`` has shape (..., C, n), i.e. the n streams are the innermost
dimension, and the projection and RMSNorm weights are in the matching flattened (C, n) order:

    layer_input, H_post, H_res = mhc_generate_mix_and_aggregate(x, phi, alpha, beta)
    layer_output = layer(layer_input)  # Attention / FFN
    x = mhc_expand_combine(layer_output, None, H_post, x, H_res)

``mhc`` wraps these two calls around a sub-layer function.
"""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from transformer_engine.jax.triton_extensions.mhc import (
    DEFAULT_NORM_EPS,
    mhc_projection_fwd,
    mhc_projection_bwd,
    mhc_scale_fwd,
    mhc_scale_bwd,
    mhc_sinkhorn_fwd,
    mhc_sinkhorn_bwd,
    mhc_aggregate_fwd,
    mhc_aggregate_bwd,
    mhc_expand_combine_fwd,
    mhc_expand_combine_bwd,
)

__all__ = [
    "mhc_projection",
    "mhc_scale",
    "mhc_sinkhorn",
    "mhc_aggregate",
    "mhc_expand_combine",
    "mhc_generate_mix_and_aggregate",
    "mhc",
]


def mhc_projection(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused projection: H = x @ (phi * norm_weight)^T and ms = mean(x^2, dim=-1).

    Parameters
    ----------
    x : jnp.ndarray
        Input of shape (..., K), where K = n * C is the flattened (C, n) hyper-connection input.
    phi : jnp.ndarray
        Projection matrix of shape (N, K), where N = 2n + n*n (= 24 for n = 4).
    norm_weight : Optional[jnp.ndarray]
        RMSNorm weight of shape (K,), absorbed into phi.
    use_tf32 : bool
        Whether to use TF32 for the matmuls.
    use_split_k : bool
        Whether to use split-K (projection) and split-M (phi gradient) reductions with atomic
        adds. Faster for large K but non-deterministic.

    Returns
    -------
    H : jnp.ndarray
        fp32 array of shape (..., 32), where only the first N columns are valid.
    ms : jnp.ndarray
        fp32 mean square of shape (...,).
    """
    return _mhc_projection(x, phi, norm_weight, use_tf32, use_split_k)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _mhc_projection(x, phi, norm_weight, use_tf32, use_split_k):
    """Internal mhc_projection with custom VJP."""
    outputs, _ = _mhc_projection_fwd_rule(x, phi, norm_weight, use_tf32, use_split_k)
    return outputs


def _mhc_projection_fwd_rule(x, phi, norm_weight, use_tf32, use_split_k):
    """Forward pass rule for mhc_projection."""
    h, ms = mhc_projection_fwd(x, phi, norm_weight, use_tf32, use_split_k)
    return (h, ms), (x, phi, norm_weight)


def _mhc_projection_bwd_rule(use_tf32, use_split_k, residuals, g):
    """Backward pass rule for mhc_projection."""
    x, phi, norm_weight = residuals
    grad_h, grad_ms = g
    return mhc_projection_bwd(grad_h, grad_ms, x, phi, norm_weight, use_tf32, use_split_k)


_mhc_projection.defvjp(_mhc_projection_fwd_rule, _mhc_projection_bwd_rule)


def mhc_scale(
    h: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    ms: jnp.ndarray,
    n: int = 4,
    eps: float = DEFAULT_NORM_EPS,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fused scale producing the mixing coefficients:

    H_pre  = sigmoid(H[..., 0:n] * alpha[0] / sqrt(ms + eps) + beta[0:n])
    H_post = 2 * sigmoid(H[..., n:2n] * alpha[1] / sqrt(ms + eps) + beta[n:2n])
    H_res  = H[..., 2n:2n+n*n] * alpha[2] / sqrt(ms + eps) + beta[2n:2n+n*n]

    Parameters
    ----------
    h, ms : jnp.ndarray
        Outputs of `mhc_projection`, of shapes (..., 32) and (...,).
    alpha : jnp.ndarray
        Scaling factors of shape (3,).
    beta : jnp.ndarray
        Bias with 2n + n*n elements, e.g. of shape (1, 2n + n*n).
    n : int
        Number of hyper connections (only n=4 is supported).
    eps : float
        RMSNorm epsilon. Defaults to the fp32 machine epsilon used by the PyTorch API.

    Returns
    -------
    H_pre (..., n), H_post (..., n) and H_res (..., n*n), in h's dtype.
    """
    out = _mhc_scale(h, alpha, beta, ms, n, eps)
    return out[..., :n], out[..., n : 2 * n], out[..., 2 * n : 2 * n + n * n]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _mhc_scale(h, alpha, beta, ms, n, eps):
    """Internal mhc_scale with custom VJP, returning the padded (..., 32) output."""
    out, _ = _mhc_scale_fwd_rule(h, alpha, beta, ms, n, eps)
    return out


def _mhc_scale_fwd_rule(h, alpha, beta, ms, n, eps):
    """Forward pass rule for mhc_scale."""
    out = mhc_scale_fwd(h, alpha, beta, ms, n, eps)
    return out, (h, alpha, beta, ms, out)


def _mhc_scale_bwd_rule(n, eps, residuals, grad_out):
    """Backward pass rule for mhc_scale."""
    h, alpha, beta, ms, out = residuals
    grad_h, grad_alpha, grad_beta, grad_ms = mhc_scale_bwd(grad_out, h, alpha, ms, out, n, eps)
    return (
        grad_h.astype(h.dtype),
        grad_alpha,
        grad_beta.reshape(beta.shape).astype(beta.dtype),
        grad_ms.astype(ms.dtype),
    )


_mhc_scale.defvjp(_mhc_scale_fwd_rule, _mhc_scale_bwd_rule)


def mhc_sinkhorn(
    h_res: jnp.ndarray,
    n: int = 4,
    recompute_hist: bool = True,
    iters: int = 20,
) -> jnp.ndarray:
    """
    Log-space Sinkhorn normalization of H_res into a doubly stochastic matrix.

    Parameters
    ----------
    h_res : jnp.ndarray
        Input of shape (..., n, n).
    n : int
        Number of hyper connections (only n=4 is supported).
    recompute_hist : bool
        Whether to recompute the f/g history in the backward pass instead of saving
        2 * (iters + 1) * n floats per token.
    iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    Doubly stochastic matrix of shape (..., n, n) in h_res's dtype.
    """
    return _mhc_sinkhorn(h_res, n, recompute_hist, iters)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _mhc_sinkhorn(h_res, n, recompute_hist, iters):
    """Internal mhc_sinkhorn with custom VJP."""
    out, _ = _mhc_sinkhorn_fwd_rule(h_res, n, recompute_hist, iters)
    return out


def _mhc_sinkhorn_fwd_rule(h_res, n, recompute_hist, iters):
    """Forward pass rule for mhc_sinkhorn."""
    out, hist = mhc_sinkhorn_fwd(h_res, n, recompute_hist, iters)
    return out, (h_res, out, hist)


def _mhc_sinkhorn_bwd_rule(n, recompute_hist, iters, residuals, grad_out):
    """Backward pass rule for mhc_sinkhorn."""
    del recompute_hist
    h_res, out, hist = residuals
    return (mhc_sinkhorn_bwd(grad_out, h_res, out, hist, n, iters),)


_mhc_sinkhorn.defvjp(_mhc_sinkhorn_fwd_rule, _mhc_sinkhorn_bwd_rule)


def mhc_aggregate(
    x: jnp.ndarray,
    h_pre: jnp.ndarray,
    use_tf32: bool = True,
) -> jnp.ndarray:
    """
    Aggregate the n streams: out = x @ H_pre, (..., C, n) @ (..., n, 1) -> (..., C).

    Returns an array of shape (..., C) in x's dtype.
    """
    return _mhc_aggregate(x, h_pre, use_tf32)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _mhc_aggregate(x, h_pre, use_tf32):
    """Internal mhc_aggregate with custom VJP."""
    out, _ = _mhc_aggregate_fwd_rule(x, h_pre, use_tf32)
    return out


def _mhc_aggregate_fwd_rule(x, h_pre, use_tf32):
    """Forward pass rule for mhc_aggregate."""
    del use_tf32
    return mhc_aggregate_fwd(x, h_pre), (x, h_pre)


def _mhc_aggregate_bwd_rule(use_tf32, residuals, grad_out):
    """Backward pass rule for mhc_aggregate."""
    x, h_pre = residuals
    return mhc_aggregate_bwd(grad_out, x, h_pre, use_tf32)


_mhc_aggregate.defvjp(_mhc_aggregate_fwd_rule, _mhc_aggregate_bwd_rule)


def mhc_expand_combine(
    f: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    h_post: jnp.ndarray,
    x: jnp.ndarray,
    h_res: jnp.ndarray,
    use_tf32: bool = True,
) -> jnp.ndarray:
    """
    Expand the sub-layer output back to n streams and mix the residual streams:

    out = (f [+ bias]) @ H_post + x @ H_res

    (..., C, n, n) = (..., C, 1) @ (..., 1, n) + (..., C, n) @ (..., n, n)

    Parameters
    ----------
    f : jnp.ndarray
        Sub-layer (attention / FFN) output of shape (..., C).
    bias : Optional[jnp.ndarray]
        Optional bias of shape (C,) of the sub-layer's last linear layer, fused here.
    h_post : jnp.ndarray
        H_post of shape (..., n).
    x : jnp.ndarray
        Hyper-connection input of shape (..., C, n).
    h_res : jnp.ndarray
        Sinkhorn-normalized H_res of shape (..., n, n).
    use_tf32 : bool
        Whether to use TF32 for the backward matmuls.

    Returns
    -------
    Array of shape (..., C, n) in x's dtype.
    """
    return _mhc_expand_combine(f, bias, h_post, x, h_res, use_tf32)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _mhc_expand_combine(f, bias, h_post, x, h_res, use_tf32):
    """Internal mhc_expand_combine with custom VJP."""
    out, _ = _mhc_expand_combine_fwd_rule(f, bias, h_post, x, h_res, use_tf32)
    return out


def _mhc_expand_combine_fwd_rule(f, bias, h_post, x, h_res, use_tf32):
    """Forward pass rule for mhc_expand_combine."""
    del use_tf32
    out = mhc_expand_combine_fwd(f, bias, h_post, x, h_res)
    return out, (f, bias, h_post, x, h_res)


def _mhc_expand_combine_bwd_rule(use_tf32, residuals, grad_out):
    """Backward pass rule for mhc_expand_combine."""
    f, bias, h_post, x, h_res = residuals
    return mhc_expand_combine_bwd(grad_out, f, bias, h_post, x, h_res, use_tf32)


_mhc_expand_combine.defvjp(_mhc_expand_combine_fwd_rule, _mhc_expand_combine_bwd_rule)


def mhc_generate_mix_and_aggregate(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
    norm_eps: float = DEFAULT_NORM_EPS,
    pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    recompute_sinkhorn_hist: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate H_pre, H_post and H_res and aggregate the n streams of x with H_pre.

    This chains projection, scale, Sinkhorn and aggregate. The defaults match
    `mhc_generate_mix_and_aggregate` of the PyTorch API.

    Parameters
    ----------
    x : jnp.ndarray
        Hyper-connection input of shape (..., C, n).
    phi : jnp.ndarray
        Projection matrix of shape (N, C*n) with N = 2n + n*n, with columns in the flattened
        (C, n) order of x and rows for [H_pre (n), H_post (n), H_res (n*n)]. H_res is
        row-major: row 2n + k*n + m gives H_res[k, m], the weight of input stream k in
        output stream m.
    alpha : jnp.ndarray
        Scaling factors of shape (3,) for H_pre, H_post and H_res.
    beta : jnp.ndarray
        Bias with N elements, in the row order of phi.
    norm_weight : Optional[jnp.ndarray]
        RMSNorm weight of shape (n*C,), in the flattened (C, n) order of x.
    use_tf32, use_split_k :
        See `mhc_projection`.
    norm_eps : float
        RMSNorm epsilon.
    pre_eps : float
        Constant added to H_pre after its sigmoid.
    sinkhorn_iters, recompute_sinkhorn_hist :
        See `mhc_sinkhorn`.

    Returns
    -------
    layer_input : jnp.ndarray
        Aggregated input of shape (..., C) for the attention / FFN sub-layer, in x's dtype.
    H_post : jnp.ndarray
        fp32 array of shape (..., n) for `mhc_expand_combine`.
    H_res : jnp.ndarray
        fp32 array of shape (..., n, n) for `mhc_expand_combine`.
    """
    tokens = x.shape[:-2]
    C, n = x.shape[-2:]
    assert n == 4, "Only n=4 is supported in this implementation"
    h, ms = mhc_projection(x.reshape(tokens + (C * n,)), phi, norm_weight, use_tf32, use_split_k)
    h_pre, h_post, h_res = mhc_scale(h, alpha, beta, ms, n, norm_eps)
    if pre_eps:
        h_pre = h_pre + pre_eps
    h_res = mhc_sinkhorn(h_res.reshape(tokens + (n, n)), n, recompute_sinkhorn_hist, sinkhorn_iters)
    layer_input = mhc_aggregate(x, h_pre, use_tf32)
    return layer_input, h_post, h_res


def mhc(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    branch_fn: Callable[[jnp.ndarray], Any],
    norm_weight: Optional[jnp.ndarray] = None,
    *,
    has_aux: bool = False,
    use_tf32: bool = True,
    use_split_k: bool = False,
    norm_eps: float = DEFAULT_NORM_EPS,
    pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    recompute_sinkhorn_hist: bool = True,
) -> Any:
    """
    Manifold-constrained hyper connection around `branch_fn`:

        layer_input, H_post, H_res = mhc_generate_mix_and_aggregate(x, phi, alpha, beta, ...)
        out = mhc_expand_combine(branch_fn(layer_input), None, H_post, x, H_res)

    i.e. layer_input[c] = sum_k H_pre[k] * x[c, k] and
    out[c, m] = H_post[m] * branch_fn(layer_input)[c] + sum_k H_res[k, m] * x[c, k].
    Only n = 4 streams are supported.

    Parameters
    ----------
    x : jnp.ndarray
        Hyper-connection input of shape (..., C, n).
    phi, alpha, beta, norm_weight :
        See `mhc_generate_mix_and_aggregate`.
    branch_fn : Callable
        The wrapped sub-layer (e.g. pre-norm + attention / FFN), called as
        `branch_fn(layer_input)` with layer_input of shape (..., C).
    has_aux : bool
        Whether `branch_fn` returns `(layer_out, aux)`, in which case `(out, aux)` is returned.
    use_tf32, use_split_k, norm_eps, pre_eps, sinkhorn_iters, recompute_sinkhorn_hist :
        See `mhc_generate_mix_and_aggregate`.

    Returns
    -------
    Output of shape (..., C, n) in x's dtype, or `(output, aux)` if `has_aux`.
    """
    layer_input, h_post, h_res = mhc_generate_mix_and_aggregate(
        x,
        phi,
        alpha,
        beta,
        norm_weight,
        use_tf32=use_tf32,
        use_split_k=use_split_k,
        norm_eps=norm_eps,
        pre_eps=pre_eps,
        sinkhorn_iters=sinkhorn_iters,
        recompute_sinkhorn_hist=recompute_sinkhorn_hist,
    )
    if has_aux:
        layer_out, aux = branch_fn(layer_input)
    else:
        layer_out, aux = branch_fn(layer_input), None
    out = mhc_expand_combine(layer_out, None, h_post, x, h_res, use_tf32)
    return (out, aux) if has_aux else out
