# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for mHC (manifold Hyper-Connection) Triton kernels."""

import os
import torch
import triton

from transformer_engine.common.triton.mhc import (
    _mhc_scale_fwd_fused,
    _mhc_scale_bwd_fused,
    _mhc_expand_combine_with_bias_fwd,
    _mhc_expand_combine_with_bias_bwd,
    _mhc_expand_combine_fwd,
    _mhc_expand_combine_bwd,
    _mhc_aggregate_fwd,
    _mhc_aggregate_bwd,
    _mhc_projection_fwd_fused,
    _mhc_projection_bwd_fused,
    _mhc_sinkhorn_fwd_fused,
    _mhc_sinkhorn_fwd_fused_recompute,
    _mhc_sinkhorn_bwd_fused,
    _mhc_sinkhorn_bwd_fused_recompute,
)
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


def check_deterministic(operator: str):
    """
    Checks if the non-deterministic algorithm is allowed for the given operator. If not, raises an assertion error with instructions on how to allow it.
    Since atomic add is used in this mHC implementation, it breaks the determinism guarantee due to non-associativity of floating point addition.
    """
    allow_nondeterministic = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "1"
    assert allow_nondeterministic, (
        f"[{operator}]: This operation uses atomic add which violates determinism. Set"
        " NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 to allow this non-deterministic behavior."
    )


def mhc_fused_sinkhorn(
    H_res: torch.Tensor, n: int = 4, recompute_hist: bool = True, iters: int = 20
):
    """
    Sinkhorn operation to compute the final H_res matrix (see eq. 19, section 4.3.1 of the DeepSeek mHC paper):

    The Sinkhorn operation conducts an iterative normalization process that alternately rescales rows and columns to sum to 1.
    This kernel performs this operation in the log space for numerical stability.

    Parameters
    ----------
    H_res : torch.Tensor
        input H_res matrix of shape (s, b, n, n) that needs to be normalized into a doubly stochastic matrix.
    n : int
        number of hyper connections, where only n=4 is supported in the current implementation
    recompute_hist : bool
        whether to recompute the intermediate history in the backward pass to save memory
    iters : int
        number of Sinkhorn iterations, according to the DeepSeek paper 20 is enough for convergence

    Returns
    -------
    out : torch.Tensor
        out of shape (s, b, n, n), which is the final H_res after Sinkhorn normalization
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    out = mHCSinkhornOp.apply(H_res, n, recompute_hist, iters)
    return out


def mhc_fused_scale(
    H: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, ms: torch.Tensor, n: int
):
    """
    Fused scale operation to compute the scaled H matrices (see eq. 16-18, section 4.3.1 of the DeepSeek mHC paper):

    H_pre = H[:, 0:n] * alpha[0] / sqrt(ms) + beta[0:n]
    H_post = H[:, n:2n] * alpha[1] / sqrt(ms) + beta[n:2n]
    H_res = H[:, 2n:2n+n*n] * alpha[2] / sqrt(ms) + beta[2n:2n+n*n]

    H_pre = sigmoid(H_pre)
    H_post = 2*sigmoid(H_post)

    Parameters
    ----------
    H : torch.Tensor
        input H matrix of shape (M, 32), where M=s*b, and only the first N elements in the last dimension are valid
    alpha : torch.Tensor
        scaling factor for H, of shape (3,), where
        alpha[0] is applied to H[:, 0:n] for H_pre
        alpha[1] is applied to H[:, n:2n] for H_post
        alpha[2] is applied to H[:, 2n:2n+n*n] for H_res
    beta : torch.Tensor
        bias term for H, of shape (1, 2*n+n*n), where
        beta[0, 0:n] is applied to H[:, 0:n] for H_pre
        beta[0, n:2n] is applied to H[:, n:2n] for H_post
        beta[0, 2n:2n+n*n] is applied to H[:, 2n:2n+n*n] for H_res
    ms : torch.Tensor
        mean square for each row of H from the projection kernel, of shape (M,), used for RMSNorm scaling
    n : int
        number of hyper connections, where only n=4 is supported in the current implementation

    Returns
    -------
    h_pre : torch.Tensor
        Scaled H_pre of shape (M, n), which aggregates (s, b, C, n) input of a Hyper Connection block into (s, b, n) as the input of attention / MLP
    h_post : torch.Tensor
        Scaled H_post of shape (M, n), which expands the output of attention / MLP of shape (s, b, n) back to (s, b, C, n) for the residual connection
    h_res : torch.Tensor
        Scaled H_res of shape (M, n*n), which mixes the n streams of the (s, b, C, n) input of a Hyper Connection block

    """
    assert n == 4, "Only n=4 is supported in this implementation"
    check_deterministic("mhc_fused_scale")
    out = mHCScaleFusedOp.apply(H, alpha, beta, ms, n)
    h_pre = out[..., :n]
    h_post = out[..., n : 2 * n]
    h_res = out[..., 2 * n : n * n + 2 * n]
    return h_pre, h_post, h_res


def mhc_fused_aggregate(x: torch.Tensor, H_pre: torch.Tensor, n: int, use_tf32: bool = True):
    """
    Aggregate operation to merge n activation streams into one (see section 4.3.1 of the DeepSeek mHC paper):
    out = x @ H_pre: (s, b, C, n) @ (s, b, n, 1) -> (s, b, C, 1) -> (s, b, C) after squeezing the last dimension

    Parameters
    ----------
    x : torch.Tensor
        input activation tensor of shape (s, b, C, n),
        where s is the sequence length, b is the batch size, C is the hidden dimension per hyper connection, and n is the number of hyper connections. Note that C is equal to the original hidden dimension divided by n.
    H_pre: torch.Tensor
        input H_pre matrix of shape (s, b, n)
    n: int
        number of hyper connections, where only n=4 is supported in the current implementation
    use_tf32: bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    Returns
    -------
    out: torch.Tensor
         output activation tensor of shape (s, b, C), which is the aggregated output after merging n hyper connections
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    check_deterministic("mhc_fused_aggregate")
    out = mHCAggregateOp.apply(x, H_pre, n, use_tf32)
    return out


def mhc_fused_expand_combine(
    f: torch.Tensor,
    bias: torch.Tensor,
    H_post: torch.Tensor,
    x: torch.Tensor,
    H_res: torch.Tensor,
    n: int,
    use_tf32: bool = True,
):
    """
    Expand and combine operation for merging n hyper connections (see section 4.3.1 of the DeepSeek mHC paper):

    out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)

    Parameters
    ----------
    f : torch.Tensor
        input activation tensor of shape (s, b, C), which is the output from the attention / FFN sub-layer in a transformer block
    bias : torch.Tensor or None
        optional bias tensor of shape (C,) from the last linear layer, where f + bias is fused in this kernel for better performance
    H_post : torch.Tensor
        input H_post matrix of shape (s, b, n)
    x : torch.Tensor
        input activation tensor of shape (s, b, C, n), which is the hyper connection input before the aggregation operation
    H_res : torch.Tensor
        input H_res matrix of shape (s, b, n, n)
    n : int
        number of hyper connections
    use_tf32 : bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    Returns
    -------
    out : torch.Tensor
        out of shape (s, b, C, n), which is the expanded and combined output after merging n hyper connections
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    check_deterministic("mhc_fused_expand_combine")
    out = mHCExpandCombineOp.apply(
        f,
        bias,
        H_post,
        x,
        H_res,
        n,
        use_tf32,
    )
    return out


def mhc_fused_projection(x: torch.Tensor, phi: torch.Tensor, use_tf32: bool = True):
    """
    Fused projection operation to compute H matrices and mean square for RMSNorm (see eq. 14-15, section 4.3.1 of the DeepSeek mHC paper):

    H = x @ phi^T: (M, K) @ (K, N) -> (M, N), which is padded to (M, 32) for better memory access pattern in the next kernels.
    ms = mean(x^2, dim=-1): (M,)

    Note: the current implementation only supports n=4

    Parameters
    ----------
    x : torch.Tensor
        input tensor of shape (M, K), where M=s*b is the batch size and K=nC is the hidden dimension after expansion.
    phi : torch.Tensor
        projection matrix of shape (N, K), where N=2n+n*n (=24 for n=4)
    use_tf32 : bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail.

    Returns
    -------
    H : torch.Tensor
        Projected matrix of shape (M, 32), where only the first N elements in the last dimension are valid.
    ms : torch.Tensor
        Mean square of shape (M,), which is used for RMSNorm in the next kernel.
    """
    assert (
        phi.shape[0] == 24
    ), "Currently only n=4 is supported, which means phi should have 24 in its first dimension"
    check_deterministic("mhc_fused_projection")
    H, ms = mHCProjectionOp.apply(x, phi, use_tf32)
    return H, ms


class mHCProjectionOp(torch.autograd.Function):
    """
    PyTorch operator for the fused projection operation in mHC, whose wrapper API is mhc_fused_projection.
    """

    @staticmethod
    def forward(ctx, x, phi, use_tf32=True):
        """
        The forward pass of the fused projection operation. Computes H = x @ phi^T and the mean
        square ms = mean(x^2, dim=-1) for RMSNorm in a single fused kernel.

        Parameters:
        ctx : The context object.
        x (tensor): The input tensor of shape (M, K), where M=s*b is the flattened batch dimension and K=nC is the hidden dimension after expansion.
        phi (tensor): The projection matrix of shape (N, K), where N=2n+n*n (=24 for n=4).
        use_tf32 (bool): Whether to use TF32 precision for matmul operations. If False, uses IEEE for better precision.

        Returns:
        tuple: A tuple of (H, ms) where H is the projected matrix of shape (M, 32) padded for memory alignment (only the first N elements are valid), and ms is the mean square of shape (M,) in FP32.
        """
        x = x.contiguous()
        phi = phi.contiguous()

        ctx.use_tf32 = use_tf32
        ctx.dtype = x.dtype

        M, K = x.shape
        device = x.device

        N = phi.shape[0]

        # Pad H to (s, b, 32) for better memory access pattern in the kernel, but only the first N elements in the last dimension are valid
        H = torch.zeros((M, 32), device=device, dtype=torch.float32)
        ms = torch.zeros(
            (M,), device=device, dtype=torch.float32
        )  # Mean square for x, used to compute RMSNorm in the next kernel

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

        _mhc_projection_fwd_fused[grid](
            x_ptr=x,  # (M, K)
            phi_ptr=phi,  # (N, K)
            h_ptr=H,  # (M, 32)
            ms_ptr=ms,  # (M,)
            M=M,
            N=N,
            K=K,
            stride_xm=K,
            stride_xk=1,
            stride_phin=K,
            stride_phik=1,
            stride_hm=32,
            stride_hn=1,
            stride_ms=1,
            BLOCK_SIZE_N=32,
            precision="tf32" if use_tf32 else "ieee",
        )

        ctx.save_for_backward(x, phi, ms)
        ctx.phi_dtype = phi.dtype

        return H.to(ctx.dtype), ms  # Keep ms in fp32

    @staticmethod
    def backward(ctx, grad_H, grad_ms):
        """
        The backward pass of the fused projection operation. Computes gradients for x and phi.

        grad_phi = grad_H^T @ x, truncated to the first N rows.
        grad_x = grad_H @ phi + 2 * x * grad_ms / K, where the second term is the gradient contribution from
        the mean square computation fused in the forward pass.

        Parameters:
        ctx : The context object with saved tensors.
        grad_H (tensor): The gradient of the loss with respect to H, of shape (M, 32).
        grad_ms (tensor): The gradient of the loss with respect to the mean square, of shape (M,).

        Returns:
        tuple: A tuple with the gradients (grad_x, grad_phi, None).
        """
        x, phi, ms = ctx.saved_tensors
        M, K = x.shape
        device = x.device

        N = phi.shape[0]

        grad_H = grad_H.contiguous().view(M, -1)
        grad_ms = grad_ms.contiguous().view(
            M,
        )
        ms = ms.contiguous().view(
            M,
        )

        grad_x = torch.empty((M, K), device=device, dtype=x.dtype)

        grad_x = torch.empty((M, K), device=device, dtype=x.dtype)
        grad_phi = general_gemm(x, grad_H, out_dtype=torch.float32, layout="NT")[0][:N, :].to(
            phi.dtype
        )  # (2n + n^2, M) @ (M, nC) = (2n + n^2, nC); grad_H's last dim is padded to 32

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

        _mhc_projection_bwd_fused[grid](
            x_ptr=x,
            grad_x_ptr=grad_x,  # (M, K)
            phi_ptr=phi,  # (N, K)
            grad_h_ptr=grad_H,  # (M, 32)
            grad_ms_ptr=grad_ms,  # (M,)
            M=M,
            N=N,
            K=K,
            stride_xm=K,
            stride_xk=1,
            stride_grad_xm=K,
            stride_grad_xk=1,
            stride_phin=K,
            stride_phik=1,
            stride_grad_phin=K,
            stride_grad_phik=1,
            stride_grad_hm=32,
            stride_grad_hn=1,
            stride_grad_ms=1,
            BLOCK_SIZE_N=32,
            precision="tf32" if ctx.use_tf32 else "ieee",
        )

        return grad_x.to(ctx.dtype), grad_phi.to(ctx.dtype), None


class mHCScaleFusedOp(torch.autograd.Function):
    """
    PyTorch operator for the fused scale operation in mHC, whose wrapper API is mhc_fused_scale.
    """

    @staticmethod
    def forward(ctx, H, alpha, beta, ms, n):
        """
        The forward pass of the fused scale operation. Applies RMSNorm scaling, bias, and activation
        functions to produce H_pre, H_post, and H_res:

        H_pre  = sigmoid(H[:, 0:n] * alpha[0] / sqrt(ms) + beta[0:n])
        H_post = 2 * sigmoid(H[:, n:2n] * alpha[1] / sqrt(ms) + beta[n:2n])
        H_res  = H[:, 2n:2n+n*n] * alpha[2] / sqrt(ms) + beta[2n:2n+n*n]

        Parameters:
        ctx : The context object.
        H (tensor): The input H matrix of shape (M, 32), where only the first N=2n+n*n elements are valid.
        alpha (tensor): The scaling factors of shape (3,), one for each of H_pre, H_post, H_res.
        beta (tensor): The bias terms of shape (1, 2n+n*n).
        ms (tensor): The mean square from the projection kernel, of shape (M,), used for RMSNorm scaling.
        n (int): The number of hyper connections (only n=4 is supported).

        Returns:
        tensor: The scaled output of shape (M, 32), where only the first N elements are valid.
        """

        ctx.dtype = H.dtype
        H = H.to(torch.float32)
        alpha = alpha.to(torch.float32)
        beta = beta.to(torch.float32)
        ms = ms.to(torch.float32)

        M, _ = H.shape

        H = H.contiguous()
        beta = beta.contiguous()
        ms = ms.contiguous()

        out = torch.empty(
            (M, 32), device=H.device, dtype=H.dtype
        )  # Pad the output to 32 in the last dimension

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_scale_fwd_fused[grid](
            h_ptr=H,  # (M, N), which is padded to (M, 32)
            b_ptr=beta,  # (N,)
            a_ptr=alpha,  # (N,)
            ms_ptr=ms,  # (M,)
            out_ptr=out,  # (M, N), which is padded to (M, 32)
            M=M,
            n=n,
            stride_hm=32,
            stride_hn=1,
            stride_a=1,
            stride_b=1,
            stride_ms=1,
            stride_out_m=32,
            stride_out_n=1,  # strides for out, which is padded to 32 in the last dimension
            BLOCK_SIZE_N=32,
            eps=torch.finfo(ms.dtype).eps,
        )

        ctx.save_for_backward(H, alpha, ms, out)
        ctx.n = n

        return out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):
        """
        The backward pass of the fused scale operation. Computes gradients for H, alpha, beta, and ms
        by backpropagating through the sigmoid activations, RMSNorm scaling, and bias additions.

        Parameters:
        ctx : The context object with saved tensors.
        grad_out (tensor): The gradient of the loss with respect to the output, of shape (M, 32).

        Returns:
        tuple: A tuple with the gradients (grad_H, grad_alpha, grad_beta, grad_ms, None).
        """
        H, alpha, ms, out = ctx.saved_tensors
        n = ctx.n

        grad_out = grad_out.contiguous()
        grad_out = grad_out.to(torch.float32)

        M, _ = grad_out.shape
        N = 2 * n + n * n

        grad_h = torch.zeros(
            (M, 32), device=grad_out.device, dtype=grad_out.dtype
        )  # Pad the grad_h to 32 in the last dimension
        grad_alpha = torch.zeros((3,), device=grad_out.device, dtype=grad_out.dtype)
        grad_beta_padded = torch.zeros((1, 32), device=grad_out.device, dtype=grad_out.dtype)
        grad_beta = grad_beta_padded[
            :, :N
        ]  # Use only the first N elements for grad_beta, the rest are just padding
        grad_ms = torch.zeros((M,), device=grad_out.device, dtype=grad_out.dtype)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_scale_bwd_fused[grid](
            grad_out_ptr=grad_out,
            out_ptr=out,
            grad_h_ptr=grad_h,
            h_ptr=H,
            grad_a_ptr=grad_alpha,
            a_ptr=alpha,
            grad_b_ptr=grad_beta,
            grad_ms_ptr=grad_ms,
            ms_ptr=ms,
            M=M,
            n=n,
            stride_grad_out_m=32,
            stride_grad_out_n=1,
            stride_out_m=32,
            stride_out_n=1,
            stride_grad_hm=32,
            stride_grad_hn=1,
            stride_hm=32,
            stride_hn=1,
            stride_grad_a=1,
            stride_a=1,
            stride_grad_b=1,
            stride_grad_ms=1,
            stride_ms=1,
            BLOCK_SIZE_N=32,
            eps=torch.finfo(ms.dtype).eps,
        )

        return (
            grad_h.to(ctx.dtype),
            grad_alpha.to(ctx.dtype),
            grad_beta.to(ctx.dtype),
            grad_ms.to(ctx.dtype),
            None,
        )


class mHCSinkhornOp(torch.autograd.Function):
    """
    PyTorch operator for the Sinkhorn operation in mHC, whose wrapper API is mhc_fused_sinkhorn.
    """

    @staticmethod
    def forward(ctx, H_res, n=4, recompute_hist=True, iters=20):
        """
        The forward pass of the Sinkhorn operation. Performs iterative row-column normalization
        in log space to convert H_res into a doubly stochastic matrix. Each iteration alternately
        rescales rows and columns to sum to 1:

        f = log_mu - logsumexp(H_res + g, dim=cols)
        g = log_nu - logsumexp(H_res + f, dim=rows)
        output = exp(f + H_res + g)

        Parameters:
        ctx : The context object.
        H_res (tensor): The input H_res matrix of shape (s, b, n, n).
        n (int): The number of hyper connections (only n=4 is supported).
        recompute_hist (bool): Whether to recompute the intermediate f/g history in the backward pass to save memory. If False, stores history buffers of shape (iters+1, s, b, n).
        iters (int): The number of Sinkhorn iterations (20 is enough for convergence per the DeepSeek paper).

        Returns:
        tensor: The doubly stochastic matrix of shape (s, b, n, n).
        """

        s, b, _, _ = H_res.shape

        ctx.dtype = H_res.dtype
        H_res = H_res.to(torch.float32)

        H_res = H_res.contiguous().view(s * b, n * n)

        hist_f, hist_g = None, None
        if not recompute_hist:
            # History buffers: (iters+1, s, b, n)
            hist_f = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
        H_res_out = torch.empty_like(H_res)  # (s*b, n*n)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(s * b * n * n, META["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_fwd_fused_recompute[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=s * b,
                n=n,
                iters=iters,
            )
        else:
            _mhc_sinkhorn_fwd_fused[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=s * b,
                n=n,
                iters=iters,
            )

        if recompute_hist:
            ctx.save_for_backward(H_res, H_res_out)
        else:
            ctx.save_for_backward(H_res, H_res_out, hist_f, hist_g)
        ctx.recompute_hist = recompute_hist
        ctx.iters = iters
        ctx.n = n

        H_res_out = H_res_out.view(s, b, n, n)
        return H_res_out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):
        """
        The backward pass of the Sinkhorn operation. Backpropagates through the iterative
        normalization by reversing through the f/g update steps. If recompute_hist is True,
        the forward pass history is recomputed to save memory.

        Parameters:
        ctx : The context object with saved tensors.
        grad_out (tensor): The gradient of the loss with respect to the output, of shape (s, b, n, n).

        Returns:
        tuple: A tuple with the gradients (grad_H_res, None, None, None).
        """

        s, b, n, _ = grad_out.shape
        M = s * b

        hist_f, hist_g = None, None
        recompute_hist = ctx.recompute_hist
        iters = ctx.iters
        if recompute_hist:
            H_res, H_res_out = ctx.saved_tensors
            hist_f = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
        else:
            H_res, H_res_out, hist_f, hist_g = ctx.saved_tensors

        n = ctx.n

        grad_res_out = grad_out.clone().contiguous().view(M, n * n)

        grad_res = torch.empty_like(H_res)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(M * n * n, META["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_bwd_fused_recompute[grid](
                grad_out_ptr=grad_res_out,
                output_ptr=H_res_out,
                grad_x_ptr=grad_res,
                x_ptr=H_res,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_grad_out_m=n * n,
                stride_grad_out_n=1,
                stride_out_m=n * n,
                stride_out_n=1,
                stride_grad_xm=n * n,
                stride_grad_xn=1,
                stride_xm=n * n,
                stride_xn=1,
                M=M,
                n=n,
                iters=iters,
            )
        else:
            _mhc_sinkhorn_bwd_fused[grid](
                grad_out_ptr=grad_res_out,
                output_ptr=H_res_out,
                grad_x_ptr=grad_res,
                x_ptr=H_res,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_grad_out_m=n * n,
                stride_grad_out_n=1,
                stride_out_m=n * n,
                stride_out_n=1,
                stride_grad_xm=n * n,
                stride_grad_xn=1,
                stride_xm=n * n,
                stride_xn=1,
                M=M,
                n=n,
                iters=iters,
            )

        grad_res = grad_res.view(s, b, n, n)

        return grad_res.to(ctx.dtype), None, None, None


class mHCAggregateOp(torch.autograd.Function):
    """
    PyTorch operator for the aggregate operation in mHC, whose wrapper API is mhc_fused_aggregate.
    """

    @staticmethod
    def forward(ctx, x, H_pre, n, use_tf32=True):
        """
        The forward pass of the aggregate operation. Merges n activation streams into one by
        computing a weighted sum using H_pre:

        out = x @ H_pre: (s, b, C, n) @ (s, b, n, 1) -> (s, b, C)

        Parameters:
        ctx : The context object.
        x (tensor): The input activation tensor of shape (s, b, C, n).
        H_pre (tensor): The pre-connection matrix of shape (s, b, n), used as weights for aggregation.
        n (int): The number of hyper connections (only n=4 is supported).
        use_tf32 (bool): Whether to use TF32 precision for matmul operations.

        Returns:
        tensor: The aggregated output of shape (s, b, C).
        """

        x = x.contiguous()
        H_pre = H_pre.contiguous()

        s, b, C, n = x.shape
        nC = n * C
        M = s * b

        out = torch.empty((s, b, C), device=x.device, dtype=x.dtype)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        _mhc_aggregate_fwd[grid](
            x_ptr=x,
            H_pre_ptr=H_pre,
            output_ptr=out,
            M=M,
            C=C,
            n=n,
            stride_xm=nC,
            stride_xCn=1,
            stride_output_m=C,
            stride_output_c=1,
        )

        ctx.save_for_backward(x, H_pre)
        ctx.n = n
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the aggregate operation. Computes gradients for x and H_pre:

        grad_x[:, :, :, i] = grad_output * H_pre[:, :, i] for each stream i
        grad_H_pre[:, :, i] = sum_C(grad_output * x[:, :, :, i]) for each stream i

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The gradient of the loss with respect to the output, of shape (s, b, C).

        Returns:
        tuple: A tuple with the gradients (grad_x, grad_H_pre, None, None).
        """
        grad_output = grad_output.contiguous()

        x, H_pre = ctx.saved_tensors
        n = ctx.n

        s, b, C, n = x.shape
        nC = n * C
        assert n == 4, "Only n=4 is supported in this implementation"
        M = s * b

        grad_x = torch.empty_like(x)
        grad_H_pre = torch.zeros(
            (s, b, n), dtype=torch.float32, device=H_pre.device
        )  # We need to use atomic_add for this so we need higher precision

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        _mhc_aggregate_bwd[grid](
            grad_output_ptr=grad_output,
            H_pre_ptr=H_pre,
            grad_H_pre_ptr=grad_H_pre,
            x_ptr=x,
            grad_x_ptr=grad_x,
            M=M,
            C=C,
            n=n,
            stride_grad_output_m=C,
            stride_grad_output_c=1,
            stride_xm=nC,
            stride_xCn=1,
            stride_grad_xm=nC,
            stride_grad_xCn=1,
            precision="tf32" if ctx.use_tf32 else "ieee",
        )

        grad_H_pre = grad_H_pre.to(H_pre.dtype)  # Cast back to the original dtype of H_pre

        return grad_x, grad_H_pre, None, None


class mHCExpandCombineOp(torch.autograd.Function):
    """
    PyTorch operator for the expand and combine operation in mHC, whose wrapper API is mhc_fused_expand_combine.
    """

    @staticmethod
    def forward(ctx, f, bias, H_post, x, H_res, n, use_tf32=True):
        """
        The forward pass of the expand and combine operation. Expands the sub-layer output f back
        to n streams using H_post, and combines with the residual connections using H_res:

        out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)

        Parameters:
        ctx : The context object.
        f (tensor): The sub-layer output tensor of shape (s, b, C).
        bias (tensor or None): Optional bias tensor of shape (C,) from the last linear layer, fused in this kernel.
        H_post (tensor): The post-connection matrix of shape (s, b, n).
        x (tensor): The hyper connection input tensor of shape (s, b, C, n) before aggregation.
        H_res (tensor): The residual connection matrix of shape (s, b, n, n).
        n (int): The number of hyper connections (only n=4 is supported).
        use_tf32 (bool): Whether to use TF32 precision for matmul operations.

        Returns:
        tensor: The expanded and combined output of shape (s, b, C, n).
        """

        x = x.contiguous()
        f = f.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        H_post = H_post.contiguous()
        H_res = H_res.contiguous()

        s, b, C, n = x.shape
        Cn = C * n
        M = s * b

        out = torch.empty((s, b, C, n), device=x.device, dtype=x.dtype)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        if bias is None:
            _mhc_expand_combine_fwd[grid](
                f_ptr=f,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )
        else:
            _mhc_expand_combine_with_bias_fwd[grid](
                f_ptr=f,
                bias_ptr=bias,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_bias=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )

        ctx.n = n
        ctx.have_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(f, bias, H_post, x, H_res)
        else:
            ctx.save_for_backward(f, H_post, x, H_res)
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the expand and combine operation. Computes gradients for f, bias,
        H_post, x, and H_res by backpropagating through the outer product and matrix multiply:

        grad_f = sum_n(grad_output * H_post) [+ reduce grad_bias over (s, b)]
        grad_H_post[:, :, i] = sum_C(grad_output[:, :, :, i] * (f [+ bias]))
        grad_x = grad_output @ H_res^T
        grad_H_res[:, :, i, j] = sum_C(grad_output[:, :, :, j] * x[:, :, :, i])

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The gradient of the loss with respect to the output, of shape (s, b, C, n).

        Returns:
        tuple: A tuple with the gradients (grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None).
        """
        grad_output = grad_output.contiguous()
        s, b, C, n = grad_output.shape

        if ctx.have_bias:
            f, bias, H_post, x, H_res = ctx.saved_tensors
        else:
            bias = None
            f, H_post, x, H_res = ctx.saved_tensors
        M = s * b

        grad_f = torch.empty_like(f)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
        grad_H_post = torch.zeros_like(
            H_post, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision
        grad_x = torch.empty_like(x)
        grad_H_res = torch.zeros_like(
            H_res, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        if bias is None:
            _mhc_expand_combine_bwd[grid](
                grad_output_ptr=grad_output,
                f_ptr=f,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                grad_H_post_ptr=grad_H_post,
                grad_f_ptr=grad_f,
                grad_H_res_ptr=grad_H_res,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=n * C,
                stride_grad_output_Cn=1,
                stride_fm=C,
                stride_fc=1,
                stride_xm=n * C,
                stride_xCn=1,
                stride_grad_fm=C,
                stride_grad_fc=1,
                stride_grad_xm=n * C,
                stride_grad_xCn=1,
                precision="tf32" if ctx.use_tf32 else "ieee",
            )
        else:
            _mhc_expand_combine_with_bias_bwd[grid](
                grad_output_ptr=grad_output,
                f_ptr=f,
                bias_ptr=bias,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                grad_H_post_ptr=grad_H_post,
                grad_f_ptr=grad_f,
                grad_bias_ptr=grad_bias,
                grad_H_res_ptr=grad_H_res,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=n * C,
                stride_grad_output_Cn=1,
                stride_fm=C,
                stride_fc=1,
                stride_bias=1,
                stride_xm=n * C,
                stride_xCn=1,
                stride_grad_fm=C,
                stride_grad_fc=1,
                stride_grad_bias=1,
                stride_grad_xm=n * C,
                stride_grad_xCn=1,
                precision="tf32" if ctx.use_tf32 else "ieee",
            )

        grad_H_post = grad_H_post.to(H_post.dtype)  # Cast back to the original dtype of H_post
        grad_H_res = grad_H_res.to(H_res.dtype)  # Cast back to the original dtype of H_res
        if bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        return grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None
