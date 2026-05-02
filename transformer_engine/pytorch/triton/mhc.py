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
    _mhc_projection_bwd_fused_norm_weight,
    _mhc_sinkhorn_fwd_fused,
    _mhc_sinkhorn_fwd_fused_recompute,
    _mhc_sinkhorn_bwd_fused,
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


def mhc_generate_mix_and_aggregate(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    norm_weight: torch.Tensor = None,
    use_tf32: bool = True,
):
    """
    Generate the mix matrix H_pre, H_post, H_res and apply H_pre to x to aggregate n streams
    This wraps projection, scale, sinkhorn, and aggregate operations into one function.

    To use mHC in your model:
    ```
    layer_input, H_post, H_res = mhc_generate_mix_and_aggregate(x, phi, alpha, beta)
    layer_output = layer(layer_input) # Attn / FFN layer
    x = mhc_fused_expand_combine(layer_input, bias, H_post, x, H_res)
    ```

    This API accepts both BF16 and FP32 parameters, though the DeepSeek V4 recipe is:
    - x: BF16
    - phi, alpha, beta: FP32

    Parameters
    ----------
    x : torch.Tensor,
        input tensor of shape (s, b, C, n), where s is the sequence length, b is the batch size, C is the hidden dimension per hyper connection, and n is the number of hyper connections,
        dtype is torch.float16 or torch.float32
        Note that C is equal to the original hidden dimension divided by n.
    phi : torch.Tensor
        projection matrix of shape (N, nC), where N=2n+n*n (=24 for n=4), and nC is the hidden dimension after expansion (n times of C),
        dtype is torch.float16 or torch.float32
        Note: If user wants to use the main grad optimization for Megatron-LM, phi should be padded to (32, nC) so we can accumulate to its main grad buffer during the GEMM,
              which will padded N to 32 for better memory accessing pattern.
    norm_weight : torch.Tensor or None
        optional, the weight for RMSNorm, of shape (K,), which is the learnable per-element affine parameters (gamma) applied to RMSNorm
        dtype is torch.float16 or torch.float32
    alpha : torch.Tensor
        scaling factor for H, of shape (3,), where
        alpha[0] is applied to H[:, 0:n] for H_pre
        alpha[1] is applied to H[:, n:2n] for H_post
        alpha[2] is applied to H[:, 2n:2n+n*n] for H_res
        dtype: torch.float16 or torch.float32
    beta : torch.Tensor
        bias term for H, of shape (1, 2*n+n*n), where
        beta[0, 0:n] is applied to H[:, 0:n] for H_pre
        beta[0, n:2n] is applied to H[:, n:2n] for H_post
        beta[0, 2n:2n+n*n] is applied to H[:, 2n:2n+n*n] for H_res
        dtype is torch.float16 or torch.float32
    use_tf32 : bool
        whether to use TF32 for matrix multiplications

    Returns
    -------
    out : torch.Tensor
        out of shape (s, b, C), which is the aggregated result after applying H_pre to x, which will be fed into attention / FFN
        with the same dtype as x
    H_post : torch.Tensor
        H_post of shape (s, b, n), which will be used in the post-processing after attention / FFN in `mhc_fused_expand_combine`
        with dtype float32
    H_res : torch.Tensor
        H_res of shape (s, b, n, n), which will be used to mix the residual connection in `mhc_fused_expand_combine`
        with dtype float32
    """
    s, b, C, n = x.shape
    assert (
        n == 4
    ), "Only n=4 is supported in this implementation, where n is the Hyper Connection number"
    nC = n * C
    N = 2 * n + n * n
    H, ms = mhc_fused_projection(x.view(s * b, nC), phi, norm_weight, use_tf32=use_tf32)
    h_pre, h_post, h_res = mhc_fused_scale(H, alpha, beta, ms, n)
    H_pre = h_pre.view(s, b, n)
    H_post = h_post.view(s, b, n)
    H_res = h_res.view(s, b, n, n)
    H_res = mhc_fused_sinkhorn(H_res, n, recompute_hist=True, iters=20)
    out = mhc_fused_aggregate(x, H_pre.view(s, b, n), n, use_tf32=use_tf32)
    return out, H_post, H_res


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
        dtype is torch.float16 or torch.float32
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
        with the same dtype as H_res
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
        Scaled H_pre of shape (M, n), which aggregates (s, b, C, n) input of a Hyper Connection block into (s, b, n) as the input of attention / MLP,
        with the same dtype as H
    h_post : torch.Tensor
        Scaled H_post of shape (M, n), which expands the output of attention / MLP of shape (s, b, n) back to (s, b, C, n) for the residual connection,
        with the same dtype as H
    h_res : torch.Tensor
        Scaled H_res of shape (M, n*n), which mixes the n streams of the (s, b, C, n) input of a Hyper Connection block,
        with the same dtype as H

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
        dtype is torch.float16 or torch.float32
    H_pre: torch.Tensor
        input H_pre matrix of shape (s, b, n)
        dtype is torch.float16 or torch.float32
    n: int
        number of hyper connections, where only n=4 is supported in the current implementation
    use_tf32: bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    Returns
    -------
    out: torch.Tensor
         output activation tensor of shape (s, b, C), which is the aggregated output after merging n hyper connections,
         with the same dtype as x
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
    use_tf32: bool = True,
    fuse_grad_x_acc: bool = False,
):
    """
    Expand and combine operation for merging n hyper connections (see section 4.3.1 of the DeepSeek mHC paper):

    out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)

    Parameters
    ----------
    f : torch.Tensor
        input activation tensor of shape (s, b, C), which is the output from the attention / FFN sub-layer in a transformer block
        dtype is torch.float16 or torch.float32
    bias : torch.Tensor or None
        optional bias tensor of shape (C,) from the last linear layer, where f + bias is fused in this kernel for better performance
        dtype is torch.float16 or torch.float32
    H_post : torch.Tensor
        input H_post matrix of shape (s, b, n)
        dtype is torch.float16 or torch.float32
    x : torch.Tensor
        input activation tensor of shape (s, b, C, n), which is the hyper connection input before the aggregation operation
        dtype is torch.float16 or torch.float32
    H_res : torch.Tensor
        input H_res matrix of shape (s, b, n, n)
        dtype is torch.float16 or torch.float32
    use_tf32 : bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail
    fuse_grad_x_acc : bool
        Use the same buffer for inplace gradient accumulation to avoid PyTorch autograd overhead.
        If enable, triton kernels will accumulate the gradient of x in the same buffer to avoid copying the gradient by PyTorch.

    Returns
    -------
    out : torch.Tensor
        out of shape (s, b, C, n), which is the expanded and combined output after merging n hyper connections,
        with the same dtype as x
    """
    _, _, _, n = x.shape
    assert n == 4, "Only n=4 is supported in this implementation"
    check_deterministic("mhc_fused_expand_combine")
    out = mHCExpandCombineOp.apply(f, bias, H_post, x, H_res, n, use_tf32, fuse_grad_x_acc)
    return out


def mhc_fused_projection(
    x: torch.Tensor, phi: torch.Tensor, norm_weight: torch.Tensor = None, use_tf32: bool = True
):
    """
    Fused projection operation to compute H matrices and mean square for RMSNorm (see eq. 14-15, section 4.3.1 of the DeepSeek mHC paper):

    H = x @ phi^T: (M, K) @ (K, N) -> (M, N), which is padded to (M, 32) for better memory access pattern in the next kernels.
    ms = mean(x^2, dim=-1): (M,)

    If norm_weight is provided, it will be absorbed into phi. In this case, the operation becomes:
    Projection:
    - H = x @ (phi.T * norm_weight) = x @ phi.T * norm_weight
    - ms = mean(x^2, dim=-1)
    - H = H / sqrt(ms) = x @ (phi.T * norm_weight) / sqrt(ms), where this step is fused into `mhc_fused_scale`
    which is equivalent to performing the computation in the normal order:
    - x_normalized = RMSNorm(x) = x * norm_weight / sqrt(ms)
    - H = x_normalized @ phi.T = (x / sqrt(ms) @ phi.T) * norm_weight

    Note: the current implementation only supports n=4

    Parameters
    ----------
    x : torch.Tensor
        input tensor of shape (M, K), where M=s*b is the batch size and K=nC is the hidden dimension after expansion.
        dtype is torch.float16 or torch.float32
    phi : torch.Tensor
        projection matrix of shape (N, K), where N=2n+n*n (=24 for n=4)
        dtype is torch.float16 or torch.float32
        Note: If user wants to use the main grad optimization for Megatron-LM, phi should be padded to (32, K) so we can accumulate to its main grad buffer during the GEMM,
              which will padded N to 32 for better memory accessing pattern.
    norm_weight : torch.Tensor or None
        optional, the weight for RMSNorm, of shape (K,), which is the learnable per-element affine parameters (gamma) applied to RMSNorm
        dtype is torch.float16 or torch.float32
    use_tf32 : bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail.

    Returns
    -------
    H : torch.Tensor
        Projected matrix of shape (M, 32), where only the first N elements in the last dimension are valid,
        with dtype float32
    ms : torch.Tensor
        Mean square of shape (M,), which is used for RMSNorm in the next kernel,
        with dtype float32
    """
    assert phi.shape[0] == 24 or (phi.shape[0] == 32 and hasattr(phi, "main_grad")), (
        "Currently only n=4 is supported, which means phi should have 24 (or 32 if you padded phi)"
        " in its first dimension"
    )
    check_deterministic("mhc_fused_projection")
    H, ms = mHCProjectionOp.apply(x, phi, norm_weight, use_tf32)
    return H, ms


class mHCProjectionOp(torch.autograd.Function):
    """
    PyTorch operator for the fused projection operation in mHC, whose wrapper API is mhc_fused_projection.
    """

    @staticmethod
    def forward(ctx, x, phi, norm_weight=None, use_tf32=True):
        """
        The forward pass of the fused projection operation. Computes H = x @ phi^T and the mean
        If norm_weight is provided, it will be absorbd by phi
        square ms = mean(x^2, dim=-1) for RMSNorm in a single fused kernel.

        Parameters:
        ctx : The context object.
        x (tensor): The input tensor of shape (M, K), where M=s*b is the flattened batch dimension and K=nC is the hidden dimension after expansion.
        phi (tensor): The projection matrix of shape (N, K), where N=2n+n*n (=24 for n=4).
            If user wants to use the main grad optimization for Megatron-LM, phi should be padded to (32, K) so we can accumulate to its main grad buffer during the GEMM
        norm_weight (tensor or None): Optional, or tensor of shape (K,). RMSNorm's learnable per-element affine parameters
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

        precision = "tf32" if ctx.use_tf32 else "ieee"
        # If upcasting from bf16 to fp32 takes place inside the triton kernel, triton will ignore "ieee" precision and use tf32 anyway
        # See https://github.com/triton-lang/triton/issues/10176 for detail.
        # Therefore, we need to use tf32x3 instead which at least has better accuracy than tf32 just to make the tests pass. In production
        # precision should be tf32 so it's not affected.
        if precision == "ieee" and x.dtype == torch.bfloat16:
            # When we have x is bf16, and either
            # - phi is fp32, or
            # - phi is bf16 but norm_weight is not None, where in this case inside the triton kernel,
            #   we will promote phi to fp32 because we want better precision for phi * norm_weight
            # In both cases we will need to upcast x to fp32 inside the kernel, and trigger the issue mentioned above
            if norm_weight is not None or phi.dtype == torch.float32:
                precision = "tf32x3"

        _mhc_projection_fwd_fused[grid](
            x_ptr=x,  # (M, K)
            phi_ptr=phi,  # (N, K)
            h_ptr=H,  # (M, 32)
            ms_ptr=ms,  # (M,)
            norm_weight_ptr=norm_weight,
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
            stride_norm_weight=1,
            BLOCK_SIZE_N=32,
            precision=precision,
            HAS_NORM_WEIGHT=norm_weight is not None,
        )

        ctx.save_for_backward(x, phi, ms, norm_weight)
        ctx.phi_dtype = phi.dtype
        ctx.precision = precision
        ctx.phi_main_grad = getattr(phi, "main_grad", None)
        ctx.norm_weight_main_grad = getattr(norm_weight, "main_grad", None)

        return H, ms  # Keep both in fp32, which will be passed to sigmoid in mHCScaleFusedOp

    @staticmethod
    def backward(ctx, grad_H, grad_ms):
        """
        The backward pass of the fused projection operation. Computes gradients for x and phi.

        - grad_psi = grad_H^T @ x: (2n + n^2, M) @ (M, nC) = (2n + n^2, nC), where grad_H's last dim is padded to 32
        If norm_weight is None:
        - grad_phi = grad_psi
        Otherwise,
        - grad_phi = grad_psi * norm_weight: (2n + n^2, nC) * (nC,) = (2n + n^2, nC)
        - grad_norm_weight = sum(grad_psi * phi, dim=0): ((2n + n^2, nC) * (2n + n^2, nC)).sum(dim=0) -> (nC,)
        Reorder a bit:
        - grad_phi = grad_H^T @ x * norm_weight
        - grad_norm_weight = sum((grad_H^T @ x) * phi, dim=0)

        - grad_x = grad_H @ phi + 2 * x * grad_ms / K, where the second term is the gradient contribution from
        the mean square computation fused in the forward pass.

        Parameters:
        ctx : The context object with saved tensors.
        grad_H (tensor): The gradient of the loss with respect to H, of shape (M, 32).
        grad_ms (tensor): The gradient of the loss with respect to the mean square, of shape (M,).

        Returns:
        tuple: A tuple with the gradients (grad_x, grad_phi, grad_norm_weight, None).
        """
        x, phi, ms, norm_weight = ctx.saved_tensors
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

        fuse_grad_x_acc = hasattr(x.untyped_storage(), "grad_x_acc")
        if fuse_grad_x_acc:
            grad_x = x.untyped_storage().grad_x_acc.view_as(x)
        else:
            grad_x = torch.empty((M, K), device=device, dtype=x.dtype)

        if norm_weight is not None:
            # With norm_weight, we need a fused kernel to perform GEMM and output both phi & norm_weight gradients
            grid = lambda META: (
                triton.cdiv(K, META["BLOCK_SIZE_K"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )

            if ctx.phi_main_grad is not None and ctx.phi_main_grad.shape == phi.shape:
                grad_phi = ctx.phi_main_grad
            else:
                grad_phi = torch.zeros_like(phi, dtype=torch.float32)
            if ctx.norm_weight_main_grad is not None:
                grad_norm_weight = ctx.norm_weight_main_grad
            else:
                grad_norm_weight = torch.zeros_like(norm_weight, dtype=torch.float32)

            _mhc_projection_bwd_fused_norm_weight[grid](
                x_ptr=x,  # (M, K)
                grad_H_ptr=grad_H,  # (M, 32)
                phi_ptr=phi,  # (N, K)
                norm_weight_ptr=norm_weight,  # (K,)
                grad_phi_ptr=grad_phi,  # (N, K)
                grad_norm_weight_ptr=grad_norm_weight,  # (K,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_grad_Hm=32,
                stride_grad_Hn=1,
                stride_phin=K,
                stride_phik=1,
                stride_norm_weight=1,
                stride_grad_phin=K,
                stride_grad_phik=1,
                stride_grad_norm_weight=1,
                BLOCK_SIZE_N=32,
                precision="tf32" if ctx.use_tf32 else "ieee",
            )

            if ctx.phi_main_grad is not None:
                grad_phi = None
            else:
                grad_phi = grad_phi.to(phi.dtype)
            if ctx.norm_weight_main_grad is not None:
                grad_norm_weight = None
            else:
                grad_norm_weight = grad_norm_weight.to(norm_weight.dtype)
        else:
            # Without norm_weight, this is only a GEMM with no fusion needed so we let cuBLAS handle it
            if ctx.phi_main_grad is not None:
                grad_phi = general_gemm(
                    x.to(grad_H.dtype),
                    grad_H,
                    out_dtype=torch.float32,
                    layout="NT",
                    accumulate=True,
                    out=ctx.phi_main_grad,
                )[0]
            else:
                grad_phi = general_gemm(
                    x.to(grad_H.dtype), grad_H, out_dtype=torch.float32, layout="NT"
                )[0][:N, :]
                grad_phi = grad_phi.to(phi.dtype)
            grad_norm_weight = None

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

        _mhc_projection_bwd_fused[grid](
            x_ptr=x,
            grad_x_ptr=grad_x,  # (M, K)
            phi_ptr=phi,  # (N, K)
            norm_weight_ptr=norm_weight,  # (K,)
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
            stride_norm_weight=1,
            stride_grad_phin=K,
            stride_grad_phik=1,
            stride_grad_hm=32,
            stride_grad_hn=1,
            stride_grad_ms=1,
            BLOCK_SIZE_N=32,
            precision=ctx.precision,
            FUSE_GRAD_X_ACC=fuse_grad_x_acc,
            HAS_NORM_WEIGHT=norm_weight is not None,
        )

        if fuse_grad_x_acc:
            del x.untyped_storage().grad_x_acc

        return grad_x.to(x.dtype), grad_phi, grad_norm_weight, None


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
        ctx.alpha_main_grad = getattr(alpha, "main_grad", None)
        ctx.beta_main_grad = getattr(beta, "main_grad", None)
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

        M, _ = grad_out.shape
        N = 2 * n + n * n

        grad_H = torch.zeros(
            (M, 32), device=grad_out.device, dtype=H.dtype
        )  # Pad the grad_H to 32 in the last dimension

        if ctx.alpha_main_grad is not None:
            grad_alpha = ctx.alpha_main_grad
        else:
            grad_alpha = torch.zeros((3,), device=grad_out.device, dtype=torch.float32)
        if ctx.beta_main_grad is not None:
            grad_beta = ctx.beta_main_grad
        else:
            grad_beta = torch.zeros((1, N), device=grad_out.device, dtype=torch.float32)
        grad_ms = torch.zeros((M,), device=grad_out.device, dtype=ms.dtype)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_scale_bwd_fused[grid](
            grad_out_ptr=grad_out,
            out_ptr=out,
            grad_H_ptr=grad_H,
            H_ptr=H,
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
            stride_grad_Hm=32,
            stride_grad_Hn=1,
            stride_Hm=32,
            stride_Hn=1,
            stride_grad_a=1,
            stride_a=1,
            stride_grad_b=1,
            stride_grad_ms=1,
            stride_ms=1,
            BLOCK_SIZE_N=32,
            eps=torch.finfo(ms.dtype).eps,
        )

        if ctx.alpha_main_grad is not None:
            grad_alpha = None
        else:
            grad_alpha = grad_alpha.to(alpha.dtype)
        if ctx.beta_main_grad is not None:
            grad_beta = None
        else:
            grad_beta = grad_beta.to(alpha.dtype)  # alpha and beta should have the same dtype

        return (
            grad_H,
            grad_alpha,
            grad_beta,
            grad_ms,
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

        grad_res_out = grad_out.contiguous().view(M, n * n)

        grad_res = torch.empty_like(H_res)

        # pylint: disable=unnecessary-lambda-assignment
        grid = lambda META: (triton.cdiv(M * n * n, META["BLOCK_SIZE"]),)

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
            RECOMPUTE=recompute_hist,
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

        fuse_grad_x_acc = hasattr(x.untyped_storage(), "grad_x_acc")
        if fuse_grad_x_acc:
            grad_x = x.untyped_storage().grad_x_acc.view_as(x)
        else:
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
            FUSE_GRAD_X_ACC=fuse_grad_x_acc,
        )

        grad_H_pre = grad_H_pre.to(H_pre.dtype)  # Cast back to the original dtype of H_pre

        if fuse_grad_x_acc:
            grad_x = None

        return grad_x, grad_H_pre, None, None


class mHCExpandCombineOp(torch.autograd.Function):
    """
    PyTorch operator for the expand and combine operation in mHC, whose wrapper API is mhc_fused_expand_combine.
    """

    @staticmethod
    def forward(ctx, f, bias, H_post, x, H_res, n, use_tf32=True, fuse_grad_x_acc=True):
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
        fuse_grad_x_acc (bool): Use the same buffer for inplace gradient accumulation to avoid PyTorch autograd overhead.

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
        ctx.fuse_grad_x_acc = fuse_grad_x_acc
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

        if ctx.fuse_grad_x_acc:
            # When fused x gradient accumulation is enabled, use fp32 for the accumulation buffer
            x.untyped_storage().grad_x_acc = grad_x.to(torch.float32)
            grad_x = None

        return grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None, None
