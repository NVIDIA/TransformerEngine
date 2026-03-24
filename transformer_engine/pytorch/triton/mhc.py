# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import triton.language as tl

from transformer_engine.common.triton.mhc import (
    _mhc_elementwise_fwd_fused,
    _mhc_elementwise_bwd_fused,
    _mhc_post_res_fwd,
    _mhc_post_res_bwd,
    _mhc_pre_fwd,
    _mhc_pre_bwd,
    _mhc_projection_fwd_fused,
    _mhc_projection_bwd_fused,
    _mhc_sinkhorn_knopp_fwd_fused,
    _mhc_sinkhorn_knopp_fwd_fused_recompute,
    _mhc_sinkhorn_knopp_bwd_fused,
    _mhc_sinkhorn_knopp_bwd_fused_recompute,
)


class mHCProjectionOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, phi, use_tf32=True):
        """
        Fused mHC projection matrices building operation.
        This kernel performs a fused matrix multiplication and a reduction to compute the projection matrices H:
        H_pre, H_post, H_res and the normalization factor r.

        :param x: input tensor x of shape (B*T, nC), and it should be in bfloat16 for better performance
        :param phi: projection matrix phi of shape (2n + n^2, nC), which consists of the following matrices
            - phi_pre: (n, nC)
            - phi_post: (n, nC)
            - phi_res: (n^2, nC)

        H = x @ phi.T: (B*T, nC) @ (nC, 2n + n^2) = (B*T, 2n + n^2)
        r = sum(x^2) / sqrt(nC): (B*T,)

        :return H: projection matrices of shape (B*T, 2n + n^2), which is padded to 32 with zeroes in the last dimension
        :return r: normalization factor of shape (B*T,)
        """
        x = x.contiguous()

        ctx.use_tf32 = use_tf32
        ctx.dtype = x.dtype

        M, K = x.shape
        device = x.device

        N = phi.shape[0]

        # Pad H to (B, T, 32) for better memory access pattern in the kernel, but only the first N elements in the last dimension are valid
        H = torch.zeros((M, 32), device=device, dtype=x.dtype)
        r = torch.zeros((M), device=device, dtype=x.dtype)

        # Launch triton kernel
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        if use_tf32:
            _mhc_projection_fwd_fused[grid](
                x_ptr=x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                h_ptr=H,  # (M, 32)
                r_ptr=r,  # (M,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,  # strides for x
                stride_phin=K,
                stride_phik=1,  # strides for phi
                stride_hm=32,
                stride_hn=1,  # strides for H, remember it's padded to 32!
                stride_r=1,  # strides for r, which is a 1D tensor
                BLOCK_SIZE_N=32,
                out_dtype=tl.float32 if x.dtype == torch.float32 else tl.bfloat16,
                precision="tf32",
                eps=torch.finfo(torch.float32).eps,
            )
        else:
            _mhc_projection_fwd_fused[grid](
                x_ptr=x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                h_ptr=H,  # (M, 32)
                r_ptr=r,  # (M,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,  # strides for x
                stride_phin=K,
                stride_phik=1,  # strides for phi
                stride_hm=32,
                stride_hn=1,  # strides for H, remember it's padded to 32!
                stride_r=1,  # strides for r, which is a 1D tensor
                BLOCK_SIZE_N=32,
                out_dtype=tl.float32 if x.dtype == torch.float32 else tl.bfloat16,
                precision="ieee",
                eps=torch.finfo(torch.float32).eps,
            )

        ctx.save_for_backward(x, phi, r)
        ctx.phi_dtype = phi.dtype

        return H.to(ctx.dtype), r.to(ctx.dtype)

    @staticmethod
    def backward(ctx, grad_H, grad_r):
        """
        Fused backward pass for the fused mHC projection operation.

        grad_H: (B * T, 2n + n^2), which is padded to 32 with zeroes in the last dimension
        grad_r: (B * T)

        grad_x, x: (B * T, nC)
        grad_phi, phi: (2n + n^2, nC)

        Forward:
        - H = x @ phi,
        - r = (x * x).sum(dim=1) / sqrt(nC)

        Backward:
        - grad_phi = x^T @ grad_H: (nC, B*T) @ (B*T, 2n + n^2) = (nC, 2n + n^2)
            -This is left for pytorch to handle

        - grad_x = (grad_H @ phi) + (x * (grad_r * 2  / sqrt(nC))):
            - grad_H @ phi: (B*T, 2n + n^2) @ (2n + n^2, nC) = (B*T, nC)
            - x * (grad_r * 2  / sqrt(nC)): (B*T, nC) * (B*T, 1) = (B*T, nC)
        """

        x, phi, r = ctx.saved_tensors  # Note phi here is still column-major
        M, K = x.shape
        device = x.device

        N = phi.shape[0]

        grad_H = grad_H
        grad_r = grad_r

        grad_H = grad_H.contiguous().view(M, -1)
        grad_r = grad_r.contiguous().view(
            M,
        )
        r = r.contiguous().view(
            M,
        )

        grad_x = torch.zeros((M, K), device=device, dtype=x.dtype)
        grad_phi = (grad_H[:, :N].T @ x).to(
            ctx.phi_dtype
        )  # (2n + n^2, M) @ (M, nC) = (2n + n^2, nC), note that the last dimension of grad_H is already padded to 32

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

        # Compute grad_x =  (grad_H @ phi^T) + (x * (grad_r * 2  / sqrt(nC))), we can fuse the GeMM and the element-wise add
        if ctx.use_tf32:
            _mhc_projection_bwd_fused[grid](
                x_ptr=x,
                dx_ptr=grad_x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                dh_ptr=grad_H,  # (M, 32)
                r_ptr=r,
                dr_ptr=grad_r,  # (M,),
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_dxm=K,
                stride_dxk=1,
                stride_phin=K,
                stride_phik=1,
                stride_dphin=K,
                stride_dphik=1,
                stride_dhm=32,
                stride_dhn=1,  # strides for grad_H, remember it's padded to 32!
                stride_dr=1,
                BLOCK_SIZE_N=32,
                precision="tf32",
            )
        else:
            _mhc_projection_bwd_fused[grid](
                x_ptr=x,
                dx_ptr=grad_x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                dh_ptr=grad_H,  # (M, 32)
                r_ptr=r,
                dr_ptr=grad_r,  # (M,),
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_dxm=K,
                stride_dxk=1,
                stride_phin=K,
                stride_phik=1,
                stride_dphin=K,
                stride_dphik=1,
                stride_dhm=32,
                stride_dhn=1,  # strides for grad_H, remember it's padded to 32!
                stride_dr=1,
                BLOCK_SIZE_N=32,
                precision="ieee",
            )

        return grad_x.to(ctx.dtype), grad_phi.to(ctx.dtype), None


class mHCElementwiseOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, H, alpha, beta, r, n):
        """
        Reference operator for mHC's pre and post calculations

        :param: H: (B * T, 2n + n^2), the unprocessed H matrices, which is padded to 32 with zeroes in the last dimension
        :param: alpha: (3,), three scalar parameters
        :param: beta: (1, 2n + n^2), bias term
        :param: r: (B * T), the denominator for RMSNorm
        :param: n: int, the width of Hyper-Connection

        :return out: (B * T, 2n + n^2), the processed H matrices, , which is padded to 32 with zeroes in the last dimension
        """

        ctx.dtype = H.dtype
        H = H.to(torch.float32)
        alpha = alpha.to(torch.float32)
        beta = beta.to(torch.float32)
        r = r.to(torch.float32)

        M, _ = H.shape

        H = H.contiguous()
        beta = beta.contiguous()
        r = r.contiguous()

        out = torch.zeros(
            (M, 32), device=H.device, dtype=H.dtype
        )  # Pad the output to 32 in the last dimension

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_elementwise_fwd_fused[grid](
            h_ptr=H,  # (M, N), which is padded to (M, 32)
            b_ptr=beta,  # (N,)
            a_ptr=alpha,  # (N,)
            r_ptr=r,  # (M,)
            out_ptr=out,  # (M, N), which is padded to (M, 32)
            M=M,
            n=n,
            stride_hm=32,
            stride_hn=1,
            stride_a=1,
            stride_b=1,
            stride_r=1,
            stride_out_m=32,
            stride_out_n=1,  # strides for out, which is padded to 32 in the last dimension
            BLOCK_SIZE_N=32,
        )

        ctx.save_for_backward(H, alpha, r, out)
        ctx.n = n

        return out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for mHC's pre and post calculations

        :param grad_out: (B * T, 2n + n^2), gradients of the output H matrices, which is padded to 32 with zeroes in the last dimension

        :return grad_H: (B * T, 2n + n^2), gradients of the input H matrices, which is padded to 32 with zeroes in the last dimension
        :return grad_alpha: (3,), gradients of the alpha parameters
        :return grad_beta: (1, 2n + n^2), gradients of the beta parameters
        :return grad_r: (B * T), gradients of the r values
        :return None: placeholder for the non-tensor n argument
        """
        H, alpha, r, out = ctx.saved_tensors
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
        grad_r = torch.zeros((M,), device=grad_out.device, dtype=grad_out.dtype)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_elementwise_bwd_fused[grid](
            grad_out_ptr=grad_out,
            out_ptr=out,
            grad_h_ptr=grad_h,
            h_ptr=H,
            grad_a_ptr=grad_alpha,
            a_ptr=alpha,
            grad_b_ptr=grad_beta,
            grad_r_ptr=grad_r,
            r_ptr=r,
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
            stride_grad_r=1,
            stride_r=1,
            BLOCK_SIZE_N=32,
        )

        return (
            grad_h.to(ctx.dtype),
            grad_alpha.to(ctx.dtype),
            grad_beta.to(ctx.dtype),
            grad_r.to(ctx.dtype),
            None,
        )


class mHCSinkhornOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, H_res, n=4, recompute_hist=True, iters=20):
        """
        H_res: (B, T, n, n)
        """
        B, T, _, _ = H_res.shape

        ctx.dtype = H_res.dtype
        H_res = H_res.to(torch.float32)

        H_res = H_res.contiguous().clone().view(B * T, n * n)  # (B*T, n*n)
        assert n == 4, "This implementation only supports n=4 for now due to BLOCK_SIZE constraints"

        hist_f, hist_g = None, None
        if not recompute_hist:
            # History buffers: (iters+1, B, T, n)
            hist_f = torch.zeros((iters + 1, B, T, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.zeros((iters + 1, B, T, n), device=H_res.device, dtype=H_res.dtype)
        H_res_out = torch.zeros_like(H_res)  # (B*T, n*n)

        grid = lambda meta: (triton.cdiv(B * T * n * n, meta["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_knopp_fwd_fused_recompute[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=B * T,
                n=n,
                iters=iters,
            )
        else:
            _mhc_sinkhorn_knopp_fwd_fused[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=B * T,
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

        H_res_out = H_res_out.view(B, T, n, n)
        return H_res_out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):

        B, T, n, _ = grad_out.shape
        M = B * T

        hist_f, hist_g = None, None
        recompute_hist = ctx.recompute_hist
        iters = ctx.iters
        if recompute_hist:
            H_res, H_res_out = ctx.saved_tensors
            hist_f = torch.zeros((iters + 1, B, T, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.zeros((iters + 1, B, T, n), device=H_res.device, dtype=H_res.dtype)
        else:
            H_res, H_res_out, hist_f, hist_g = ctx.saved_tensors

        iters = ctx.iters
        n = ctx.n

        grad_res_out = grad_out.clone().contiguous().view(M, n * n)

        grad_res = torch.zeros_like(H_res)

        grid = lambda meta: (triton.cdiv(M * n * n, meta["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_knopp_bwd_fused_recompute[grid](
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
            _mhc_sinkhorn_knopp_bwd_fused[grid](
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

        grad_res = grad_res.view(B, T, n, n)

        return grad_res.to(ctx.dtype), None, None, None


class mHCPreOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, H_pre, n):
        """
        Perform the pre-mHC transformation using the pre matrix H_pre (which is the first n columns of H) on the input x.

        :param x: (B, T, n, C) where nC = n * C
        :param H_pre: (B, T, n) pre-mHC transformation matrix, which is a slice: H[:, :, :4], and H is padded to have 32 columns
                      we will use contiguous() to create a copy here to simplify things for autograd since this tensor is not large
                      so making a copy should be fine
        :param n: the number of hyper connections, which must be 4 in this implementation
        """

        x = x.contiguous()
        H_pre = (
            H_pre.contiguous()
        )  # This likely will incur a copy but H_pre is fairly small (B*T*4), so it should be fine

        B, T, n, C = x.shape
        nC = n * C
        assert n == 4 and nC % n == 0, "Only n=4 is supported in this implementation"
        M = B * T

        out = torch.zeros((B, T, C), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )

        _mhc_pre_fwd[grid](
            x_ptr=x,
            H_pre_ptr=H_pre,
            output_ptr=out,
            M=M,
            C=C,
            n=n,
            stride_xm=nC,
            stride_xn=C,
            stride_xc=1,
            stride_H_pre_m=n,
            stride_H_pre_n=1,  # A slice has the same strides as its parent tensor
            stride_output_m=C,
            stride_output_c=1,
        )

        ctx.save_for_backward(x, H_pre)
        ctx.n = n

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradients for the pre-mHC transformation.

        :param grad_output: (B, T, C) gradient of the output

        :return: gradients with respect to the inputs of the forward function, which are:
        - grad_x: (B, T, n, C) gradient with respect to x
        - grad_H: (B, T, 2*n + n*n) gradient with respect to H, which is padded to 32
        - None for n since it's an integer and we don't need gradients for it
        """
        grad_output = grad_output.contiguous()

        x, H_pre = ctx.saved_tensors
        n = ctx.n

        B, T, n, C = x.shape
        nC = n * C
        assert n == 4 and nC % n == 0, "Only n=4 is supported in this implementation"
        M = B * T

        grad_x = torch.zeros_like(x)
        grad_H_pre = torch.zeros(
            (B, T, n), dtype=torch.float32, device=H_pre.device
        )  # We need to use atomic_add for this so we need higher precision

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )

        _mhc_pre_bwd[grid](
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
            stride_H_pre_m=n,
            stride_H_pre_n=1,
            stride_grad_H_pre_m=n,
            stride_grad_H_pre_n=1,
            stride_xm=nC,
            stride_xn=C,
            stride_xc=1,
            stride_grad_xm=nC,
            stride_grad_xn=C,
            stride_grad_xc=1,
        )

        grad_H_pre = grad_H_pre.to(H_pre.dtype)  # Cast back to the original dtype of H_pre

        return grad_x, grad_H_pre, None


class mHCPostResOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, H_post, x, H_res, n):
        """
        Perform the fused post-mHC and residual transformation in the forward pass.

        :param x: the original transformer input to the manifold connection, with shape (B, T, n, C)
        :param H_post: the post-transformation matrix, with shape (B, T, n)
        :param H_res: the residual transformation matrix, with shape (B, T, n, n)
        :param f: the output of the attention/FFN module before the post transformation, with shape (B, T, C)
        :param n: the number of hyper connection streams, which is a constant 4 in our implementation.

        :return out = H_res @ x + H_post @ f, where f is the output of the attention/FFN module before the post transformation.
        """

        x = x.contiguous()
        f = f.contiguous()
        H_post = H_post.contiguous()
        H_res = H_res.contiguous()

        B, T, n, C = x.shape
        nC = n * C
        assert n == 4 and nC % n == 0, "Only n=4 is supported in this implementation"
        M = B * T

        out = torch.zeros((B, T, n, C), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )

        _mhc_post_res_fwd[grid](
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
            stride_H_post_m=n,
            stride_H_post_n=1,
            stride_xm=n * C,
            stride_xn=C,
            stride_xc=1,
            stride_H_res_m=n * n,
            stride_H_res_n1=n,
            stride_H_res_n2=1,
            stride_output_m=n * C,
            stride_output_n=C,
            stride_output_c=1,
        )

        ctx.n = n
        ctx.save_for_backward(f, H_post, x, H_res)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradients for the fused post-mHC and residual transformation in the backward pass.
        """

        grad_output = grad_output.contiguous()
        B, T, n, C = grad_output.shape

        f, H_post, x, H_res = ctx.saved_tensors
        n = ctx.n
        M = B * T

        grad_f = torch.zeros_like(f)
        grad_H_post = torch.zeros_like(
            H_post, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision
        grad_x = torch.zeros_like(x)
        grad_H_res = torch.zeros_like(
            H_res, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )

        _mhc_post_res_bwd[grid](
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
            stride_grad_output_n=C,
            stride_grad_output_c=1,
            stride_fm=C,
            stride_fc=1,
            stride_H_post_m=n,
            stride_H_post_n=1,
            stride_xm=n * C,
            stride_xn=C,
            stride_xc=1,
            stride_H_res_m=n * n,
            stride_H_res_n1=n,
            stride_H_res_n2=1,
            stride_grad_H_post_m=n,
            stride_grad_H_post_n=1,
            stride_grad_fm=C,
            stride_grad_fc=1,
            stride_grad_H_res_m=n * n,
            stride_grad_H_res_n1=n,
            stride_grad_H_res_n2=1,
            stride_grad_xm=n * C,
            stride_grad_xn=C,
            stride_grad_xc=1,
        )

        grad_H_post = grad_H_post.to(H_post.dtype)  # Cast back to the original dtype of H_post
        grad_H_res = grad_H_res.to(H_res.dtype)  # Cast back to the original dtype of H_res

        return grad_f, grad_H_post, grad_x, grad_H_res, None
