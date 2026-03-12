# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from attr import dataclass
import pytest
import torch
import torch.nn.functional as F

from tests.pytorch.utils import reset_rng_states
from transformer_engine.pytorch.triton.mhc import (
    mHCElementwiseOp,
    mHCPostResOp,
    mHCPreOp,
    mHCProjectionOp,
    mHCSinkhornOp,
)

seed = 1234
reset_rng_states()

# Enable TF32 for matmul to ensure consistency between the fused and reference implementations
torch.backends.cuda.matmul.allow_tf32 = False

@torch.compile
def mHCProjectionRef(x, phi):
    """
    Reference operator for mHC's projection building operation.

    x: (M, nC) where M = B * T
    phi: (2n + n^2, nC), which consists of the following matrices
        - phi_pre: (n, nC)
        - phi_post: (n, nC)
        - phi_res: (n^2, nC)
    n: number of Hyper Connection streams
    C: hidden dimension per stream
    """
    eps = torch.finfo(torch.float32).eps
    x_dtype = x.dtype
    x = x.to(torch.float32)
    phi = phi.to(torch.float32)

    Hs = x @ phi.T  # (M, 2n + n^2)

    x_fp32 = x.to(torch.float32) # Use fp32 for better numerical stability in variance calculation
    var = (x_fp32 * x_fp32).mean(dim=1)
    r = torch.sqrt(var + eps) # (M,)

    return Hs.to(x_dtype), r.to(x_dtype)

@torch.compile
def mHCElementwiseRef(H, alpha, beta, r, n):
    """
    Reference operator for mHC's pre and post calculations

    :param: H: (M, 2n + n^2), the unprocessed H matrices where M = B * T
    :param: alpha: (3,), three scalar parameters
    :param: beta: (1, 2n + n^2), bias term
    :param: r: (M,), the denominator for RMSNorm
    :param: n: int, the width of Hyper-Connection

    :return Hs: (M, 2n + n^2), the processed H matrices
    """

    M, _ = H.shape
    H_dtype = H.dtype
    H = H.to(torch.float32)
    alpha = alpha.to(torch.float32)
    beta = beta.to(torch.float32)
    r = r.to(torch.float32)

    H_pre = H[:, :n]  # (M, n)
    H_post = H[:, n:2*n]  # (M, n)
    H_res = H[:, 2*n:]  # (M, n^2)

    beta_pre = beta[0, :n]
    beta_post = beta[0, n:2*n]
    beta_res = beta[0, 2*n:2*n +  n*n]

    alpha_pre, alpha_post, alpha_res = alpha[0], alpha[1], alpha[2]

    H_pre = H_pre * alpha_pre
    H_post = H_post * alpha_post
    H_res = H_res * alpha_res

    H_pre = H_pre / r[:, None]
    H_post = H_post / r[:, None]
    H_res = H_res / r[:, None]

    H_pre = H_pre + beta_pre
    H_post = H_post + beta_post
    H_res = H_res + beta_res

    H_pre = F.sigmoid(H_pre)
    H_post = 2 * F.sigmoid(H_post)

    out = torch.cat([H_pre, H_post, H_res], dim=-1) # (M, 2n + n^2)

    return out.to(H_dtype)

@torch.compile
def mHCSinkhornRef(H_res, n=4, iterations=20):
    """
    Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.

    :param H_res: a tensor of shape (B, T, n, n)
    :return: a tensor of shape (B, T, n, n)
    """
    B, T = H_res.shape[:2]
    device = H_res.device
    dtype = H_res.dtype

    H_res_f = H_res.to(torch.float32).clone() # Use float32 for better numerical stability during Sinkhorn iterations

    log_mu = torch.zeros(B, T, n, device=device, dtype=torch.float32)
    log_nu = torch.zeros(B, T, n, device=device, dtype=torch.float32)

    f = torch.zeros(B, T, n, device=device, dtype=torch.float32)
    g = torch.zeros(B, T, n, device=device, dtype=torch.float32)

    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res_f + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res_f + f.unsqueeze(3), dim=2)

    log_P = f.unsqueeze(3) + H_res_f + g.unsqueeze(2)
    H_res_out = torch.exp(log_P).to(dtype) # Convert back to original dtype

    return H_res_out

@torch.compile
def mHCPreRef(x, H_pre, n):
    """
    Reference operator for applying mHC's pre matrix H to a vector x.

    x: (B, T, n, C)
    H_pre: (B, T, n)
    """
    H_pre = H_pre.contiguous()

    B, T, n, C = x.shape
    H_pre = H_pre.view(B, T, 1, n)  # (B, T, 1, n)

    out = (H_pre @ x).view(B, T, C) # (B, T, C)

    return out

@torch.compile
def mHCPostResRef(f, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's post transformation and residual transformation

    f: (B, T, C)
    H_post: (B, T, n)
    x: (B, T, n, C)
    H_res: (B, T, n, n)
    """

    B, T, n, C = x.shape

    f = f.view(B, T, 1, C)
    H_post = H_post.view(B, T, n, 1)

    out = H_post @ f + H_res @ x # (B, T, n, C)

    return out


@dataclass
class MHCConfig:
    B: int = 32 # Batch size
    T: int = 2048 # Sequence length
    C: int = 1024 # Hidden dimension
    n: int = 4 # Number of Hyper Connection streams

    allow_n = [4,]

    def __init__(self, B, T, C, n=4):
        assert n in self.allow_n, f"n must be one of {self.allow_n}"
        self.B = B
        self.T = T
        self.C = C
        self.n = n

    @staticmethod
    def desc(cfg):
        return f"B{cfg.B}_T{cfg.T}_C{cfg.C}_n{cfg.n}"

mhc_configs = [
    MHCConfig(8, 32, 32),
    MHCConfig(8, 128, 16 * 64),
    MHCConfig(4, 128, 16 * 64,),
    MHCConfig(2, 2048, 24 * 128),
    MHCConfig(1, 2048, 24 * 128,),
    MHCConfig(8, 1, 16 * 128,),
    MHCConfig(8, 1, 16 * 256,),
    MHCConfig(8, 1, 16 * 192,),
    MHCConfig(8, 128, 16 * 192,),
    MHCConfig(8, 1, 16 * 512,),
    MHCConfig(8, 128, 16 * 512,),
    MHCConfig(8, 1, 16 * 1024,),
    MHCConfig(8, 128, 16 * 1024,),
]

def get_tols(dtype):
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    else:
        tols = dict(atol=5e-3, rtol=5e-3)
    return tols

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_projection(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C
    N = 2*n + n*n

    tols = get_tols(dtype)
    use_tf32 = False

    x = torch.randn(B*T, nC, device='cuda', requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device='cuda')

    x_ref = x.detach().clone().requires_grad_(True)
    phi_ref = phi.detach().clone().requires_grad_(True)

    ref_out_Hs, ref_out_r = mHCProjectionRef(x_ref, phi_ref)
    fused_out_Hs_padded, fused_out_r = mHCProjectionOp.apply(x, phi, use_tf32)
    fused_out_Hs = fused_out_Hs_padded[:, :N]

    torch.testing.assert_close(fused_out_Hs, ref_out_Hs, **tols)
    torch.testing.assert_close(fused_out_r, ref_out_r, **tols)

    (ref_out_Hs.sum() + ref_out_r.sum()).backward()
    (fused_out_Hs.sum() + fused_out_r.sum()).backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(phi.grad, phi_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
def test_mhc_elementwise(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    N = 2*n + n*n

    tols = get_tols(dtype)

    H_padded = torch.randn(B*T, 32, device='cuda', requires_grad=True, dtype=dtype)
    H = H_padded[:, :N]
    alpha = torch.randn(3, device='cuda', requires_grad=True, dtype=dtype)
    beta = torch.randn(1, 2*n + n*n, device='cuda', requires_grad=True, dtype=dtype)
    r_raw = torch.randn(B*T, device='cuda', dtype=dtype) + 1.0
    r = r_raw.detach().clone().requires_grad_(True)

    H_ref = H.detach().clone().requires_grad_(True)
    alpha_ref = alpha.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    r_ref = r.detach().clone().requires_grad_(True)

    ref_out = mHCElementwiseRef(H_ref[:, :N], alpha_ref, beta_ref, r_ref, n)
    fused_out_padded = mHCElementwiseOp.apply(H_padded, alpha, beta, r, n)
    fused_out = fused_out_padded[:, :N]

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(H_padded.grad[:, :N], H_ref.grad, **tols)
    torch.testing.assert_close(alpha.grad, alpha_ref.grad, **tols)
    torch.testing.assert_close(beta.grad, beta_ref.grad, **tols)
    torch.testing.assert_close(r.grad, r_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_combined(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    N = 2*n + n*n
    nC = n * C

    tols = get_tols(dtype)

    tols = get_tols(dtype)
    use_tf32 = False

    x = torch.randn(B*T, nC, device='cuda', requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device='cuda')

    alpha = torch.randn(3, device='cuda', requires_grad=True, dtype=dtype)
    beta = torch.randn(1, 2*n + n*n, device='cuda', requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    phi_ref = phi.detach().clone().requires_grad_(True)

    alpha_ref = alpha.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    ref_out_H, ref_out_r = mHCProjectionRef(x_ref, phi_ref)
    fused_out_H_padded, fused_out_r = mHCProjectionOp.apply(x, phi, use_tf32)

    ref_out = mHCElementwiseRef(ref_out_H[:, :N], alpha_ref, beta_ref, ref_out_r, n)
    fused_out_padded = mHCElementwiseOp.apply(fused_out_H_padded, alpha, beta, fused_out_r, n)
    fused_out = fused_out_padded[:, :N]

    def mhc_combined(x_ref, phi_ref, alpha_ref, beta_ref):
        dtype = x_ref.dtype
        x_ref = x_ref.to(torch.float32)
        phi_ref = phi_ref.to(torch.float32)
        alpha_ref = alpha_ref.to(torch.float32)
        beta_ref = beta_ref.to(torch.float32)

        x_rmsnorm = F.rms_norm(x_ref, normalized_shape=(nC,))
        H = x_rmsnorm @ phi_ref.T
        H_pre = H[:, :n]
        H_post = H[:, n:2*n]
        H_res = H[:, 2*n:]

        out_pre = H_pre * alpha_ref[0] + beta_ref[:, :n]
        out_post = H_post * alpha_ref[1] + beta_ref[:, n:2*n]
        out_res = H_res * alpha_ref[2] + beta_ref[:, 2*n:]

        out_pre = out_pre.sigmoid()
        out_post = 2 * out_post.sigmoid()
        out_res = out_res

        return out_pre.to(dtype), out_post.to(dtype), out_res.to(dtype)

    H_pre_combined, H_post_combined, _ = mhc_combined(x_ref, phi_ref, alpha_ref, beta_ref)

    torch.testing.assert_close(H_pre_combined, ref_out[:, :n], **tols)
    torch.testing.assert_close(H_post_combined, ref_out[:, n:2*n], **tols)

    torch.testing.assert_close(H_pre_combined, fused_out[:, :n], **tols)
    torch.testing.assert_close(H_post_combined, fused_out[:, n:2*n], **tols)


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_sinkhorn_knopp(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n

    tols = get_tols(dtype)

    x = torch.randn(B, T, n, n, device='cuda', requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)

    ref_out = mHCSinkhornRef(x_ref, n)
    fused_out = mHCSinkhornOp.apply(x, n)

    print(f"ref_out.dtype: {ref_out.dtype}, fused_out.dtype: {fused_out.dtype}") # --- IGNORE ---

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_pre(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C

    tols = get_tols(dtype)

    x = torch.randn(B, T, n, C, device='cuda', requires_grad=True, dtype=dtype)
    H_pre = torch.randn(B, T, n, device='cuda', requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    H_pre_ref = H_pre.detach().clone().requires_grad_(True)

    ref_out = mHCPreRef(x_ref, H_pre_ref, n)
    fused_out = mHCPreOp.apply(x, H_pre, n)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_pre.grad, H_pre_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_post_res(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C

    tols = get_tols(dtype)

    f = torch.randn(B, T, C, device='cuda', requires_grad=True, dtype=dtype)
    H_post = torch.randn(B, T, n, device='cuda', requires_grad=True, dtype=dtype)
    x = torch.randn(B, T, n, C, device='cuda', requires_grad=True, dtype=dtype)
    H_res = torch.randn(B, T, n, n, device='cuda', requires_grad=True, dtype=dtype)

    f_ref = f.detach().clone().requires_grad_(True)
    H_post_ref = H_post.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    H_res_ref = H_res.detach().clone().requires_grad_(True)

    ref_out = mHCPostResRef(f_ref, H_post_ref, x_ref, H_res_ref, n)
    fused_out = mHCPostResOp.apply(f, H_post, x, H_res, n)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(f.grad, f_ref.grad, **tols)
    torch.testing.assert_close(H_post.grad, H_post_ref.grad, **tols)
    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_res.grad, H_res_ref.grad, **tols)
