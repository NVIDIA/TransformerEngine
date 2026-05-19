"""
Profile that the dedicated swizzle kernels (swizzle_{row,col}_scaling_kernel
in transformer_engine/common/swizzle/swizzle.cu) disappear from the timeline
when NVFP4 RHT cast-fusion emits SF in the GEMM-swizzled layout directly
(optimize_for_gemm=True).

Test setup:
  - NVFP4 + RHT + post-RHT amax (same as te.Linear sets up internally)
  - rowwise=True AND columnwise=True (covers BOTH swizzle_row_scaling_kernel
    and swizzle_col_scaling_kernel; this is what tex.Linear's input quantizer
    needs during training because the rowwise tensor is used by the fwd GEMM
    and the columnwise tensor is used by the dgrad GEMM)
  - tex.swizzle_scales_for_gemm_(t) is what te.Linear -> tex.generic_gemm
    calls just before the cuBLAS LT NVFP4 GEMM dispatch
"""

import torch
import transformer_engine.pytorch as te  # noqa: F401 must be first
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer


def make_quantizer(optimize_for_gemm: bool) -> NVFP4Quantizer:
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
    )
    q.optimize_for_gemm = optimize_for_gemm
    return q


import re

# Match ONLY the standalone swizzle pass kernels in
# transformer_engine/common/swizzle/swizzle.cu — NOT RHT cast-fusion kernels
# whose mangled name happens to contain "Swizzle" because of the
# `template <..., bool kEnableSwizzleSFOutput, ...>` parameter substring.
STANDALONE_SWIZZLE_RE = re.compile(
    r"(?:multi_tensor_(?:un)?swizzle|(?:un)?swizzle)_(?:row|col)_scaling_kernel"
)


def dump_kernel_counts(prof, label: str) -> dict:
    print(f"\n=== {label} ===")
    counts: dict[str, int] = {}
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        counts[ev.name] = counts.get(ev.name, 0) + 1
    standalone_swizzle_total = 0
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        marker = ""
        if STANDALONE_SWIZZLE_RE.search(name):
            marker = "  <-- STANDALONE SWIZZLE PASS"
            standalone_swizzle_total += c
        # Truncate long mangled CUTLASS names for readability
        short = name if len(name) <= 110 else name[:107] + "..."
        print(f"  {c:4d}  {short}{marker}")
    print(f"  -- standalone swizzle kernel total: {standalone_swizzle_total}")
    return counts


def profile_path(optimize_for_gemm: bool, x: torch.Tensor, n_iters: int = 20):
    q = make_quantizer(optimize_for_gemm=optimize_for_gemm)
    # warm-up
    for _ in range(3):
        t = q(x)
        tex.swizzle_scales_for_gemm_(t)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iters):
            t = q(x)
            tex.swizzle_scales_for_gemm_(t)
        torch.cuda.synchronize()
    return prof


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = "cuda"
    # Shape that hits the production RHT cast-fusion fast-path
    # (rows % 64 == 0, cols % 128 == 0, BF16, SM100/110).
    M, N = 8192, 4096
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)

    print(f"Shape: M={M}, N={N}, dtype=bf16, RHT=True, post_rht_amax=True")
    print(f"iters: 20 (after 3 warm-up)")

    prof_baseline = profile_path(optimize_for_gemm=False, x=x)
    counts_baseline = dump_kernel_counts(
        prof_baseline, "BASELINE: optimize_for_gemm=False (separate swizzle kernel)"
    )

    prof_swf = profile_path(optimize_for_gemm=True, x=x)
    counts_swf = dump_kernel_counts(
        prof_swf, "SUT: optimize_for_gemm=True (quant emits swizzled SF directly)"
    )

    print("\n=== VERDICT ===")
    base_swizzle = sum(c for n, c in counts_baseline.items() if STANDALONE_SWIZZLE_RE.search(n))
    swf_swizzle = sum(c for n, c in counts_swf.items() if STANDALONE_SWIZZLE_RE.search(n))
    print(f"  baseline standalone swizzle kernel launches: {base_swizzle}")
    print(f"  SUT standalone swizzle kernel launches:      {swf_swizzle}")
    if swf_swizzle == 0 and base_swizzle > 0:
        print(
            "  PASS: standalone swizzle pass disappears from timeline under optimize_for_gemm=True"
        )
    else:
        print(
            "  FAIL: expected baseline > 0 and SUT == 0; check whether SUT actually "
            "set with_gemm_swizzled_scales=True on the output tensor"
        )


if __name__ == "__main__":
    main()
