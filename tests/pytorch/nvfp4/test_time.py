#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import math
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

# ====== 与你提供的 SR 脚本一致的帮助函数（LUT、unpack、反量化） ======
def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated

_FP4_LUT = torch.tensor(
    [ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 ],
    dtype=torch.float32,
)

def fp4_to_fp32(fp4: torch.Tensor) -> torch.Tensor:
    fp4_lut = _FP4_LUT.to(fp4.device)
    return fp4_lut[fp4.to(torch.long)]

def dequantize_fp4(qx: torch.Tensor, sx: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    # sx 是 inv-scale 的 fp8 存储，这里和你脚本保持一致：还原 scale 再乘 LUT 值
    sf = sx.repeat_interleave(16, dim=1).view(torch.float8_e4m3fn).to(torch.float32)
    dqx = fp4_to_fp32(unpack_fp4(qx))
    sf = sf[: dqx.shape[0], : dqx.shape[1]]
    dequant = dqx * sf * (amax / (6.0 * 448))
    return dequant

# ===================== 基础计时工具 =====================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_ms(fn, repeat=1):
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    cuda_sync()
    return (time.perf_counter() - t0) * 1000.0 / max(repeat, 1)

# ===================== 量化与反量化（NVFP4Quantizer） =====================
def quantize_nvfp4_tensor(x: torch.Tensor, use_sr: bool, use_2d: bool, use_rht: bool):
    # 与你提供的量化构造保持一致
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=use_rht,
        with_post_rht_amax=True,
        stochastic_rounding=use_sr,
        with_2d_quantization=use_2d,
    )
    x_nvfp4 = q(x)  # NVFP4Tensor
    # 拆出 rowwise / columnwise 数据与 scale（与你脚本一致）
    assert x_nvfp4._rowwise_data is not None
    qx = x_nvfp4._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4._rowwise_scale_inv is not None
    sx = x_nvfp4._rowwise_scale_inv
    assert x_nvfp4._columnwise_data is not None
    qx_t = x_nvfp4._columnwise_data.view(dtype=torch.uint8)
    assert x_nvfp4._columnwise_scale_inv is not None
    sx_t = x_nvfp4._columnwise_scale_inv
    return qx, sx, qx_t, sx_t

def bench_qdq(M, N, dtype, device, iters, use_sr, use_2d, use_rht):
    # RHT 仅支持 bfloat16：不合法的组合直接“跳过”
    if use_rht and dtype != torch.bfloat16:
        return dict(quant_ms=float("nan"), dequant_ms=float("nan"), qdq_ms=float("nan"),
                    notes="skip: RHT requires bfloat16")

    torch.manual_seed(1234)
    x = torch.randn((M, N), device=device, dtype=dtype) * 2 - 1
    amax = torch.max(torch.abs(x)).float()

    # 预热一次（构建 quantizer 与第一次运行）
    qx, sx, qx_t, sx_t = quantize_nvfp4_tensor(x, use_sr, use_2d, use_rht)
    _ = dequantize_fp4(qx,   sx,   amax)
    _ = dequantize_fp4(qx_t, sx_t, amax)

    # 分开计时：Quant
    def _quant_only():
        nonlocal qx, sx, qx_t, sx_t
        qx, sx, qx_t, sx_t = quantize_nvfp4_tensor(x, use_sr, use_2d, use_rht)

    # 分开计时：Dequant（对 rowwise 与 columnwise 都做，和验证脚本一致）
    def _dequant_only():
        _ = dequantize_fp4(qx,   sx,   amax)
        _ = dequantize_fp4(qx_t, sx_t, amax)

    quant_ms   = timed_ms(_quant_only,   repeat=iters)
    dequant_ms = timed_ms(_dequant_only, repeat=iters)

    # 合并计时：QDQ（单轮里先 quant 再 dequant）
    def _qdq_once():
        qx2, sx2, qx_t2, sx_t2 = quantize_nvfp4_tensor(x, use_sr, use_2d, use_rht)
        _ = dequantize_fp4(qx2,   sx2,   amax)
        _ = dequantize_fp4(qx_t2, sx_t2, amax)

    qdq_ms = timed_ms(_qdq_once, repeat=iters)

    return dict(quant_ms=quant_ms, dequant_ms=dequant_ms, qdq_ms=qdq_ms, notes="")

# ===================== GEMM 对比 =====================
def bench_gemm(M, N, P, dtype, device, iters, warmup=5):
    torch.manual_seed(1234)
    A = torch.randn((M, N), device=device, dtype=dtype)
    B = torch.randn((N, P), device=device, dtype=dtype)
    # 预热
    for _ in range(warmup):
        _ = A @ B
    cuda_sync()
    # 计时
    return timed_ms(lambda: A.matmul(B), repeat=iters)

# ===================== 主过程 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--P", type=int, default=None, help="GEMM 的列数，默认等于 N")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    P = args.P if args.P is not None else args.N

    avail, reason = te.is_nvfp4_available(return_reason=True)
    print(f"# nvfp4_available={avail}  reason='{reason}'")
    print(f"# device={device}  dtype={dtype}  A=({args.M},{args.N})  B=({args.N},{P})  iters={args.iters}")

    # 先测 GEMM（多次求平均）
    _ = bench_gemm(args.M, args.N, P, dtype, device, iters=5, warmup=args.warmup)
    gemm_ms = bench_gemm(args.M, args.N, P, dtype, device, iters=args.iters, warmup=0)

    # 方案集合：vanilla / +SR / +RHT / +2D / 组合
    settings = [
        dict(name="vanilla",       sr=False, rht=False, use2d=False),
        dict(name="vanilla+SR",    sr=True,  rht=False, use2d=False),
        dict(name="RHT",           sr=False, rht=True,  use2d=False),
        dict(name="RHT+SR",        sr=True,  rht=True,  use2d=False),
        dict(name="2D",            sr=False, rht=False, use2d=True),
        dict(name="2D+SR",         sr=True,  rht=False, use2d=True),
        dict(name="RHT+2D",        sr=False, rht=True,  use2d=True),
        dict(name="RHT+2D+SR",     sr=True,  rht=True,  use2d=True),
    ]

    print("case,M,N,P,dtype,RHT,SR,Use2D,quant_ms,dequant_ms,qdq_ms,gemm_ms,qdq/gemm,notes")

    # 预热每个 setting（避免首次构建开销影响）
    for s in settings:
        _ = bench_qdq(args.M, args.N, dtype, device, iters=5, use_sr=s["sr"], use_2d=s["use2d"], use_rht=s["rht"])

    for s in settings:
        res = bench_qdq(args.M, args.N, dtype, device, iters=args.iters,
                        use_sr=s["sr"], use_2d=s["use2d"], use_rht=s["rht"])
        quant_ms   = res["quant_ms"]
        dequant_ms = res["dequant_ms"]
        qdq_ms     = res["qdq_ms"]
        note       = res["notes"]
        ratio = (qdq_ms / gemm_ms) if (not math.isnan(qdq_ms)) else float("nan")
        print(f"{s['name']},{args.M},{args.N},{P},{args.dtype},{s['rht']},{s['sr']},{s['use2d']},"
              f"{quant_ms:.4f},{dequant_ms:.4f},{qdq_ms:.4f},{gemm_ms:.4f},{ratio:.6f},{note}")

if __name__ == "__main__":
    main()
