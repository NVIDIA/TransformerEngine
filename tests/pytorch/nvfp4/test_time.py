#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import time
import torch
from torch.cuda import Event, nvtx

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType


# ===================== 计时工具 =====================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_timing_ms(fn, warmup=5, iters=50, label=""):
    # 预热
    for _ in range(warmup):
        fn()
    cuda_sync()

    start, end = Event(enable_timing=True), Event(enable_timing=True)
    total = 0.0
    for _ in range(iters):
        if label:
            nvtx.range_push(label)
        start.record()
        fn()
        end.record()
        end.synchronize()
        if label:
            nvtx.range_pop()
        total += start.elapsed_time(end)  # ms
    return total / max(1, iters)


# ===================== NVFP4 构造 =====================
def make_quantizer(stochastic_rounding: bool, with_rht: bool, with_2d: bool):
    # rowwise/columnwise 都开：以便 X/W 两条路径都能生成对应布局
    return NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,  # RHT 后做 amax 更合理
        stochastic_rounding=stochastic_rounding,
        with_2d_quantization=with_2d,
    )


def quantize_tensor(q: NVFP4Quantizer, x: torch.Tensor):
    """返回 NVFP4Tensor（包含 packed data 与 scale），用于后续 generic_gemm。"""
    # 这里走 allocate+update 两步，便于把构造开销与真正量化开销分离
    out = q.make_empty(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    return q.update_quantized(x, out)


# ===================== GEMM（NVFP4Tensor） =====================
def generic_gemm_nvfp4(x_nvfp4, w_nvfp4, out_dtype, accumulate=False, out_tensor=None):
    # tex.generic_gemm 接口（参考 TE 代码习惯）
    # 返回 (out, bias_grad, gelu_input, extra_output)
    workspace = torch.empty(4, dtype=torch.uint8, device=x_nvfp4.device)
    transa = True  # W layout: (N,K) rowwise => 以 A^T 形式传
    transb = False  # X layout: (M,K) rowwise
    out_quantizer = None
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False

    out = out_tensor if accumulate else None
    y = tex.generic_gemm(
        w_nvfp4,
        transa,
        x_nvfp4,
        transb,
        out,
        out_quantizer,
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        use_gelu,
        gelu_input,
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )[0]
    return y


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--in_features", type=int, default=4096)
    ap.add_argument("--out_features", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device(args.device)
    x_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out_dtype = torch.bfloat16  # 端到端 GEMM 输出 dtype，可改参数化
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    avail, reason = te.is_nvfp4_available(return_reason=True)
    print(f"# nvfp4_available={avail} reason='{reason}'")
    print(
        f"# device={device} x_dtype={x_dtype} out_dtype={out_dtype} "
        f"bz={args.batch} seq={args.seq} in={args.in_features} out={args.out_features} "
        f"iters={args.iters}"
    )

    B, S, K, N = args.batch, args.seq, args.in_features, args.out_features
    M = B * S  # 将 (B,S,K) 展平为 (M,K)，便于与 generic_gemm 对齐
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # 准备输入与权重（bf16/fp32）
    X = torch.randn((M, K), device=device, dtype=x_dtype)
    W = torch.randn((N, K), device=device, dtype=x_dtype)  # 与 tex.generic_gemm 期望一致：(N,K)

    # baseline：bf16 GEMM（无量化） —— 用于对比端到端加速/开销比例
    def _bf16_gemm():
        _ = X.to(torch.bfloat16) @ W.t().to(torch.bfloat16)

    bf16_ms = cuda_timing_ms(_bf16_gemm, warmup=args.warmup, iters=args.iters, label="bf16_gemm")

    settings = [
        dict(name="vanilla", sr=False, rht=False, use2d=False),
        dict(name="SR", sr=True, rht=False, use2d=False),
        dict(name="RHT", sr=False, rht=True, use2d=False),
        dict(name="RHT+SR", sr=True, rht=True, use2d=False),
        dict(name="2D", sr=False, rht=False, use2d=True),
        dict(name="2D+SR", sr=True, rht=False, use2d=True),
        dict(name="RHT+2D", sr=False, rht=True, use2d=True),
        dict(name="RHT+2D+SR", sr=True, rht=True, use2d=True),
    ]

    print(
        "setting,bz,seq,in,out,dtype,SR,RHT,Use2D,"
        "X_quant_ms,W_quant_cold_ms,W_quant_hot_ms,"
        "QGEMM_ms,bf16_gemm_ms,QGEMM/bf16,notes"
    )

    # 预热每个 setting（构建/首次内核开销）
    for s in settings:
        if s["rht"] and x_dtype != torch.bfloat16:
            continue
        qx = make_quantizer(stochastic_rounding=s["sr"], with_rht=s["rht"], with_2d=s["use2d"])
        qw = make_quantizer(stochastic_rounding=s["sr"], with_rht=False, with_2d=s["use2d"])
        # 预热：一次量化 + 一次端到端
        x_nv = quantize_tensor(qx, X)
        w_nv = quantize_tensor(qw, W)
        _ = generic_gemm_nvfp4(x_nv, w_nv, out_dtype, accumulate=False, out_tensor=None)
        cuda_sync()

    for s in settings:
        notes = []
        if s["rht"] and x_dtype != torch.bfloat16:
            # RHT 仅支持 bf16：跳过
            print(
                f"{s['name']},{B},{S},{K},{N},{args.dtype},{s['sr']},{s['rht']},{s['use2d']},"
                f"nan,nan,nan,nan,{bf16_ms:.4f},nan,skip:RHT requires bf16"
            )
            continue

        # 分别构造 X/W 量化器
        qx = make_quantizer(stochastic_rounding=s["sr"], with_rht=s["rht"], with_2d=s["use2d"])
        # 权重通常不开 RHT；是否 2D 由 use2d 控制
        qw = make_quantizer(stochastic_rounding=s["sr"], with_rht=False, with_2d=s["use2d"])

        # -------- X 量化（每步都需要）---------
        def _quant_x_once():
            _ = quantize_tensor(qx, X)

        x_quant_ms = cuda_timing_ms(
            _quant_x_once, warmup=args.warmup, iters=args.iters, label=f"{s['name']}_quantX"
        )

        # -------- W 量化（冷态 vs 热态）---------
        # 冷态：等价第一步或者权重更新后首次量化
        def _quant_w_cold_once():
            _ = quantize_tensor(qw, W)

        w_quant_cold_ms = cuda_timing_ms(
            _quant_w_cold_once,
            warmup=args.warmup,
            iters=args.iters,
            label=f"{s['name']}_quantW_cold",
        )

        # 热态：模拟缓存后重复量化（若实际训练复用缓存，此项可近似为 0）
        # 这里用相同 API 近似测“重复调用”的时间下界
        w_nv_cached = quantize_tensor(qw, W)  # 先做一次，作为缓存

        def _quant_w_hot_once():
            # 真实缓存是复用 NVFP4 权重；这里重新 update 作为“热路径近似”
            _ = qw.update_quantized(W, w_nv_cached)

        w_quant_hot_ms = cuda_timing_ms(
            _quant_w_hot_once, warmup=args.warmup, iters=args.iters, label=f"{s['name']}_quantW_hot"
        )

        # -------- 端到端（Q→GEMM）---------
        # 每次循环：量化 X（W 复用上面缓存），随后 generic_gemm
        def _qgemm_once():
            x_nv = quantize_tensor(qx, X)
            # 这里复用已量化的 w_nv_cached，符合“权重量化缓存”的真实训练形态
            _ = generic_gemm_nvfp4(x_nv, w_nv_cached, out_dtype, accumulate=False, out_tensor=None)

        qgemm_ms = cuda_timing_ms(
            _qgemm_once, warmup=args.warmup, iters=args.iters, label=f"{s['name']}_QGEMM"
        )

        ratio = qgemm_ms / bf16_ms if bf16_ms > 0 else float("nan")

        print(
            f"{s['name']},{B},{S},{K},{N},{args.dtype},{s['sr']},{s['rht']},{s['use2d']},"
            f"{x_quant_ms:.4f},{w_quant_cold_ms:.4f},{w_quant_hot_ms:.4f},"
            f"{qgemm_ms:.4f},{bf16_ms:.4f},{ratio:.6f},{'|'.join(notes)}"
        )


if __name__ == "__main__":
    main()
