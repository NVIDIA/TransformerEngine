import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

M, N = 8192, 7168  # your actual shape
x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
split_sections = torch.tensor([128] * (M // 128), dtype=torch.int64, device="cuda")

for optimize_for_gemm in [False, True]:
    q = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)
    q.optimize_for_gemm = optimize_for_gemm

    # warmup
    for _ in range(10):
        tex.group_quantize(x, q, split_sections.shape[0], split_sections)
    torch.cuda.synchronize()

    # time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        tex.group_quantize(x, q, split_sections.shape[0], split_sections)
    end.record()
    torch.cuda.synchronize()
    print(f"optimize_for_gemm={optimize_for_gemm}: {start.elapsed_time(end) / 100 * 1000:.1f} μs")
