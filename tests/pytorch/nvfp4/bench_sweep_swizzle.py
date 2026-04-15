import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
import torch
import torch.cuda.nvtx as nvtx                                                                                                                                                                             
 
N = 7168                                                                                                                                                                                                   
num_experts = 64
ITERS = 50

M_VALUES = [8192, 16384, 32768, 65536, 131072]

                                                                                                                                                                                                           
def bench(fn, label, iters=ITERS):
    # warmup                                                                                                                                                                                               
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nvtx.range_push(label)                                                                                                                                                                                 
    start.record()
    for _ in range(iters):                                                                                                                                                                                 
        fn()    
    end.record()
    nvtx.range_pop()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / iters * 1000
    print(f"  {label}: {us:.1f} us")                                                                                                                                                                       
    return us
                                                                                                                                                                                                           
                
print(f"N={N}, num_experts={num_experts}")
print("-" * 60)

for M in M_VALUES:                                                                                                                                                                                         
    if M % num_experts != 0:
        print(f"M={M}: skipped (not divisible by num_experts={num_experts})")                                                                                                                              
        continue

    rows_per_expert = M // num_experts                                                                                                                                                                     
    split_sections = [rows_per_expert] * num_experts
    split_section_tensor = torch.tensor(split_sections, dtype=torch.int64, device="cuda")                                                                                                                  
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
                                                                                                                                                                                                           
    print(f"\nM={M} ({rows_per_expert} rows/expert):")
                                                                                                                                                                                                           
    label_prefix = f"M{M}"

    # --- graph-safe, swizzle ON ---
    q_on = NVFP4Quantizer(
        rowwise=True,                                                                                                                                                                                      
        columnwise=True,
        with_rht=True,                                                                                                                                                                                     
        with_post_rht_amax=True,
    )
    q_on.optimize_for_gemm = True
    bench(                                                                                                                                                                                                 
        lambda: tex.group_quantize(x, q_on, num_experts, split_section_tensor),
        f"{label_prefix}_graph_safe_swizzle_ON",                                                                                                                                                           
    )           

    # --- graph-safe, swizzle OFF ---                                                                                                                                                                      
    q_off = NVFP4Quantizer(
        rowwise=True,                                                                                                                                                                                      
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
    )
    q_off.optimize_for_gemm = False
    bench(                                                                                                                                                                                                 
        lambda: tex.group_quantize(x, q_off, num_experts, split_section_tensor),
        f"{label_prefix}_graph_safe_swizzle_OFF",                                                                                                                                                          
    )           

    # --- non-graph-safe ---                                                                                                                                                                               
    q_list = [
        NVFP4Quantizer(                                                                                                                                                                                    
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
        )
        for _ in range(num_experts)
    ]                                                                                                                                                                                                      
    bench(
        lambda: tex.split_quantize(x, split_sections, q_list),                                                                                                                                             
        f"{label_prefix}_non_graph_safe",
    )

