import transformer_engine.pytorch as te                                                                                                                                                                    
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer                                                                                                                                                      
import torch
import torch.cuda.nvtx as nvtx                                                                                                                                                                             
                
N = 7168                                                                                                                                                                                                   
num_experts = 64
ITERS = 50                                                                                                                                                                                                 
                
M_VALUES = [8192, 16384, 32768, 65536, 131072]                                                                                                                                                             
 
                                                                                                                                                                                                           
def make_unequal_splits(M, num_experts):
    base = M // num_experts
    splits = []                                                                                                                                                                                            
    for i in range(num_experts):
        if i % 2 == 0:                                                                                                                                                                                     
            splits.append(base - 128)
        else:
            splits.append(base + 128)                                                                                                                                                                      
    # fix rounding so sum == M
    diff = M - sum(splits)                                                                                                                                                                                 
    splits[-1] += diff                                                                                                                                                                                     
    return splits
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def bench(fn, label, iters=ITERS):
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
    if M % num_experts != 0 or (M // num_experts) <= 128:
        print(f"M={M}: skipped")                                                                                                                                                                           
        continue                                                                                                                                                                                           
                                                                                                                                                                                                           
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")                                                                                                                                             
    label_prefix = f"M{M}"
                                                                                                                                                                                                           
    print(f"\nM={M}:")                                                                                                                                                                                     
 
    # --- graph-safe, equal splits (O(1) division) ---                                                                                                                                                     
    equal_splits = [M // num_experts] * num_experts
    equal_tensor = torch.tensor(equal_splits, dtype=torch.int64, device="cuda")                                                                                                                            
    q_eq = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)
    q_eq.optimize_for_gemm = False                                                                                                                                                                         
    bench(                                                                                                                                                                                                 
        lambda: tex.group_quantize(x, q_eq, num_experts, equal_tensor),                                                                                                                                    
        f"{label_prefix}_graph_safe_equal_O1",                                                                                                                                                             
    )           
                                                                                                                                                                                                           
    # --- graph-safe, unequal splits (binary search) ---                                                                                                                                                   
    unequal_splits = make_unequal_splits(M, num_experts)
    unequal_tensor = torch.tensor(unequal_splits, dtype=torch.int64, device="cuda")                                                                                                                        
    q_uneq = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)
    q_uneq.optimize_for_gemm = False                                                                                                                                                                       
    bench(                                                                                                                                                                                                 
        lambda: tex.group_quantize(x, q_uneq, num_experts, unequal_tensor),                                                                                                                                
        f"{label_prefix}_graph_safe_unequal_bsearch",                                                                                                                                                      
    )           

    # --- non-graph-safe (linear scan) ---                                                                                                                                                                 
    q_list = [
        NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)                                                                                                              
        for _ in range(num_experts)                                                                                                                                                                        
    ]
    bench(                                                                                                                                                                                                 
        lambda: tex.split_quantize(x, equal_splits, q_list),
        f"{label_prefix}_non_graph_safe_linear",
    )                                                   
