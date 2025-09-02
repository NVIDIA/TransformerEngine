import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_model_init
from transformer_engine.common import recipe
import time
import csv

def run_once(in_features, out_features, batch_size, iters=500):
    
    model = te.Linear(in_features, out_features, bias=True, device="cuda").half()
    inp = torch.randn(batch_size, in_features, device="cuda").half()
    torch_model = torch.nn.Linear(in_features, out_features, bias=True).cuda().half()
    
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

   
    # Warm up for te model
    for _ in range(50):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = model(inp, is_first_microbatch=True)
        # out = model(inp, is_first_microbatch=True)
    # Warm up for torch model
    for _ in range(50):
        with torch.no_grad():
            out = torch_model(inp)
            torch.cuda.synchronize()

    
    times = []
    with torch.no_grad():
        start = time.time()
        for i in range(iters):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(inp, is_first_microbatch=False)
            # out = model(inp, is_first_microbatch=False)
        torch.cuda.synchronize()
        end = time.time()
        avg_time_te = (end - start) / iters

    start = time.time()
    torch_times = []
    for i in range(iters):
        with torch.no_grad():
            out = torch_model(inp)
    torch.cuda.synchronize()
    end = time.time()
    avg_time_torch = (end - start) / iters

    del model
    del inp
    del torch_model
    torch.cuda.empty_cache()

    return avg_time_te, avg_time_torch

if __name__ == "__main__":
    in_feature_list = [1280, 5120, 10240]
    out_features_list = [1280, 5120, 10240]
    batch_size_list = [512, 1024, 2048]

    # in_feature_list = [5120]
    # out_features_list = [5120]
    # batch_size_list = [1024]

    results = []
    for in_features in in_feature_list:
        for out_features in out_features_list:
            for batch_size in batch_size_list:
                print(f"=============Test in_features X out_features X batch_size {in_features}x{out_features}x{batch_size}")
                te_time, torch_time = run_once(in_features, out_features, batch_size)
                print(f"te linear average costime: {te_time*1000:.6f} ms")
                print(f"torch linear average costime: {torch_time*1000:.6f} ms")
                results.append([in_features, out_features, batch_size, te_time*1000, torch_time*1000])

   
    import csv
    import sys

    # Print CSV header and rows in a better format using csv.writer
    writer = csv.writer(sys.stdout)
    writer.writerow(["in_features", "out_features", "batch_size", "te_time(ms)", "torch_time(ms)"])
    for row in results:
        writer.writerow(row)

    # print out speedup over torch with problem shape
    for row in results:
        in_features, out_features, batch_size, te_time, torch_time = row
        print(f"Shape: in_features={in_features}, out_features={out_features}, batch_size={batch_size} | Speedup over torch: {torch_time/te_time}")