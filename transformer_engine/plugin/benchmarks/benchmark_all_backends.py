#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
import os
import sys
import torch
import time
import numpy as np
from datetime import datetime
from typing import Dict, List


from transformer_engine.plugin.test_utils import get_available_backends, get_backend


class BenchmarkResult:
    def __init__(self, backend_name: str, operation_name: str, shape: tuple,
                 mean_time: float, std_time: float, min_time: float, max_time: float,
                 gflops: float = None, bandwidth: float = None):
        self.backend_name = backend_name
        self.operation_name = operation_name
        self.shape = shape
        self.mean_time = mean_time
        self.std_time = std_time
        self.min_time = min_time
        self.max_time = max_time
        self.gflops = gflops
        self.bandwidth = bandwidth

    def __str__(self):
        gflops_str = f"{self.gflops:.2f} GFLOPS" if self.gflops else "N/A"
        bandwidth_str = f"{self.bandwidth:.2f} GB/s" if self.bandwidth else "N/A"
        return (f"{self.backend_name:12s} {self.mean_time:8.4f}±{self.std_time:6.4f} ms  "
                f"[{self.min_time:7.4f}, {self.max_time:7.4f}]  "
                f"{gflops_str:15s} {bandwidth_str:12s}")


def time_operation(func, warmup_iters=10, benchmark_iters=100):
    for _ in range(warmup_iters):
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(benchmark_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        func()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def compute_gflops(operation: str, shape: tuple, time_ms: float) -> float:
    if operation in ['gelu', 'relu', 'silu']:
        flops = np.prod(shape) * 5
    elif operation == 'layernorm':
        total_elements = np.prod(shape)
        hidden_size = shape[-1]
        flops = total_elements * (3 + 2 * hidden_size)
    elif operation == 'rmsnorm':
        total_elements = np.prod(shape)
        hidden_size = shape[-1]
        flops = total_elements * (2 + hidden_size)
    elif operation == 'gemm':
        M, N, K = shape
        flops = 2 * M * N * K
    else:
        return None

    return (flops / 1e9) / (time_ms / 1000)


def compute_bandwidth(operation: str, shape: tuple, time_ms: float) -> float:
    bytes_per_element = 4

    if operation in ['gelu', 'relu', 'silu']:
        total_bytes = np.prod(shape) * 2 * bytes_per_element
    elif operation in ['layernorm', 'rmsnorm']:
        total_bytes = np.prod(shape) * 5 * bytes_per_element
    elif operation == 'gemm':
        M, N, K = shape
        total_bytes = (M*K + K*N + M*N) * bytes_per_element
    else:
        return None

    return (total_bytes / 1e9) / (time_ms / 1000)


def benchmark_activations(backends: List[str], shapes: List[tuple], device: str) -> List[BenchmarkResult]:
    print("\n" + "="*80)
    print("Activation Function Performance Test")
    print("="*80)

    results = []
    operations = [
        ('gelu', 'GELU'),
        ('relu', 'ReLU'),
        ('silu', 'SiLU'),
    ]

    for shape in shapes:
        print(f"\nShape: {shape}")
        x = torch.randn(shape, dtype=torch.float32, device=device)

        for op_method, op_name in operations:
            print(f"\n  {op_name}:")
            print(f"    {'Backend':<12s} {'Time (ms)':<20s} {'Range (ms)':<25s} {'GFLOPS':<15s} {'Bandwidth'}")
            print(f"    {'-'*85}")

            for backend_name in backends:
                backend = get_backend(backend_name)

                try:
                    func = lambda: getattr(backend, op_method)(x, None)
                    timing = time_operation(func)

                    gflops = compute_gflops(op_method, shape, timing['mean'])
                    bandwidth = compute_bandwidth(op_method, shape, timing['mean'])

                    result = BenchmarkResult(
                        backend_name, op_method, shape,
                        timing['mean'], timing['std'], timing['min'], timing['max'],
                        gflops, bandwidth
                    )
                    results.append(result)
                    print(f"    {result}")

                except Exception as e:
                    print(f"    {backend_name:12s} SKIPPED ({type(e).__name__}: {str(e)[:40]})")

    return results


def benchmark_normalization(backends: List[str], shapes: List[tuple], device: str) -> List[BenchmarkResult]:
    print("\n" + "="*80)
    print("Normalization Performance Test")
    print("="*80)

    results = []
    eps = 1e-5

    for shape in shapes:
        print(f"\nShape: {shape}")
        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=torch.float32, device=device)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=device)
        bias = torch.zeros(hidden_size, dtype=torch.float32, device=device)

        print(f"\n  LayerNorm forward:")
        print(f"    {'Backend':<12s} {'Time (ms)':<20s} {'Range (ms)':<25s} {'GFLOPS':<15s} {'Bandwidth'}")
        print(f"    {'-'*85}")

        for backend_name in backends:
            backend = get_backend(backend_name)

            try:
                func = lambda: backend.layernorm_fwd(x, weight, bias, eps, None, None, torch.float32, 0, False)
                timing = time_operation(func)

                gflops = compute_gflops('layernorm', shape, timing['mean'])
                bandwidth = compute_bandwidth('layernorm', shape, timing['mean'])

                result = BenchmarkResult(
                    backend_name, 'layernorm_fwd', shape,
                    timing['mean'], timing['std'], timing['min'], timing['max'],
                    gflops, bandwidth
                )
                results.append(result)
                print(f"    {result}")

            except Exception as e:
                print(f"    {backend_name:12s} SKIPPED ({type(e).__name__})")

        print(f"\n  RMSNorm forward:")
        print(f"    {'Backend':<12s} {'Time (ms)':<20s} {'Range (ms)':<25s} {'GFLOPS':<15s} {'Bandwidth'}")
        print(f"    {'-'*85}")

        for backend_name in backends:
            backend = get_backend(backend_name)

            try:
                func = lambda: backend.rmsnorm_fwd(x, weight, eps, None, None, torch.float32, 0, False)
                timing = time_operation(func)

                gflops = compute_gflops('rmsnorm', shape, timing['mean'])
                bandwidth = compute_bandwidth('rmsnorm', shape, timing['mean'])

                result = BenchmarkResult(
                    backend_name, 'rmsnorm_fwd', shape,
                    timing['mean'], timing['std'], timing['min'], timing['max'],
                    gflops, bandwidth
                )
                results.append(result)
                print(f"    {result}")

            except Exception as e:
                print(f"    {backend_name:12s} SKIPPED ({type(e).__name__})")

    return results


def benchmark_gemm(backends: List[str], configs: List[tuple], device: str) -> List[BenchmarkResult]:
    print("\n" + "="*80)
    print("GEMM Performance Test")
    print("="*80)

    results = []

    for M, N, K in configs:
        print(f"\nConfig: M={M}, N={N}, K={K}")
        print(f"  {'Backend':<12s} {'Time (ms)':<20s} {'Range (ms)':<25s} {'GFLOPS':<15s} {'Bandwidth'}")
        print(f"  {'-'*85}")

        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)
        D = torch.empty(M, N, dtype=torch.float32, device=device)
        workspace = torch.empty(1024, dtype=torch.uint8, device=device)

        for backend_name in backends:
            backend = get_backend(backend_name)

            try:
                func = lambda: backend.generic_gemm(
                    A, False, B, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )
                timing = time_operation(func)

                gflops = compute_gflops('gemm', (M, N, K), timing['mean'])
                bandwidth = compute_bandwidth('gemm', (M, N, K), timing['mean'])

                result = BenchmarkResult(
                    backend_name, 'gemm', (M, N, K),
                    timing['mean'], timing['std'], timing['min'], timing['max'],
                    gflops, bandwidth
                )
                results.append(result)
                print(f"  {result}")

            except Exception as e:
                print(f"  {backend_name:12s} SKIPPED ({type(e).__name__})")

    return results


def print_summary(all_results: List[BenchmarkResult]):
    print("\n" + "="*80)
    print("Performance Comparison Summary")
    print("="*80)

    from collections import defaultdict
    by_operation = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        by_operation[result.operation_name][result.backend_name].append(result)

    print("\nAverage Performance (all shapes):")
    print(f"{'Operation':<20s} {'Backend':<12s} {'Avg Time (ms)':<15s} {'Avg GFLOPS':<15s}")
    print("-"*65)

    for op_name, backends_data in sorted(by_operation.items()):
        for backend_name, results in sorted(backends_data.items()):
            avg_time = np.mean([r.mean_time for r in results])
            gflops_list = [r.gflops for r in results if r.gflops is not None]
            avg_gflops = np.mean(gflops_list) if gflops_list else None

            gflops_str = f"{avg_gflops:.2f}" if avg_gflops else "N/A"
            print(f"{op_name:<20s} {backend_name:<12s} {avg_time:<15.4f} {gflops_str:<15s}")

    print("\n" + "="*80)
    print("Fastest Backend (by operation)")
    print("="*80)

    for op_name, backends_data in sorted(by_operation.items()):
        backend_avg_times = {}
        for backend_name, results in backends_data.items():
            backend_avg_times[backend_name] = np.mean([r.mean_time for r in results])

        if backend_avg_times:
            fastest = min(backend_avg_times.items(), key=lambda x: x[1])
            print(f"{op_name:<20s} → {fastest[0]:<12s} ({fastest[1]:.4f} ms)")


def save_results_csv(results: List[BenchmarkResult], filename: str):
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Backend', 'Operation', 'Shape', 'Mean(ms)', 'Std(ms)',
            'Min(ms)', 'Max(ms)', 'GFLOPS', 'GB/s'
        ])

        for result in results:
            writer.writerow([
                result.backend_name,
                result.operation_name,
                str(result.shape),
                f"{result.mean_time:.4f}",
                f"{result.std_time:.4f}",
                f"{result.min_time:.4f}",
                f"{result.max_time:.4f}",
                f"{result.gflops:.2f}" if result.gflops else "N/A",
                f"{result.bandwidth:.2f}" if result.bandwidth else "N/A",
            ])

    print(f"\nResults saved to: {filename}")


def main():
    print("\n" + "="*80)
    print(" "*25 + "Multi-Backend Performance Comparison Test")
    print("="*80)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nDevice: CUDA - {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print(f"\nDevice: CPU")
    print(f"PyTorch version: {torch.__version__}")

    backends = get_available_backends()
    print(f"\nAvailable backends: {', '.join(backends)}")
    print(f"Total: {len(backends)} backends")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmark_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")

    activation_shapes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    normalization_shapes = [
        (8, 512, 768),
        (16, 512, 1024),
        (32, 512, 2048),
    ]

    gemm_configs = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    all_results = []

    results = benchmark_activations(backends, activation_shapes, device)
    all_results.extend(results)
    save_results_csv(results, f"{output_dir}/activations.csv")

    results = benchmark_normalization(backends, normalization_shapes, device)
    all_results.extend(results)
    save_results_csv(results, f"{output_dir}/normalization.csv")

    results = benchmark_gemm(backends, gemm_configs, device)
    all_results.extend(results)
    save_results_csv(results, f"{output_dir}/gemm.csv")

    print_summary(all_results)

    save_results_csv(all_results, f"{output_dir}/all_results.csv")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
