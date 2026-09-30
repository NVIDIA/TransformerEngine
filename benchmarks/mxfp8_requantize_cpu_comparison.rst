.. Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. See LICENSE for license information.

MXFP8 requantization CPU dispatch overhead
=========================================

Measured on September 30, 2026 against commit
``3afc38835c8048a01afbb71577fcc4c5d98e0b57``. CuTeDSL has modestly higher CPU
launch cost than the branch's native CUDA kernel: approximately 0.6-0.8 us
per preallocated common C API call when enqueuing bounded launch batches.

One NVIDIA GB200 (physical GPU 0, SM100); main host thread pinned to CPU 0
on ARM Neoverse-V2 (144 available cores, CPU max 3.384 GHz, boost disabled).
Driver 580.178.04, CUDA 13.4.2, PyTorch
``2.15.0a0+875d815502.nvinternal.main``, cuDNN 9.27.0, CUTLASS DSL 4.8.0,
TVM FFI 0.1.14.post1, Python 3.12, aarch64. TE is the current editable
checkout with ``NVTE_WITH_CUTEDSL=1`` and architecture ``100a``. A runtime
backend toggle selects native CUDA or cached CuTeDSL through the same public
``nvte_grouped_requantize`` C API.

Source inputs are normal BF16 activations (seed 1234), independently quantized
to compact rowwise E4M3. Groups are uniform and 128-aligned. Both backends use
BF16 intermediates and produce E4M3 data and both GEMM-swizzled scale layouts.
All outputs match exactly through the C API and PyTorch binding.

The helper times each C API call inside C++, excluding Python and ctypes.
Inputs, outputs and descriptors are preallocated. First use/JIT and 50 warmup
launches per backend are excluded. The statistic is the median of nine means,
each containing 1000 calls; backend order alternates every repetition.

Each call records CPU-side elapsed time with ``steady_clock`` and thread CPU
work with ``CLOCK_THREAD_CPUTIME_ID``. The timer-only wall overhead is about
0.03 us and is included in the table; calibrated thread CPU measurements
agree with the batched wall times within approximately 0.1 us. CPU clock
measurement overhead (about 0.28-0.30 us) is separately recorded in the raw
results. No GPU completion wait is inside any timed interval.

Three C API modes clarify the launch state:

* Bounded asynchronous enqueue: synchronize before each batch of up to 64
  launches, outside timing. This avoids a long queue forcing the host to wait
  for GPU execution while retaining ordinary steady-state enqueue behavior.
* Drained stream: synchronize before each call, outside timing. This measures
  dispatch to an idle stream, including its CUDA/driver submission behavior.
* Graph capture: time adding each node to a graph; kernels do not execute.
  Graph creation, end-capture and destruction are outside timing. This is
  capture overhead, not graph replay time.

.. list-table:: CPU-side elapsed time per C API call, microseconds
   :header-rows: 1

   * - Rows x hidden / groups
     - Native, batched
     - CuTeDSL, batched
     - Native, drained
     - CuTeDSL, drained
     - Native, capture
     - CuTeDSL, capture
   * - 1024 x 128 / 8
     - 2.42
     - 3.12
     - 2.59
     - 4.03
     - 1.30
     - 1.70
   * - 1024 x 256 / 4
     - 2.44
     - 3.13
     - 2.61
     - 3.89
     - 1.33
     - 1.75
   * - 1024 x 4096 / 8
     - 2.41
     - 3.05
     - 2.62
     - 3.97
     - 1.33
     - 1.74
   * - 4096 x 4096 / 8
     - 2.40
     - 3.02
     - 2.84
     - 4.06
     - 1.37
     - 1.79
   * - 32768 x 4096 / 64
     - 2.41
     - 3.09
     - 3.02
     - 4.63
     - 1.35
     - 1.84
   * - 131072 x 7168 / 256
     - 2.50
     - 3.27
     - 3.84
     - 5.38
     - 1.38
     - 1.95

The public PyTorch binding additionally allocates three output buffers,
constructs grouped descriptors and updates Python-visible tensor fields. Its
CPU timing includes all of that work and the Python-to-binding call. Restoring
the compact input fields is outside timing; the stream is synchronized before
each call. Explicit cached GPU offsets avoid an extra prefix-sum kernel. A
prequantized ``GroupedTensor`` is constructed directly rather than calling the
separate group-quantize API, whose descriptor limit is irrelevant here.

.. list-table:: CPU-side elapsed time per PyTorch binding call, microseconds
   :header-rows: 1

   * - Rows x hidden / groups
     - Native
     - CuTeDSL
     - Added cost
   * - 1024 x 128 / 8
     - 16.24
     - 19.83
     - 3.59
   * - 1024 x 256 / 4
     - 16.98
     - 20.48
     - 3.50
   * - 1024 x 4096 / 8
     - 16.59
     - 20.90
     - 4.31
   * - 4096 x 4096 / 8
     - 17.02
     - 20.59
     - 3.57
   * - 32768 x 4096 / 64
     - 17.80
     - 21.54
     - 3.74
   * - 131072 x 7168 / 256
     - 20.34
     - 23.74
     - 3.41

These measurements do not isolate each internal cost within the CuTeDSL
adapter. They quantify the complete cached dispatch path, including its
config-key/cache lookup, tensor-view construction, TVM FFI argument handling
and CUDA launch. They should not be interpreted as pure TVM FFI crossing cost.
The largest shapes retain their much larger GPU time savings; for tiny tensors,
the added CPU time can consume a meaningful fraction of the GPU savings.
Captured graph replay avoids re-entering per-op C++/FFI dispatch on each replay.

First CuTeDSL use plus synchronization took roughly 0.49-0.55 seconds per
specialization in this process; it is recorded separately and excluded from
steady-state timings. The warm cache does not re-enter Python compilation.
These timings are from CPU clocks, not GPU events or Nsight kernel durations.

Reproduce from the repository root with the validated editable build::

    g++ -O3 -shared -fPIC -std=c++17 benchmarks/mxfp8_requantize_cpu.cpp -Itransformer_engine/common/include -I/usr/local/cuda/include -L/workspace/repos/TransformerEngine -ltransformer_engine -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/workspace/repos/TransformerEngine -Wl,-rpath,/usr/local/cuda/lib64 -o /tmp/mxfp8_requantize_cpu.so
    CUDA_VISIBLE_DEVICES=0 NVTE_ENABLE_CUTEDSL_BACKEND=0 python benchmarks/benchmark_mxfp8_requantize_cpu.py --pytorch --shapes 1024,128,8 1024,256,4 1024,4096,8 4096,4096,8 32768,4096,64 131072,7168,256 --iterations 1000 --repeats 9 --warmup 50 --output /tmp/requantize_cpu_final.json

Full batch statistics, p95 latency, timer calibration, software metadata and
first-call times are in ``mxfp8_requantize_cpu_results.json``. The host timing
helper is built with optimization and calls the current common shared library.
