.. Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. See LICENSE for license information.

MXFP8 row-to-column requantization comparison
============================================

Measured on September 30, 2026. The adapted common CuTeDSL kernel accepts
compact input scales, uses BF16 intermediates, and emits TE's optional rowwise
swizzled scales in the same launch. It is integrated into the common C++/TVM
FFI dispatcher and improves all nine measured shapes over the current grouped
CUDA kernel (1.44-2.27x). Large shapes achieve approximately 5.9-6.4 TB/s of
effective logical bandwidth. Nsight Compute measured 6.15 TB/s of physical
DRAM traffic on the largest shape.

Source and environment
----------------------

* TE: ``5505fbf033e110bbd0ce06832d3ca97c8d313e7a``. Both
  ``nvte_grouped_requantize`` (current grouped path) and
  ``nvte_group_requantize`` (older dense path) were measured.
* `Pinned cuDNN Frontend kernel
  <https://github.com/NVIDIA/cudnn-frontend/blob/347e186e661fef677f78b2a576c9a64cb2e0a8b2/python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_col_requant.py>`_:
  ``Mxfp8ColRequant.ws_kernel``. SHA256:
  ``70fea913ac9c6ffb1f709e30dd1e880c7af6ed30e9500db6836b71b99fe23fb7``.
  The linked develop source and installed cuDNN Frontend 1.30.0 copy matched
  byte for byte. The revision above was resolved and its file verified too.
* One NVIDIA GB200, compute capability 10.0, 152 SMs, approximately 185 GiB
  device memory, 1200 W power limit. Four GPUs exist on this host; timing used
  only physical GPU 0. Clock settings were not changed. NVLink connects the
  four GPUs (NV18); the benchmark performs no communication.
* Driver 580.178.04; CUDA toolkit 13.4.2; PyTorch
  ``2.15.0a0+875d815502.nvinternal.main`` (CUDA 13.4); cuDNN 9.27.0;
  CUTLASS DSL 4.8.0; TVM FFI 0.1.14.post1; Python 3.12 on aarch64.
* TE imported from this checkout and loaded
  ``/workspace/repos/TransformerEngine/libtransformer_engine.so``. The
  existing editable native build already contained the branch changes.
  Its configuration was Release, PyTorch, ``NVTE_CUDA_ARCHS=100a``,
  ``NVTE_WITH_CUTEDSL=OFF``, NCCL EP on. The initial benchmark invoked CuTeDSL directly.
  Integration was subsequently built with ``NVTE_WITH_CUTEDSL=1`` and the same
  architecture, framework and optional-feature configuration.
  CuTeDSL registration/dispatch code is already present on this branch;
  no main merge was needed.
* ``CUDA_VISIBLE_DEVICES=0``, ``CUDA_CACHE_DISABLE=1``,
  ``NVTE_FRAMEWORK=pytorch``, ``NVTE_USE_CCACHE=1``.
  The C API benchmark selects BF16 fast math explicitly. It does not use the
  Python ``NVTE_FUSED_GROUP_REQUANTIZE`` switch to choose its measured variants.

Functionality
-------------

All three implementations decode rowwise MXFP8, find a new maximum per 32
rows of each column, select a power-of-two E8M0 scale, and encode a new FP8
payload. This is requantization, not a lossless transpose of the original
quantized values. TE's columnwise payload remains in row-major memory order.

.. list-table:: Contracts
   :header-rows: 1
   :widths: 24 38 38

   * - Behavior
     - Branch grouped kernel
     - cuDNN kernel
   * - Input scales
     - Compact E8M0
     - Dispatch-pool GEMM-swizzled 128x4 atoms
   * - Output payload
     - Row-major with columnwise scaling
     - Same by default; optional global K-major transpose
   * - Output column scales
     - Compact or per-group GEMM-swizzled
     - Per-expert GEMM-swizzled; matches TE for aligned groups
   * - Rowwise scale output
     - Fused compact-to-swizzled conversion when requested
     - No rowwise scale output; benchmark conversion supplies it
   * - Intermediate
     - BF16 with fast math, otherwise FP32
     - BF16
   * - FP8 formats
     - E4M3 or E5M2 input, E4M3 output
     - E4M3 to E4M3 or E5M2 to E5M2
   * - Shapes
     - Four grouped shape representations, including varying hidden dimensions
     - Common hidden dimension; expert counts define padded row pools
   * - Alignment and padding
     - Every group's rows and columns divisible by 128
     - Hidden divisible by 128; token padding 128 or 256, valid counts may be unaligned
   * - Capacity tails
     - Leaves unused rows and scales untouched
     - Same without retired-row clearing; can optionally clear retired K-major rows
   * - Dequantized output
     - Older dense API can also emit BF16
     - Not provided
   * - Blackwell conversion
     - CUDA-version-gated packed scaled up-conversion with a fallback
     - Portable down-conversion already exists; scaled down-conversion is gated to sm_107a

The shared benchmark contract is E4M3 input/output, BF16 intermediate, common
hidden dimension, 128-aligned expert counts, row-major data, and swizzled output
scales in both directions. The input swizzle is timed every invocation and
restricted to live rows. It is not hidden in setup or credited as pre-existing
work. Neither path materializes a global BF16 intermediate. cuDNN's optional
K-major mode is recorded separately in the JSON and is not used in speedup
claims because it produces a different data layout.

The cuDNN source's portable path compiles and executes on SM100a with this
software stack, even though the file lives under a Rubin directory. This does
not validate older CUDA/CUTLASS releases or actual Rubin hardware. Its
up-conversion still uses the scaled instruction unconditionally, unlike TE's
CUDA-version-gated fallback.

Measurement and numerical reference
-----------------------------------

``benchmark_mxfp8_requantize.py`` calls the common C APIs with preallocated
buffers and descriptors. It passes the same non-default CUDA stream to C++,
CuTeDSL and CUDA graph capture. Five individual warm-up calls precede graph
capture, followed by three warm-up graph replays. Each timing sample is a
100-invocation CUDA graph measured with CUDA events; the table gives the median
of nine samples, in microseconds per invocation. Synchronization occurs before
measurement and at each ending event. No concurrent workload ran on GPU 0.

This measures steady-state GPU time. Allocation, Python tensor mutation,
host TMA descriptor construction, Python launch overhead and JIT compilation
are excluded. Per-shape CuTeDSL compilation took approximately 0.4 seconds per
variant and is recorded separately. The same inputs/outputs are reused between
invocations; smaller working sets can remain in L2. These are not rotating
buffer or whole-layer training timings.

Inputs are BF16 normal random values with independent powers of two in
[-8, 8] per rowwise 32-element block (seed 1234). Native PyTorch constructs the
rowwise E4M3/E8M0 input, independently decodes it to BF16, computes the per-group
columnwise quantization, and constructs the expected scale swizzle. Payload
bytes, both scale directions and capacity-tail sentinels are checked before
timing. This avoids comparing the two kernels only against each other.

Full-capacity results
---------------------

.. list-table:: Median microseconds per invocation
   :header-rows: 1

   * - Rows x hidden
     - Groups
     - TE grouped
     - TE dense
     - cuDNN raw
     - cuDNN + swizzle
     - Speedup over grouped
   * - 1,024 x 128
     - 8
     - 5.25
     - 2.42
     - 2.41
     - 3.34
     - 1.57x
   * - 1,024 x 256
     - 4
     - 5.22
     - 2.61
     - 3.30
     - 4.42
     - 1.18x
   * - 1,024 x 4,096
     - 8
     - 6.25
     - 4.33
     - 4.43
     - 5.95
     - 1.05x
   * - 4,096 x 4,096
     - 8
     - 11.16
     - 10.74
     - 7.15
     - 9.57
     - 1.17x
   * - 8,192 x 4,096
     - 64
     - 19.58
     - 20.30
     - 11.65
     - 14.96
     - 1.31x
   * - 32,768 x 4,096
     - 64
     - 70.49
     - 87.26
     - 46.06
     - 54.55
     - 1.29x
   * - 32,768 x 7,168
     - 256
     - 119.42
     - 163.89
     - 79.57
     - 94.64
     - 1.26x
   * - 65,536 x 8,192
     - 64
     - 259.09
     - 324.92
     - 169.46
     - 196.88
     - 1.32x
   * - 131,072 x 7,168
     - 256
     - 443.99
     - 631.63
     - 298.42
     - 354.98
     - 1.25x

Half-capacity, uneven groups
---------------------------

This includes many empty groups. Conversion of scales beyond the live row
bound is skipped, and both payload and scale tails are checked for writes.

.. list-table:: Median microseconds per invocation
   :header-rows: 1

   * - Capacity x hidden
     - Live rows
     - Groups
     - Empty groups
     - TE grouped
     - cuDNN + swizzle
     - Speedup
   * - 1,024 x 4,096
     - 512
     - 64
     - 60
     - 6.40
     - 4.97
     - 1.29x
   * - 32,768 x 4,096
     - 16,384
     - 64
     - 27
     - 38.53
     - 32.35
     - 1.19x
   * - 131,072 x 7,168
     - 65,536
     - 256
     - 123
     - 236.89
     - 193.56
     - 1.22x

Optimization experiments
-----------------------

* Swept hidden tiles 128/256/512, one/two pipeline stages and persistent grids
  from 152 to 2432 CTAs (capped at the actual number of tiles). Every candidate
  was validated. For 4096 x 4096, tile 128, one stage and grid 1024 improved
  cuDNN plus swizzle from 9.57 to 8.51 us. For 32768 x 4096, tile 256, two
  stages and grid 304 gave 53.05 us versus approximately 54.55 us at the
  upstream default. These are per-shape sweep winners, not a validated universal
  dispatch rule. Raw samples and all candidates are retained.
* A direct global-to-register compact-scale prototype fused rowwise swizzling
  but was slower: 169-231 us at 32768 x 4096, versus 70.7 us for TE grouped.
* A producer-warp prototype preserved cuDNN's TMA pipeline and converted
  compact scales into its shared-memory scale layout while emitting rowwise
  scales. It was also slower: 219.4 us at 32768 x 4096. Both prototypes are
  isolated in ``requantize_mxfp8_experimental.py`` behind
  ``--compact-experiment``. Neither is proposed for production use.

Integrated compact-input kernel
-------------------------------

The licensed port lives in
``transformer_engine/common/CuTeDSL/cast/mxfp8/requantize_mxfp8.py``. It has no
runtime dependency on private cuDNN Python modules or on either framework.
The original BF16 register arithmetic is retained; TMA directly loads compact
scales, avoiding a producer-side software swizzle. Four-byte scale packs form
the optional rowwise GEMM atoms in shared memory and are bulk-stored alongside
the columnwise output. Redundant valid-row metadata and masked arithmetic were
removed because the TE contract requires every group to be 128-aligned.

C++ caches the compiled TVM FFI callable and its failures, so compilation and
Python registration happen only on the first specialization. With
``NVTE_ENABLE_CUTEDSL_BACKEND=1``, the grouped C API selects it for BF16 fast math,
E4M3/E5M2 input, E4M3 output, constant hidden dimensions, and compact or swizzled
output scales. Rowwise data remains an optional alias. Uniform groups need no
GPU offsets; varying row counts use device offsets without CPU synchronization.
Empty groups and unused capacity are supported. FP32 math, varying hidden
sizes, misaligned scale pointers and unsupported TMA strides retain CUDA
fallbacks. Wide hidden dimensions must be divisible by 512; 128/256/384 use
full-width compact scale bulk loads. Compilation failure also falls back.
Scaled decode was validated on Blackwell with CUDA 13.4; older toolchains
without the conversion instruction fall back. Rubin hardware was unavailable.

.. list-table:: Full-capacity medians in microseconds, with both TE scale outputs
   :header-rows: 1

   * - Rows x hidden / groups
     - Branch grouped CUDA
     - Adapted CuTeDSL
     - Speedup
     - Effective TB/s
   * - 1024 x 128 / 8
     - 5.25
     - 2.31
     - 2.27x
     - 0.12
   * - 1024 x 256 / 4
     - 5.21
     - 3.12
     - 1.67x
     - 0.18
   * - 1024 x 4096 / 8
     - 6.22
     - 3.49
     - 1.78x
     - 2.51
   * - 4096 x 4096 / 8
     - 11.15
     - 7.29
     - 1.53x
     - 4.82
   * - 8192 x 4096 / 64
     - 19.56
     - 12.14
     - 1.61x
     - 5.79
   * - 32768 x 4096 / 64
     - 70.65
     - 47.71
     - 1.48x
     - 5.89
   * - 32768 x 7168 / 256
     - 119.61
     - 81.44
     - 1.47x
     - 6.04
   * - 65536 x 8192 / 64
     - 259.11
     - 175.32
     - 1.48x
     - 6.41
   * - 131072 x 7168 / 256
     - 444.41
     - 308.46
     - 1.44x
     - 6.38

Effective bandwidth counts FP8 payload read/write plus compact input scales,
rowwise output scales and columnwise output scales:
``live_rows * hidden * (2 + 3/32) / time``. It is decimal TB/s, with no
high-precision global intermediate, and excludes group metadata and redundant
TMA transactions. It is distinct from physical DRAM counters.

The cached C++ dispatch was separately profiled and confirmed to launch exactly
one ``GroupedRequantize.ws_kernel``. At 32768 x 4096 it measured 47.64 us versus
71.15 us for CUDA; at 131072 x 7168, 306.41 us versus 444.42 us. The direct full
sweep above uses CUDA events around graph replay (100 launches, 5 warmups,
3 warmup replays, 9 measured replays; median per-launch batch time). Compile
latency is recorded separately. Nsight Systems graph-node traces independently
measured exclusive median kernel durations of 47.584 and 308.448 us, respectively.
Those medians include warmup and measured launches but exclude gaps between
nodes; the event medians include those gaps.

A controlled output ablation explains the difference from raw cuDNN:

.. list-table:: Same-run medians, microseconds
   :header-rows: 1

   * - Rows x hidden
     - Original cuDNN, columnwise output
     - Port, columnwise output
     - Port, both TE scale outputs
   * - 32768 x 4096
     - 46.44
     - 46.63
     - 47.72
   * - 65536 x 8192
     - 169.16
     - 170.17
     - 174.53
   * - 131072 x 7168
     - 298.43
     - 298.68
     - 306.40

At the largest size, compact-scale adaptation adds only 0.09% versus raw
cuDNN; the extra rowwise scale output adds 7.72 us (2.58%). Both outputs together
cost 2.67% more than raw cuDNN. The added output writes ``M*N/32`` bytes and
requires a shared-memory pack and bulk stores, explaining the small overhead.
Raw cuDNN omits that requested TE output and also requires pre-swizzled input.
The original plus input swizzle remains slower than the adapted single launch.

Nsight Compute measured 969,074,176 DRAM read bytes and 946,233,856 DRAM write
bytes over 311,232 ns: 6.154 TB/s. Input generation matches the main benchmark
(seed 1234, normal BF16 activations, independent blockwise powers from -8 to 8,
rowwise E4M3 quantization). This is one profiled warmed launch, with Nsight
Compute's metric replay/cache behavior; physical counters can differ from
logical bytes because of caches, compression and memory writeback behavior.

Integration validation: 55 direct independent-reference/capacity cases,
52 focused PyTorch tests with the backend enabled, and 146 common C++ tests
with CuTeDSL enabled passed. Compute Sanitizer passed 13 boundary/extreme cases
with zero errors. Pre-commit and focused common Python/C++ lint pass.
The full PyTorch Python lint launcher reports an existing import-order issue
in ``transformer_engine/pytorch/attention/fused_mla_q_uproj.py:19``. The full
license launcher fails only on pre-existing untracked ``nccl_ep/include``
headers; all contributed files pass the focused license check. JAX does not
currently call this grouped C API path. The initial comparison below remains
useful as the record of the upstream layouts and prototype exploration.

Reproduce the integrated measurements from this checkout::

    NVTE_WITH_CUTEDSL=1 NVTE_CUDA_ARCHS=100a NVTE_FRAMEWORK=pytorch NVTE_BUILD_MAX_JOBS=72 NVTE_BUILD_THREADS_PER_JOB=2 python -m pip install -e . -v --no-build-isolation
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --adapted --shapes 1024,128,8 1024,256,4 1024,4096,8 4096,4096,8 8192,4096,64 32768,4096,64 32768,7168,256 65536,8192,64 131072,7168,256 --iterations 100 --repeats 9 --output /tmp/requantize_adapted_full.json
    NVTE_ENABLE_CUTEDSL_BACKEND=0 CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --cutedsl-dispatch --profile --shapes 1024,128,8 4096,4096,8 32768,4096,64 131072,7168,256 --iterations 100 --repeats 9 --output /tmp/requantize_dispatch.json
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --adapted --adapted-column-only --shapes 4096,4096,8 32768,4096,64 65536,8192,64 131072,7168,256 --iterations 100 --repeats 9 --output /tmp/requantize_adapted_overhead.json
    CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true --output=/tmp/requantize_adapted_nsys python benchmarks/benchmark_mxfp8_requantize.py --adapted --shapes 32768,4096,64 131072,7168,256 --iterations 50 --repeats 3 --output /tmp/requantize_adapted_nsys.json
    nsys stats --report cuda_gpu_kern_sum --format csv /tmp/requantize_adapted_nsys.nsys-rep
    CUDA_VISIBLE_DEVICES=0 ncu --profile-from-start off --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --csv --log-file /tmp/requantize_ncu.csv python benchmarks/profile_mxfp8_requantize.py --rows 131072 --hidden 7168 --groups 256
    CUDA_VISIBLE_DEVICES=2 python -m pytest -q tests/pytorch/mxfp8/test_mxfp8_requantize_cutedsl.py
    NVTE_ENABLE_CUTEDSL_BACKEND=1 CUDA_VISIBLE_DEVICES=3 python -m pytest -q tests/pytorch/mxfp8/test_mxfp8_group_quantize_graph_safe.py -k prequantized
    NVTE_ENABLE_CUTEDSL_BACKEND=1 CUDA_VISIBLE_DEVICES=2 tests/cpp/build/operator/test_operator --gtest_filter='*Requantize*'

Initial comparison validation and reproduction
----------------------------------------------

* Existing PyTorch requantization tests: 52 passed.
* Common C++ requantization tests: 146 passed across both APIs, four grouped
  shape representations, E4M3/E5M2 input and FP32/BF16 intermediates.
* Benchmark reference checks passed for all nine full-capacity shapes, three
  half-capacity shapes, and all tuned candidates. Separate checks cover
  interior empty groups, all-zero inputs, signed zeros, positive/negative NaNs,
  E8M0 codes 0/1/2/100/127/245/254/255, and scales mixed within a column block.
  Extreme-value results matched byte for byte; their timings are not quoted.
* GPU profiling confirmed ``group_requantize_mxfp8_kernel`` for TE and
  ``Mxfp8ColRequant.ws_kernel`` for the external variants.
* The first memory-check run reported handled CUDA API version probes
  (Python bindings request 13041; driver advertises 13040), rather than kernel
  memory faults. A separate memory check excludes API-probe reporting and
  finishes with zero errors; its log and result are listed below.
* Pre-commit, Black formatting, Python compilation and the benchmark-directory
  license check pass. The repository-wide license check fails on the pre-existing
  untracked ``nccl_ep/include/`` headers; the added Python files pass it.

Run from the repository root with the current editable build, CUTLASS DSL,
CUDA Python, Triton and a cuDNN Frontend package containing this kernel::

    curl -L --fail https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/347e186e661fef677f78b2a576c9a64cb2e0a8b2/python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_col_requant.py -o /tmp/glu_mxfp8_col_requant_pinned.py
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,128,8 1024,256,4 1024,4096,8 4096,4096,8 8192,4096,64 32768,4096,64 32768,7168,256 65536,8192,64 131072,7168,256 --iterations 100 --repeats 9 --cudnn-source /tmp/glu_mxfp8_col_requant_pinned.py --output /tmp/requantize_baseline.json
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,4096,64 32768,4096,64 131072,7168,256 --occupancy 0.5 --imbalance uneven --iterations 100 --repeats 9 --cudnn-source /tmp/glu_mxfp8_col_requant_pinned.py --output /tmp/requantize_capacity_final.json
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,256,4 4096,4096,8 32768,4096,64 --tune --iterations 100 --repeats 7 --output /tmp/requantize_tuning.json
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,256,4 4096,4096,8 32768,4096,64 --compact-experiment --iterations 100 --repeats 7 --output /tmp/requantize_tma_experiment.json
    CUDA_VISIBLE_DEVICES=3 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,256,8 --splits 0,128,0,256,128,0,512,0 --pattern special --profile --iterations 20 --repeats 3 --output /tmp/requantize_special_final.json
    CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,256,8 --splits 0,128,0,256,128,0,512,0 --pattern special_mixed --profile --iterations 20 --repeats 3 --output /tmp/requantize_special_mixed.json
    CUDA_VISIBLE_DEVICES=2 compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 99 python benchmarks/benchmark_mxfp8_requantize.py --shapes 1024,256,8 --splits 0,128,0,256,128,0,512,0 --pattern zeros --iterations 2 --repeats 1 --output /tmp/requantize_zeros_memcheck.json
    NVTE_FLASH_ATTN_V2=1 NVTE_FLASH_ATTN_V3=0 NVTE_FLASH_ATTN_V4=0 python -m pytest -q tests/pytorch/mxfp8/test_mxfp8_group_quantize_graph_safe.py -k requantize
    cmake -GNinja -S tests/cpp -B tests/cpp/build -DCMAKE_CUDA_ARCHITECTURES=100a
    cmake --build tests/cpp/build --target test_operator -j48
    CUDA_VISIBLE_DEVICES=1 tests/cpp/build/operator/test_operator --gtest_filter='*Requantize*:*Requantization*'

All numeric samples, compile times, source hashes, kernel names, exact split
sizes and variant configurations are saved in ``mxfp8_requantize_results.json``.
Local logs are ``/tmp/requantize_cpp_test.log``, ``/tmp/requantize_pytest.log``,
``/tmp/requantize_memcheck_final.log`` and ``/tmp/requantize_license.log``.
