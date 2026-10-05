# Transformer Engine v2.20.2 Release Notes

## Key Features and Enhancements

- [Common, JAX, PyTorch] Added an experimental CuTeDSL backend for MXFP8 quantization, enabled at build time with `NVTE_WITH_CUTEDSL=1` and selected at runtime with `NVTE_ENABLE_CUTEDSL_BACKEND=1`, with automatic fallback to CUDA C++ kernels. ([#3137](https://github.com/NVIDIA/TransformerEngine/pull/3137))
- [Common, PyTorch] Added fused grouped MXFP8 paths for weighted SwiGLU and clamped-SwiGLU activation with columnwise quantization and for in-place requantization. ([#3315](https://github.com/NVIDIA/TransformerEngine/pull/3315)) ([#3359](https://github.com/NVIDIA/TransformerEngine/pull/3359))
- [Common] Improved low-precision quantization performance with wider NVFP4 stochastic-rounding instructions, columnwise MXFP8 dBias reduction, register-resident MXFP8 cast kernels, and removal of an unused MXFP8 tile fetch. ([#3357](https://github.com/NVIDIA/TransformerEngine/pull/3357)) ([#3439](https://github.com/NVIDIA/TransformerEngine/pull/3439)) ([#3459](https://github.com/NVIDIA/TransformerEngine/pull/3459)) ([#3500](https://github.com/NVIDIA/TransformerEngine/pull/3500))
- [Common, PyTorch] Added opt-in Quantile Balancing histogram accumulation to the fused sigmoid router, including two-kernel and fused-atomic modes. ([#3395](https://github.com/NVIDIA/TransformerEngine/pull/3395))
- [Common, PyTorch] Added native, row-scaled, and feature-gated fused grouped-MLP support for the SiTU-GLU activation. ([#3402](https://github.com/NVIDIA/TransformerEngine/pull/3402))
- [PyTorch] Enabled `torch.compile(fullgraph=True)` for `Linear` with supported FP8, MXFP8, and NVFP4 recipes and adopted current opaque custom-class registration where available. ([#3053](https://github.com/NVIDIA/TransformerEngine/pull/3053)) ([#3495](https://github.com/NVIDIA/TransformerEngine/pull/3495))
- [PyTorch] Added per-sequence mask, window, and diagonal-alignment policies to packed THD attention, with automatic padded or grouped dispatch. ([#3274](https://github.com/NVIDIA/TransformerEngine/pull/3274))
- [Common, PyTorch] Added a CUDA graph-capturable fused Expert Parallelism prepare-and-dispatch path and reduced eager dispatch/combine CPU overhead. ([#3282](https://github.com/NVIDIA/TransformerEngine/pull/3282)) ([#3380](https://github.com/NVIDIA/TransformerEngine/pull/3380))
- [PyTorch] Added `NVTE_PLUGIN` support for loading alternate PyTorch backend implementations and overriding FlashAttention and attention-backend selection. ([#3401](https://github.com/NVIDIA/TransformerEngine/pull/3401))
- [PyTorch] Reduced grouped-layer CPU overhead by avoiding redundant quantizer validation, reusing grouped-MLP fusion plans, and eliminating repeated device discovery. ([#3405](https://github.com/NVIDIA/TransformerEngine/pull/3405)) ([#3410](https://github.com/NVIDIA/TransformerEngine/pull/3410))
- [PyTorch] Added paged-stashing metadata for saved `GroupedLinear` activation storage, including unquantized BF16 and FP16 rowwise tensors. ([#3423](https://github.com/NVIDIA/TransformerEngine/pull/3423))
- [PyTorch] Reduced CUDA graph memory retention by releasing warmup and capture outputs when consumed and clearing per-callable replay state on reset. ([#3427](https://github.com/NVIDIA/TransformerEngine/pull/3427))
- [PyTorch] Added `ScaledTanhSReLU`, a tanh-soft-clamped squared ReLU with per-row scaling and feature-gated cuDNN fused grouped-MLP support. ([#3463](https://github.com/NVIDIA/TransformerEngine/pull/3463))
- [PyTorch] Added experimental cuDNN-backed `GatedDeltaProductAttention`. ([#3539](https://github.com/NVIDIA/TransformerEngine/pull/3539))
- [JAX] Added Hopper BF16 grouped GEMM support. ([#3083](https://github.com/NVIDIA/TransformerEngine/pull/3083))
- [JAX] Added `sqrtsoftplus` scoring to fused MoE routing. ([#3448](https://github.com/NVIDIA/TransformerEngine/pull/3448))
- [JAX] Enabled Expert Parallelism to borrow XLA's NCCL communicator when supported, added single-process multi-device execution, and defaulted bootstrap to the AllGather-scan metadata path with fused scan-dispatch support. ([#3452](https://github.com/NVIDIA/TransformerEngine/pull/3452)) ([#3522](https://github.com/NVIDIA/TransformerEngine/pull/3522)) ([#3561](https://github.com/NVIDIA/TransformerEngine/pull/3561))
- [Build] Improved source-build dependency discovery for CUDA, NVCC, headers, and libraries installed through Python packages. ([#3251](https://github.com/NVIDIA/TransformerEngine/pull/3251))
- [Build] Reduced highly parallel source-build time by splitting grouped activation CUDA translation units. ([#3430](https://github.com/NVIDIA/TransformerEngine/pull/3430))

## Fixed Issues

- [Common] Fixed illegal memory accesses and incorrect results in TMA-based quantization by preserving shared-memory pointer provenance and correcting Grouped MXFP8 work mapping and descriptor synchronization. ([#3482](https://github.com/NVIDIA/TransformerEngine/pull/3482)) ([#3483](https://github.com/NVIDIA/TransformerEngine/pull/3483))
- [Common] Fixed BF16 GEMM precision loss on L40 by excluding cuBLASLt split-K algorithms that store partial reductions in BF16. ([#3440](https://github.com/NVIDIA/TransformerEngine/pull/3440))
- [Common, JAX] Fixed race conditions between zeroing and routed gradient stores in Triton permutation kernels. ([#3581](https://github.com/NVIDIA/TransformerEngine/pull/3581))
- [PyTorch] Fixed fused grouped-MLP correctness for E5M2 backward gradients, padded single-group SReLU inputs, and unsupported GEGLU or SiTU-GLU configurations on Rubin. ([#3352](https://github.com/NVIDIA/TransformerEngine/pull/3352)) ([#3485](https://github.com/NVIDIA/TransformerEngine/pull/3485)) ([#3567](https://github.com/NVIDIA/TransformerEngine/pull/3567))
- [PyTorch] Fixed cuDNN grouped-MLP weight-gradient compilation, CUDA graph capture, and FC1/FC2 corruption by retaining separate persistent descriptor workspaces. ([#3579](https://github.com/NVIDIA/TransformerEngine/pull/3579))
- [PyTorch] Fixed quantized parameter integration by supporting MXFP8 master-weight casts on ranks with empty shards and preserving quantized-tensor subclasses across `detach()` and C++ bindings. ([#3348](https://github.com/NVIDIA/TransformerEngine/pull/3348)) ([#3393](https://github.com/NVIDIA/TransformerEngine/pull/3393))
- [PyTorch] Fixed delayed-scaling FP8 activation-recompute metadata pairing for evaluation mode, mode transitions, nested checkpoints, and checkpoint entry under disabled autograd. ([#3394](https://github.com/NVIDIA/TransformerEngine/pull/3394))
- [PyTorch] Fixed delayed weight-gradient computation when an external consumer clears an eagerly computed FP8 bias gradient before `backward_dw()`. ([#3400](https://github.com/NVIDIA/TransformerEngine/pull/3400))
- [PyTorch] Fixed `FusedAdam` with different parameter and gradient dtypes in capturable mode and prevented tensor-handle pool exhaustion for highly fragmented parameter groups. ([#3414](https://github.com/NVIDIA/TransformerEngine/pull/3414)) ([#3418](https://github.com/NVIDIA/TransformerEngine/pull/3418))
- [PyTorch] Fixed Quantile Balancing router capture and replay with mutable bin bounds through an explicit trusted-producer validation handshake. ([#3426](https://github.com/NVIDIA/TransformerEngine/pull/3426))
- [PyTorch] Fixed non-fused and BF16 activation paths from failing when `activation_recompute_in_mlp=True`; unsupported paths now warn and proceed without recomputation. ([#3436](https://github.com/NVIDIA/TransformerEngine/pull/3436))
- [PyTorch] Fixed NCCL-EP zero-copy MXFP8 scale offsets for symmetric-memory windows and made offset resolution compatible with older and newer PyTorch APIs. ([#3447](https://github.com/NVIDIA/TransformerEngine/pull/3447)) ([#3466](https://github.com/NVIDIA/TransformerEngine/pull/3466))
- [PyTorch] Made fused grouped-MLP `dprob` bit-exact when deterministic algorithms are requested and a supporting cuDNN Frontend is installed. ([#3407](https://github.com/NVIDIA/TransformerEngine/pull/3407))
- [PyTorch] Replaced deprecated TorchScript fallbacks with eager execution when `torch.compile` is disabled or unsupported. ([#3502](https://github.com/NVIDIA/TransformerEngine/pull/3502))
- [JAX] Fixed fused-attention backward output-gradient sharding, rotated THD metadata handling, and compound DP/FSDP test-data placement. ([#3516](https://github.com/NVIDIA/TransformerEngine/pull/3516))
- [PyTorch] Fixed NVFP4 stochastic-quantization corruption and invalid memory reads by retaining the owning RNG tensors through dispatch. ([#3533](https://github.com/NVIDIA/TransformerEngine/pull/3533))
- [Common, PyTorch] Fixed distributed Newton-Schulz numerical correctness, workspace sizing and lifetime, and validation of unsupported matrix dimensions. ([#3541](https://github.com/NVIDIA/TransformerEngine/pull/3541))
- [Build, PyTorch] Fixed source builds to gate NCCL-EP consistently on targeted GPU architectures and available PyTorch symmetric-memory support. ([#3396](https://github.com/NVIDIA/TransformerEngine/pull/3396))
- [Common, JAX, PyTorch] Fixed NCCL-EP synchronization, timeout, and shared-workspace failures and added validated compatibility for builds using NCCL 2.30.5–2.30.x with newer NCCL runtimes through `NCCL_DEV_API_JIT`. ([#3619](https://github.com/NVIDIA/TransformerEngine/pull/3619))

## Breaking Changes in This Release

- [Common, JAX, PyTorch] Raised the minimum supported cuDNN version from 9.3 to 9.12.0, added configure-time, build-time, import-time, and runtime enforcement, and deprecated the now-obsolete experimental `nvte_get_runtime_num_segments`. ([#3236](https://github.com/NVIDIA/TransformerEngine/pull/3236))
- [Common, PyTorch] Added MXFP8 support to cuBLASMp communication-overlap GEMMs and raised the minimum cuBLASMp version for `NVTE_WITH_CUBLASMP=1` from 0.8.0 to 0.8.1. ([#3145](https://github.com/NVIDIA/TransformerEngine/pull/3145))
- [Common, PyTorch] Reduced P2P context-parallel attention forward temporary storage from O(CP) to O(1); callers of experimental `nvte_cp_thd_out_correction` must pass the new `old_lse` argument. ([#2916](https://github.com/NVIDIA/TransformerEngine/pull/2916))
- [PyTorch] Added attention-logit softcapping to `DotProductAttention`, `MultiheadAttention`, and `TransformerLayer`; callers passing optional arguments after `bottom_right_diagonal` positionally must switch to keyword arguments. ([#3391](https://github.com/NVIDIA/TransformerEngine/pull/3391))
- [PyTorch] Added experimental `GatedDeltaNet2Attention` and a shared linear-attention base API; experimental `GatedDeltaNetAttention` no longer accepts `name` or derives from `TransformerEngineBaseModule`, and source builds now require `nvidia-cudnn-frontend>=1.29.0`. ([#3521](https://github.com/NVIDIA/TransformerEngine/pull/3521))

## Deprecated Features

- [Common, JAX, PyTorch] Added opaque-handle v2 fused-attention C APIs, deprecated the legacy `nvte_get_fused_attn_backend`, `nvte_fused_attn_fwd`, and `nvte_fused_attn_bwd` shims, and removed `NVTE_FUSED_ATTN_BACKEND` in favor of automatic backend selection. ([#2964](https://github.com/NVIDIA/TransformerEngine/pull/2964))

## Known Issues in This Release

- [Common] cuBLAS 13.7.0 through 13.8.0 may silently leave grouped-GEMM output tiles unwritten on Blackwell and Rubin when NCCL kernels run concurrently, producing incorrect gradients. No Transformer Engine workaround is available; upgrade to [cuBLAS 13.8.1](https://developer.nvidia.com/cublas-13-8-1-download-archive) or later. ([#3578](https://github.com/NVIDIA/TransformerEngine/pull/3578))
- [Common, Build] Building or running Transformer Engine with cuSOLVERMp support enabled against CUDA Toolkit 13.4 Update 1 and cuSOLVERMp 0.9.1 or earlier may fail with unresolved cuSOLVER symbols while linking, loading, or using the Transformer Engine library. If affected, use the base CUDA Toolkit 13.4 release instead.