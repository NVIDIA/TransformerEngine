# Transformer Engine v2.19 Release Notes

## Key Features and Enhancements

- [Common] Added Rubin SM107a support when building with CUDA 13.4 or later. ([#3275](https://github.com/NVIDIA/TransformerEngine/pull/3275))
- [PyTorch] Added experimental per-direction hybrid quantization through `CustomRecipe`, allowing rowwise and columnwise paths to use different quantizers. ([#2817](https://github.com/NVIDIA/TransformerEngine/pull/2817))
- [Common, PyTorch] Added opt-in 2D block scaling for MXFP8 weights through `MXFP8BlockScaling(enable_2d_quantization=True)`. ([#2634](https://github.com/NVIDIA/TransformerEngine/pull/2634))
- [Common, PyTorch] Added MXFP8 quantization support to NCCL-EP dispatch forward and combine backward, with `GroupedLinear` and fused grouped MLP support for prequantized grouped inputs. ([#3270](https://github.com/NVIDIA/TransformerEngine/pull/3270)) ([#3244](https://github.com/NVIDIA/TransformerEngine/pull/3244))
- [Common, PyTorch] Added cuDNN FP8 fused attention support for packed THD inputs and context-parallel execution; FP8 THD sink-softmax backward requires cuDNN 9.26 or later. ([#2994](https://github.com/NVIDIA/TransformerEngine/pull/2994)) ([#3441](https://github.com/NVIDIA/TransformerEngine/pull/3441))
- [PyTorch] Added experimental cuDNN-backed `GatedDeltaNetAttention` for dense and THD inputs. ([#3351](https://github.com/NVIDIA/TransformerEngine/pull/3351))
- [PyTorch] Enabled `torch.compile(fullgraph=True)` for supported FlashAttention and the non-FP8 unfused attention backend. ([#3153](https://github.com/NVIDIA/TransformerEngine/pull/3153)) ([#3189](https://github.com/NVIDIA/TransformerEngine/pull/3189)) ([#3201](https://github.com/NVIDIA/TransformerEngine/pull/3201)) ([#3286](https://github.com/NVIDIA/TransformerEngine/pull/3286))
- [PyTorch] Added per-callable `capture_time_hooks` to `make_graphed_callables` for non-capturable warmup and graph-construction work. ([#2831](https://github.com/NVIDIA/TransformerEngine/pull/2831))
- [PyTorch] Added forward and backward support to `FusedMLAQUpProjRopeQuant` for fused MLA Q up-projection, RoPE, and MXFP8 quantization. ([#3303](https://github.com/NVIDIA/TransformerEngine/pull/3303)) ([#3330](https://github.com/NVIDIA/TransformerEngine/pull/3330))
- [PyTorch] Expanded THD context parallelism with hierarchical `a2a+p2p`, experimental no-load-balance AllGather, and FlashAttention 4 on `p2p`, `all_gather`, and `a2a` paths. ([#3290](https://github.com/NVIDIA/TransformerEngine/pull/3290)) ([#3221](https://github.com/NVIDIA/TransformerEngine/pull/3221)) ([#3438](https://github.com/NVIDIA/TransformerEngine/pull/3438)) ([#3149](https://github.com/NVIDIA/TransformerEngine/pull/3149))
- [PyTorch] Enabled row-scaled NVFP4 weight-gradient computation in dense `Linear` and `GroupedLinear` layers. ([#3206](https://github.com/NVIDIA/TransformerEngine/pull/3206)) ([#3324](https://github.com/NVIDIA/TransformerEngine/pull/3324))
- [Common, PyTorch] Reduced MXFP8 grouped MLP CPU overhead. ([#3338](https://github.com/NVIDIA/TransformerEngine/pull/3338)) ([#3267](https://github.com/NVIDIA/TransformerEngine/pull/3267)) ([#3238](https://github.com/NVIDIA/TransformerEngine/pull/3238)) ([#3293](https://github.com/NVIDIA/TransformerEngine/pull/3293))
- [PyTorch] Added named extra-output channels to `te.ops.Sequential`, allowing later fusible operations to consume outputs from earlier operations. ([#3320](https://github.com/NVIDIA/TransformerEngine/pull/3320))
- [PyTorch] Added `newton_schulz_tp` for tensor-parallel Newton-Schulz orthogonalization of replicated or sharded matrices. ([#2920](https://github.com/NVIDIA/TransformerEngine/pull/2920))
- [PyTorch] Optimized the performance and reduced `parallel_cross_entropy` memory use. Added `overwrite_input` option to reduce the memory usage even further by reusing the input storage. ([#3273](https://github.com/NVIDIA/TransformerEngine/pull/3273))
- [Common, PyTorch] Added NVFP4 stochastic-rounding and RHT split-quantization support on SM120 and SM121 GPUs. ([#3281](https://github.com/NVIDIA/TransformerEngine/pull/3281)) ([#3265](https://github.com/NVIDIA/TransformerEngine/pull/3265))
- [JAX] Added per-head maximum-logit outputs to `fused_attn` and Flax `DotProductAttention`. ([#3112](https://github.com/NVIDIA/TransformerEngine/pull/3112))
- [JAX] Optimized experimental MoE Block to improve E2E perf, MXFP8 quantization, overflow detection, reduced other overheads ([#3354](https://github.com/NVIDIA/TransformerEngine/pull/3354))
- [JAX] Expert Parallelism APIs with reduced receive capacity and overflow detection ([#3277](https://github.com/NVIDIA/TransformerEngine/pull/3277))
- [JAX] Added `ep_bootstrap` support for meshes with axes orthogonal to expert parallelism by deriving communicator domains from the active mesh. ([#3226](https://github.com/NVIDIA/TransformerEngine/pull/3226))
- [Build] Added CMake discovery of NCCL headers installed by `nvidia-nccl-cu12` and `nvidia-nccl-cu13`. ([#3227](https://github.com/NVIDIA/TransformerEngine/pull/3227))
- [Build] Enabled NVRTC to locate CUDA headers in Python `site-packages` at runtime, improving JIT compilation in relocated and build-isolated installations. ([#3252](https://github.com/NVIDIA/TransformerEngine/pull/3252))
- [Build] Reduced Transformer Engine library, wheel, and container sizes by compressing embedded CUDA binaries. ([#3465](https://github.com/NVIDIA/TransformerEngine/pull/3465))

## Fixed Issues

- [Common] Disabled an affected cuBLAS 13.7 grouped GEMM algorithm on B300 and Rubin GPUs to prevent silent data corruption. ([#3475](https://github.com/NVIDIA/TransformerEngine/pull/3475))
- [PyTorch] Fixed FlashAttention 2 local and post-release validation, FlashAttention 3 sliding-window context-parallel checks, and FlashAttention 4 fallback for unsupported GPUs, head dimensions, or CUTLASS DSL configurations. ([#3356](https://github.com/NVIDIA/TransformerEngine/pull/3356)) ([#3341](https://github.com/NVIDIA/TransformerEngine/pull/3341))
- [PyTorch] Fixed `FusedAdam` checkpoint step counters for parameter groups that are empty on some ranks. ([#3318](https://github.com/NVIDIA/TransformerEngine/pull/3318))
- [PyTorch] Fixed delayed-scaling FP8 activation recomputation when checkpointed callables enable FP8 in an inner autocast context. ([#3284](https://github.com/NVIDIA/TransformerEngine/pull/3284))
- [PyTorch] Fixed CUDA graph replay parameter-gradient lifetime by cloning returned parameter gradients by default, preventing later replays from overwriting retained gradients. ([#2937](https://github.com/NVIDIA/TransformerEngine/pull/2937))
- [PyTorch] Fixed a possible first-call segfault during lazy extension initialization by acquiring the GIL before Python imports. ([#3255](https://github.com/NVIDIA/TransformerEngine/pull/3255))
- [PyTorch] Fixed THD fused RoPE launch scaling for high sequence counts in token-linear forward and backward kernels. ([#3057](https://github.com/NVIDIA/TransformerEngine/pull/3057))
- [JAX] Fixed GEMM partition inference for tuple-valued mesh-axis specifications, including nested contracting axes and DP+FSDP layouts. ([#3429](https://github.com/NVIDIA/TransformerEngine/pull/3429)) ([#3272](https://github.com/NVIDIA/TransformerEngine/pull/3272))
- [Build] Fixed NCCL-EP JIT in installed packages by shipping the runtime library and required JIT headers and discovering matching NCCL headers at runtime. ([#3434](https://github.com/NVIDIA/TransformerEngine/pull/3434))
- [Build] Fixed NCCL-EP source builds to respect `MAX_JOBS` and use a valid unlimited-parallel make invocation when no limit is set. ([#3138](https://github.com/NVIDIA/TransformerEngine/pull/3138)) ([#3243](https://github.com/NVIDIA/TransformerEngine/pull/3243))

## Breaking Changes in This Release

- [PyTorch] Changed `mhc_fused_projection` so its `H` output is always FP32 instead of matching the input dtype; cast it explicitly where BF16 is required. ([#2978](https://github.com/NVIDIA/TransformerEngine/pull/2978))
- [PyTorch] Changed the NCCL-EP Python API to support eager dispatch and drop-on-overflow: pass `num_topk` to `ep_bootstrap`, update positional `ep_bootstrap` and `EpBuffer` calls, and replace `EpBuffer.token_counts` with `EpBuffer.tokens_per_expert`. ([#3229](https://github.com/NVIDIA/TransformerEngine/pull/3229))

## Deprecated Features

- [PyTorch] Deprecated the unused `is_cg_capturable` argument to `parallel_cross_entropy`; the operation is always CUDA graph capturable. ([#3455](https://github.com/NVIDIA/TransformerEngine/pull/3455))
- [PyTorch] Deprecated selecting `GroupedLinear`'s grouped-tensor path with `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM`; pass `use_grouped_tensor=True` or `False` to `GroupedLinear` instead. ([#3224](https://github.com/NVIDIA/TransformerEngine/pull/3224))
- [PyTorch] Deprecated implicit pointer-based layout detection when `DotProductAttention` receives Q/K/V views of packed buffers; pass the packed buffer through `qkv_layer` or `kv_layer` and set `qkv_interleave_dim` instead. ([#3200](https://github.com/NVIDIA/TransformerEngine/pull/3200))

## Known Issues in This Release

- [PyTorch] GDN attention may generate a `NaN` in the backward pass when used with `head_dim==32`. The [fix](https://github.com/NVIDIA/cudnn-frontend/pull/994) has been merged into `nvidia-cudnn-frontend` and will be resolved by the next release.
