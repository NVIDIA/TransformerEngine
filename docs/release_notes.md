# Transformer Engine v2.19 Release Notes

## Key Features and Enhancements

- [Common] Added Rubin SM107a code generation and stochastic-rounding support with CUDA 13.4 and later. ([#3275](https://github.com/NVIDIA/TransformerEngine/pull/3275))
- [Common, PyTorch] Added experimental per-direction hybrid quantization through `CustomRecipe`, allowing rowwise and columnwise paths to use different quantizers. ([#2817](https://github.com/NVIDIA/TransformerEngine/pull/2817))
- [Common, PyTorch] Added opt-in 2D block scaling for MXFP8 weights through `MXFP8BlockScaling(enable_2d_quantization=True)`. ([#2634](https://github.com/NVIDIA/TransformerEngine/pull/2634))
- [Common, PyTorch] Added MXFP8 quantization to NCCL-EP dispatch forward and combine backward, with `GroupedLinear` and fused grouped MLP support for prequantized grouped inputs. ([#3270](https://github.com/NVIDIA/TransformerEngine/pull/3270)) ([#3244](https://github.com/NVIDIA/TransformerEngine/pull/3244))
- [Common, PyTorch] Added cuDNN FP8 fused attention for packed THD inputs and context-parallel execution. ([#2994](https://github.com/NVIDIA/TransformerEngine/pull/2994))
- [PyTorch] Added experimental cuDNN-backed `GatedDeltaNetAttention` for dense and THD inputs with cuDNN Frontend 1.28 and later. ([#3351](https://github.com/NVIDIA/TransformerEngine/pull/3351))
- [PyTorch] Added a `DistributedWeight` protocol and dispatch hooks for external generalized tensor-parallel weight materialization. ([#3005](https://github.com/NVIDIA/TransformerEngine/pull/3005))
- [PyTorch] Enabled `torch.compile(fullgraph=True)` for attention-backend selection, supported FlashAttention implementations, and the non-FP8 unfused attention backend. ([#3153](https://github.com/NVIDIA/TransformerEngine/pull/3153)) ([#3189](https://github.com/NVIDIA/TransformerEngine/pull/3189)) ([#3201](https://github.com/NVIDIA/TransformerEngine/pull/3201)) ([#3286](https://github.com/NVIDIA/TransformerEngine/pull/3286))
- [PyTorch] Added per-callable `capture_time_hooks` to `make_graphed_callables` for non-capturable warmup and graph-construction work. ([#2831](https://github.com/NVIDIA/TransformerEngine/pull/2831))
- [PyTorch] Added forward and backward support to `FusedMLAQUpProjRopeQuant` for fused MLA Q up-projection, RoPE, and MXFP8 quantization. ([#3303](https://github.com/NVIDIA/TransformerEngine/pull/3303)) ([#3330](https://github.com/NVIDIA/TransformerEngine/pull/3330))
- [PyTorch] Expanded THD context parallelism with hierarchical `a2a+p2p`, experimental no-load-balance AllGather, and FlashAttention 4 on `p2p`, `all_gather`, and `a2a` paths. ([#3290](https://github.com/NVIDIA/TransformerEngine/pull/3290)) ([#3221](https://github.com/NVIDIA/TransformerEngine/pull/3221)) ([#3438](https://github.com/NVIDIA/TransformerEngine/pull/3438)) ([#3149](https://github.com/NVIDIA/TransformerEngine/pull/3149))
- [PyTorch] Enabled row-scaled NVFP4 weight-gradient computation in dense `Linear` and `GroupedLinear` layers. ([#3206](https://github.com/NVIDIA/TransformerEngine/pull/3206)) ([#3324](https://github.com/NVIDIA/TransformerEngine/pull/3324))
- [Common, PyTorch] Reduced MXFP8 grouped MLP overhead through direct scale swizzling, single-group fast paths, fused scaled activations, and removal of unnecessary probability tensors. ([#3338](https://github.com/NVIDIA/TransformerEngine/pull/3338)) ([#3267](https://github.com/NVIDIA/TransformerEngine/pull/3267)) ([#3238](https://github.com/NVIDIA/TransformerEngine/pull/3238)) ([#3293](https://github.com/NVIDIA/TransformerEngine/pull/3293))
- [PyTorch] Added `newton_schulz_tp` for tensor-parallel Newton-Schulz orthogonalization of replicated or sharded matrices. ([#2920](https://github.com/NVIDIA/TransformerEngine/pull/2920))
- [PyTorch] Reduced `parallel_cross_entropy` memory use by reconstructing derivatives during backward and added `overwrite_input` to reuse input storage. ([#3273](https://github.com/NVIDIA/TransformerEngine/pull/3273))
- [Common, PyTorch] Added NVFP4 stochastic-rounding and RHT split-quantization support on SM120 and SM121 GPUs. ([#3281](https://github.com/NVIDIA/TransformerEngine/pull/3281)) ([#3265](https://github.com/NVIDIA/TransformerEngine/pull/3265))
- [JAX] Added per-head maximum-logit outputs to `fused_attn` and Flax `DotProductAttention`. ([#3112](https://github.com/NVIDIA/TransformerEngine/pull/3112))
- [JAX] Improved experimental MoE and Expert Parallelism APIs with grouped MXFP8 quantization, reduced receive capacity, overflow reporting, optional token dropping, and communicator re-bootstrap. ([#3354](https://github.com/NVIDIA/TransformerEngine/pull/3354)) ([#3277](https://github.com/NVIDIA/TransformerEngine/pull/3277))
- [Build] Enabled NVRTC to locate CUDA headers in Python `site-packages` at runtime, improving JIT compilation in relocated and build-isolated installations. ([#3252](https://github.com/NVIDIA/TransformerEngine/pull/3252))
- [Build] Reduced Transformer Engine library, wheel, and container sizes by compressing embedded CUDA binaries. ([#3465](https://github.com/NVIDIA/TransformerEngine/pull/3465))

## Fixed Issues

- [Common] Fixed mHC Triton kernel out-of-memory failures on L40 GPUs with Triton 3.8. ([#3442](https://github.com/NVIDIA/TransformerEngine/pull/3442))
- [Common] Fixed no-op handling in FP8 blockwise and MXFP8 quantization so cached outputs are preserved and required dBias reductions still run. ([#3271](https://github.com/NVIDIA/TransformerEngine/pull/3271)) ([#3246](https://github.com/NVIDIA/TransformerEngine/pull/3246))
- [Common] Disabled an affected cuBLAS 13.7 grouped GEMM algorithm on B300 and Rubin GPUs to prevent silent data corruption. ([#3475](https://github.com/NVIDIA/TransformerEngine/pull/3475))
- [Common, PyTorch] Prevented FP8 THD sink-softmax backward from using cuDNN versions earlier than 9.26, where ragged statistics can be misindexed. ([#3441](https://github.com/NVIDIA/TransformerEngine/pull/3441))
- [Common, PyTorch] Fixed NCCL-EP eager dispatch on ranks receiving zero tokens. ([#3276](https://github.com/NVIDIA/TransformerEngine/pull/3276))
- [PyTorch] Fixed FlashAttention version detection and safely treated unusable FlashAttention 4 or CUTLASS combinations as unavailable. ([#3356](https://github.com/NVIDIA/TransformerEngine/pull/3356)) ([#3341](https://github.com/NVIDIA/TransformerEngine/pull/3341))
- [PyTorch] Fixed `FusedAdam` checkpoint step counters for parameter groups that are empty on some ranks. ([#3318](https://github.com/NVIDIA/TransformerEngine/pull/3318))
- [PyTorch] Fixed delayed-scaling FP8 activation recomputation when checkpointed callables enable FP8 in an inner autocast context. ([#3284](https://github.com/NVIDIA/TransformerEngine/pull/3284))
- [PyTorch] Fixed CUDA graph replay parameter-gradient lifetime so later replays cannot overwrite returned gradients. ([#2937](https://github.com/NVIDIA/TransformerEngine/pull/2937))
- [PyTorch] Fixed a possible first-call segfault during lazy extension initialization by acquiring the GIL before Python imports. ([#3255](https://github.com/NVIDIA/TransformerEngine/pull/3255))
- [PyTorch] Fixed THD fused RoPE launch scaling for high sequence counts in token-linear forward and backward kernels. ([#3057](https://github.com/NVIDIA/TransformerEngine/pull/3057))
- [JAX] Fixed GEMM and collective GEMM partitioning on nested or multi-axis meshes and `ep_bootstrap` when meshes include axes orthogonal to expert parallelism. ([#3429](https://github.com/NVIDIA/TransformerEngine/pull/3429)) ([#3272](https://github.com/NVIDIA/TransformerEngine/pull/3272)) ([#3226](https://github.com/NVIDIA/TransformerEngine/pull/3226))
- [Build] Fixed NCCL and NCCL-EP source and JIT builds by packaging NCCL-EP runtime files, finding pip-installed NCCL headers, and respecting `MAX_JOBS`. ([#3434](https://github.com/NVIDIA/TransformerEngine/pull/3434)) ([#3227](https://github.com/NVIDIA/TransformerEngine/pull/3227)) ([#3138](https://github.com/NVIDIA/TransformerEngine/pull/3138)) ([#3243](https://github.com/NVIDIA/TransformerEngine/pull/3243))

## Breaking Changes in This Release

- [PyTorch] Changed `mhc_fused_projection` so its `H` output is always FP32 instead of matching the input dtype; cast it explicitly where BF16 is required. ([#2978](https://github.com/NVIDIA/TransformerEngine/pull/2978))
- [PyTorch] Changed the NCCL-EP Python API to support eager dispatch and drop-on-overflow: pass `num_topk` to `ep_bootstrap`, update positional `ep_bootstrap` and `EpBuffer` calls, and replace `EpBuffer.token_counts` with `EpBuffer.tokens_per_expert`. ([#3229](https://github.com/NVIDIA/TransformerEngine/pull/3229))

## Deprecated Features

- [PyTorch] Deprecated the unused `is_cg_capturable` argument to `parallel_cross_entropy`; the operation is always CUDA graph capturable. ([#3455](https://github.com/NVIDIA/TransformerEngine/pull/3455))
- [PyTorch] Deprecated selecting `GroupedLinear`'s grouped-tensor path with `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM`; pass `use_grouped_tensor=True` or `False` to `GroupedLinear` instead. ([#3224](https://github.com/NVIDIA/TransformerEngine/pull/3224))
- [PyTorch] Deprecated implicit pointer-based layout detection when `DotProductAttention` receives Q/K/V views of packed buffers; pass the packed buffer through `qkv_layer` or `kv_layer` and set `qkv_interleave_dim` instead. ([#3200](https://github.com/NVIDIA/TransformerEngine/pull/3200))
