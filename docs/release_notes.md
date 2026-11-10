# Transformer Engine v2.21 Release Notes

## Key Features and Enhancements

- [PyTorch] Added experimental cuDNN-backed `KimiDeltaAttention`. ([#3548](https://github.com/NVIDIA/TransformerEngine/pull/3548))
- [Common, PyTorch] Added prototype NVFP4 quantization with UE5M3 block scales. ([#3325](https://github.com/NVIDIA/TransformerEngine/pull/3325))
- [PyTorch] Expanded `torch.compile(fullgraph=True)` support to FP16 and BF16 FusedAttention, fusible `BasicLinear` and `Bias` operations including Linear + Bias forward fusion, and FP8 weight caching in compiled `Linear` calls using `is_first_microbatch`. ([#3472](https://github.com/NVIDIA/TransformerEngine/pull/3472)) ([#3547](https://github.com/NVIDIA/TransformerEngine/pull/3547)) ([#3505](https://github.com/NVIDIA/TransformerEngine/pull/3505))
- [PyTorch] Added FlashAttention 3 and FlashAttention 4 support for THD attention with `pad_between_seqs=True`, including P2P and all-to-all context parallelism. ([#3205](https://github.com/NVIDIA/TransformerEngine/pull/3205))
- [PyTorch] Added support for FlashAttention 2.8.4. ([#3403](https://github.com/NVIDIA/TransformerEngine/pull/3403))
- [PyTorch] Added support for backward with head dimension 256 in THD and context parallel attention when using FlashAttention 4.0.0b33+ ([#3540](https://github.com/NVIDIA/TransformerEngine/pull/3540))
- [PyTorch] Improved THD training performance with dropout on SM100 and SM103. ([#3313](https://github.com/NVIDIA/TransformerEngine/pull/3313))
- [PyTorch] Added `MoeDispatch` and `MoeCombine` fusible operations for building MoE blocks with `te.ops.Sequential` and improved the performance of it. ([#3503](https://github.com/NVIDIA/TransformerEngine/pull/3503))
- [Common, PyTorch] Added a deterministic mode to `fused_moe_aux_loss` through `deterministic=True`. ([#3587](https://github.com/NVIDIA/TransformerEngine/pull/3587))
- [PyTorch] Extended the fused grouped MLP with deterministic `dprob` for GLU activations and fused activation recompute for `ScaledTanhSReLU`. ([#3635](https://github.com/NVIDIA/TransformerEngine/pull/3635)) ([#3473](https://github.com/NVIDIA/TransformerEngine/pull/3473)) ([#3666](https://github.com/NVIDIA/TransformerEngine/pull/3666))
- [Common, PyTorch] Added `fused_mla_kv_rope_mxfp8`, which fuses multi-latent attention key/value RoPE with MXFP8 quantization for DeepSeek-V3 head dimensions on SM100 and later. ([#3672](https://github.com/NVIDIA/TransformerEngine/pull/3672))
- [Common, PyTorch] Added a fused RMSNorm forward kernel with MXFP8 output that writes compact or GEMM-swizzled scaling factors directly. ([#3657](https://github.com/NVIDIA/TransformerEngine/pull/3657))
- [Common] Improved low-precision quantization performance with 128x128-tile grouped MXFP8 requantization, one-CTA-per-tile MXFP8 scale swizzling, and single-pass row and column amax for single and grouped row-scaled NVFP4. ([#3656](https://github.com/NVIDIA/TransformerEngine/pull/3656)) ([#3658](https://github.com/NVIDIA/TransformerEngine/pull/3658)) ([#3454](https://github.com/NVIDIA/TransformerEngine/pull/3454)) ([#3467](https://github.com/NVIDIA/TransformerEngine/pull/3467))
- [Common, PyTorch] Added NVFP4 Random Hadamard Transform support on SM120 and SM121 GPUs. ([#3300](https://github.com/NVIDIA/TransformerEngine/pull/3300))
- [Common, PyTorch] Added `return_argmax=True` to `parallel_cross_entropy`, returning the predicted token index without a second pass over the logits. ([#3671](https://github.com/NVIDIA/TransformerEngine/pull/3671))
- [PyTorch] Added `zero_tensor_dummy_wgrad`, which returns a storage-free dummy weight gradient for parameters using `fuse_wgrad_accumulation`, avoiding redundant additions when a weight is used more than once. ([#3670](https://github.com/NVIDIA/TransformerEngine/pull/3670))
- [Common, PyTorch] Reduced CPU overhead in `Linear`, `LayerNormLinear`, `LayerNormMLP`, and `GroupedLinear` by removing unnecessary tensor views and a redundant device-to-host split-size synchronization. ([#3523](https://github.com/NVIDIA/TransformerEngine/pull/3523)) ([#3554](https://github.com/NVIDIA/TransformerEngine/pull/3554))
- [JAX] Optimized the MoE block with fused cuDNN Frontend MXFP8 grouped GEMM and activation kernels, quantization before FSDP all-gather, and user-controlled checkpoint names on block intermediates. ([#3638](https://github.com/NVIDIA/TransformerEngine/pull/3638)) ([#3589](https://github.com/NVIDIA/TransformerEngine/pull/3589))
- [Build] Reduced source-build time by compiling NVFP4 quantization kernels with NVRTC. ([#3301](https://github.com/NVIDIA/TransformerEngine/pull/3301))
- [Build] Made aarch64 Transformer Engine libraries compatible with BOLT post-link optimization. ([#3383](https://github.com/NVIDIA/TransformerEngine/pull/3383)) ([#3627](https://github.com/NVIDIA/TransformerEngine/pull/3627))
- [Docs] Added a datatype and hardware support matrix and a Mixture of Experts guide with PyTorch and JAX examples. ([#3195](https://github.com/NVIDIA/TransformerEngine/pull/3195)) ([#3494](https://github.com/NVIDIA/TransformerEngine/pull/3494))
- [JAX] Added an opt-in fused Expert Parallelism prepare-and-dispatch path through `NVTE_EP_FUSED_PREPARE_DISPATCH=1`. ([#3674](https://github.com/NVIDIA/TransformerEngine/pull/3674))

## Fixed Issues

- [Common] Fixed shared-memory reuse races and incomplete TMA store synchronization in FP8, MXFP8, NVFP4, and Hadamard-fusion quantization kernels. ([#3417](https://github.com/NVIDIA/TransformerEngine/pull/3417))
- [Common] Fixed incorrect column-wise scales in grouped MXFP8 quantization with varying last dimensions and GEMM-swizzled scales. ([#3599](https://github.com/NVIDIA/TransformerEngine/pull/3599))
- [Common] Fixed kernel launch failures on GPUs with smaller per-block shared-memory or register limits by falling back from gated-activation TMA kernels and bounding multi-tensor swizzle kernel registers. ([#3600](https://github.com/NVIDIA/TransformerEngine/pull/3600)) ([#3622](https://github.com/NVIDIA/TransformerEngine/pull/3622))
- [Common] Fixed silent out-of-bounds bias reads in non-FP8 GEMMs by rejecting bias tensors whose dtype differs from the output dtype. ([#3565](https://github.com/NVIDIA/TransformerEngine/pull/3565))
- [Common] Fixed integer overflow in fused MoE auxiliary-loss kernels for inputs larger than `INT_MAX` elements. ([#3569](https://github.com/NVIDIA/TransformerEngine/pull/3569))
- [Common] Fixed loss of small sigmoid scores in biased fused top-k routing. ([#3637](https://github.com/NVIDIA/TransformerEngine/pull/3637))
- [Common, PyTorch] Fixed THD context-parallel kernel launch failures with large packed-document counts by keeping sequence-boundary tables within shared-memory limits. ([#3698](https://github.com/NVIDIA/TransformerEngine/pull/3698))
- [Common] Fixed segmentation faults at process exit after NVRTC kernels or CUDA graph-captured cuDNN fused attention were used. ([#3733](https://github.com/NVIDIA/TransformerEngine/pull/3733))
- [Common] Disabled FusedAttention for THD training with learnable softmax on cuDNN versions earlier than 9.26 to avoid incorrect `dSink` results and out-of-bounds reads. ([#3470](https://github.com/NVIDIA/TransformerEngine/pull/3470))
- [Common] Fixed `Float8CurrentScaling` and `Float8BlockScaling` ignoring the `use_power_2_scales` and `use_f32_scales` constructor arguments. ([#3605](https://github.com/NVIDIA/TransformerEngine/pull/3605))
- [Build, Common] Fixed source builds for Blackwell targets with CUDA 12.8. ([#3602](https://github.com/NVIDIA/TransformerEngine/pull/3602))
- [PyTorch] Fixed attention backend selection so a broken FlashAttention 2 or 3 installation no longer breaks `import transformer_engine.pytorch`, and FlashAttention 4 is skipped for deterministic backward on SM12x. ([#3553](https://github.com/NVIDIA/TransformerEngine/pull/3553)) ([#3595](https://github.com/NVIDIA/TransformerEngine/pull/3595))
- [PyTorch] Made FlashAttention 4 backward on Hopper fail with an actionable error instead of hanging when `quack-kernels` 0.6.0 through 0.6.3 is installed with CUTLASS DSL 4.6.2 or later. ([#3762](https://github.com/NVIDIA/TransformerEngine/pull/3762))
- [PyTorch] Fixed NVFP4 `torch.compile` failures for non-matrix `Linear` inputs and incorrect gradients from compiled Linear operations with bias. ([#3628](https://github.com/NVIDIA/TransformerEngine/pull/3628))
- [PyTorch] Fixed ONNX export of MXFP8 block quantization for TensorRT 11.4, FP8 current-scaling quantization output metadata, and the registered `layernorm` operator. ([#3592](https://github.com/NVIDIA/TransformerEngine/pull/3592)) ([#3659](https://github.com/NVIDIA/TransformerEngine/pull/3659))
- [PyTorch] Fixed deferred initialization of fusible operations by adding `reset_parameters()` to `te.ops.Linear`, `te.ops.Bias`, and `te.ops.Sequential`. ([#3327](https://github.com/NVIDIA/TransformerEngine/pull/3327))
- [PyTorch] Fixed `GroupedLinear` distributed weights on the grouped-tensor path, including backward `main_grad` handling and FP8 requantization of re-gathered weights. ([#3517](https://github.com/NVIDIA/TransformerEngine/pull/3517))
- [PyTorch] Fixed NVFP4 weight views in the single-group fused grouped MLP. ([#3662](https://github.com/NVIDIA/TransformerEngine/pull/3662))
- [PyTorch] Fixed `GroupedTensor` gradient propagation through autograd views and stream tracking of its backing tensors. ([#3701](https://github.com/NVIDIA/TransformerEngine/pull/3701))
- [PyTorch] Fixed `LayerNormLinear` silently disabling Userbuffers all-gather overlap when the forward pass runs under `torch.no_grad()`. ([#3546](https://github.com/NVIDIA/TransformerEngine/pull/3546))
- [PyTorch] Fixed `FusedSGD.zero_grad(set_to_none=False)` setting gradients to `None` instead of zeroing them. ([#3549](https://github.com/NVIDIA/TransformerEngine/pull/3549))
- [PyTorch] Fixed NCCL-EP bootstrap and Newton-Schulz with PyTorch's `ProcessGroupNCCL2` backend by borrowing the NCCL communicator in C++. ([#3586](https://github.com/NVIDIA/TransformerEngine/pull/3586))
- [JAX] Fixed intermittent distributed softmax hangs by initializing softmax custom calls before execution. ([#3612](https://github.com/NVIDIA/TransformerEngine/pull/3612))
- [JAX] Avoided CUDA graph stream-capture failures by disabling the experimental CuTeDSL backend in JAX. ([#3784](https://github.com/NVIDIA/TransformerEngine/pull/3784))
- [Common, JAX] Fixed Expert Parallelism handles when XLA relocates handle memory, such as under `lax.scan`, by reconstructing handles with `ncclEpImportHandle` and defaulting `topk_idx` to int32. ([#3639](https://github.com/NVIDIA/TransformerEngine/pull/3639))
- [Build, JAX] Added the missing Triton dependency to the JAX extension. ([#3610](https://github.com/NVIDIA/TransformerEngine/pull/3610))

## Breaking Changes in This Release

- [Build, Common] Builds with the experimental `NVTE_WITH_CUTEDSL=1` option now require `nvidia-cutlass-dsl` 4.8.0 or later. ([#3620](https://github.com/NVIDIA/TransformerEngine/pull/3620))

## Deprecated Features

- [PyTorch] Deprecated `NVTE_CUTEDSL_FUSED_GROUPED_MLP` in favor of passing `split_alignment=256` to `te.ops.GroupedLinear`; single grouped weights and biases no longer require `NVTE_GROUPED_LINEAR_SINGLE_PARAM`. ([#3584](https://github.com/NVIDIA/TransformerEngine/pull/3584))
- [Common, JAX, PyTorch] Removed the PyTorch Expert Parallelism handle cache that could exhaust its 4096-handle limit and deprecated the handle cache, `NVTE_EP_HANDLE_CACHE_SIZE`, and the non-v2 `nvte_ep_dispatch`, `nvte_ep_combine`, `nvte_ep_dispatch_bwd`, and `nvte_ep_combine_bwd` APIs for removal in Transformer Engine 3.0. ([#3760](https://github.com/NVIDIA/TransformerEngine/pull/3760))
- [JAX] Added an opt-in fused Expert Parallelism prepare-and-dispatch path through `NVTE_EP_FUSED_PREPARE_DISPATCH=1` and deprecated `ep_prepare` and `ep_dispatch_fwd` in favor of `ep_prepare_and_dispatch_fwd`, with removal planned in Transformer Engine 3.0. ([#3674](https://github.com/NVIDIA/TransformerEngine/pull/3674))
- [Common, PyTorch] Removed device-side validation of Quantile Balancing bin bounds and deprecated the now-redundant `mark_qb_bin_bounds_validated` and unvalidated QB C API variants. ([#3734](https://github.com/NVIDIA/TransformerEngine/pull/3734))
