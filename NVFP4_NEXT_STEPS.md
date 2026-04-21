# Per-Token NVFP4 Grouped GEMM — What's Missing and Next Steps

## Current State

The end-to-end plumbing is complete: recipe -> quantizer -> fused op -> cuDNN kernel. Smoke tests pass for both forward and backward (with `backward_override`). However, several pieces are placeholders or approximations.

---

## What's Missing

### 1. Per-Token Quantization Kernel (CUDA)

**Status:** Placeholder only (`quantize_pertoken_nvfp4.cuh`)

**What exists today:** The standard NVFP4 quantizer computes a single per-tensor amax and derives one global scale `s_global = amax / (fp8_max * fp4_max)`. All tokens share this scale.

**What's needed:** A kernel that computes per-row (per-token) amax and derives per-row global scales.

**Files to create/modify:**
- `transformer_engine/common/cast/nvfp4/quantize_pertoken_nvfp4.cuh` — implement kernel
- `transformer_engine/common/cast/nvfp4/group_quantize_pertoken_nvfp4.cuh` — grouped variant for MoE
- `transformer_engine/common/include/transformer_engine/cast.h` — add C API declarations
- `transformer_engine/common/cast.cu` — add dispatch for per-token path
- `transformer_engine/pytorch/csrc/extensions/cast.cpp` — add `group_quantize_nvfp4_pertoken_impl` and wire into `group_quantize()`

**Kernel spec:**
```
Input:  (M, K) tensor, BF16/FP32
Output: (M, K/2) packed FP4 data, uint8
        (round_to_128(M), ceil(K/16)/4) block scales, FP8 E4M3
        (M,) per-token global scales, FP32
```

**Key difference from standard NVFP4:** Step 1 computes `amax` per row instead of per tensor. This requires a row-wise parallel reduction (one warp per row or similar).

### 2. NVFP4Quantizer Per-Token Support

**Status:** Not started

**What's needed:** The `NVFP4Quantizer` and `NVFP4Tensor` need to support per-token amax/global_scale instead of per-tensor.

**Files to modify:**
- `transformer_engine/pytorch/tensor/nvfp4_tensor.py`
  - `NVFP4Quantizer.make_empty()` — allocate `amax_rowwise` as `(M,)` instead of `(1,)` when per-token mode is enabled
  - `NVFP4Tensor` — property to expose `per_token_global_scale` as `(M,)` FP32 tensor
  - `NVFP4Quantizer.get_scale_shape()` — may need adjustment for per-token layout
- `transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py` — storage format for per-token scales

### 3. Fused Op: Per-Token Global Scale from Quantizer Output

**Status:** Approximation (broadcast per-tensor amax to all tokens)

**What's in place:** `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4.fuser_forward()` extracts `grouped_fc1_x.amax` (per-tensor, shape `(1,)`) and broadcasts it to `(valid_m, 1, 1)` as `global_scale_tensor`.

**What's needed once per-token quantizer exists:**
```python
# Replace this:
global_scale_val = nvfp4_amax.float() / (fp4_max * fp8_max)
global_scale_tensor = global_scale_val.expand(in_shape[0]).reshape(-1, 1, 1)

# With this:
global_scale_tensor = grouped_fc1_x.per_token_global_scale.reshape(-1, 1, 1)
```

**File:** `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` (inside `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4.fuser_forward()`)

### 4. Fused FC1->FC2 Handoff (Optimization)

**Status:** Not optimized — FC1 outputs BF16, then FC2 re-quantizes to NVFP4

**What exists in MXFP8:** FC1 kernel produces FP8 output + SFD (scale factor D) in a single kernel call (`discrete_col_sfd=True`). FC2 consumes them directly with zero re-quantization.

**What's needed for NVFP4:** Enable `discrete_col_sfd=True` with FP4 output dtype in the cuDNN kernel, so FC1 directly produces NVFP4 output + block scales + per-token global scales. Then FC2 can consume them without re-quantizing. This requires:
- cuDNN kernel: verify FP4 output with SFD generation works (may already work, needs testing)
- TE fused op: update FC2 input path to use FC1's SFD output instead of re-quantizing

### 5. Backward Pass

**Status:** By design, backward uses higher precision via `backward_override`.

The `NVFP4PerTokenBlockScaling` recipe intentionally runs forward in NVFP4 for throughput and backward in BF16 for training stability. This is controlled by `backward_override`:
- `"high_precision"`: saves original BF16 tensors for backward (more memory, best accuracy)
- `"dequantized"`: dequantizes saved FP4 tensors to BF16 for backward (less memory, slightly less accurate)

No fused NVFP4 backward kernels are needed. The unfused BF16 backward path handles gradient computation. This matches the pattern established by MXFP8 in PR #2644.

### 7. Tests

**Status:** Basic smoke tests pass. Comprehensive tests missing.

**Needed:**
- cuDNN Frontend: more test configs in `test_grouped_gemm_glu_nvfp4.py` (FP8 + global_scale, varying per-token values, discrete mode, class API)
- TE: dedicated NVFP4 per-token test cases in `tests/pytorch/test_backward_override.py` for `NVFP4PerTokenBlockScaling`
- Numerical accuracy comparison: NVFP4 per-token vs per-tensor vs BF16 baseline on real MoE workloads

---

## Recommended Execution Order

### Phase 1: Make per-token approximation production-ready
1. Add comprehensive cuDNN Frontend tests for `global_scale_tensor` (more configs, edge cases)
2. Add TE test cases for `NVFP4PerTokenBlockScaling` in backward override test suite
3. Optimize FC1->FC2 handoff (test if `discrete_col_sfd=True` works with FP4 output)

### Phase 2: Implement true per-token quantization kernel
4. Implement `quantize_pertoken_nvfp4.cuh` CUDA kernel
5. Add grouped variant `group_quantize_pertoken_nvfp4.cuh`
6. Add C API and C++ bindings
7. Update `NVFP4Quantizer` to support per-token mode
8. Update fused op to use real per-token global scales

### Phase 3: Benchmarking and validation
9. Benchmark per-token vs per-tensor NVFP4 on DeepSeek-V3 / Mixtral MoE workloads
10. Compare training loss curves: NVFP4 per-token vs MXFP8 vs BF16
11. Measure throughput: fused NVFP4 per-token vs unfused NVFP4 vs MXFP8 fused
