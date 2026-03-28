# TransformerEngine Stable ABI Migration Plan

## Goal

Ship a single compiled `te_stable_abi.so` that works across PyTorch versions by migrating from the unstable pybind11 extension (`transformer_engine_torch.cpython-*.so`) to the PyTorch libtorch stable ABI (`torch/csrc/stable/`).

## What's Been Done

### Infrastructure (complete)

- **102 stable ABI C++ ops** in `csrc/stable/` (16 source files, ~3,200 lines)
  - Registered via `STABLE_TORCH_LIBRARY` / `STABLE_TORCH_LIBRARY_IMPL`
  - Compiled as `te_stable_abi.so` with `setuptools.Extension` (not `CppExtension`)
  - No ATen, pybind11, c10 headers — only `torch/csrc/stable/` and NVTE C API
- **`stable_common.h`** utility header:
  - `DType ↔ ScalarType` converters
  - `getCurrentCUDAStreamRaw()`, `getSMCount()`
  - `makeTransformerEngineTensor()`, `makeQuantizedTensorWrapper()`
  - `allocateStableTensor()`, `runWithWorkspace()`
- **Build system**: `setup_pytorch_stable_extension()` in `build_tools/pytorch.py`
- **Common library changes**: Public accessors on `CommOverlapCore` (`get_tp_id()`, `get_ubuf()`, etc.)

### Python routing layer (complete)

- **`_tex.py`**: All 47 Python files import through `_tex.py` instead of `transformer_engine_torch` directly. When the pybind `.so` is removed, `_tex.py` imports everything from `_stable_torch_module.py`.
- **`_stable_torch_module.py`** (~1,150 lines): Pure Python implementations of all 150 symbols exported by the pybind module, including:
  - Python `IntEnum` replacements for DType, NVTE_QKV_Layout, NVTE_Bias_Type, etc.
  - `FP8TensorMeta` as a Python class
  - CommOverlap stub classes
  - Activation wrappers with 4-path quantizer dispatch (FULLY_FUSED / UNFUSED / FUSED_AMAX_FP8 / FUSED_AMAX_NVFP4)
  - Multi-tensor ops via pointer-pack pattern
  - GEMM via `extract_tensor_data` + `_ops.gemm`
- **`_extract.py`**: Extracts raw `(data, te_dtype, scale_inv, scaling_mode)` from quantized tensor types (Float8, Float8Block, MXFP8, NVFP4)
- **`_quantize_stable.py`**: `quantize_into()` and `quantize_new()` that call stable ABI quantize ops

### Quantizer patches (complete, zero regressions)

All 4 quantizer types patched to call stable ops directly:
- `Float8Quantizer` — `update_quantized` + `quantize_impl` (with `scale_inv = 1/scale` init)
- `Float8BlockQuantizer` — `update_quantized` + `quantize_impl`
- `MXFP8Quantizer` — `update_quantized` + `quantize_impl`
- `NVFP4Quantizer` — `update_quantized` + `quantize_impl`

Files: `tensor/float8_tensor.py`, `tensor/float8_blockwise_tensor.py`, `tensor/mxfp8_tensor.py`, `tensor/nvfp4_tensor.py`

### Key bugs found and fixed

1. **NVTEScalingMode enum values were wrong**: MXFP8=1 (not 3), BLOCK_1D=2 (not 1), BLOCK_2D=3 (not 2). Caused MXFP8 activations to enter BLOCK_SCALING_2D kernel path.
2. **RoPE `empty_like` preserves non-contiguous strides**: Use `allocateStableTensor` instead for contiguous output.
3. **Float8Quantizer.quantize_impl needs `scale_inv = 1/scale`**: C++ `create_tensor` does `reciprocal(scale)`, Python `make_empty` doesn't.
4. **GEMM column-major convention**: Output D must be allocated as `(N, M)` not `(M, N)`.
5. **GEMM output dtype**: Must match input dtype (cuBLAS requires compatible types).

### Test results

| Configuration | Passed | Failed | Notes |
|--------------|--------|--------|-------|
| With pybind `.so` (baseline) | 10,983/10,984 | 1 | 1 pre-existing failure |
| With pybind `.so` + quantizer patches + `_tex.py` routing | 19,167/19,168 | 1 | Same pre-existing failure |
| **Without pybind `.so`** | **3,684/10,984** | **7,300** | All non-FP8 forward passes work |

## Remaining Issues

### Issue 1: Backward GEMM dimension mismatches (~600 non-FP8 failures)

**Symptom**: Assertion `(transb == CUBLAS_OP_T ? B0 : B1) == k` fails for backward GEMM calls in complex modules (microbatching, skip_dgrad/skip_wgrad combinations).

**Root cause**: The NVTE cuBLAS wrapper uses column-major convention where TensorWrapper shape `(rows, cols)` is `(cols, rows)` in matrix terms. Our `generic_gemm` Python implementation allocates output D correctly for simple cases but the backward pass in TE modules passes tensors with shapes that don't match the column-major expectation.

**Fix**: The pybind `generic_gemm` in `gemm.cpp` handles this transparently because `makeTransformerEngineTensor(at::Tensor)` stores the PyTorch shape as-is, and the cuBLAS code interprets it in column-major. Our stable path does the same. The issue is likely that the backward pass creates intermediate tensors (via reshape/view) that have unexpected shapes for our Python `generic_gemm`'s M/N computation. Need to trace the exact failing call to identify which tensor is wrong.

### Issue 2: FP8 scale swizzling not in stable GEMM (~5,000 FP8 failures)

**Symptom**: FP8 GEMM fails because scales aren't in the format cuBLAS expects.

**Root cause**: The pybind `generic_gemm` calls `swizzle_scales_for_gemm(A_tensor, ...)` and `swizzle_scales_for_gemm(B_tensor, ...)` before `nvte_cublas_gemm_v2`. Our stable GEMM doesn't do this.

**Fix**: Add scale swizzling to the stable GEMM C++ code (`csrc/stable/gemm.cpp`). The NVTE C API `nvte_swizzle_scaling_factors()` is a pure C function — add a call before the GEMM, or add a separate stable op `swizzle_scales` and call from Python before GEMM.

### Issue 3: Grouped GEMM not implemented (~200 failures)

**Symptom**: `NotImplementedError` for `te_general_grouped_gemm` and variants.

**Root cause**: No stable ABI ops for grouped GEMM. The NVTE C APIs (`nvte_multi_tensor_gemm`, `nvte_grouped_gemm`, etc.) are pure C functions taking `NVTETensor*` arrays and `NVTEGroupedTensor` handles.

**Fix**: Add stable C++ ops using the pointer-pack pattern (same as multi_tensor ops). Python side packs tensor data_ptrs into int64 tensors. C++ reconstructs `TensorWrapper` arrays and calls the NVTE C API. Also needs `GroupedTensorWrapper` construction for the grouped variants.

### Issue 4: Fused attention not implemented (~500 failures)

**Symptom**: `NotImplementedError` for `fused_attn_fwd` and `fused_attn_bwd`.

**Root cause**: `fused_attn_fwd_noalloc` stable op exists (47 args, registered). `fused_attn_bwd_noalloc` exists in C++ but can't be registered (77 args > 64 limit).

**Fix for fwd**: Write Python wrapper that extracts raw buffers from Q/K/V/S/O, converts NVTE enum args to `int()`, calls `_ops.fused_attn_fwd_noalloc(...)`, unpacks the 11-tuple return.

**Fix for bwd**: Split into two stable ops (`fused_attn_bwd_setup` + `fused_attn_bwd_execute`), each under 64 args. Or pack quantization metadata into fewer tensors to reduce arg count.

### Issue 5: `multi_tensor_quantize` / `split_quantize` / `group_quantize` (minor)

**Symptom**: `NotImplementedError` stubs.

**Fix**: Add stable C++ op using pointer-pack pattern. The fused kernel `nvte_multi_cast_transpose` is a pure C function taking `NVTETensor*` arrays. Python detects quantizer types, calls `make_empty` for outputs, packs pointers. C++ calls `nvte_multi_cast_transpose` for fused path (Float8 delayed scaling) or per-tensor `nvte_quantize_v2` for unfused.

### Issue 6: `swizzle_scales_for_gemm_` (minor)

**Symptom**: `NotImplementedError` stub.

**Fix**: Add stable C++ op that takes raw scale_inv tensors + scaling_mode, calls `nvte_swizzle_scaling_factors`, returns swizzled tensors. Python wrapper sets tensor attributes in-place.

### Issue 7: `clamped_swiglu` / `clamped_dswiglu` UNFUSED path (minor)

**Symptom**: UNFUSED path produces tensors that GEMM rejects for NVFP4/Float8Block quantizers.

**Fix**: The UNFUSED path does `hp_activation → quantize_new`. The `quantize_new` output's TensorWrapper metadata doesn't match what GEMM expects. Need to ensure `make_empty` + `quantize_into` produces identical metadata to C++ `create_tensor` + `quantize`.

### Issue 8: CommOverlap requires c10d::ProcessGroup (known limitation)

**Symptom**: Distributed comm overlap tests fail.

**Fix**: Future work — the CommOverlap hot path runs in `libtransformer_engine.so` (no PyTorch dependency). The `CommOverlapHelper` initialization needs `c10d::ProcessGroup` for allgather/barrier. Fix: move allgather/barrier callbacks to Python using `torch.distributed`.

## Recommended Next Steps

### Priority 1: Fix backward GEMM (Issue 1)
The highest-impact fix. Trace the exact failing backward GEMM call to find which tensor has the wrong shape. The fix is likely in how the TE module code reshapes tensors for backward — our `generic_gemm` may need to handle the case where D is pre-allocated by the caller (pass through without re-allocating).

### Priority 2: Add scale swizzling to stable GEMM (Issue 2)
Enables all FP8 tests. Add `nvte_swizzle_scaling_factors()` call in `csrc/stable/gemm.cpp` before `nvte_cublas_gemm_v2()`, matching the pybind `generic_gemm` behavior.

### Priority 3: Implement fused attention forward (Issue 4)
Write the Python wrapper over `fused_attn_fwd_noalloc`. For backward, split the 77-arg function into two stable ops.

### Priority 4: Implement grouped GEMM (Issue 3)
Add stable C++ ops using pointer-pack pattern for `nvte_multi_tensor_gemm` and `NVTEGroupedTensor` construction for `nvte_grouped_gemm`.

### Priority 5: Implement multi_tensor_quantize (Issue 5)
Add stable C++ op for `nvte_multi_cast_transpose` using pointer-pack pattern.

## Key Reference

### NVTEScalingMode values (transformer_engine.h)
```
NVTE_DELAYED_TENSOR_SCALING = 0
NVTE_MXFP8_1D_SCALING = 1
NVTE_BLOCK_SCALING_1D = 2
NVTE_BLOCK_SCALING_2D = 3
NVTE_NVFP4_1D_SCALING = 4
```

### TE DType values (transformer_engine.h)
```
kByte=0, kInt16=1, kInt32=2, kInt64=3, kFloat32=4, kFloat16=5,
kBFloat16=6, kFloat8E4M3=7, kFloat8E5M2=8, kFloat8E8M0=9, kFloat4E2M1=10
```

### Key files
```
transformer_engine/pytorch/_tex.py                    — routing module (remove star import here)
transformer_engine/pytorch/_stable_torch_module.py    — all Python implementations
transformer_engine/pytorch/tensor/_extract.py         — tensor metadata extraction
transformer_engine/pytorch/tensor/_quantize_stable.py — quantize dispatch
transformer_engine/pytorch/csrc/stable/              — 16 stable ABI C++ source files
transformer_engine/pytorch/csrc/stable_common.h      — shared C++ utilities
build_tools/pytorch.py                               — build config for te_stable_abi.so
```
