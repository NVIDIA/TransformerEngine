# Stable ABI: Missing Fusions and Performance Regressions

This document tracks pure-Python implementations in `_stable_torch_module.py` that
replace fused C++ kernels from the pybind11 extension, causing performance regressions.

## Critical — Python loops replacing fused kernels

### `_group_quantize_fallback` (NVFP4 path)
- **Issue**: Per-chunk quantize loop for non-MXFP8 quantizers (NVFP4, Float8Block)
- **pybind11**: `nvte_group_quantize` handles all chunks in one fused kernel
- **Impact**: O(num_tensors) kernel launches instead of 1
- **Fix**: Extend the C++ `group_quantize` op in `cast.cpp` to handle NVFP4

### `split_quantize`
- **Issue**: Per-split quantize loop with no bulk allocation optimization
- **pybind11**: Bulk-allocation for MXFP8/NVFP4 + fused kernel
- **Impact**: O(num_splits) separate quantization kernels
- **Fix**: Add C++ `split_quantize` op or bulk-allocation path

### `te_general_grouped_gemm`
- **Issue**: Per-GEMM loop calling `_ops.gemm()` individually
- **pybind11**: `nvte_multi_tensor_gemm` batches all GEMMs
- **Impact**: Loss of stream-level parallelism, ~10-30% throughput regression
- **Fix**: Add C++ wrapper for `nvte_multi_tensor_gemm`

### `multi_tensor_quantize`
- **Issue**: Stub that raises `NotImplementedError`
- **pybind11**: `nvte_multi_cast_transpose` fused kernel
- **Impact**: Runtime crash if called
- **Fix**: Add C++ wrapper for `nvte_multi_cast_transpose`

### NVFP4 multi-tensor ops (4 functions)
- `nvfp4_multi_tensor_fused_scale`
- `nvfp4_2d_multi_tensor_transpose`
- `nvfp4_multi_tensor_2d_partial_cast`
- `nvfp4_multi_tensor_compute_partial_amax`
- **Issue**: Per-tensor Python loops calling single-tensor stable ops
- **pybind11**: Direct C++ multi-tensor operations
- **Impact**: O(list_length) kernel launches each
- **Fix**: Add C++ wrappers accepting tensor lists

## Medium — Missing kernel fusion

### `layernorm_fwd` / `rmsnorm_fwd` quantize fusion
- **Issue**: Only fuses norm+quantize for Float8 delayed scaling; Block/MXFP8/NVFP4
  fall back to separate norm then quantize kernels
- **Impact**: 2 kernel launches instead of 1, every layer
- **Fix**: Extend `_try_fused_norm_quantize_*` to support more quantizer types

### Activation forward + quantize (NVFP4)
- **Issue**: Unfused path for NVFP4 activations — computes activation then quantizes
- **Impact**: Extra kernel launch per activation
- **Fix**: Extend fused activation+quantize to NVFP4

### `_make_dbias_dact` backward fusion
- **Issue**: For non-MXFP8, falls back to unfused dact + bias reduction + quantize
- **pybind11**: Single fused `dact_dbias_noalloc` kernel
- **Impact**: 3 operations instead of 1 in backward pass
- **Fix**: Extend stable ABI `dactivation_dbias_noalloc` to support more scaling modes

## Low — Minor inefficiencies

### `generic_gemm` on-the-fly transpose
- **Issue**: Computes FP8 transpose at GEMM time when columnwise data is missing
- **Impact**: 1-2 extra transpose kernels per GEMM (depends on tensor lifecycle)
- **Mitigation**: Only when `_transpose_invalid=True`

### NVFP4 stochastic rounding
- **Issue**: `quantize_into` does not pass stochastic rounding flag to C++ kernel
- **Impact**: Missing feature, not a perf regression — tests skip/fail
- **Fix**: Add stochastic rounding parameter to stable ABI quantize ops
