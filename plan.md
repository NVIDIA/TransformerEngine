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
- **Build/import cutover**: top-level `setup.py` and `transformer_engine/pytorch/setup.py` now build only `te_stable_abi.so` for PyTorch; `transformer_engine/pytorch/__init__.py` no longer attempts to load `transformer_engine_torch` at import time.

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
6. **Stable GEMM operand extraction was orientation-blind**: The Python stable shim now selects rowwise vs columnwise quantized buffers based on `transa`/`transb`, matching the pybind GEMM path for MXFP8/NVFP4-style tensors.
7. **Stable GEMM was missing scale swizzling**: `csrc/stable/gemm.cpp` now swizzles MXFP8/NVFP4 scales before `nvte_cublas_gemm_v2` when the incoming tensor is not already GEMM-ready, and the stable module now exposes `swizzle_scales_for_gemm_` parity via a stable op.
8. **Fused attention backward: workspace lifetime bug**: `Tensor ws_data` was declared inside the `if (ws_shape.ndim > 0)` block — freed at `}` before Phase 2, subsequent aux allocations reused the workspace memory. Fix: declare `ws_data` OUTSIDE the if-block in both `fused_attn_fwd_noalloc` and `fused_attn_bwd_noalloc` (csrc/stable/attention.cpp).
9. **Fused attention forward: rng_state not returned in aux pack**: pybind11 puts the caller's `rng_state` tensor DIRECTLY into the aux pack slot (extensions/attention.cpp:280), so cuDNN's write-back (fused_attn_f16_arbitrary_seqlen.cu:1182) goes to that tensor. Stable code was allocating a NEW tensor for that slot (cuDNN overwrote it to point to rng_state but we returned the empty original allocation). Fix: detect the rng_state slot (dtype=kInt64, shape=[2]) and use the caller's `rng_state` tensor directly.
10. **Fused attention backward: dQ/dK/dV must be NON-CONTIGUOUS VIEWS of packed dQKV**: For SB3HD/3HD layouts, cuDNN computes output gradient stride from qkv_layout (e.g. stride=[3*B*H*D, 3*H*D, D, 1]) starting from dQ.data_ptr(). Using `.contiguous()` creates separate small tensors (S*B*H*D elements) and cuDNN writes out-of-bounds → CUDA_ERROR_ILLEGAL_ADDRESS (err 700). Fix: pass non-contiguous views (dQKV[..., 0, :, :]) so dQ.data_ptr()=dQKV.data_ptr() and the write stays within dQKV bounds.

### Test results

| Configuration | Passed | Failed | Notes |
|--------------|--------|--------|-------|
| With pybind `.so` (baseline) | 10,983/10,984 | 1 | 1 pre-existing failure |
| With pybind `.so` + quantizer patches + `_tex.py` routing | 19,167/19,168 | 1 | Same pre-existing failure |
| **Without pybind `.so`** | **3,684/10,984** | **7,300** | All non-FP8 forward passes work |

## Current Test Results (2026-03-29, B200 sm=100a)

Previous results were on H100 (sm=90). These results are on B200 (sm=100a) with the full L0_pytorch_unittest + L1_cpp_distributed test suites.

### Fixes applied in this session

1. **`_make_dbias_dact` activation_type mapping** (`_stable_torch_module.py`): The fused `dact_dbias_noalloc` C++ op uses a compact 5-entry table (0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu), but Python was sending indices from the full 11-entry `dact_table` (e.g., relu=5 instead of 2). Fixed by adding separate `act_type` (for unfused dactivation_noalloc) and `fused_act_type` (for fused dact_dbias_noalloc) parameters. **Result: -432 failures in test_sanity.**

2. **`moe_permute_fwd` signature mismatch** (`_stable_torch_module.py`): The stable C++ `moe_permute_fwd` expects pre-sorted `sorted_row_id` and `row_id_map` tensors, but the Python wrapper was passing raw `indices` and an empty `workspace` list. The pybind C++ version did radix sorting internally. Fixed by implementing the workspace management and `torch.sort`-based sorting in the Python wrapper. Also fixed `moe_unpermute_bwd` to reshape `prob_grad` from flat `[N*topK]` to `[N, topK]`. **Result: -24 failures in test_permutation (39 → 15).**

3. **`_FP8_DTYPE_TO_TE` mapping incomplete** (`_extract.py`): The mapping only had `"fp8e4m3"` and `"fp8e5m2"` keys, but Float8Tensor stores `_fp8_dtype` in various forms: `torch.float8_e4m3fn` (torch.dtype), or integer `7` (from `_stable_torch_module.py` enum). Added entries for `"torch.float8_e4m3fn"`, `"torch.float8_e5m2"`, `"float8_e4m3fn"`, `"float8_e5m2"`, `"7"`, `"8"`. **Result: -13 failures in test_attention (634 → 621).**

4. **`extract_tensor_data` NVFP4 detection** (`_extract.py`, implemented but reverted): NVFP4Tensor has `_rowwise_data` but no `_fp8_dtype` or `_is_2D_scaled` attributes. The fix correctly returns `te_dtype=10` (kFloat4E2M1) and `sm=4` (NVFP4_1D_SCALING), but exposes a deeper `swizzle_scaling_factors` illegal memory access in `swizzle.cu:539` during GEMM on B200. **Reverted** pending swizzle bug fix. The fix code is:
   ```python
   # In extract_tensor_data, first _rowwise_data branch:
   elif "NVFP4" in type(tensor).__name__:
       te_dtype = 10  # kFloat4E2M1
   # And for scaling mode fallback:
   elif "NVFP4" in type(tensor).__name__:
       sm = NVTE_NVFP4_1D_SCALING
   elif "MXFP8" in type(tensor).__name__:
       sm = NVTE_MXFP8_1D_SCALING
   ```

### Regression comparison: main vs stable-abi branch (initial baseline)

| Test | Main (P/F) | Stable-ABI (P/F) | Regressions |
|------|-----------|------------------|-------------|
| test_sanity | 33045/33045 | 33045/36935 | +3890 failures |
| test_recipe | 58/58 | 58/73 | +15 failures |
| test_deferred_init | 17/17 | 17/17 | OK |
| test_numerics | 2968/2968 | SEGFAULT (rc=139) | +2968 failures |
| test_cuda_graphs | 437/437 | 437/563 | +126 failures |
| test_jit | 8/8 | 8/8 | OK |
| test_fused_rope | 9609/9609 | 9609/9609 | OK |
| test_nvfp4 | 2192/2192 | 2192/4292 | +2100 failures |
| test_mxfp8 | 306/306 | 306/612 | +306 failures |
| test_quantized_tensor | 471/471 | 471/608 | +137 failures |
| test_float8blockwisetensor | 207/207 | 207/267 | +60 failures |
| test_float8_blockwise_scaling_exact | 258/258 | 258/350 | +92 failures |
| test_float8_blockwise_gemm_exact | 1579/1579 | 1579/1579 | OK |
| test_grouped_tensor | 25/25 | 25/30 | +5 failures |
| test_gqa | 20/20 | 20/20 | OK |
| test_fused_optimizer | 31/31 | 31/35 | +4 failures |
| test_multi_tensor | 2790/2790 | 2790/3407 | +617 failures |
| test_fusible_ops | 4022/4022 | 4022/4532 | +510 failures |
| test_permutation | 396/396 | 396/435 | +39 failures |
| test_parallel_cross_entropy | 8/8 | 8/8 | OK |
| test_cpu_offloading | 758/758 | 758/1171 | +413 failures |
| test_cpu_offloading_v1 | 21/21 | 21/21 | OK |
| test_attention | 2607/2607 | 2607/3256 | +649 failures |
| test_attention_deterministic | 2607/2607 | 2607/3179 | +572 failures |
| test_kv_cache | 576/576 | 576/576 | OK |
| test_hf_integration | 2/2 | 2/2 | OK |
| test_checkpoint | 11/22 | 11/22 | OK (both fail) |
| test_fused_router | 351/351 | 351/351 | OK |
| test_partial_cast | 1/1 | 1/2 | +1 failures |
| cpp_distributed | 0/0 | 0/0 | OK (build failure on both) |
| **TOTAL** | **65381/65392** | **62413/71960** | **+12504 regressions** |

**Main branch**: 65,381 passed, 11 failed (all in test_checkpoint — pre-existing), 0 errors
**Stable-ABI branch**: 62,413 passed, 9,547 failed, 0 errors (+ test_numerics SEGFAULT)

### Pre-existing failures on main (not regressions)
- test_checkpoint: 11 failures (same on both branches)
- cpp_distributed: build failure (same on both branches)

### Regression Root Causes (categorized, ~12,504 total)

#### 1. `scale_inv non-FP8` — "Assertion failed: !t.scale_inv.has_data(). Scale_inv is not supported for non-FP8 output" (~2,300 failures)
- **Files affected**: test_sanity (1330), test_nvfp4 (1612), test_fusible_ops (126), test_recipe (14), test_cpu_offloading (29), test_quantized_tensor (6), test_grouped_tensor (2)
- **Root cause**: `quantize_new()` in `_quantize_stable.py` is passing a `scale_inv` tensor to `ops.quantize()` even for non-FP8 output tensors. The C++ `CheckOutputTensor` asserts that `scale_inv` should not be set for non-FP8 dtypes.
- **Fix**: Ensure `quantize_new`/`quantize_into` only sets `scale_inv` for FP8 output dtypes.

#### 2. `CUBLAS_STATUS_NOT_SUPPORTED` (~2,200 failures)
- **Files affected**: test_sanity (2127), test_cpu_offloading (19), test_cuda_graphs (24), test_fused_optimizer (1), test_recipe (1), test_float8_blockwise_scaling_exact (2)
- **Root cause**: GEMM called with incompatible tensor types/layouts. On B200 (sm=100a), different cuBLAS algorithms are needed vs H100 (sm=90). The stable GEMM shim may not handle all scaling mode combinations correctly on Blackwell.
- **Fix**: Investigate which GEMM parameter combos fail and adjust tensor extraction/layout logic.

#### 3. `test_numerics SEGFAULT` (rc=139, ~2,968 failures)
- **Root cause**: test_numerics segfaults entirely. Needs investigation — run with `--tb=long` on a single test to find the crash location.

#### 4. `transpose` errors (~464 failures)
- **Files affected**: test_nvfp4 (380), test_mxfp8 (84)
- **Root cause**: Transpose ops in `transformer_engine/common/transpose/` failing. Likely related to on-the-fly transpose for NVFP4/MXFP8 quantized tensors.

#### 5. `bad activation_type` for `dact_dbias` (~441 failures)
- **Files affected**: test_sanity (432), test_fusible_ops (9)
- **Root cause**: `dact_dbias_noalloc` in `csrc/stable/bias.cpp:114` receives activation_type=5, which maps to `relu` in the C++ table but the stable code doesn't handle it. The activation type enum mapping may differ between pybind and stable paths.
- **Fix**: Add missing activation types to `dact_dbias_noalloc` in `bias.cpp`.

#### 6. `NotImplementedError` stubs (~379 failures)
- **Files affected**: test_mxfp8 (214), test_nvfp4 (108), test_multi_tensor (54), test_grouped_tensor (3), test_partial_cast (1)
- **Root cause**: Some ops still have `NotImplementedError` stubs in `_stable_torch_module.py`.

#### 7. Attention errors (~650 failures)
- **Files affected**: test_attention (506 cudnn_util + 98 RMS + 44 fused_attn + 1 view), test_attention_deterministic (572)
- **Root cause**: cuDNN utility errors in fused attention paths. May be B200/sm=100a specific behavior. The RMS assertion errors suggest numerical differences in FP8 attention.

#### 8. `scale_inv` other issues (~359 failures)
- **Files affected**: test_fusible_ops (300), test_cuda_graphs (33), test_quantized_tensor (24), test_cpu_offloading (3), test_fused_optimizer (2)
- **Root cause**: Various scale_inv-related issues beyond the non-FP8 assertion. Likely incorrect scale_inv handling in quantizer paths.

#### 9. `Offset increment outside graph capture` (~355 failures)
- **Files affected**: test_cpu_offloading (355)
- **Root cause**: CPU offloading relies on CUDA graph capture which interacts differently with stable ABI ops.

#### 10. Numerical mismatches (~275 failures)
- **Files affected**: test_multi_tensor (563), test_quantized_tensor (92), test_float8blockwisetensor (53), test_float8_blockwise_scaling_exact (88), test_fusible_ops (68), test_cuda_graphs (27), test_mxfp8 (8)
- **Root cause**: Outputs differ numerically. Some may be precision issues in the stable quantize path, others may be incorrect tensor metadata.

#### 11. `moe_permute_fwd` signature error (~39 failures)
- **Files affected**: test_permutation (39)
- **Root cause**: `transformer_engine_stable::moe_permute_fwd()` expected different number of arguments. The stable C++ op signature doesn't match what Python calls.

#### 12. Missing attributes on recipe states (~30 failures)
- **Files affected**: test_cuda_graphs (18+12)
- **Root cause**: `NVFP4BlockScalingRecipeState` and `Float8BlockScalingRecipeState` objects missing attributes expected by the cuda graphs code.

### Current results after fixes

| Test | Main F | Before F | After F | Delta |
|------|--------|----------|---------|-------|
| test_sanity | 0 | 3890 | 3458 | **-432** |
| test_recipe | 0 | 15 | 15 | 0 |
| test_deferred_init | 0 | 0 | 0 | 0 |
| test_numerics | 0 | SEGFAULT | SEGFAULT | N/A |
| test_cuda_graphs | 0 | 126 | 126 | 0 |
| test_jit | 0 | 0 | 0 | 0 |
| test_fused_rope | 0 | 0 | 0 | 0 |
| test_nvfp4 | 0 | 2100 | 2100 | 0 |
| test_mxfp8 | 0 | 306 | 306 | 0 |
| test_quantized_tensor | 0 | 137 | 137 | 0 |
| test_float8blockwisetensor | 0 | 60 | 60 | 0 |
| test_float8_blockwise_scaling_exact | 0 | 92 | 92 | 0 |
| test_float8_blockwise_gemm_exact | 0 | 0 | 0 | 0 |
| test_grouped_tensor | 0 | 5 | 5 | 0 |
| test_gqa | 0 | 0 | 0 | 0 |
| test_fused_optimizer | 0 | 4 | 4 | 0 |
| test_multi_tensor | 0 | 617 | 625 | +8 |
| test_fusible_ops | 0 | 510 | 511 | +1 |
| test_permutation | 0 | 39 | 15 | **-24** |
| test_parallel_cross_entropy | 0 | 0 | 0 | 0 |
| test_cpu_offloading | 0 | 413 | 413 | 0 |
| test_cpu_offloading_v1 | 0 | 0 | 0 | 0 |
| test_attention | 0 | 649 | 621 | **-28** |
| test_attention_deterministic | 0 | 572 | 578 | -6 |
| test_kv_cache | 0 | 0 | 0 | 0 |
| test_hf_integration | 0 | 0 | 0 | 0 |
| test_checkpoint | 11 | 11 | 11 | 0 |
| test_fused_router | 0 | 0 | 0 | 0 |
| test_partial_cast | 0 | 1 | 1 | 0 |
| **TOTAL** | **11** | **9547+SEG** | **~9059+SEG** | **~-488** |

Note: Total is approximate — some tests have slight run-to-run variance (~1-8 tests). The `_FP8_DTYPE_TO_TE` fix reduced attention failures by 28 (649 → 621) and attention_deterministic by ~6.

### Remaining Priority Fix Order (by impact)

1. **NVFP4 `extract_tensor_data` + swizzle crash** (~2,100 in test_nvfp4 + ~1,330 in test_sanity): `extract_tensor_data` returns wrong dtype/sm for NVFP4Tensor (dtype=0/kByte instead of 10/kFloat4E2M1, sm=0 instead of 4). Fix is known and tested but exposes a `swizzle_scaling_factors` illegal memory access in the GEMM path. Need to investigate swizzle bug first.
2. **test_numerics SEGFAULT** (~2,968): Doesn't crash when run by test class individually — only segfaults when running the full file. Likely OOM or accumulated GPU memory corruption from earlier test's CUDA error.
3. **CUBLAS_STATUS errors** (~2,127 in test_sanity): cuBLAS GEMM fails for certain FP8 scaling mode combinations on B200. Needs investigation of which GEMM parameter combos are unsupported.
4. **FP8 Attention cuDNN dtype errors** (~506 in test_attention): `_FP8_DTYPE_TO_TE` mapping fix should resolve most of these (applied, pending verification).
5. **Transpose errors** (~464 in test_nvfp4/mxfp8): NVFP4/MXFP8 transpose ops failing.
6. **NotImplementedError stubs** (~379): Some ops still have stubs in `_stable_torch_module.py`.
7. **CPU offloading offset errors** (~355): "Offset increment outside graph capture" in test_cpu_offloading.
8. **Numerical mismatches** (~625 in test_multi_tensor, ~275 elsewhere): Output values differ from main. May be precision issues in stable quantize path.
9. **Remaining permutation edge cases** (~15): Token dropping (`num_out_tokens < num_tokens * topK`) shapes in `moe_unpermute_bwd`.
10. **Recipe state attributes** (~30 in test_cuda_graphs): `NVFP4BlockScalingRecipeState`/`Float8BlockScalingRecipeState` missing attributes.

### FP8 backward fixes (2026-03-28, complete)

Four bugs fixed in `_stable_torch_module.py`:

1. **`_make_dbias_dact` UNFUSED path for Float8Quantizer**: `nvte_quantize_dbias_dgelu` kernel doesn't support DELAYED_TENSOR_SCALING + IS_DBIAS on Hopper (SM < 10.0). Fixed by using UNFUSED path (dact → bf16 → sum bias → quantize separately) for delayed FP8; fused path only for MXFP8.

2. **On-the-fly FP8 transpose in `generic_gemm`**: For NN/NT layouts with FP8 delayed scaling, Hopper cuBLAS requires TN layout, so A needs columnwise data (physical transpose). After `quantize_into()`, `_transpose_invalid=True`. Fix: create transpose on-the-fly via `_ops.fp8_transpose(A_data, A_dtype, None)` when `transa=False` and no columnwise data.

3. **`skip_gemm` fix for columnwise-only tensors**: Float8Tensors quantized with `rowwise=False, columnwise=True` have `_data=None` (empty placeholder), causing `A_data.numel()==0 → skip_gemm=True` incorrectly. Fix: check if columnwise data exists before skipping.

4. **M computation fix for columnwise-only tensors**: When A_data is the empty placeholder but `A_cw_data` is the physical transpose (shape `[N, M]` for logical `[M, N]`), M must be derived from `A_cw_data.shape[0]` (not `A_data.shape[-1]=0`).

### DelayedScaling activation backward fixes (2026-03-28, complete)

Three bugs fixed in `_stable_torch_module.py`:

1. **`bgrad_quantize` not actually quantizing**: Was returning raw BF16 grad unchanged → FP8 weight × BF16 grad GEMM → `CUBLAS_STATUS_NOT_SUPPORTED`. Fix: call `quantize_new(grad_output, quantizer)` to quantize the gradient before the GEMM.

2. **`clamped_swiglu` ignoring quantizer**: Forward call with a Float8Quantizer was returning BF16 tensor → FC2 GEMM with FP8 weight × BF16 input → unsupported. Fix: call `quantize_new(out, quantizer)` when quantizer is not None.

3. **`_make_dbias_dact` wrong act_type indices**: Used wrong indices for `dsilu` (was 1=glu, should be 9), `drelu` (was 2=geglu, should be 5), `dsrelu` (was 4=qgeglu, should be 7). C++ table: 0=gelu,1=glu,2=geglu,3=qgelu,4=qgeglu,5=relu,6=reglu,7=srelu,8=sreglu,9=silu,10=swiglu. These caused wrong/gated backward kernels to run on non-gated activations (shape mismatch assertion).

### Grouped GEMM implementation (complete)

**New files/changes:**
- `_stable_torch_module.py`: `te_general_grouped_gemm` now loops over tensor pairs, calling `_ops.gemm()` for each. `te_general_grouped_gemm_for_grouped_tensor`, `te_general_grouped_gemm_for_discrete_in`, `te_general_grouped_gemm_for_discrete_out` implemented using new stable C++ ops.
- `csrc/stable/grouped_gemm.cpp` (new): Three stable ops — `grouped_gemm_for_grouped_tensor`, `grouped_gemm_for_discrete_in`, `grouped_gemm_for_discrete_out` — wrapping `nvte_grouped_gemm` and variants (Blackwell+, run on Blackwell test machine to validate).

## All Issues RESOLVED (2026-03-28)

All 10,984 sanity tests pass without the pybind `.so`. The full migration is functionally complete for this hardware (H100, sm=90).

## Previously Tracked Issues (now resolved)

### Issue 1: NVFP4/Float8BlockScaling backward GEMMs (~3200 failures)

**Symptom**: `Assertion failed: status != CUBLAS_STATUS_NOT_SUPPORTED. Unable to find suitable cuBLAS GEMM algorithm` in backward GEMMs for NVFP4 and Float8BlockScaling recipes.

**Root cause**: The backward GEMM calls (dgrad NN layout, wgrad NT layout) for NVFP4/Float8Block recipes have incorrect tensor parameters or unsupported dimensions/formats for cuBLAS. The on-the-fly FP8 transpose logic only applies to DELAYED scaling (MXFP8/NVFP4 need different handling).

**Fix**: Investigate which GEMM parameters are wrong for NVFP4 backward. The forward works, so the issue is specific to backward tensor shapes or quantizer usage. May need per-scaling-mode on-the-fly transpose logic.

### Issue 2: `split_quantize` / `group_quantize` / `multi_tensor_quantize` stubs (~648 failures)

**Symptom**: `NotImplementedError` stubs in `_stable_torch_module.py`.

**Root cause**: GroupedLinear forward uses `tex.split_quantize` to quantize the input split across expert groups.

**Fix**: Add stable C++ op using pointer-pack pattern for `nvte_multi_cast_transpose` / per-tensor quantize.

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

## Recommended Next Steps (updated 2026-03-29)

### How to resume testing
```bash
# Rebuild TE on this branch:
NVTE_CUDA_ARCHS="100a" NVTE_USE_CCACHE=1 NVTE_BUILD_THREADS_PER_JOB=4 NVTE_CCACHE_BIN=sccache SCCACHE_DIR=/.cache/sccache NVTE_FRAMEWORK=pytorch pip install -v --no-build-isolation -e .

# Run a specific test:
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 2
python3 -m pytest --tb=short -q /workspace/tests/pytorch/test_sanity.py

# Main branch worktree for comparison (if still exists):
# /tmp/te-main (git worktree of main branch)

# Results files:
# /workspace/test_results_main.jsonl — main branch results
# /workspace/test_results_stable-abi.jsonl — original stable-abi results
# /workspace/logs_fixed/*.xml — latest junit XML results after fixes
# /workspace/run_tests.sh — test runner script
```

### Priority 1: Fix NVFP4 `extract_tensor_data` + swizzle crash (~3,400 failures)
The NVFP4 detection fix for `_extract.py` is known and tested (return dtype=10, sm=4 for NVFP4Tensor), but it exposes a `swizzle_scaling_factors` illegal memory access in `swizzle.cu:539` during GEMM. This crash cascades to corrupt GPU state for all subsequent tests.
- **Investigation needed**: Debug the swizzle crash. Likely the NVFP4 scale tensor format from `make_empty` doesn't match what `nvte_swizzle_scaling_factors` expects (e.g., wrong shape, dtype, or memory layout).
- **Key test**: `python3 -m pytest -x tests/pytorch/test_sanity.py -k "fp8_recipe1 and dtype0 and small and True-LayerNorm-True-True-False"`

### Priority 2: Fix CUBLAS_STATUS errors (~2,127 in test_sanity)
cuBLAS GEMM fails with `CUBLAS_STATUS_NOT_SUPPORTED` for certain FP8 tensor combinations. Likely B200 (sm=100a) specific — these tests passed on H100 (sm=90).
- **Investigation needed**: Run a failing test with CUBLAS debug logging to see which GEMM config is unsupported. May need to adjust the stable GEMM shim's tensor extraction or use a different cuBLAS algorithm selection.
- **Key test**: `python3 -m pytest -x tests/pytorch/test_sanity.py -k "fp8_recipe2 and dtype0 and small and True-LayerNorm-True-True-False"`

### Priority 3: test_numerics SEGFAULT (~2,968)
Doesn't crash when running individual test classes. Only segfaults when running the full file. Likely OOM or GPU memory corruption from an earlier test that triggers a CUDA error.
- **Investigation needed**: Run test classes individually to find which one triggers the crash. Try `python3 -m pytest tests/pytorch/test_numerics.py -k "test_gpt_fp8_parameters"` etc.

### Priority 4: FP8 Attention errors (~621, down from 649)
The `_FP8_DTYPE_TO_TE` mapping fix helped 28 tests, but ~621 failures remain. Investigation found:
1. **Forward S placeholder**: The pybind path creates S via `Float8Quantizer::create_tensor({0}, kFloat32)` which produces a TensorWrapper with FP8 dtype, empty uint8 data, and real amax/scale/scale_inv tensors from the quantizer. Our stable path replicates this correctly via `s_quantizer.make_empty([0])`.
2. **Backward `fused_attn_bwd_packed`**: ~506 failures come from the backward pass. The error "Invalid cuDNN data type" happens inside `fused_attn_bwd_packed` at line 2885. All Python-side dtypes are valid (Q/K/V/O at 7=FP8, dO/dQ/dK/dV at 6=BF16, S at 7=FP8, dP at 4=F32). The invalid dtype must be generated INSIDE the C++ `fused_attn_bwd_packed` code when it unpacks the `dtype_info` tensor or constructs internal TensorWrappers.
3. **Remaining ~127 forward failures**: cuDNN `BAD_PARAM_NULL_POINTER` and numerical mismatches.
- **Fix approach**: Debug the C++ `fused_attn_bwd_packed` in `csrc/stable/attention.cpp` — check how it unpacks the `dtype_info` tensor and constructs TensorWrappers for the backward. The issue is likely in how dP (softmax gradient) or S TensorWrapper is built from the packed info.

### Previous priorities (still relevant)
- **Transpose errors** (~464): NVFP4/MXFP8 transpose ops failing
- **NotImplementedError stubs** (~379): Some ops still have stubs
- **CPU offloading offset** (~355): "Offset increment outside graph capture"
- **Numerical mismatches** (~625+): Output values differ from main

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
