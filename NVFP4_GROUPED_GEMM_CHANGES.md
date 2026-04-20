# Per-Token NVFP4 Grouped GEMM — Change Summary

## Overview

Added per-token NVFP4 global scale support to the cuDNN Frontend grouped GEMM kernels, and a new `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` fused op class in TransformerEngine to use it.

**Scope:** Forward pass only. NVFP4 (FP4 E2M1 data + FP8 E4M3 block scales + FP32 per-token global scale). Backward falls back to unfused path.

---

## cuDNN Frontend Changes

### New Parameter: `global_scale_tensor`

A new optional `global_scale_tensor` parameter was added to both the GLU and quant grouped GEMM kernels. It carries a per-token FP32 global scale that is applied to the accumulator after the per-expert `alpha` multiply and before the activation function.

- **Shape:** `(valid_m, S, 1)` where S=1 for per-token, S>1 for future subchannel scaling
- **Default:** `None` (no-op, zero overhead — compile-time `const_expr` guard)
- **Kernel behavior:** `acc = acc * alpha[expert] * global_scale[token] -> activation(acc)`

### Files Modified

| File | Change |
|------|--------|
| `python/cudnn/grouped_gemm/grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py` | Added `enable_global_scale` to `__init__`, `global_scale` param to `__call__`/kernel. Per-token load via `get_gmem_tensor("global_scale", ...)`, FP32 multiply on accumulator after alpha. |
| `python/cudnn/grouped_gemm/grouped_gemm_quant/grouped_gemm_quant.py` | Same kernel changes for the quant (FC2) path. |
| `python/cudnn/grouped_gemm/moe_sched_extension.py` | Registered `"global_scale"` in the M-dimension tensor category (alongside `prob`, `c`, `d`) for both contiguous and discrete extensions. |
| `python/cudnn/grouped_gemm/grouped_gemm_glu/api.py` | Added `sample_global_scale`/`global_scale_tensor` to `GroupedGemmGluSm100.__init__`, shape validation, dense+discrete compile paths, `tensor_api` closures, `execute`, `grouped_gemm_glu_wrapper_sm100`, and cache keys. |
| `python/cudnn/grouped_gemm/grouped_gemm_quant/api.py` | Same API plumbing for `GroupedGemmQuantSm100` and `grouped_gemm_quant_wrapper_sm100`. |

### Files Created

| File | Description |
|------|-------------|
| `test/python/fe_api/test_grouped_gemm_glu_nvfp4.py` | 3 L0 tests: backward compat (`None`), identity (`ones`), functional scaling (`2x`). All pass on B200. |

### Test Results (cuDNN Frontend)

| Test Suite | Result |
|------------|--------|
| `test_grouped_gemm_swiglu.py` | 58 passed, 94 skipped (no regression) |
| `test_grouped_gemm_glu.py` | 312 passed, 233 skipped (no regression) |
| `test_grouped_gemm_glu_nvfp4.py` | 3 passed (new) |

---

## TransformerEngine Changes

### New Class: `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4`

A new fused operation class for NVFP4 forward grouped MLP, modeled after `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8`.

**Enabled by:** `NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1` environment variable.

### Files Modified

| File | Change |
|------|--------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | Added `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` class (~300 lines), `fuse_forward_ops_nvfp4` registration function. Imported `NVFP4Quantizer` and `NVFP4_BLOCK_SCALING_SIZE`. |
| `transformer_engine/pytorch/ops/fused/__init__.py` | Added `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` to exports. |

### MXFP8 vs NVFP4 Fused Op Comparison

| Aspect | MXFP8 | NVFP4 |
|--------|-------|-------|
| Env var | `NVTE_CUTEDSL_FUSED_GROUPED_MLP` | `NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4` |
| Data dtype | `float8_e4m3fn` | `float4_e2m1fn_x2` (via `.view()` from `uint8`) |
| Scale dtype | `float8_e8m0fnu` | `float8_e4m3fn` (via `.view()` from `uint8`) |
| Block size | 32 (`MXFP8_BLOCK_SCALING_SIZE`) | 16 (`NVFP4_BLOCK_SCALING_SIZE`) |
| `sf_vec_size` | 32 | 16 |
| FC1 `d_dtype` | `float8_e4m3fn` (re-quant for FC2) | `bfloat16` (no SFD generation) |
| `discrete_col_sfd` | `True` | `False` (FC2 input re-quantized separately) |
| `global_scale_tensor` | Not used (`None`) | Per-token FP32 from `amax / (fp4_max * fp8_max)` |
| FC2 input | Direct from FC1 SFD output (zero-copy) | Re-quantized BF16 -> NVFP4 |

### Data Flow

```
MXFP8 path:
  Input(BF16) -> MXFP8 quant -> FC1 GEMM+SwiGLU -> FP8 output + SFD scales
                                                     |
                                                     v (zero-copy)
                                              FC2 GEMM+quant -> Output(BF16)

NVFP4 path:
  Input(BF16) -> NVFP4 quant -> FC1 GEMM+SwiGLU -> BF16 output
                  + global_scale    + global_scale    |
                                                      v (re-quantize to NVFP4)
                                              FC2 GEMM+quant -> Output(BF16)
                                                + global_scale
```

---

## Per-Token NVFP4 Recipe and Backward Override

### New Recipe: `NVFP4PerTokenBlockScaling`

Subclass of `NVFP4BlockScaling` that enables per-token global scaling in the forward grouped GEMM path. Backward precision is controlled by `NVTE_BACKWARD_OVERRIDE` (same as MXFP8 per PR #2644).

**Usage:**
```python
from transformer_engine.common.recipe import NVFP4PerTokenBlockScaling

# Forward: NVFP4 per-token, Backward: high-precision (BF16)
recipe = NVFP4PerTokenBlockScaling(backward_override="high_precision")

# Forward: NVFP4 per-token, Backward: dequantized
recipe = NVFP4PerTokenBlockScaling(backward_override="dequantized")

# Or via env var:
# NVTE_BACKWARD_OVERRIDE=high_precision
recipe = NVFP4PerTokenBlockScaling()
```

**Env var to enable fused path:** `NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1`

### Files Modified/Created for Recipe

| File | Change |
|------|--------|
| `transformer_engine/common/recipe/__init__.py` | Added `NVFP4PerTokenBlockScaling` recipe class (subclass of `NVFP4BlockScaling`) and `nvfp4_pertoken()` class method on `Recipe`. |
| `transformer_engine/pytorch/quantization.py` | Added `NVFP4PerTokenBlockScalingRecipeState` (inherits from `NVFP4BlockScalingRecipeState`). Registered in factory before `nvfp4()` check. |
| `transformer_engine/pytorch/ops/_common.py` | Updated `fuse_grouped_mlp_ops` recipe check: `recipe.mxfp8() or recipe.nvfp4_pertoken()`. |

### Per-Token Quantization Kernel Placeholder

| File | Description |
|------|-------------|
| `transformer_engine/common/cast/nvfp4/quantize_pertoken_nvfp4.cuh` | **New placeholder** — CUDA kernel header for per-token NVFP4 quantization. Documents the scaling hierarchy, parameters, and TODO items for implementation. |

### Backward Override Flow

The backward override is inherited from `NVFP4BlockScaling` and works identically to the MXFP8 pattern (PR #2644):

1. **Forward:** `grouped_linear.py` reads `recipe.backward_override`
2. **If `"high_precision"`:** saves original high-precision input before quantization
3. **If `"dequantized"`:** saves quantized input, dequantizes in backward
4. **If `None`:** standard NVFP4 backward (unfused, since backward kernels don't support `global_scale_tensor` yet)

No module-level changes needed — `grouped_linear.py` automatically respects the `backward_override` field from any `Recipe` subclass.

---

## Open Items

1. **Per-token quantization kernel** — `quantize_pertoken_nvfp4.cuh` is a placeholder. Currently, the per-tensor amax is broadcast to all tokens as an approximation. The kernel needs to: (a) compute per-row amax via parallel reduction, (b) derive per-row global_scale, (c) quantize with per-row scales. Also requires changes to `NVFP4Quantizer.make_empty()` to allocate `(M,)` shaped amax and C++ bindings in `cast.cpp`.

2. **FC2 input re-quantization overhead** — The MXFP8 path avoids re-quantization by having FC1 output SFD (scale factor D) directly in FP8 format. The NVFP4 path outputs BF16 from FC1 and re-quantizes to NVFP4 for FC2 input. This can be optimized by enabling `discrete_col_sfd=True` with NVFP4 output dtype in a future iteration.

3. **Backward pass** — `global_scale_tensor` is forward-pass only. The backward kernels (`grouped_gemm_dglu`, `grouped_gemm_dswiglu`) do not yet support it. Backward falls back to the unfused path.

4. **No runtime overhead for existing MXFP8 path** — `enable_global_scale` is a compile-time constant (`cutlass.const_expr`). When `False`, the compiler eliminates dead branches entirely.

---

## Verification Commands

Run these in an environment with both TE (built with C++ extensions) and cuDNN Frontend installed.

### Prerequisites

```bash
# Ensure cudnn-frontend source is on PYTHONPATH (for the global_scale_tensor changes)
export PYTHONPATH=/path/to/cudnn-frontend/python:$PYTHONPATH
```

### 1. Verify Imports and Recipe

```bash
python -c "
from transformer_engine.common.recipe import (
    NVFP4BlockScaling,
    NVFP4PerTokenBlockScaling,
)

# Recipe class hierarchy
r = NVFP4PerTokenBlockScaling()
print('nvfp4():', r.nvfp4())                   # True (subclass of NVFP4BlockScaling)
print('nvfp4_pertoken():', r.nvfp4_pertoken())  # True
print('mxfp8():', r.mxfp8())                   # False

# Backward override
r_hp = NVFP4PerTokenBlockScaling(backward_override='high_precision')
print('backward_override:', r_hp.backward_override)  # high_precision

r_dq = NVFP4PerTokenBlockScaling(backward_override='dequantized')
print('backward_override:', r_dq.backward_override)  # dequantized
"
```

### 2. Verify Fused Op Class Loads

```bash
NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1 python -c "
from transformer_engine.pytorch.ops.fused.forward_grouped_mlp import (
    ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8,
    ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4,
)
print('MXFP8 supported:', ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported())
print('NVFP4 supported:', ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4.is_supported())
# Both should be True on Blackwell (SM100) with cuDNN frontend installed
"
```

### 3. Verify Recipe State Factory

```bash
python -c "
from transformer_engine.pytorch.quantization import make_recipe_state
from transformer_engine.common.recipe import (
    NVFP4BlockScaling,
    NVFP4PerTokenBlockScaling,
)

# Standard NVFP4 -> NVFP4BlockScalingRecipeState
state1 = make_recipe_state(NVFP4BlockScaling(), mode='forward')
print('NVFP4:', type(state1).__name__)

# Per-token NVFP4 -> NVFP4PerTokenBlockScalingRecipeState
state2 = make_recipe_state(NVFP4PerTokenBlockScaling(), mode='forward')
print('NVFP4 PerToken:', type(state2).__name__)
"
```

### 4. Verify Fusion Gate Accepts NVFP4 Per-Token Recipe

```bash
python -c "
from transformer_engine.common.recipe import (
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    NVFP4PerTokenBlockScaling,
)

# Simulate the check in fuse_grouped_mlp_ops
for recipe_cls in [MXFP8BlockScaling, NVFP4BlockScaling, NVFP4PerTokenBlockScaling]:
    r = recipe_cls()
    passes = r.mxfp8() or r.nvfp4_pertoken()
    print(f'{recipe_cls.__name__:40s} fusion gate: {passes}')
# Expected:
#   MXFP8BlockScaling        -> True  (mxfp8)
#   NVFP4BlockScaling        -> False (neither)
#   NVFP4PerTokenBlockScaling -> True  (nvfp4_pertoken)
"
```

### 5. Run cuDNN Frontend NVFP4 Tests (global_scale_tensor)

```bash
cd /path/to/cudnn-frontend/test/python
conda activate cudnn-dev  # or your env with cudnn-frontend built

# New NVFP4 global_scale tests
python -m pytest fe_api/test_grouped_gemm_glu_nvfp4.py -v --tb=short

# Regression: existing tests should still pass
python -m pytest fe_api/test_grouped_gemm_swiglu.py -v --tb=short
python -m pytest fe_api/test_grouped_gemm_glu.py -v --tb=short
```

### 6. Run TE Grouped Linear Tests (requires full TE build)

```bash
cd /path/to/TransformerEngine

# Existing MXFP8 grouped MLP tests (regression check)
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python -m pytest test/pytorch/test_grouped_linear.py -v --tb=short -k "mxfp8" 2>&1 | tail -20

# NVFP4 per-token path (end-to-end, requires fused op + kernel support)
NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1 python -m pytest test/pytorch/test_grouped_linear.py -v --tb=short -k "nvfp4" 2>&1 | tail -20
```

### 7. Smoke Test: NVFP4 Per-Token Forward Pass (manual)

```bash
NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1 python -c "
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4PerTokenBlockScaling

recipe = NVFP4PerTokenBlockScaling(backward_override='high_precision')

# Simple MoE-like grouped linear test
num_groups = 4
in_features = 256
out_features = 512
batch = 128

with te.fp8_autocast(fp8_recipe=recipe):
    fc1 = te.GroupedLinear(
        in_features, out_features, num_groups,
        bias=False, params_dtype=torch.bfloat16,
    ).cuda()

    x = torch.randn(batch, in_features, dtype=torch.bfloat16, device='cuda')
    split_sizes = torch.tensor([32, 32, 32, 32], dtype=torch.int64, device='cuda')

    y = fc1(x, extra_inputs=(split_sizes,))
    print(f'Input: {x.shape}, Output: {y.shape}')
    print(f'Output dtype: {y.dtype}')
    print('Forward pass OK')
"
```

### 8. Backward Override Smoke Test

```bash
NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4=1 python -c "
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4PerTokenBlockScaling

# Test high_precision backward override
recipe = NVFP4PerTokenBlockScaling(backward_override='high_precision')

num_groups = 4
in_features = 256
out_features = 512

with te.fp8_autocast(fp8_recipe=recipe):
    fc1 = te.GroupedLinear(
        in_features, out_features, num_groups,
        bias=False, params_dtype=torch.bfloat16,
    ).cuda()

    x = torch.randn(32 * num_groups, in_features, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    split_sizes = torch.tensor([32] * num_groups, dtype=torch.int64, device='cuda')

    y = fc1(x, extra_inputs=(split_sizes,))
    loss = y.sum()
    loss.backward()
    print(f'Grad shape: {x.grad.shape}')
    print(f'Grad dtype: {x.grad.dtype}')
    print('Backward pass (high_precision override) OK')
"
```
