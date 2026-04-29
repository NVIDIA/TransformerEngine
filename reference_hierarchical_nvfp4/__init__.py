"""NVFP4 reference: same *recipe* as TE (S_enc, S_dec fp8, bsi, FP4) except
``S_enc = (fp8_max*fp4_max)/amax`` uses each **1x64** window's max instead of
per-tensor global amax. See ``fp8_e4m3_utils_np`` and ``core_nvfp4.cuh``.

- PyTorch: symbols below (requires ``torch`` + ``numpy`` for E4M3).
- CPU: ``hierarchical_nvfp4_ref_numpy`` (``numpy`` only), or run
  ``python reference_hierarchical_nvfp4/hierarchical_nvfp4_ref_numpy.py``.
"""

from .hierarchical_nvfp4_ref import (
    COARSE,
    FINE,
    HierarchicalNVFP4Colwise,
    HierarchicalNVFP4Rowwise,
    dequantize_colwise,
    dequantize_rowwise,
    fp4_e2m1_grid_torch,
    quantize_columnwise_1x64_1x16,
    quantize_rowwise_1x64_1x16,
    reference_matmul_tn,
    roundtrip_error,
)

__all__ = [
    "COARSE",
    "FINE",
    "HierarchicalNVFP4Colwise",
    "HierarchicalNVFP4Rowwise",
    "dequantize_colwise",
    "dequantize_rowwise",
    "fp4_e2m1_grid_torch",
    "quantize_columnwise_1x64_1x16",
    "quantize_rowwise_1x64_1x16",
    "reference_matmul_tn",
    "roundtrip_error",
]
