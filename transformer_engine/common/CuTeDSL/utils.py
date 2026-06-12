import cutlass

_CUTLASS_DTYPE_FROM_STR = {
    "fp32": cutlass.Float32,
    "fp16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
}
_STR_FROM_CUTLASS_DTYPE = {v: k for k, v in _CUTLASS_DTYPE_FROM_STR.items()}

def str_to_cutlass_dtype(dtype_str: str):
    """Convert a string dtype to a cutlass dtype, or None if unknown."""
    return _CUTLASS_DTYPE_FROM_STR.get(dtype_str, None)

def cutlass_dtype_to_str(dtype):
    """Convert a cutlass dtype back to its protocol string, or None if unknown."""
    return _STR_FROM_CUTLASS_DTYPE.get(dtype, None)