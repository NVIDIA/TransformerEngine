# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Triton utilities for JAX primitives.

This module provides utility functions for integrating Triton kernels into
JAX primitives. Triton is only imported when this module is used.

Triton Package Compatibility:
    There are two Triton packages that can be used:

    1. 'triton' (from OpenAI/PyPI): Standard package, works with JAX out of the box.
       Install with: pip install triton

    2. 'pytorch-triton' (from PyTorch's index): Bundled with PyTorch, includes
       PyTorch-specific patches. Version format: "3.0.0+<commit_sha>"

       IMPORTANT: The 'pytorch-triton' package on PyPI (version 0.0.1) is a
       placeholder that will NOT work. The real pytorch-triton is only available
       from PyTorch's package index and is auto-installed with PyTorch:
           pip install torch --index-url https://download.pytorch.org/whl/cu121

       pytorch-triton has been tested to work with JAX Triton kernels.

Environment Variables:
    NVTE_USE_PYTORCH_TRITON: If set to "1", explicitly acknowledge using
        pytorch-triton for JAX Triton kernels (suppresses warnings). This is
        useful when both JAX and PyTorch are installed in the same environment.
        Default is "0".
"""

import hashlib
import os
import warnings
from typing import Any, Callable, Mapping
import zlib

from jax import core
import jax
import jax.numpy as jnp


# Placeholder package version on PyPI that should never be used
_PYTORCH_TRITON_PLACEHOLDER_VERSION = "0.0.1"


def _detect_triton_package():
    """Detect which Triton package is installed and validate compatibility.

    Returns:
        tuple: (triton_version: str or None, is_pytorch_triton: bool, is_placeholder: bool)

    The function detects:
    - None: Triton not installed
    - Standard triton from OpenAI (versions like "3.1.0")
    - Real pytorch-triton from PyTorch's index (versions like "3.0.0+45fff310c8")
    - Placeholder pytorch-triton from PyPI (version "0.0.1" - broken, raises RuntimeError)
    """
    try:
        import triton

        triton_version = getattr(triton, "__version__", "unknown")
    except ImportError:
        return None, False, False
    except RuntimeError as e:
        # The placeholder pytorch-triton package from PyPI raises:
        # RuntimeError: "Should never be installed"
        if "Should never be installed" in str(e):
            return _PYTORCH_TRITON_PLACEHOLDER_VERSION, False, True
        raise

    # Check for placeholder package (version 0.0.1 from PyPI)
    is_placeholder = triton_version == _PYTORCH_TRITON_PLACEHOLDER_VERSION

    # Real pytorch-triton versions have a commit SHA suffix like "3.0.0+45fff310c8"
    is_pytorch_triton = "+" in triton_version and len(triton_version.split("+")[-1]) >= 8

    return triton_version, is_pytorch_triton, is_placeholder


def _check_triton_compatibility():
    """Check Triton package compatibility and emit warnings if necessary.

    This function handles the case where both JAX and PyTorch may be installed,
    each expecting different Triton packages:
    - JAX typically uses the standard 'triton' package from OpenAI
    - PyTorch uses 'pytorch-triton' which is versioned with commit SHAs

    The NVTE_USE_PYTORCH_TRITON environment variable can be used to explicitly
    acknowledge using pytorch-triton with JAX (suppresses warnings).

    Raises:
        ImportError: If triton is not installed or the placeholder package is detected.
    """
    triton_version, is_pytorch_triton, is_placeholder = _detect_triton_package()

    # Handle placeholder package from PyPI
    if is_placeholder:
        raise ImportError(
            "Detected the placeholder 'pytorch-triton' package (version 0.0.1) from PyPI.\n"
            "This is NOT a functional Triton installation.\n\n"
            "The placeholder package exists to prevent namespace conflicts. To fix this:\n\n"
            "Option 1 - Use standard Triton (recommended for JAX-only environments):\n"
            "    pip uninstall pytorch-triton triton\n"
            "    pip install triton\n\n"
            "Option 2 - Use real pytorch-triton (for mixed JAX+PyTorch environments):\n"
            "    pip uninstall pytorch-triton triton\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "    # pytorch-triton is automatically installed as a torch dependency\n\n"
            "Note: Do NOT run 'pip install pytorch-triton' directly - this installs\n"
            "the broken placeholder. The real pytorch-triton only comes from PyTorch's index."
        )

    if triton_version is None:
        raise ImportError(
            "Triton is required for transformer_engine.jax.triton_extensions.\n\n"
            "Option 1 - Install standard Triton (recommended for JAX-only):\n"
            "    pip install triton\n\n"
            "Option 2 - Install PyTorch with pytorch-triton (for mixed environments):\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n\n"
            "If you don't need Triton, use transformer_engine.jax.cpp_extensions instead."
        )

    use_pytorch_triton_explicit = bool(int(os.environ.get("NVTE_USE_PYTORCH_TRITON", "0")))

    if is_pytorch_triton:
        if use_pytorch_triton_explicit:
            # User explicitly opted in - just log info (no warning)
            pass  # Silent acknowledgment, no warning needed
        else:
            # pytorch-triton detected but user didn't explicitly opt in
            warnings.warn(
                f"Detected pytorch-triton package (version {triton_version}) instead of the"
                " standard 'triton' package from OpenAI. This typically happens when PyTorch is"
                " installed alongside JAX.\n\npytorch-triton is compatible with JAX Triton"
                " kernels. To suppress this warning, set:\n    export"
                " NVTE_USE_PYTORCH_TRITON=1\n\nAlternatively, for a JAX-only environment:\n  - Use"
                " separate virtual environments for JAX and PyTorch, or\n  - Use"
                " transformer_engine.jax.cpp_extensions instead (CUDA-based, no Triton needed)",
                category=UserWarning,
                stacklevel=3,
            )

    return triton_version, is_pytorch_triton


# Perform compatibility check and get triton info
_TRITON_VERSION, _IS_PYTORCH_TRITON = _check_triton_compatibility()

try:
    from jax._src.lib import gpu_triton
    from triton.compiler import compiler as tc
    from triton.backends.nvidia import compiler as cb
    from triton.runtime import autotuner
except ImportError as e:
    raise ImportError(
        "Triton is required for transformer_engine.jax.triton_extensions. "
        "Install with: pip install triton\n"
        "If you don't need Triton, use transformer_engine.jax.cpp_extensions instead."
    ) from e


__all__ = ["triton_call_lowering", "get_triton_info"]

# Triton kernel cache (module-level, shared across all kernels)
_TRITON_KERNEL_CACHE = {}


def get_triton_info():
    """Get information about the installed Triton package.

    Returns:
        dict: Dictionary containing:
            - version (str): Triton version string (e.g., "3.1.0" or "3.0.0+45fff310c8")
            - is_pytorch_triton (bool): True if using real pytorch-triton from PyTorch's index
            - is_openai_triton (bool): True if using standard triton from OpenAI/PyPI
            - env_acknowledged (bool): True if NVTE_USE_PYTORCH_TRITON=1 is set
            - source (str): "pytorch" or "openai" indicating the package source

    Example:
        from transformer_engine.jax.triton_extensions import get_triton_info
        info = get_triton_info()
        print(f"Triton version: {info['version']} (from {info['source']})")
        if info['is_pytorch_triton']:
             print("Using pytorch-triton - compatible with both PyTorch and JAX")
    """
    env_acknowledged = bool(int(os.environ.get("NVTE_USE_PYTORCH_TRITON", "0")))

    return {
        "version": _TRITON_VERSION,
        "is_pytorch_triton": _IS_PYTORCH_TRITON,
        "is_openai_triton": not _IS_PYTORCH_TRITON,
        "env_acknowledged": env_acknowledged and _IS_PYTORCH_TRITON,
        "source": "pytorch" if _IS_PYTORCH_TRITON else "openai",
    }


def get_triton_dtype(aval):
    """Convert JAX dtype to Triton type string.

    Args:
        aval: JAX ShapedArray

    Returns:
        Triton type string (e.g., "*fp32" for pointer, "i32" for scalar)
    """
    dtype_map = {
        jnp.dtype("bfloat16"): "bf16",
        jnp.dtype("float64"): "fp64",
        jnp.dtype("float32"): "fp32",
        jnp.dtype("float16"): "fp16",
        jnp.dtype("float8_e4m3fn"): "fp8e4nv",
        jnp.dtype("float8_e5m2"): "fp8e5",
        jnp.dtype("int64"): "i64",
        jnp.dtype("int32"): "i32",
        jnp.dtype("int16"): "i16",
        jnp.dtype("int8"): "i8",
        jnp.dtype("uint64"): "u64",
        jnp.dtype("uint32"): "u32",
        jnp.dtype("uint16"): "u16",
        jnp.dtype("uint8"): "u8",
        jnp.dtype("bool"): "i1",
    }

    assert isinstance(aval, core.ShapedArray), "aval must be a JAX ShapedArray"
    return f"*{dtype_map[aval.dtype]}"


def compile_triton(
    kernel_fn: Callable,
    signature: Mapping[str, str],
    constants: Mapping[str, Any],
    num_warps: int,
    num_stages: int,
    num_ctas: int,
    compute_capability: int,
    enable_fp_fusion: bool = False,
):
    """Compile a Triton kernel to PTX.

    Kernels are cached to avoid recompilation.

    Args:
        kernel_fn: Triton kernel function (decorated with @triton.jit)
        signature: Dict mapping arg names to types (e.g., {"x_ptr": "*fp32", "n": "i32"})
        constants: Dict of compile-time constants
        num_warps: Number of warps per block
        num_stages: Number of pipeline stages
        num_ctas: Number of CTAs (cooperative thread arrays)
        compute_capability: CUDA compute capability
        enable_fp_fusion: Enable FP fusion optimizations (default False for accuracy)

    Returns:
        TritonKernel object for JAX
    """
    # Create cache key
    cache_key = hashlib.md5(
        str(
            (
                kernel_fn.__name__,
                tuple(sorted(signature.items())),
                tuple(sorted(constants.items())),
                num_warps,
                num_stages,
                num_ctas,
                enable_fp_fusion,
                compute_capability,
            )
        ).encode()
    ).hexdigest()

    if cache_key in _TRITON_KERNEL_CACHE:
        return _TRITON_KERNEL_CACHE[cache_key]

    # Compile kernel
    options = cb.CUDAOptions(
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        cluster_dims=(1, 1, 1),
        debug=False,
        enable_fp_fusion=enable_fp_fusion,
    )

    # Mark constants as constexpr in signature
    signature_with_constexpr = dict(signature)
    for const_name in constants.keys():
        if const_name in signature_with_constexpr:
            signature_with_constexpr[const_name] = "constexpr"

    src = tc.ASTSource(
        fn=kernel_fn,
        constexprs=constants,
        signature=signature_with_constexpr,
    )

    compiled = tc.compile(
        src,
        target=tc.GPUTarget("cuda", compute_capability, 32),
        options=options.__dict__,
    )

    # Create kernel object for JAX
    # From jax/jaxlib/gpu/triton_kernels.cc:
    from packaging import version

    if version.parse(jax.__version__) >= version.parse("0.8.2"):
        kernel = gpu_triton.TritonKernel(
            compiled.name,  # arg0: kernel_name (str)
            num_warps,  # arg1: num_warps (int)
            num_ctas,  # arg2: num_ctas (int)
            compiled.metadata.shared,  # arg3: shared_mem_bytes (int)
            compiled.asm["ptx"],  # arg4: ptx (str)
            "",  # arg5: ttir (str) - empty
            compute_capability,  # arg6: compute_capability (int)
        )
    else:
        kernel = gpu_triton.TritonKernel(
            compiled.name,
            num_warps,
            compiled.metadata.shared,
            compiled.asm["ptx"],
            "",  # ttir
            compute_capability,
            1,
            1,
            1,
        )

    _TRITON_KERNEL_CACHE[cache_key] = kernel
    return kernel


def triton_call_lowering(
    ctx,
    kernel_fn: Callable,
    *array_args,
    grid,
    input_output_aliases: Mapping[int, int] = None,
    constexprs: Mapping[str, Any] = None,
):
    """Helper for MLIR lowering that calls a Triton kernel.

    Use this in your primitive's lowering method to call Triton kernels.

    Args:
        ctx: MLIR lowering context
        kernel_fn: Triton kernel function
        *array_args: Input arrays (from ctx)
        grid: Grid dimensions (int or tuple)
        input_output_aliases: Mapping of input to output aliases
        constexprs: Compile-time constants for the kernel. This includes both
                    tl.constexpr arguments AND scalar runtime arguments (like
                    num_tokens, strides) that are known at JAX trace time.

    Returns:
        MLIR lowering result

    Example:
        @staticmethod
        def lowering(ctx, x, *, block_size):
            from ..triton_extensions import triton_call_lowering
            n = ctx.avals_in[0].size
            return triton_call_lowering(
                ctx, my_kernel, x,
                grid=(triton.cdiv(n, block_size),),
                constexprs={
                    "n_elements": n,  # scalar arg (not tl.constexpr in kernel)
                    "BLOCK_SIZE": block_size,  # tl.constexpr arg
                },
            )
    """
    # Get compute capability using gpu_triton
    compute_capability = gpu_triton.get_compute_capability(0)  # device 0

    # Build signature dict: map arg names to types
    # Get arg names from kernel function
    if isinstance(kernel_fn, autotuner.Autotuner):
        arg_names = kernel_fn.fn.arg_names
    else:
        arg_names = kernel_fn.arg_names

    # Build signature for tensor arguments only (inputs + outputs)
    # Scalar arguments should be passed via constexprs and will be
    # specialized into the kernel at compile time
    all_avals = list(ctx.avals_in) + list(ctx.avals_out)
    constexpr_names = set(constexprs.keys()) if constexprs else set()
    tensor_arg_names = [n for n in arg_names if n not in constexpr_names]
    signature = {n: get_triton_dtype(a) for n, a in zip(tensor_arg_names, all_avals)}

    # Normalize grid to 3D
    if isinstance(grid, int):
        grid_tuple = (grid, 1, 1)
    elif len(grid) == 1:
        grid_tuple = (grid[0], 1, 1)
    elif len(grid) == 2:
        grid_tuple = (grid[0], grid[1], 1)
    else:
        grid_tuple = grid[:3]

    # Default values for the kernel
    actual_kernel_fn = kernel_fn
    num_warps = 32
    num_stages = (
        1  # TODO(Phuong): consider if it is beneficial to expose num_warps, num_stages, num_ctas
    )
    num_ctas = 1
    kernel_constexprs = constexprs if constexprs is not None else {}

    # Handle autotuned kernels - compile all configs
    if isinstance(kernel_fn, autotuner.Autotuner):
        # Compile all configs for runtime selection
        kernel_calls = []
        actual_kernel_fn = kernel_fn.fn

        for config in kernel_fn.configs:
            # Extract parameters from config
            config_num_warps = config.num_warps if config.num_warps is not None else num_warps
            config_num_stages = config.num_stages if config.num_stages is not None else num_stages
            config_num_ctas = config.num_ctas if config.num_ctas is not None else num_ctas

            # Merge config kwargs with user constexprs
            config_constexprs = {**config.kwargs, **(constexprs if constexprs else {})}

            # Compile this config
            config_kernel = compile_triton(
                actual_kernel_fn,
                signature,
                config_constexprs,
                config_num_warps,
                config_num_stages,
                config_num_ctas,
                compute_capability,
                enable_fp_fusion=False,
            )

            # Create kernel call for this config
            config_params = []
            for _ in list(ctx.avals_in) + list(ctx.avals_out):
                config_params.append(gpu_triton.create_array_parameter(0, 16))

            config_call = gpu_triton.TritonKernelCall(
                config_kernel,
                grid_tuple[0],
                grid_tuple[1],
                grid_tuple[2],
                config_params,
            )

            kernel_calls.append((config_call, str(config)))

        # Create autotuned kernel call
        # Convert input_output_aliases to format with sizes
        if input_output_aliases is None:
            input_output_aliases = {}

        input_output_aliases_with_sizes = tuple(
            (
                input_idx,
                output_idx,
                ctx.avals_in[input_idx].size * ctx.avals_in[input_idx].dtype.itemsize,
            )
            for input_idx, output_idx in input_output_aliases.items()
        )

        kernel_call = gpu_triton.TritonAutotunedKernelCall(
            f"{actual_kernel_fn.__name__}_autotuned",
            kernel_calls,
            input_output_aliases_with_sizes,
        )

    else:
        # Regular kernel: compile single config
        kernel = compile_triton(
            actual_kernel_fn,
            signature,
            kernel_constexprs,
            num_warps,
            num_stages,
            num_ctas,
            compute_capability,
            enable_fp_fusion=False,
        )

        kernel_params = []
        for _ in list(ctx.avals_in) + list(ctx.avals_out):
            kernel_params.append(gpu_triton.create_array_parameter(0, 16))

        kernel_call = gpu_triton.TritonKernelCall(
            kernel,
            grid_tuple[0],
            grid_tuple[1],
            grid_tuple[2],
            kernel_params,
        )

    serialized_metadata = b""
    call_proto = kernel_call.to_proto(actual_kernel_fn.__name__, serialized_metadata)

    if input_output_aliases is None:
        input_output_aliases = {}

    # Use JAX FFI lowering with compressed protobuf
    rule = jax.ffi.ffi_lowering(
        "triton_kernel_call",  # Custom call target registered in gpu_triton.py
        api_version=2,
        backend_config=zlib.compress(call_proto),
        operand_output_aliases=input_output_aliases,
    )

    return rule(ctx, *array_args)
