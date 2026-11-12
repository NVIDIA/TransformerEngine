# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Lifetime of TE-owned CuTeDSL executors registered through TVM-FFI."""

import tvm_ffi


# Keep executors alive until native references and registry entries are released.
_registered_kernels = {}


def register_kernel(name: str, compiled) -> None:
    """Register a compiled executor and retain its Python owner until cleanup."""
    native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
    tvm_ffi.register_global_func(name, native if native is not None else compiled, override=True)
    _registered_kernels[name] = compiled


def release_cuda_state(device=None) -> None:
    """Remove TE-owned compiled entries globally or for one live cleanup callback."""
    for name in tuple(_registered_kernels):
        if device is not None and f"_device_{device}_generation_" not in name:
            continue
        tvm_ffi.remove_global_func(name)
        del _registered_kernels[name]
