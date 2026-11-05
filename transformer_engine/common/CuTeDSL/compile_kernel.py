# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compile a CuTeDSL kernel and export it as a TVM-FFI shared library.

The C++ dispatcher (common/tvm_ffi_bridge.cpp) runs this in a separate Python process, so
compilation never re-enters the caller's interpreter, and then loads the library itself.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def _parse_arg(value: str):
    """Entrypoint arguments arrive as strings; JSON literals (true, 12, ...) are decoded."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main(argv=None) -> int:
    """Return 0 if the kernel was exported to `--output`, nonzero otherwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", required=True, help="Kernel factory in CuTeDSL/__init__")
    parser.add_argument("--name", required=True, help="Symbol the kernel is exported under")
    parser.add_argument("--output", required=True, help="Path of the shared library to write")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to compile for")
    parser.add_argument("args", nargs="*", help="Positional arguments of the entrypoint")
    ns = parser.parse_args(argv)

    # pylint: disable=import-outside-toplevel,no-name-in-module
    from cuda.bindings import runtime as cudart
    from cutlass import cute

    from transformer_engine.common import CuTeDSL

    cudart.cudaSetDevice(ns.device)
    compiled = getattr(CuTeDSL, ns.entrypoint)(*[_parse_arg(a) for a in ns.args])
    if compiled is None:
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        object_file = os.path.join(tmp_dir, "kernel.o")
        compiled.export_to_c(object_file, function_name=ns.name)
        runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
        rpaths = [f"-Wl,-rpath,{os.path.dirname(lib)}" for lib in runtime_libs]
        compiler = os.environ.get("CC", "cc")
        subprocess.run(
            [compiler, "-shared", "-o", ns.output, object_file, *runtime_libs, *rpaths], check=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
