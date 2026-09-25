# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

import ctypes
import functools
import os
from ctypes.util import find_library
from importlib import metadata
from pathlib import Path
from typing import Optional, Tuple
import transformer_engine.common

# Minimum NCCL version for the runtime-loaded NCCL EP backend.
_NCCL_EP_MIN_VERSION = (2, 30, 4)


@functools.lru_cache(maxsize=1)
def _nccl_runtime_version() -> Optional[Tuple[int, int, int]]:
    """Return runtime (major, minor, patch) from libnccl.so.2, or None if unavailable."""
    try:
        libnccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_LOCAL)
        ncclGetVersion = libnccl.ncclGetVersion
    except (OSError, AttributeError):
        return None
    ver = ctypes.c_int(0)
    if ncclGetVersion(ctypes.byref(ver)) != 0:
        return None
    v = ver.value
    return (v // 10000, (v // 100) % 100, v % 100)


def _nccl_ep_library_installed() -> bool:
    if nccl_ep_home := os.getenv("NCCL_EP_HOME"):
        home = Path(nccl_ep_home)
        if any(
            library.is_file()
            for lib_dir in ("lib", "lib64")
            for library in (home / lib_dir).glob("libnccl_ep.so*")
        ):
            return True

    try:
        transformer_engine.common._get_shared_object_file("nccl_ep")
    except FileNotFoundError:
        return find_library("nccl_ep") is not None
    else:
        return True


def is_nccl_ep_available() -> bool:
    """Return True if the NCCL EP library and a compatible NCCL runtime are available."""
    if not _nccl_ep_library_installed():
        return False
    cur = _nccl_runtime_version()
    return cur is not None and cur >= _NCCL_EP_MIN_VERSION


def require_nccl_ep() -> None:
    """Raise RuntimeError if NCCL EP cannot run on the current libnccl."""
    if not _nccl_ep_library_installed():
        raise RuntimeError(
            "NCCL EP library libnccl_ep.so is not installed. Build Transformer Engine "
            "with NVTE_WITH_NCCL_EP=1."
        )
    mn = ".".join(str(x) for x in _NCCL_EP_MIN_VERSION)
    cur = _nccl_runtime_version()
    if cur is None:
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {mn}; could not load libnccl.so.2 or query its "
            "version. Install NCCL or ensure libnccl.so.2 is on the loader path."
        )
    if cur < _NCCL_EP_MIN_VERSION:
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {mn} at runtime; found "
            f"{'.'.join(str(x) for x in cur)}. Upgrade NCCL to a compatible version."
        )


try:
    from . import pytorch
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "pytorch" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `torch` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.jax` on a system that
            # also has a PyTorch installation. In order to enable that use case, we issue a warning here
            # about the missing PyTorch extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a PyTorch installation but could not find the shared object file for the "
                "Transformer Engine PyTorch extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[pytorch]` or "
                "build from source with `NVTE_FRAMEWORK=pytorch`.",
                category=RuntimeWarning,
            )

try:
    from . import jax
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "jax" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `jax` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.pytorch` on a system
            # that also has a Jax installation. In order to enable that use case, we issue a warning here
            # about the missing Jax extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a Jax installation but could not find the shared object file for the "
                "Transformer Engine Jax extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[jax]` or "
                "build from source with `NVTE_FRAMEWORK=jax`.",
                category=RuntimeWarning,
            )

__version__ = str(metadata.version("transformer_engine"))


def smoke_test(verbose: bool = True) -> bool:
    """Run smoke checks to verify the Transformer Engine installation.

    Confirms the package is installed correctly and that each framework extension
    that should be present actually loaded. The top-level import deliberately
    swallows a missing framework extension and only warns, so a successful
    ``import transformer_engine`` does not imply ``transformer_engine.pytorch`` or
    ``transformer_engine.jax`` is usable. Because Transformer Engine is built per
    framework, a framework that is installed without its extension is reported as
    skipped when another extension did load, and as a failure only when none did.
    For PyTorch a minimal forward pass runs on the current device. Checks that need
    a GPU are skipped when no GPU is available, so the call is safe to run anywhere
    as an installation sanity check.

    Parameters
    ----------
    verbose : bool, default = True
        Print the result of each check.

    Returns
    -------
    bool
        ``True`` if every executed check passed, ``False`` otherwise. Skipped checks
        do not affect the result.
    """
    # Deliberately a smoke test rather than the full suite: install integrity, that
    # each framework extension loaded, and one PyTorch forward pass. Deeper coverage
    # stays in tests/ and qa/.
    results = []
    # Transformer Engine is built per framework via NVTE_FRAMEWORK, so an extension
    # that is absent may well be deliberate; the module-level guard above only warns
    # for exactly that reason. Track which frameworks are installed without their
    # extension and only fail if none of them loaded.
    loaded = []
    missing = []

    def _record(name: str, passed: bool, detail: str = "") -> None:
        results.append(passed)
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}" + (f": {detail}" if detail else ""))

    def _skip(name: str, detail: str = "") -> None:
        if verbose:
            print(f"[SKIP] {name}" + (f": {detail}" if detail else ""))

    # Installation integrity (reuses the PyPI sanity check).
    try:
        from .common import sanity_checks_for_pypi_installation

        sanity_checks_for_pypi_installation()
        _record("installation", True, f"transformer_engine {__version__}")
    except Exception as err:  # pylint: disable=broad-except
        _record("installation", False, str(err))

    # PyTorch backend, checked only if it is available. A backend that is present
    # but fails at runtime is reported, not skipped.
    try:
        from . import pytorch as te
    except (ImportError, FileNotFoundError):
        # FileNotFoundError mirrors the module-level guard: torch is installed but
        # the Transformer Engine PyTorch extension shared object is missing.
        te = None
    if te is None:
        # The module-level guard only warns in this case, so the import above can
        # succeed with the extension missing. Report it rather than stay silent.
        try:
            import torch  # pylint: disable=unused-import
        except ImportError:
            pass
        else:
            missing.append(("pytorch", "torch", "PyTorch"))
    else:
        loaded.append("pytorch")
        try:
            import torch

            if not torch.cuda.is_available():
                _record("pytorch", True, "imported; no CUDA device, GPU checks skipped")
            else:
                major, minor = te.get_device_compute_capability()
                dtype = torch.bfloat16 if te.is_bf16_available() else torch.float16
                with torch.no_grad():
                    model = te.Linear(16, 16, params_dtype=dtype, device="cuda")
                    out = model(torch.zeros(4, 16, dtype=dtype, device="cuda"))
                if tuple(out.shape) != (4, 16):
                    raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
                formats = [
                    name
                    for name, available in (
                        ("FP8", te.is_fp8_available()),
                        ("MXFP8", te.is_mxfp8_available()),
                        ("NVFP4", te.is_nvfp4_available()),
                    )
                    if available
                ]
                detail = f"sm_{major}{minor}, {dtype}, formats: {', '.join(formats) or 'none'}"
                _record("pytorch", True, detail)
        except Exception as err:  # pylint: disable=broad-except
            _record("pytorch", False, str(err))

    # JAX backend, same treatment. Importing the module is itself the meaningful
    # check here: it is what runs load_framework_extension, and the module-level
    # guard above turns that failure into a warning rather than an error. No
    # functional pass is attempted, since the loaded extension is what this
    # reports on.
    try:
        from . import jax as te_jax  # pylint: disable=unused-import
    except (ImportError, FileNotFoundError):
        te_jax = None
    if te_jax is None:
        try:
            import jax  # pylint: disable=unused-import
        except ImportError:
            pass
        else:
            missing.append(("jax", "jax", "JAX"))
    else:
        loaded.append("jax")
        try:
            import jax

            devices = jax.devices()
            gpus = [device for device in devices if device.platform == "gpu"]
            if not gpus:
                platforms = sorted({device.platform for device in devices}) or ["none"]
                _record("jax", True, f"extension loaded; no GPU device ({', '.join(platforms)})")
            else:
                _record("jax", True, f"extension loaded; {len(gpus)} GPU device(s)")
        except Exception as err:  # pylint: disable=broad-except
            _record("jax", False, str(err))

    # A framework whose extension was never built is only a failure when nothing
    # else loaded, since a single-framework install is a normal, supported state.
    for name, package, pretty in missing:
        if loaded:
            _skip(
                name,
                f"{package} is installed but Transformer Engine was not built for it "
                f"(built for: {', '.join(loaded)})",
            )
        else:
            _record(
                name,
                False,
                f"{package} is installed but the Transformer Engine {pretty} extension "
                "did not load",
            )

    passed = all(results)
    if verbose:
        print(f"\n{sum(results)}/{len(results)} checks passed.")
    return passed
