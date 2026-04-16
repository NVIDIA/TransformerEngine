#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install DeepEP (hybrid-ep branch).
#
# NVSHMEM support (fast nvlink communication) blocked by
# https://nvbugspro.nvidia.com/bug/5810040
#
# Environment expectations:
#   - CUDA toolkit (nvcc) available via $CUDA_HOME or /usr/local/cuda
#   - PyTorch pre-installed (NGC container or equivalent)
#   - GPU with Blackwell architecture (sm_100 / sm_120) recommended
#
# Usage:
#   bash install_hybridep.sh
#
set -euo pipefail

CLONE_DIR="$(mktemp -d)"
trap 'rm -rf "$CLONE_DIR"' EXIT

echo "============================================"
echo " DeepEP (hybrid-ep) Installation"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Install pynvml (DeepEP runtime dependency)
# ---------------------------------------------------------------------------
echo "[1/3] Installing pynvml ..."
pip install --no-cache-dir pynvml 2>&1 | tail -2
echo ""

# ---------------------------------------------------------------------------
# 2. Detect target GPU architecture
# ---------------------------------------------------------------------------
echo "[2/3] Detecting GPU architecture ..."

FORCE_ARCH="${FORCE_ARCH:-}"
GPU_ARCH="$(python3 -c "
import torch, sys, os
force = os.environ.get('FORCE_ARCH', '')
if force:
    print(force)
    sys.exit(0)
if not torch.cuda.is_available():
    print('10.0')                       # safe default (Blackwell datacenter)
    sys.exit(0)
cap = torch.cuda.get_device_capability(0)
major, minor = cap
arch = f'{major}.{minor}'
if major < 10:
    print(f'ERROR: Detected GPU compute capability {arch} (sm_{major}{minor}), '
          f'but DeepEP hybrid-ep kernels require Blackwell (sm_100+).', file=sys.stderr)
    print(f'To force a target architecture anyway, set FORCE_ARCH=10.0 '
          f'and re-run.', file=sys.stderr)
    sys.exit(1)
print(arch)
")"
echo "       Target TORCH_CUDA_ARCH_LIST=${GPU_ARCH}"
echo ""

# ---------------------------------------------------------------------------
# 3. Clone & build DeepEP (hybrid-ep branch)
# ---------------------------------------------------------------------------
echo "[3/3] Cloning and building DeepEP ..."
git clone --branch hybrid-ep --depth 1 \
    https://github.com/deepseek-ai/DeepEP.git \
    "${CLONE_DIR}/DeepEP" 2>&1 | tail -3

pushd "${CLONE_DIR}/DeepEP" > /dev/null

# Build knobs:
# - TORCH_CUDA_ARCH_LIST: compile for the detected (or default) arch.
# - NVSHMEM_DIR is intentionally NOT set.  The pip-installed NVSHMEM ships
#   libnvshmem_device.a compiled without -fPIC, whose relocations are
#   incompatible with shared-library linking via the host linker.
#   Without NVSHMEM_DIR, setup.py disables internode / low-latency features
#   in deep_ep_cpp but the hybrid_ep_cpp extension (the primary expert-
#   parallelism module) is unaffected.
# - DISABLE_AGGRESSIVE_PTX_INSTRS: automatically set to 1 by setup.py for
#   any arch != 9.0, so no need to export it here.
unset NVSHMEM_DIR 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"

# Force-disable NVSHMEM in setup.py so it does not auto-discover the pip
# package via importlib (the pip archive has link-incompatible objects).
sed -i 's/disable_nvshmem = False/disable_nvshmem = True/' setup.py

echo "       Building wheel (this may take several minutes) ..."
python3 setup.py bdist_wheel 2>&1 | tail -5
echo ""

echo "       Installing wheel ..."
pip install --no-cache-dir --no-deps dist/*.whl 2>&1 | tail -3

popd > /dev/null
echo ""

# ---------------------------------------------------------------------------
# Verify imports
# ---------------------------------------------------------------------------
echo "Verifying imports ..."
python3 -c "
import sys

failures = []

# -- deep_ep (Python package) --
try:
    import deep_ep
    print('  OK  deep_ep')
except Exception as e:
    print(f'  FAIL deep_ep: {e}')
    failures.append('deep_ep')

# -- deep_ep_cpp (C++/CUDA extension) --
try:
    import deep_ep_cpp
    print('  OK  deep_ep_cpp')
except Exception as e:
    print(f'  FAIL deep_ep_cpp: {e}')
    failures.append('deep_ep_cpp')

# -- hybrid_ep_cpp (C++/CUDA extension) --
try:
    import hybrid_ep_cpp
    print('  OK  hybrid_ep_cpp')
except Exception as e:
    print(f'  FAIL hybrid_ep_cpp: {e}')
    failures.append('hybrid_ep_cpp')

if failures:
    print(f'\nERROR: {len(failures)} import(s) failed: {failures}', file=sys.stderr)
    sys.exit(1)
else:
    print('\nAll imports succeeded.')
"

echo ""
echo "============================================"
echo " Installation complete"
echo "============================================"
