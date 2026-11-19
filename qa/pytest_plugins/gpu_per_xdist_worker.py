# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gives each pytest-xdist worker its own GPU. Load with `pytest -n <num GPUs> -p gpu_per_xdist_worker`.

pytest-xdist sets PYTEST_XDIST_WORKER (gw0, gw1, ...) before loading plugins, so this runs before
any test module initializes CUDA. Outside an xdist worker it does nothing.
"""

import os

_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker is not None:
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible:
        _gpus = _visible.split(",")
    else:
        _gpus = [str(i) for i in range(int(os.environ["PYTEST_XDIST_WORKER_COUNT"]))]
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[int(_worker[2:]) % len(_gpus)]
