# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gives each pytest-xdist worker its own GPU and an equal share of the CPUs.

Load with `pytest -n <num GPUs> -p gpu_per_xdist_worker`. pytest-xdist sets PYTEST_XDIST_WORKER
(gw0, gw1, ...) before loading plugins, so this runs before any test module imports a framework or
initializes CUDA. Outside an xdist worker it does nothing.
"""

import os

_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker is not None:
    _num_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible:
        _gpus = _visible.split(",")
    else:
        # All GPUs are visible and test.sh launches one worker per GPU, so worker count == GPU count.
        _gpus = [str(i) for i in range(_num_workers)]
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[int(_worker[2:]) % len(_gpus)]

    # Each worker's OpenMP pool otherwise defaults to every core, so the workers oversubscribe the
    # CPUs and small CPU ops (e.g. float64 reference computations) slow down by orders of magnitude.
    if "OMP_NUM_THREADS" not in os.environ:
        _num_cpus = len(os.sched_getaffinity(0))
        os.environ["OMP_NUM_THREADS"] = str(max(1, _num_cpus // _num_workers))
