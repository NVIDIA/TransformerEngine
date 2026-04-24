# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""config for collective_gemm tests"""
import pytest

import transformer_engine.jax  # noqa: F401 - must load libtransformer_engine.so before transformer_engine_jax
from transformer_engine_jax import nvte_built_with_cublasmp


def pytest_addoption(parser):
    """Pytest hook for collective_gemm tests"""
    parser.addoption("--coordinator-address", action="store", default="localhost:12345")
    parser.addoption("--num-processes", action="store", default=1)
    parser.addoption("--process-id", action="store", default=0)
    parser.addoption("--local-device-ids", action="store", default=None)
    parser.addoption("--use-cublasmp", action="store_true", default=False)


@pytest.fixture(autouse=True)
def distributed_args(request):
    """Fixture for querying distributed initialization arguments"""
    if request.cls:
        use_cublasmp = request.config.getoption("--use-cublasmp")
        if use_cublasmp and not nvte_built_with_cublasmp():
            pytest.skip(
                "Collective GEMM cuBLASMp backend tests require Transformer Engine to be built "
                "with NVTE_WITH_CUBLASMP=1."
            )
        request.cls.coordinator_address = request.config.getoption("--coordinator-address")
        request.cls.num_processes = int(request.config.getoption("--num-processes"))
        request.cls.process_id = int(request.config.getoption("--process-id"))
        request.cls.local_device_ids = request.config.getoption("--local-device-ids")
        request.cls.num_devices_per_process = (
            1
            if request.cls.local_device_ids is None
            else len(request.cls.local_device_ids.split(","))
        )
        request.cls.use_cublasmp = use_cublasmp
