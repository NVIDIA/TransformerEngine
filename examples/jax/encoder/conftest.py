# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""config for test_multiprocessing_encoder"""
import pytest


def pytest_addoption(parser):
    """Pytest hook for test_multiprocessing_encoder"""
    parser.addoption("--num-process", action="store", default=0)
    parser.addoption("--process-id", action="store", default=0)


@pytest.fixture(autouse=True)
def multiprocessing_parses(request):
    """Fixture for querying num-process and process-id"""
    if request.cls:
        request.cls.num_process = int(request.config.getoption("--num-process"))
        request.cls.process_id = int(request.config.getoption("--process-id"))
