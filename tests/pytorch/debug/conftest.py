# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--feature_dirs", nargs="+", action="store", default="", help="List of feature directories"
    )
    parser.addoption(
        "--configs_dir",
        action="store",
        default="",
        type=str,
        help="Path to the directory with configs.",
    )


@pytest.fixture
def feature_dirs(request):
    return request.config.getoption("--feature_dirs")


@pytest.fixture
def configs_dir(request):
    return request.config.getoption("--configs_dir")
