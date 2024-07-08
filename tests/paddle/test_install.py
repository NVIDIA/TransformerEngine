# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test basic installation of Paddle extensions"""


def test_import():
    """
    Test if Paddle extension can be imported normally
    """
    import transformer_engine.paddle  # pylint: disable=unused-import
