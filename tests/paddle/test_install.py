# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test basic installation of Paddle extensions"""


def test_import():
    """
    Test if Paddle extension can be imported normally
    """
    try:
        import transformer_engine.paddle    # pylint: disable=unused-import
        te_imported = True
    except:    # pylint: disable=bare-except
        te_imported = False

    assert te_imported, 'transformer_engine import failed'
