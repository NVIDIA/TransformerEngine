# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
from functools import partial
import os

from transformer_engine.jax.cpp_extensions.misc import get_xla_flag


@pytest.fixture(autouse=True, scope="function")
def preserve_xla_flags():
    """Ensures the XLA flags environment variable is restored after any tests in this file run."""
    old_flags = os.getenv("XLA_FLAGS")
    yield
    if old_flags is not None:
        os.environ["XLA_FLAGS"] = old_flags


def test_get_xla_flag(request):
    os.environ["XLA_FLAGS"] = ""
    assert get_xla_flag("") is None
    assert get_xla_flag("--foo") is None
    assert get_xla_flag("--bar=1") is None

    os.environ["XLA_FLAGS"] = "--foo --bar=1 --baz=biz"
    assert get_xla_flag("--foo") == True
    assert get_xla_flag("--bar") == "1"
    assert get_xla_flag("--bar", cast=int) == 1
    assert get_xla_flag("--bar", cast=bool) == True
    assert get_xla_flag("--baz") == "biz"
    with pytest.raises(ValueError):
        # cast will fail
        assert get_xla_flag("--baz", cast=int)
    assert get_xla_flag("--xla") is None

    os.environ["XLA_FLAGS"] = "--xla_abc --xla_abb"
    assert get_xla_flag("--xla_abc") == True
    assert get_xla_flag("--xla_abb") == True
