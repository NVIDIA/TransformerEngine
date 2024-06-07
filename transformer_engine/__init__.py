# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

from importlib import metadata
import transformer_engine.common

try:
    from . import pytorch
except (ImportError, StopIteration) as e:
    pass

try:
    from . import jax
except (ImportError, StopIteration) as e:
    pass

try:
    from . import paddle
except (ImportError, StopIteration) as e:
    pass

try:
    import transformer_engine_jax
except ImportError:
    pass

try:
    import transformer_engine_paddle
except ImportError:
    pass

__version__ = str(metadata.version("transformer_engine"))
