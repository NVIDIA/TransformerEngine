from typing import Any, Callable
from .ops import OpGraph

COMPUTE_PIPELINE_CUSTOM_SERIALIZERS: dict[type, Callable[[Any], OpGraph]] = {}
