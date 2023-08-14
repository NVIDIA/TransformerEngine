from typing import Callable, Any

FUSIONS_INF: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_FWD: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_BWD: dict[tuple[type, ...], Callable[..., Any]] = {}
