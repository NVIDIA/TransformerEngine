# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Set


class BackendImplKind(str, Enum):
    DEFAULT = "flagos"
    REFERENCE = "reference"
    VENDOR = "vendor"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class OpImpl:
    op_name: str
    impl_id: str
    kind: BackendImplKind
    fn: Callable[..., Any]
    vendor: Optional[str] = None
    priority: int = 0
    supported_dtypes: Optional[Set[str]] = None
    min_arch: Optional[str] = None

    def __post_init__(self):
        if self.kind == BackendImplKind.VENDOR and not self.vendor:
            raise ValueError(f"OpImpl with kind=VENDOR must specify vendor name: {self.impl_id}")

    def is_available(self) -> bool:
        avail_fn = getattr(self.fn, "_is_available", None)
        if callable(avail_fn):
            try:
                return bool(avail_fn())
            except Exception:
                return False
        return True


TOKEN_PATTERNS = {
    "flagos": lambda impl: impl.kind == BackendImplKind.DEFAULT,
    "reference": lambda impl: impl.kind == BackendImplKind.REFERENCE,
    "vendor": lambda impl: impl.kind == BackendImplKind.VENDOR,
}


def match_token(impl: OpImpl, token: str) -> bool:
    if token in TOKEN_PATTERNS:
        return TOKEN_PATTERNS[token](impl)

    if token.startswith("vendor:"):
        vendor_name = token.split(":", 1)[1]
        return impl.kind == BackendImplKind.VENDOR and impl.vendor == vendor_name

    if token.startswith("impl:"):
        impl_id = token.split(":", 1)[1]
        return impl.impl_id == impl_id

    return False
