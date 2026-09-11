# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Verify that packaged NCCL EP JIT headers contain all local quoted includes."""

import re
import sys
from pathlib import Path


_INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
_EXTERNAL_HEADERS = {"nccl.h", "nccl_device.h"}
_HEADER_SUFFIXES = {".cuh", ".h", ".hh", ".hpp", ".inc", ".inl"}


def main() -> None:
    include_root = Path(sys.argv[1])
    jit_root = include_root / "nccl_ep"
    public_header = include_root / "nccl_ep.h"
    if not public_header.is_file():
        raise RuntimeError(f"Missing NCCL EP public header: {public_header}")
    if not jit_root.is_dir():
        raise RuntimeError(f"Missing NCCL EP JIT header directory: {jit_root}")

    failures = []

    headers = [public_header, *jit_root.rglob("*")]
    for header in headers:
        if not header.is_file() or header.suffix not in _HEADER_SUFFIXES:
            continue
        for line_number, line in enumerate(header.read_text().splitlines(), 1):
            match = _INCLUDE_PATTERN.match(line)
            if match is None:
                continue
            include = match.group(1)
            if include in _EXTERNAL_HEADERS:
                continue
            candidates = (
                header.parent / include,
                include_root / include,
                jit_root / include,
                jit_root / "device" / include,
            )
            if not any(candidate.is_file() for candidate in candidates):
                failures.append(f"{header.relative_to(include_root)}:{line_number}: {include}")

    if failures:
        raise RuntimeError("Missing local NCCL EP JIT headers:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
