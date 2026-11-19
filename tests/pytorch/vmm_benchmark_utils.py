# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Dependency-free size parsing and VMM benchmark reporting helpers."""

import argparse
from decimal import Decimal, InvalidOperation


MIB = 1024 * 1024
COUNTERS = (
    "allocation_requests", "free_requests", "physical_blocks_created",
    "physical_blocks_reused", "physical_blocks_released", "map_calls",
    "unmap_calls", "event_wait_calls",
)


def mib_bytes(value):
    """Accept fractional MiB only when they describe a positive whole byte count."""
    try:
        count = Decimal(str(value)) * MIB
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Expected a positive MiB size, e.g. 3 or 7.5") from exc
    if not count.is_finite() or count <= 0 or count != count.to_integral_value():
        raise argparse.ArgumentTypeError("MiB size must describe a positive whole number of bytes")
    return int(count)


def byte_shape(count):
    """Use two logical rows where possible, without dropping an odd final byte."""
    return (2, count // 2) if count % 2 == 0 else (count,)


def split_mapped_bytes(logical_bytes, block_bytes):
    half = (logical_bytes + 1) // 2
    return 2 * ((half + block_bytes - 1) // block_bytes) * block_bytes


def handles(info):
    return {(page["domain"], page["handle"]) for page in info["pages"]}


def describe(info):
    block = info["block_bytes"]
    logical = info["logical_bytes"]
    mapped = len(info["pages"]) * block
    boundary = next((page["offset"] for page in info["pages"] if page["domain"] == 1), None)
    by_domain = {str(domain): 0 for domain in (0, 1)}
    for page in info["pages"]:
        by_domain[str(page["domain"])] += max(0, min(block, logical - page["offset"]))
    return {
        "virtual_address": hex(info["base"]), "logical_bytes": logical,
        "mapped_bytes": mapped, "padding_bytes": mapped - logical,
        "reserved_bytes": info["reserved_bytes"],
        "physical_domain_boundary_bytes": boundary,
        "logical_midpoint_bytes": logical / 2,
        "logical_bytes_by_domain": by_domain,
        "physical_blocks": [
            {"domain": page["domain"], "handle": hex(page["handle"])}
            for page in info["pages"]
        ],
    }


def counter_delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in COUNTERS}
