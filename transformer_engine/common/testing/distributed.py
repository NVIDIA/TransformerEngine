# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Distributed execution context for a rank spawned by the benchmarkable harness."""

from __future__ import annotations

import os

RANK_ENV = "NVTE_BENCHMARK_DIST_RANK"
WORLD_SIZE_ENV = "NVTE_BENCHMARK_DIST_WORLD_SIZE"
RENDEZVOUS_ENV = "NVTE_BENCHMARK_DIST_RENDEZVOUS"
NODE_ID_ENV = "NVTE_BENCHMARK_DIST_NODE_ID"


def is_child() -> bool:
    """Whether this process is a rank the harness spawned, rather than the parent.

    The parent never sets these variables on itself, so their presence is what stops a
    child from launching ranks of its own for the same test.
    """
    return RANK_ENV in os.environ


def rank() -> int | None:
    """This process's rank, or ``None`` when it is not a spawned rank."""
    value = os.environ.get(RANK_ENV)
    return None if value is None else int(value)


def world_size() -> int | None:
    """The launch's world size, or ``None`` when this is not a spawned rank."""
    value = os.environ.get(WORLD_SIZE_ENV)
    return None if value is None else int(value)


def rendezvous() -> str | None:
    """The rendezvous location for this launch, or ``None`` outside one.

    Unique per launch: two configs at the same world size in one session must not share
    it, and a stale location left by a crashed run must not be reused.
    """
    return os.environ.get(RENDEZVOUS_ENV)


def expected_node_id() -> str | None:
    """The pytest node ID this rank was spawned to run, or ``None`` outside a launch.

    Checked against the test actually running so a stale ``NVTE_BENCHMARK_DIST_*``
    environment -- an old export, or a command line copied out of a log -- fails loudly
    instead of silently running one rank where the test asked for several.
    """
    return os.environ.get(NODE_ID_ENV)
