# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent metadata for contiguous linear-attention partitions."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LinearAttentionFragment:
    """Intersection of one independent document with one rank's token interval."""

    document: int
    start: int
    end: int
    first_rank: int
    last_rank: int


@dataclass(frozen=True)
class LinearAttentionCPPartition:
    """Host metadata, independent of tensor frameworks and attention flavors.

    Packed (THD) inputs intersect documents with contiguous rank intervals.
    Dense (BSHD/SBHD) inputs give each rank a nearly equal slice of every document.
    Empty documents have no fragments; their state is unchanged.
    """

    cu_seqlens: Tuple[int, ...]
    cp_rank: int
    cp_size: int
    fragments: Tuple[LinearAttentionFragment, ...]
    local_cu_seqlens: Tuple[int, ...]
    forward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    backward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    outgoing: Tuple[int, ...]
    incoming: Tuple[int, ...]
    terminal: Tuple[int, ...]
    seed_owners: Tuple[int, ...]
    empty_owned: Tuple[int, ...]
    slots: int  # One per rank for packed inputs; one per document for dense inputs.
    communicates: bool


def build_linear_attention_partition(cu_seqlens, cp_size, cp_rank, *, dense=False):
    """Build one rank's fragment and summary metadata on the host.

    ``cu_seqlens`` contains full-document boundaries as host integers. For
    P = ``cp_size``, packed rank r owns [r*T//P:(r+1)*T//P] of T total tokens.
    Dense rank r owns [r*S//P:(r+1)*S//P] of each length-S document.
    Descriptor rows are (document, first_record, count, stride). Each selects
    summaries of one document, in forward or reverse order.
    """
    cu_seqlens = tuple(cu_seqlens)
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("CP size must be positive and rank must lie inside the group.")
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0:
        raise ValueError("Global cumulative sequence lengths must start at zero.")
    if any(a > b for a, b in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("Global cumulative sequence lengths must be nondecreasing.")
    total = cu_seqlens[-1]
    boundaries = tuple(r * total // cp_size for r in range(cp_size + 1))
    slots = len(cu_seqlens) - 1 if dense else 1
    fragments, seeds, empty = [], [], []
    communicates = False
    for doc, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        if dense:
            length = end - start
            if length != cu_seqlens[1]:
                raise ValueError("Dense CP needs equal global document lengths.")
            cuts = tuple(r * length // cp_size for r in range(cp_size + 1))
            first_rank = min(cp_size - 1, bisect_right(cuts, 0) - 1) if length else 0
            last_rank = min(cp_size - 1, bisect_right(cuts, length - 1) - 1) if length else 0
            fragment_start, fragment_end = start + cuts[cp_rank], start + cuts[cp_rank + 1]
        else:
            first_rank = min(cp_size - 1, bisect_right(boundaries, start) - 1)
            last_rank = (
                min(cp_size - 1, bisect_right(boundaries, end - 1) - 1)
                if end > start
                else first_rank
            )
            fragment_start = max(start, boundaries[cp_rank])
            fragment_end = min(end, boundaries[cp_rank + 1])
        seeds.append(first_rank)
        communicates |= first_rank != last_rank
        if start == end and first_rank == cp_rank:
            empty.append(doc)
        if fragment_start < fragment_end:
            fragments.append(
                LinearAttentionFragment(doc, fragment_start, fragment_end, first_rank, last_rank)
            )
    cu = [0]
    forward, backward, outgoing, incoming, terminal = [], [], [], [], []
    for index, frag in enumerate(fragments):
        cu.append(cu[-1] + frag.end - frag.start)
        slot = frag.document if dense else 0
        forward.append(
            (frag.document, frag.first_rank * slots + slot, cp_rank - frag.first_rank, slots)
        )
        backward.append(
            (frag.document, frag.last_rank * slots + slot, frag.last_rank - cp_rank, -slots)
        )
        if cp_rank < frag.last_rank:
            outgoing.append(index)
        else:
            terminal.append(index)
        if cp_rank > frag.first_rank:
            incoming.append(index)
    return LinearAttentionCPPartition(
        cu_seqlens,
        cp_rank,
        cp_size,
        tuple(fragments),
        tuple(cu),
        tuple(forward),
        tuple(backward),
        tuple(outgoing),
        tuple(incoming),
        tuple(terminal),
        tuple(seeds),
        tuple(empty),
        slots,
        communicates,
    )
