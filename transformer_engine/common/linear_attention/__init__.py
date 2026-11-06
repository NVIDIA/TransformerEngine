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

    Packed inputs intersect document intervals with contiguous rank intervals.
    Dense inputs give every rank a contiguous, nearly equal slice of each document.
    Empty documents have no fragments; their state is unchanged.
    """

    documents: Tuple[int, ...]
    rank: int
    size: int
    fragments: Tuple[LinearAttentionFragment, ...]
    local_cu_seqlens: Tuple[int, ...]
    forward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    backward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    outgoing: Tuple[int, ...]
    incoming: Tuple[int, ...]
    terminal: Tuple[int, ...]
    seed_owners: Tuple[int, ...]
    empty_owned: Tuple[int, ...]
    slots: int
    communicates: bool


def build_linear_attention_partition(documents, size, rank, *, dense=False):
    """Build ordered fragment intersections and prefix/suffix record ranges.

    ``documents`` are global cumulative offsets. Packed rank r owns
    [floor(r*T/P), floor((r+1)*T/P)). Dense documents must all have the same
    length. Descriptor rows are (document, first, count, stride).
    Their record ranges never cross an independent document's boundary.
    """
    documents = tuple(documents)
    if size < 1 or not 0 <= rank < size:
        raise ValueError("CP size must be positive and rank must lie inside the group.")
    if len(documents) < 2 or documents[0] != 0:
        raise ValueError("Global cumulative sequence lengths must start at zero.")
    if any(a > b for a, b in zip(documents, documents[1:])):
        raise ValueError("Global cumulative sequence lengths must be nondecreasing.")
    total = documents[-1]
    boundaries = tuple(r * total // size for r in range(size + 1))
    slots = len(documents) - 1 if dense else 1
    fragments, seeds, empty = [], [], []
    communicates = False
    for doc, (start, end) in enumerate(zip(documents, documents[1:])):
        if dense:
            length = end - start
            if length != documents[1]:
                raise ValueError("Dense CP needs equal global document lengths.")
            cuts = tuple(r * length // size for r in range(size + 1))
            first = min(size - 1, bisect_right(cuts, 0) - 1) if length else 0
            last = min(size - 1, bisect_right(cuts, length - 1) - 1) if length else 0
            left, right = start + cuts[rank], start + cuts[rank + 1]
        else:
            first = min(size - 1, bisect_right(boundaries, start) - 1)
            last = min(size - 1, bisect_right(boundaries, end - 1) - 1) if end > start else first
            left, right = max(start, boundaries[rank]), min(end, boundaries[rank + 1])
        seeds.append(first)
        communicates |= first != last
        if start == end and first == rank:
            empty.append(doc)
        if left < right:
            fragments.append(LinearAttentionFragment(doc, left, right, first, last))
    cu = [0]
    forward, backward, outgoing, incoming, terminal = [], [], [], [], []
    for index, frag in enumerate(fragments):
        cu.append(cu[-1] + frag.end - frag.start)
        slot = frag.document if dense else 0
        forward.append(
            (frag.document, frag.first_rank * slots + slot, rank - frag.first_rank, slots)
        )
        backward.append(
            (frag.document, frag.last_rank * slots + slot, frag.last_rank - rank, -slots)
        )
        if rank < frag.last_rank:
            outgoing.append(index)
        else:
            terminal.append(index)
        if rank > frag.first_rank:
            incoming.append(index)
    return LinearAttentionCPPartition(
        documents,
        rank,
        size,
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
