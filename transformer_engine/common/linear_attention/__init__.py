# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Metadata for linear-attention partitions with contiguous token ownership"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LinearAttentionFragment:
    """Metadata for a fragment: the intersection of one independent sequence
    with one rank's token interval.

    For example, global cu_seqlens [0, 8, 12, 24] describes sequences
    A [0, 8), B [8, 12), and C [12, 24). With CP=4, each rank owns six
    consecutive tokens. Each row below describes one fragment:

        Rank  Rank interval  Sequence  start  end  first_rank  last_rank
        0     [0, 6)         A (0)         0    6           0          1
        1     [6, 12)        A (0)         6    8           0          1
        1     [6, 12)        B (1)         8   12           1          1
        2     [12, 18)       C (2)        12   18           2          3
        3     [18, 24)       C (2)        18   24           2          3

    ``start`` is inclusive and ``end`` is exclusive; both are global offsets.
    ``first_rank`` and ``last_rank`` identify the first and last ranks owning
    tokens of the whole sequence, not just this fragment.
    """

    document: int
    start: int
    end: int
    first_rank: int
    last_rank: int


@dataclass(frozen=True)
class LinearAttentionCPHostMetadata:
    """Global sequence boundaries and local fragment metadata for one CP rank.

    Packed (THD) inputs intersect sequences with contiguous rank intervals.
    Dense (BSHD/SBHD) inputs give each rank a nearly equal slice of every sequence.
    Empty sequences have no fragments; their state is unchanged.

    Using the fragment example from LinearAttentionFragment, packed rank 1 owns:

        fragments:                  A [6, 8), B [8, 12)
        local_cu_seqlens:           (0, 2, 6)
        forward_descriptors:        ((0, 0, 1, 1), (1, 1, 0, 1))
        incoming_fragment_indices:  (0,)     # A continues from rank 0.
        outgoing_fragment_indices:  ()      # Neither sequence continues to a later rank.
        terminal_fragment_indices:  (0, 1)   # Both sequences end on this rank.
    """

    cu_seqlens: Tuple[int, ...]  # Global cumulative sequence lengths.
    cp_rank: int
    cp_size: int
    fragments: Tuple[LinearAttentionFragment, ...]
    local_cu_seqlens: Tuple[int, ...]  # Cumulative lengths of nonempty local fragments.
    # Select preceding summaries to compose each fragment's starting memory.
    forward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    # Select following reverse summaries to compose each fragment's ending-state gradient.
    backward_descriptors: Tuple[Tuple[int, int, int, int], ...]
    outgoing_fragment_indices: Tuple[int, ...]  # Sequence continues to a later rank.
    incoming_fragment_indices: Tuple[int, ...]  # Sequence starts on an earlier rank.
    terminal_fragment_indices: Tuple[int, ...]  # Sequence ends on this rank.
    initial_state_owners: Tuple[int, ...]  # Initial-state owner rank for each global sequence.
    empty_owned: Tuple[int, ...]  # Empty global sequence IDs assigned to this rank.
    slots: int  # One per rank for packed inputs; one per sequence for dense inputs.
    communicates: bool  # Any global sequence spans multiple ranks.


def build_linear_attention_cp_host_metadata(cu_seqlens, cp_size, cp_rank, *, dense=False):
    """Build one rank's fragment and summary metadata on the host.

    ``cu_seqlens`` contains full-sequence boundaries as host integers. For
    P = ``cp_size``, packed rank r owns [r*T//P:(r+1)*T//P] of T total tokens.
    Dense rank r owns [r*S//P:(r+1)*S//P] of each length-S sequence.
    """
    cu_seqlens = tuple(cu_seqlens)
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("CP size must be positive and rank must lie inside the group.")
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0:
        raise ValueError("cu_seqlens must start at zero and contain at least two entries.")
    if any(a > b for a, b in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("cu_seqlens must be nondecreasing.")
    total = cu_seqlens[-1]
    boundaries = tuple(r * total // cp_size for r in range(cp_size + 1))
    slots = len(cu_seqlens) - 1 if dense else 1
    fragments, initial_state_owners, empty = [], [], []
    communicates = False
    for doc, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        if dense:
            length = end - start
            if length != cu_seqlens[1]:
                raise ValueError("Dense CP needs equal global sequence lengths.")
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
        initial_state_owners.append(first_rank)
        communicates |= first_rank != last_rank
        if start == end and first_rank == cp_rank:
            empty.append(doc)
        if fragment_start < fragment_end:
            fragments.append(
                LinearAttentionFragment(doc, fragment_start, fragment_end, first_rank, last_rank)
            )
    cu = [0]
    # Descriptor rows are (sequence, first_summary, count, stride). Each selects
    # summaries of one sequence, in forward or reverse order.
    forward_desc, backward_desc = [], []
    outgoing_fragment_indices = []
    incoming_fragment_indices = []
    terminal_fragment_indices = []
    for index, frag in enumerate(fragments):
        cu.append(cu[-1] + frag.end - frag.start)
        slot = frag.document if dense else 0
        forward_desc.append(
            (frag.document, frag.first_rank * slots + slot, cp_rank - frag.first_rank, slots)
        )
        backward_desc.append(
            (frag.document, frag.last_rank * slots + slot, frag.last_rank - cp_rank, -slots)
        )
        if cp_rank < frag.last_rank:
            outgoing_fragment_indices.append(index)
        else:
            terminal_fragment_indices.append(index)
        if cp_rank > frag.first_rank:
            incoming_fragment_indices.append(index)
    return LinearAttentionCPHostMetadata(
        cu_seqlens,
        cp_rank,
        cp_size,
        tuple(fragments),
        tuple(cu),
        tuple(forward_desc),
        tuple(backward_desc),
        tuple(outgoing_fragment_indices),
        tuple(incoming_fragment_indices),
        tuple(terminal_fragment_indices),
        tuple(initial_state_owners),
        tuple(empty),
        slots,
        communicates,
    )
