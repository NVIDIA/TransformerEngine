# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental VMM localization for NVFP4 post-RHT amax computation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import torch
import transformer_engine_torch as tex

from .localized_mxfp8 import _get_localization_context
from .nvfp4_tensor import NVFP4Quantizer
from .vmm import VMMRowSplitAllocator


@dataclass
class _WorkspaceSlot:
    signature: tuple
    workspace: "NVFP4VMMWorkspace"
    in_use: bool = False


class _NVFP4Scratch:
    """Reusable VMM producer output and per-domain amax storage."""

    def __init__(
        self,
        shape: Tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        quantizer: NVFP4Quantizer,
    ) -> None:
        del quantizer
        self.allocator = VMMRowSplitAllocator(device)
        self.input = self.allocator.allocate(shape, dtype)
        self.rowwise_amaxes = tuple(
            torch.empty((1,), dtype=torch.float32, device=device) for _ in range(2)
        )
        self.columnwise_amaxes = tuple(
            torch.empty((1,), dtype=torch.float32, device=device) for _ in range(2)
        )

    def close(self) -> None:
        self.allocator.close()


_WORKSPACE_POOLS: Dict[str, List[_WorkspaceSlot]] = {}
_SCRATCH_POOLS: Dict[tuple, _NVFP4Scratch] = {}
_POOL_STAGE: Optional[str] = None


def is_nvfp4_vmm_localization_eligible(
    tensor: torch.Tensor,
    quantizer: object,
    output_cols: int,
) -> bool:
    """Whether the specialized two-domain post-RHT amax path is supported."""
    if os.getenv("NVTE_NVFP4_VMM_LOCALIZATION", "0") != "1":
        return False
    if not isinstance(quantizer, NVFP4Quantizer):
        return False
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        return False
    rows = tensor.shape[0]
    return (
        quantizer.rowwise_usage
        and quantizer.columnwise_usage
        and quantizer.with_rht
        and quantizer.with_post_rht_amax
        and not quantizer.with_amax_reduction
        and not quantizer.with_2d_quantization
        and not quantizer.row_scaled_nvfp4
        and not quantizer.nvfp4_use_4over6
        and rows % 256 == 0
        and output_cols % 128 == 0
    )

class NVFP4VMMWorkspace:
    """One graph-stable output with localized BF16 producer/amax execution."""

    def __init__(
        self,
        shape: Tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        quantizer: NVFP4Quantizer,
        scratch: _NVFP4Scratch,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.quantizer = quantizer
        self.scratch = scratch
        _, _, self.streams = _get_localization_context(device.index)
        self._fork_event = torch.cuda.Event(enable_timing=False)
        self._join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
        self._capture_events = []

        self.output = quantizer.make_empty(shape, dtype=dtype, device=device)
        # BasicLinear clears internal fuser tensors after backward. The pool
        # owns these buffers and reuses their graph-stable addresses.
        self.output._do_not_clear = True
        self._rowwise_data = self.output._rowwise_data
        self._columnwise_data = self.output._columnwise_data
        self._rowwise_scale_inv = self.output._rowwise_scale_inv
        self._columnwise_scale_inv = self.output._columnwise_scale_inv
        self._amax_rowwise = self.output._amax_rowwise
        self._amax_columnwise = self.output._amax_columnwise

    def _events(self):
        if torch.cuda.is_current_stream_capturing():
            fork_event = torch.cuda.Event(enable_timing=False)
            join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
            self._capture_events.extend((fork_event, *join_events))
            return fork_event, join_events
        return self._fork_event, self._join_events

    def _launch_producer_and_amax(
        self,
        producer_input: torch.Tensor,
        grad_output: Optional[torch.Tensor],
        parent_stream: torch.cuda.Stream,
    ) -> None:
        # In-place GEMM swizzling replaces the full tensor's scale attributes.
        # Autograd saving/cleanup also detaches inner tensors from the wrapper.
        # Restore all persistent buffers before writing the next result.
        self.output._rowwise_data = self._rowwise_data
        self.output._columnwise_data = self._columnwise_data
        self.output._rowwise_scale_inv = self._rowwise_scale_inv
        self.output._columnwise_scale_inv = self._columnwise_scale_inv
        self.output._amax_rowwise = self._amax_rowwise
        self.output._amax_columnwise = self._amax_columnwise

        # Keep SwiGLU/dSwiGLU as one ordinary full-chip launch. Its output is
        # written directly into row-split VMM storage so only the following
        # Hadamard-amax kernels need programmatic localization.
        if grad_output is None:
            tex.swiglu_out(producer_input, self.scratch.input)
        else:
            tex.dswiglu_out(grad_output, producer_input, self.scratch.input)

        fork_event, join_events = self._events()
        fork_event.record(parent_stream)
        rows_per_domain = self.shape[0] // 2
        for domain, (stream, join_event) in enumerate(zip(self.streams, join_events)):
            row_start = domain * rows_per_domain
            row_end = row_start + rows_per_domain
            stream.wait_event(fork_event)
            with torch.cuda.stream(stream):
                local_output = self.scratch.input[row_start:row_end]
                tex.nvfp4_compute_amax(
                    local_output,
                    self.quantizer,
                    self.scratch.rowwise_amaxes[domain],
                    self.scratch.columnwise_amaxes[domain],
                )
            join_event.record(stream)
        for event in join_events:
            parent_stream.wait_event(event)

    def _quantize(self):
        rowwise_amax = self.output._amax_rowwise
        columnwise_amax = self.output._amax_columnwise
        # RHT is block-local over 16 rows, and each domain starts on a
        # 128-row boundary. Max-reducing the two local post-RHT amaxes is
        # therefore identical to computing the amax over the full tensor.
        torch.maximum(
            self.scratch.rowwise_amaxes[0],
            self.scratch.rowwise_amaxes[1],
            out=rowwise_amax,
        )
        torch.maximum(
            self.scratch.columnwise_amaxes[0],
            self.scratch.columnwise_amaxes[1],
            out=columnwise_amax,
        )

        # Keep the RHT+quant kernel as one ordinary full-chip launch. It reads
        # the two-domain VMM input directly and writes the standard full output,
        # preserving baseline layout and Philox consumption without assembly.
        return tex.nvfp4_quantize_with_amax_out(
            self.scratch.input,
            self.quantizer,
            rowwise_amax,
            columnwise_amax,
            self.output,
        )

    def swiglu(self, input_: torch.Tensor):
        """Produce and quantize a SwiGLU output."""
        parent_stream = torch.cuda.current_stream(self.device)
        self._launch_producer_and_amax(input_, None, parent_stream)
        output = self._quantize()
        output._nvte_vmm_workspace = self
        return output

    def dswiglu(self, grad_output: torch.Tensor, input_: torch.Tensor):
        """Produce and quantize a dSwiGLU output."""
        parent_stream = torch.cuda.current_stream(self.device)
        self._launch_producer_and_amax(input_, grad_output, parent_stream)
        output = self._quantize()
        output._nvte_vmm_workspace = self
        return output

    def reset_quantizer(self, quantizer: NVFP4Quantizer) -> None:
        self.quantizer = quantizer
        self.output._quantizer = quantizer

    def close(self) -> None:
        pass


def _scratch_signature(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    quantizer: NVFP4Quantizer,
) -> tuple:
    return (
        _POOL_STAGE,
        shape,
        dtype,
        device,
        quantizer.rowwise_usage,
        quantizer.columnwise_usage,
    )


def _get_scratch(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    quantizer: NVFP4Quantizer,
) -> _NVFP4Scratch:
    signature = _scratch_signature(shape, dtype, device, quantizer)
    scratch = _SCRATCH_POOLS.get(signature)
    if scratch is None:
        scratch = _NVFP4Scratch(
            shape,
            dtype=dtype,
            device=device,
            quantizer=quantizer,
        )
        _SCRATCH_POOLS[signature] = scratch
    return scratch


def acquire_nvfp4_vmm_workspace(
    shape: Tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    quantizer: NVFP4Quantizer,
) -> NVFP4VMMWorkspace:
    """Acquire one liveness-tracked output workspace."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    signature = (
        shape,
        dtype,
        device,
        quantizer.dtype,
        quantizer.rowwise_usage,
        quantizer.columnwise_usage,
        quantizer.with_rht,
        quantizer.with_post_rht_amax,
        quantizer.stochastic_rounding,
        quantizer.optimize_for_gemm,
    )
    scratch = _get_scratch(shape, dtype, device, quantizer)
    stage = _POOL_STAGE
    if stage is None:
        return NVFP4VMMWorkspace(
            shape,
            dtype=dtype,
            device=device,
            quantizer=quantizer,
            scratch=scratch,
        )

    pool = _WORKSPACE_POOLS.setdefault(stage, [])
    slot = next(
        (
            candidate
            for candidate in pool
            if candidate.signature == signature and not candidate.in_use
        ),
        None,
    )
    if slot is None:
        workspace = NVFP4VMMWorkspace(
            shape,
            dtype=dtype,
            device=device,
            quantizer=quantizer,
            scratch=scratch,
        )
        slot = _WorkspaceSlot(signature, workspace, in_use=True)
        pool.append(slot)
    else:
        slot.in_use = True
        workspace = slot.workspace
        workspace.reset_quantizer(quantizer)
    workspace._vmm_pool_slot = slot
    return workspace


def release_nvfp4_vmm_tensor_workspaces(*tensors: object) -> None:
    """Release unique workspaces after their final GEMM consumer."""
    released = set()
    for tensor in tensors:
        workspace = getattr(tensor, "_nvte_vmm_workspace", None)
        if workspace is None or id(workspace) in released:
            continue
        slot = getattr(workspace, "_vmm_pool_slot", None)
        if slot is None:
            workspace.close()
        else:
            slot.in_use = False
        released.add(id(workspace))


def begin_nvfp4_vmm_workspace_iteration(stage: str) -> None:
    """Begin a graph warmup or capture iteration."""
    global _POOL_STAGE
    if _POOL_STAGE is not None:
        raise RuntimeError(f"NVFP4 VMM workspace iteration {_POOL_STAGE!r} is already active")
    _POOL_STAGE = stage
    for slot in _WORKSPACE_POOLS.get(stage, ()):
        if slot.in_use:
            raise RuntimeError(f"NVFP4 VMM workspace for {stage} was not released")


def end_nvfp4_vmm_workspace_iteration(*, validate: bool = True) -> None:
    """End an iteration and validate workspace ownership."""
    global _POOL_STAGE
    stage = _POOL_STAGE
    if stage is None:
        return
    _POOL_STAGE = None
    pool = _WORKSPACE_POOLS.get(stage, ())
    live = sum(slot.in_use for slot in pool)
    if not validate:
        for slot in pool:
            slot.in_use = False
    if validate and live:
        raise RuntimeError(
            f"NVFP4 VMM workspace iteration for {stage} ended with {live} live leases"
        )


def clear_nvfp4_vmm_workspace_pools() -> None:
    """Release VMM mappings after captured graphs are destroyed."""
    if _POOL_STAGE is not None:
        raise RuntimeError("Cannot clear an active NVFP4 VMM workspace iteration")
    for pool in _WORKSPACE_POOLS.values():
        for slot in pool:
            slot.workspace.close()
    _WORKSPACE_POOLS.clear()
    for scratch in _SCRATCH_POOLS.values():
        scratch.close()
    _SCRATCH_POOLS.clear()
