# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections import defaultdict
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import math
import re
import torch

from ..distributed import (
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
    _NVFP4AllGatherAsyncHandle
)
from ..quantized_tensor import QuantizedTensor
from ..tensor import NVFP4TensorStorage, MXFP8TensorStorage
from ..utils import nvtx_range_pop, nvtx_range_push, round_up_to_nearest_multiple
from ..constants import NVFP4_BLOCK_SCALING_SIZE, MXFP8_BLOCK_SCALING_SIZE
from .base import get_dummy_wgrad

import transformer_engine_torch as tex

DEBUG_TENSOR = None


class ETPChain(str, Enum):
    """Prefetch chain identifier for an ETPShardedParam.

    GRAPHED   — fwd/bwd captured by a CUDA graph (MLM _CudaGraphRunner).
    UNGRAPHED — fwd/bwd runs eagerly; includes embedding/output_layer and
                routed grouped experts always, plus router/shared_experts
                when their scope tag is not in cuda_graph_scope.

    Chains never cross-link (prev_w/next_w stay within one chain). CG
    disabled → single UNGRAPHED chain; full-iteration graph → single GRAPHED.
    """
    GRAPHED = "ETP_graphed"
    UNGRAPHED = "ETP_ungraphed"


# Module-level cuda_graph_scope, set by MLM at init via set_cuda_graph_scope().
# None or empty → CG is disabled; every ETP param classifies as UNGRAPHED.
# Value is a set of scope tags; e.g. {"mamba","attn","moe_router"}.
_CUDA_GRAPH_SCOPE: Optional[set] = None
# Whether shared_experts are run with overlap (cannot be captured). When True,
# shared_experts stay UNGRAPHED regardless of moe_router scope inclusion, matching
# the transformer_layer.py guard that excludes them from the captured submodules.
_MOE_SHARED_EXPERT_OVERLAP: bool = False


def set_cuda_graph_scope(scope, moe_shared_expert_overlap: bool = False):
    """Record the active cuda_graph_scope for ETP chain classification.

    Called by MLM at init, BEFORE classify_etp_chains(). ``scope`` may be
    None, an empty iterable (CG disabled), or an iterable of scope tags.
    """
    global _CUDA_GRAPH_SCOPE, _MOE_SHARED_EXPERT_OVERLAP
    _CUDA_GRAPH_SCOPE = set(scope) if scope else None
    _MOE_SHARED_EXPERT_OVERLAP = bool(moe_shared_expert_overlap)


def _classify_param_chain(param_name: str) -> 'ETPChain':
    """Classify an ETPShardedParam by name + active cuda_graph_scope.

    embedding / output_layer are always UNGRAPHED. Other kinds (mamba mixer,
    self/cross_attention, shared_experts, routed experts) are GRAPHED iff
    their scope tag is present in cuda_graph_scope; otherwise UNGRAPHED.
    """
    n = param_name

    # Always ungraphed — embedding and output_layer live outside any CG runner.
    if "embedding" in n or "output_layer" in n:
        return ETPChain.UNGRAPHED

    scope = _CUDA_GRAPH_SCOPE
    if not scope:
        # CG disabled: every ETP param goes to the single UNGRAPHED chain.
        return ETPChain.UNGRAPHED

    if ".mlp.shared_experts." in n:
        if _MOE_SHARED_EXPERT_OVERLAP:
            return ETPChain.UNGRAPHED
        return ETPChain.GRAPHED if ("moe" in scope or "moe_router" in scope) else ETPChain.UNGRAPHED

    if ".mlp.experts." in n:
        return ETPChain.GRAPHED if "moe" in scope else ETPChain.UNGRAPHED

    if ".self_attention." in n or ".cross_attention." in n:
        return ETPChain.GRAPHED if "attn" in scope else ETPChain.UNGRAPHED

    if ".mixer." in n:
        return ETPChain.GRAPHED if "mamba" in scope else ETPChain.UNGRAPHED

    return ETPChain.UNGRAPHED


def classify_etp_chains(model) -> None:
    """Walk model.named_parameters() and set chain_id on every ETPShardedParam.

    Call once at init, AFTER set_cuda_graph_scope() and BEFORE the first fwd
    of any graphed param. Raises if an already chain-initialized param would
    be reclassified into a different chain (its prev/next links are already
    wired into the wrong list).
    """
    conflicts = []
    for name, param in model.named_parameters():
        if not isinstance(param, ETPShardedParam):
            continue
        target = _classify_param_chain(name).value
        if param.prefetch_initialized and param.chain_id != target:
            conflicts.append((name, param.chain_id, target))
            continue
        param.chain_id = target

        # Bwd-prefetch opt-out: embedding.word_embeddings.weight does not need
        # an AG in the bwd pass (its wgrad is a scatter-add on sharded rows
        # and its input has no dgrad). Skipping its bwd AG saves one collective.
        if "embedding" in name:
            param._need_weight_prefetch_bwd = False
    if conflicts:
        raise RuntimeError(
            "classify_etp_chains: the following params were already chain-initialized "
            "with a different chain_id than the classifier would assign — this means "
            "their chain links are already wired into the wrong list. Move classification "
            "earlier in init. Conflicts: "
            + ", ".join(f"{n}: {old!r}->{new!r}" for n, old, new in conflicts[:3])
            + ("..." if len(conflicts) > 3 else "")
        )


class ETPWeightState(Enum):
    NONE = "NONE"              # Sharded, no pending operation
    ASYNC_WAIT = "ASYNC_WAIT"  # Async all-gather in progress
    DATA_READY = "DATA_READY"  # Async all-gather complete, result in cache
    DATA_READY_SYNC = "DATA_READY_SYNC"  # Sync all-gather complete, result in cache


_STATE_TRANSITIONS = {
    ETPWeightState.NONE:       {ETPWeightState.ASYNC_WAIT, ETPWeightState.DATA_READY_SYNC},
    ETPWeightState.ASYNC_WAIT: {ETPWeightState.DATA_READY},
    ETPWeightState.DATA_READY: {ETPWeightState.NONE},
    ETPWeightState.DATA_READY_SYNC: {ETPWeightState.NONE},
}


# Global ETP buffer cache (persists across clear(); never set to None after creation).
_ETP_CACHE = None
_ETP_PARAMS = []

# Global set of ETPShardedParam with in-flight async comms (AG or RS).
_inflight_comm_params: set = set()
_AG_STREAMS: Dict[str, torch.cuda.Stream] = {}
_RS_STREAMS: Dict[str, torch.cuda.Stream] = {}

# Wgrad input buffer pool, keyed by (shape, dtype). UNGRAPHED-only: GRAPHED
# wgrad bufs need address stability for CG replay and are not pool-recycled.
_wgrad_buf_pool: Dict[tuple, list] = {}


def _wgrad_pool_get(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """Get a pool buffer or allocate fresh. Tagged so _wgrad_pool_put accepts
    only pool-owned buffers — callers that don't use _wgrad_pool_get (e.g.
    Megatron layers.py wgrad GEMM, aten F.embedding bwd) fall through to the
    caching allocator on release."""
    key = (shape, dtype)
    pool = _wgrad_buf_pool.get(key)
    if pool:
        buf = pool.pop()
    else:
        buf = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
    buf._from_etp_wgrad_pool = True
    return buf


def _wgrad_pool_put(buf: torch.Tensor):
    """Return a pool-owned buffer for reuse (no-op for untagged buffers; see
    _wgrad_pool_get)."""
    if not getattr(buf, '_from_etp_wgrad_pool', False):
        return
    key = (tuple(buf.shape), buf.dtype)
    if key not in _wgrad_buf_pool:
        _wgrad_buf_pool[key] = []
    _wgrad_buf_pool[key].append(buf)


def _stream_key(chain_id: str, group) -> tuple:
    """Key for the per-(chain, group) AG/RS stream dicts.

    Two partitioning axes:
      - chain_id: captured (GRAPHED) vs eager (UNGRAPHED) ops must not share
        a stream (eager ops would contaminate capture/replay state).
      - group: independent NCCL communicators (e.g. ETP vs EETP) get their
        own user-level stream to avoid cross-group serialization.
    """
    return (chain_id, id(group) if group is not None else 0)


def get_ag_stream(chain_id: str = ETPChain.GRAPHED.value, group=None) -> torch.cuda.Stream:
    """Return the ETP all-gather stream for (chain_id, group). See _stream_key."""
    key = _stream_key(chain_id, group)
    if key not in _AG_STREAMS:
        _AG_STREAMS[key] = torch.cuda.Stream()
    return _AG_STREAMS[key]


def get_rs_stream(chain_id: str = ETPChain.GRAPHED.value, group=None) -> torch.cuda.Stream:
    """Return the ETP reduce-scatter stream for (chain_id, group). See _stream_key."""
    key = _stream_key(chain_id, group)
    if key not in _RS_STREAMS:
        _RS_STREAMS[key] = torch.cuda.Stream()
    return _RS_STREAMS[key]


def get_all_ag_streams() -> list:
    """All AG streams created so far, across chains and groups."""
    return list(_AG_STREAMS.values())


def get_all_rs_streams() -> list:
    """All RS streams created so far, across chains and groups."""
    return list(_RS_STREAMS.values())


def get_ag_streams_for_chain(chain_id: str) -> list:
    """AG streams for one chain (all groups that chain has touched)."""
    return [s for k, s in _AG_STREAMS.items() if k[0] == chain_id]


def get_rs_streams_for_chain(chain_id: str) -> list:
    """RS streams for one chain (all groups that chain has touched)."""
    return [s for k, s in _RS_STREAMS.items() if k[0] == chain_id]

# Cached once per process: whether the TE build exposes the split-phase APIs.
_COALESCED_AMAX_TE_APIS_AVAILABLE = (
    hasattr(tex, "compute_amax_nvfp4") and hasattr(tex, "quantize_cast_only_nvfp4")
)

# Tier-2: multi-tensor amax kernel fuses N per-expert (zero_amax + amax + D2D) chains
# into two multi-tensor kernel launches.  Independent of Tier-1 coalesced allreduce.
_MULTI_AMAX_TE_API_AVAILABLE = hasattr(tex, "compute_multi_amax_nvfp4")


def _coalesced_amax_static_eligible(weights):
    """Walk the weight list once and decide whether the coalesced-amax path
    is applicable. Depends only on fields that are fixed after model
    construction (quantizer class, flags, amax_reduction_group, group size)."""
    if not _COALESCED_AMAX_TE_APIS_AVAILABLE:
        return False
    if len(weights) <= 1:
        return False

    group = None
    for w in weights:
        q = w._quantizer
        if q is None or not isinstance(w.quantized, NVFP4TensorStorage):
            return False
        if not getattr(q, "with_amax_reduction", False):
            return False
        if getattr(q, "with_rht", False):
            # RHT path does amax on RHT-rotated view, can't split compute
            # from cast the way compute_amax_only assumes.
            return False
        g = getattr(q, "amax_reduction_group", None)
        if g is None:
            return False
        if group is None:
            group = g
        elif g is not group:
            return False
    return group.size() > 1


def _quantize_with_coalesced_amax(weights, skip_weight_cast, cast_noop_flag):
    """Replace the per-weight (compute_amax + allreduce + cast) loop with:
       compute_amax loop  →  one coalesced allreduce  →  cast loop."""
    group = weights[0]._quantizer.amax_reduction_group

    # Materialize padded shards once; on padded last-rank get_padded_shard()
    # launches an F.pad kernel, and we'd otherwise pay it twice per expert.
    padded_shards = [w.get_padded_shard() for w in weights]

    # Phase 1: per-weight local amax into each w.quantized's amax buffers.
    # Keep rowwise/columnwise both populated so the group allreduce sees
    # whichever the consumer GEMM will read.
    for w in weights:
        w._quantizer.set_usage(rowwise=True, columnwise=True)
    if _MULTI_AMAX_TE_API_AVAILABLE:
        # Tier-2: single multi-tensor launch writes both rowwise and columnwise
        # amax directly (no per-expert D2D replicate), fusing N per-expert chains.
        # w._quantizer is set once by _configure_quantizer and never rebinds, so
        # cache the list on weights[0] alongside _coalesced_amax_static.  Output
        # list is NOT cached because w.quantized can rebind if the weight is
        # re-quantized externally.
        anchor = weights[0]
        quantizer_list = getattr(anchor, "_multi_amax_quantizer_list", None)
        if quantizer_list is None:
            quantizer_list = [w._quantizer for w in weights]
            anchor._multi_amax_quantizer_list = quantizer_list
        tex.compute_multi_amax_nvfp4(
            padded_shards,
            quantizer_list,
            [w.quantized for w in weights],
        )
    else:
        for w, shard in zip(weights, padded_shards):
            tex.compute_amax_nvfp4(
                tensor=shard,
                quantizer=w._quantizer,
                output=w.quantized,
            )

    # Phase 2: one coalesced allreduce across every weight's amax tensors.
    amax_tensors = []
    for w in weights:
        rw = w.quantized._amax_rowwise
        cw = w.quantized._amax_columnwise
        if rw is not None:
            amax_tensors.append(rw)
        if cw is not None and (rw is None or cw.data_ptr() != rw.data_ptr()):
            amax_tensors.append(cw)
    torch.distributed.all_reduce_coalesced(
        amax_tensors,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Phase 3: per-weight cast using the pre-reduced amax; skips the internal
    # allreduce inside the quantizer.
    for w, shard in zip(weights, padded_shards):
        tex.quantize_cast_only_nvfp4(
            tensor=shard,
            quantizer=w._quantizer,
            output=w.quantized,
            noop=cast_noop_flag,
        )
        w.did_cast_to_low_precision = True


@dataclass
class ETPConfig:
    """Global configuration for Extended Tensor Parallelism."""
    pad_for_alignment: int = 16
    check_param_states: bool = True
    weight_prefetch: bool = True
    # When True and the weight list in _all_gather_weight contains >1 NVFP4
    # shards that share an amax reduction group, coalesce their per-expert
    # amax allreduces into a single NCCL call. Requires TE with
    # tex.compute_amax_nvfp4 / tex.quantize_cast_only_nvfp4; the eligibility
    # guard in _coalesced_amax_static_eligible falls back to the per-weight
    # path when either binding is missing.
    coalesce_amax_allreduce: bool = True

ETP_CONFIG = ETPConfig()

def update_config(**kwargs):
    """Update the global ETP configuration."""
    for key, value in kwargs.items():
        if not hasattr(ETP_CONFIG, key):
            raise ValueError(f"Unknown ETP config option: {key}")
        setattr(ETP_CONFIG, key, value)


def tag_etp_params_with_names(model):
    """Populate _debug_name on every ETPShardedParam with its full dotted parameter name.

    Call once after model construction so the linking log prints human-readable names
    instead of raw tensor ids.
    """
    for name, param in model.named_parameters():
        if isinstance(param, ETPShardedParam):
            param._debug_name = name


def wrap_module_params_etp(module, weight_names, etp_group, is_grouped=None):
    """Shard and re-register all parameters of a module using ETP weight sharding."""
    if etp_group.size() == 1:
        return

    etp_size = etp_group.size()
    etp_rank = etp_group.rank()

    for idx, name in enumerate(weight_names):
        param = getattr(module, name, None)
        if param is None:
            continue

        # delete the original parameter, which will be replaced by an ETP sharded one
        delattr(module, name)

        if ETP_CONFIG.pad_for_alignment > 0:
            # Ensure each shard's dim0 is a multiple of 16 for quantization (NVFP4/FP8) by padding 
            # the last rank such that the total padded length of dim0 is a multiple of ETP size * 16
            alignment = ETP_CONFIG.pad_for_alignment * etp_size
            tensor = param.data
            dim0 = tensor.shape[0]
            pad_length = (alignment - dim0 % alignment) % alignment if alignment > 0 else 0
            padded_dim0 = dim0 + pad_length
            is_padded_last_rank = pad_length > 0 and etp_rank == etp_size - 1
            # Create the ETP sharded param, pass a clone of the shard so that the original unsharded
            # buffer may be deallocated
            shard_size = padded_dim0 // etp_size
            start_idx = etp_rank * shard_size
            end_idx = min((etp_rank + 1) * shard_size, tensor.shape[0])
            shard = tensor[start_idx: end_idx]
            etp_shard = ETPShardedParam(shard.clone())
            # finally, set attributes
            etp_shard.pad_length = pad_length
            etp_shard.is_padded_last_rank = is_padded_last_rank
        else:
            shard_size = tensor.shape[0] // etp_group.size()
            shard = tensor[etp_rank * shard_size: (etp_rank + 1) * shard_size]
            etp_shard = ETPShardedParam(shard.clone())

        if is_grouped:
            etp_shard.expert_idx = idx
            etp_shard.is_routed_expert = True
            # Grouped routed experts are UNGRAPHED unless the "moe" scope captures
            # them; classify_etp_chains() will fix this up at init time based on
            # the actual cuda_graph_scope. We set UNGRAPHED here as a safe default.
            etp_shard.chain_id = ETPChain.UNGRAPHED.value
        etp_shard.group = etp_group
        etp_shard.ps_size = etp_size
        # register the newly sharded param back to the module
        module._parameters[name] = etp_shard

        global _ETP_PARAMS
        _ETP_PARAMS.append(etp_shard)

    if is_grouped:
        allweights = [getattr(module, name) for name in weight_names]
        allweights[0].weight_list = allweights


class ETPShardHandle:

    def __init__(self, handle, etp_shards, reduce_scatter=False):
        self.handle = handle
        self.etp_shards = etp_shards
        self.reduce_scatter = reduce_scatter
        _inflight_comm_params.add(etp_shards[0])

    def wait(self):
        if self.handle is not None:
            self.handle.wait()
            self.handle = None  # Release NCCL Work and its C++ tensor references promptly
        for w in self.etp_shards:
            if self.reduce_scatter:
                w._set_rs_state(ETPWeightState.DATA_READY)
            else:
                w._set_state(ETPWeightState.DATA_READY)

        _inflight_comm_params.discard(self.etp_shards[0])


class ETPShardedParam(torch.nn.Parameter):

    _pending_rs_weight = None
    _first_weight_flag = True
    # Per-chain state: each chain_id (ETPChain.GRAPHED / ETPChain.UNGRAPHED) has
    # its own linked list. Chains never cross-link: prev_w/next_w only connect
    # params with the same chain_id.
    _chain_state: Dict[str, dict] = {}

    @classmethod
    def _get_chain_state(cls, chain_id: str) -> dict:
        if chain_id not in cls._chain_state:
            cls._chain_state[chain_id] = {
                'last_weight': None,
                'link_node_count': 0,
                'link_table_buffer': [],
                'link_table_flushed': False,
            }
        return cls._chain_state[chain_id]

    @classmethod
    def _buffer_link_table_row(cls, prev: "ETPShardedParam", curr: "ETPShardedParam", chain: dict) -> None:
        """Buffer one row of the prefetch-link table (flushed atomically on the second forward pass)."""
        _W = 70

        def _layer_id(name: str) -> str:
            m = re.search(r"\d+", name)
            return m.group() if m else "-"

        chain['link_node_count'] += 1
        if chain['link_node_count'] == 1:
            chain_id = getattr(curr, 'chain_id', ETPChain.UNGRAPHED.value)
            chain['link_table_buffer'].append(
                f"\n[{chain_id} chain]"
                f"\n{'node_id':>7} | {'layer_id':>8} | {'curr_weight_name':<{_W}} | prev_weight_name"
                f"\n{'-'*7}-+-{'-'*8}-+-{'-'*_W}-+-{'-'*_W}"
            )
            # Seed weight (first ETP param) as row 0
            chain['link_table_buffer'].append(
                f"{'0':>7} | {_layer_id(prev._debug_name):>8} | {prev._debug_name:<{_W}} | -"
            )
        chain['link_table_buffer'].append(
            f"{chain['link_node_count']:>7} | {_layer_id(curr._debug_name):>8} | "
            f"{curr._debug_name:<{_W}} | {prev._debug_name}"
        )

    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):
        requires_grad = kwargs.get('requires_grad', True)
        return super(ETPShardedParam, cls).__new__(cls, tensor, requires_grad=requires_grad)

    def __init__(self, x, *args, **kwargs):
        super().__init__()
        
        # all gather
        self.state = ETPWeightState.NONE
        self._ag_ticket_fwd = None
        self._ag_ticket_bwd = None
        self._prefetch_handle = None
        self._need_weight_prefetch = True
        # Per-direction prefetch opt-outs. Default True. The embedding weight
        # never needs an AG during bwd (its wgrad is a scatter-add indexed by
        # token ids, and its input is non-differentiable, so no dgrad either).
        # classify_etp_chains() sets this to False for embedding.word_embeddings.weight.
        self._need_weight_prefetch_bwd = True
        self.ag_event = torch.cuda.Event(external=True)
        # DDP backward hook (set by register_grad_accum_hook)
        self._grad_accum_node = None
        self._grad_accum_hook = None
        # Quantization
        self._quantizer = None
        self.did_cast_to_low_precision = False
        self.quantized = None
        # Prefetching linked list
        self.prefetch_initialized = False
        self.next_w = None
        self.prev_w = None
        # Chain identity (ETPChain.GRAPHED / ETPChain.UNGRAPHED). Defaults to
        # UNGRAPHED as a safe fallback; classify_etp_chains(model) walks the
        # model at init time (after set_cuda_graph_scope) and reclassifies
        # based on param name + active cuda_graph_scope.
        self.chain_id = ETPChain.UNGRAPHED.value
        # Grouped gemm
        self.is_routed_expert = False
        self.expert_idx = None
        self.group = None
        self.weight_list = None
        # Reduce-scatter state (set during wgrad_reduce_scatter)
        self.rs_state = ETPWeightState.NONE
        self.wgrad_rs = None
        self._wgrad_rs_handle = None
        self.rs_event = torch.cuda.Event(external=True)
        self._rs_ticket = None
        # Padding
        self.is_padded_last_rank = False
        self.pad_length = 0
        # Debug
        self._debug_name = ""

    def setup(self, weight_quantizer=None):
        """Set quantizer and create quantized shard."""

        if self._quantizer is None:
            def _configure_quantizer(q, group):
                q = q.copy()
                q.with_amax_reduction = True
                q.amax_reduction_group = group
                q.internal = False
                q.optimize_for_gemm = True
                return q

            weights = self.weight_list if self.is_routed_expert and self.weight_list is not None else [self]
            for quantizer, weight in zip(weight_quantizer, weights):
                if quantizer is None:
                    continue

                weight._quantizer = _configure_quantizer(quantizer, weight.group)
                weight.quantized = weight._quantizer.quantize(weight.get_padded_shard())
                weight.quantized.is_routed_expert = getattr(weight, 'is_routed_expert', False)

    @property
    def _weights(self):
        """Return the list of individual weight shards (self for non-routed, weight_list for routed)."""
        weights = self.weight_list if self.is_routed_expert else [self]
        # Safety: all weights must be in the same state.
        assert all(w.state == weights[0].state for w in weights)
        return list(weights)

    @property
    def _unsharded_shape_padded(self):
        out_shape = list(self.size())
        if self.pad_length > 0 and self.group.rank() == self.group.size() - 1:
            out_shape[0] = (out_shape[0]+ self.pad_length) * self.group.size()
        else:
            out_shape[0] = out_shape[0] * self.group.size()
        return tuple(out_shape)   

    @property
    def _unsharded_shape(self):
        out_shape = list(self._unsharded_shape_padded)
        out_shape[0] -= self.pad_length
        return tuple(out_shape)

    @property
    def _sharded_padded_shape(self):
        out_shape = list(self.size())
        if self.pad_length > 0 and self.group.rank() == self.group.size() - 1:
            out_shape[0] += self.pad_length
        return tuple(out_shape)

    def get_padded_shard(self):
        if self.pad_length > 0 and self.is_padded_last_rank:
            return torch.nn.functional.pad(self, (0, 0, 0, self.pad_length))  
        return self

    def _set_state(self, new_state: ETPWeightState):
        # if ETP_CONFIG.check_param_states:
        #     assert new_state in _STATE_TRANSITIONS[self.state], \
        #         f"Invalid state transition: {self.state} -> {new_state}"
        self.state = new_state

    def _set_rs_state(self, new_state: ETPWeightState):
        # if ETP_CONFIG.check_param_states:
        #     assert new_state in _STATE_TRANSITIONS[self.rs_state], \
        #         f"Invalid state transition: {self.rs_state} -> {new_state}"
        self.rs_state = new_state

    def _get_cache_key(self, dtype, fwd: bool, reduce_scatter: bool) -> tuple:
        """Build cache key using output shape + dtype.

        Weights with matching gathered shape and dtype share a buffer.
        For expert weights gathered in parallel, self.expert_idx distinguishes them so
        each gets a distinct buffer, while same-indexed experts across layers share.
        """
        
        if not isinstance(dtype, torch.dtype):
            return (self._unsharded_shape_padded, dtype, fwd, not fwd, self.expert_idx, reduce_scatter)
        return (self._unsharded_shape_padded, dtype, self.expert_idx, reduce_scatter)

    def _quantize_if_needed(self, skip_weight_cast=False, cast_noop_flag=None):
        """Re-quantize sharded weight into existing buffer. Returns quantized weight or self."""
        if self._quantizer is None:
            self.did_cast_to_low_precision = False
            return self

        self._quantizer.set_usage(rowwise=True, columnwise=True)
        if skip_weight_cast is False or cast_noop_flag is not None:
            tex.quantize(
                tensor=self.get_padded_shard(),
                quantizer=self._quantizer,
                output=self.quantized,
                noop=cast_noop_flag,
            )
        self.did_cast_to_low_precision = True

        return self.quantized

    def _strip_padding(self, tensor):
        if self.pad_length == 0:
            return tensor

        if isinstance(tensor, QuantizedTensor):
            assert isinstance(tensor, (NVFP4TensorStorage, MXFP8TensorStorage)), \
                f"Unsupported quantized tensor type for ETP padding: {type(tensor)}"

            metadata = tensor.get_metadata()
            if metadata.get("rowwise_data") is not None:
                metadata["rowwise_data"] = metadata["rowwise_data"][:-self.pad_length]
            if metadata.get("columnwise_data") is not None:
                if isinstance(tensor, NVFP4TensorStorage):
                    # NVFP4 transposes columnwise and packs 2 values per byte
                    metadata["columnwise_data"] = metadata["columnwise_data"][
                        ..., :-self.pad_length // 2
                    ].contiguous()
                else:
                    # MXFP8 columnwise is not transposed, strip first dim
                    metadata["columnwise_data"] = metadata["columnwise_data"][
                        :-self.pad_length
                    ]
            M = self._unsharded_shape[0]
            if isinstance(tensor, NVFP4TensorStorage):
                # NVFP4 scale_inv shapes (see NVFP4Quantizer.get_scale_shape):
                #   rowwise_scale_inv:    [round_up(M, 128),  round_up(ceil(K/16), 4)]
                #   columnwise_scale_inv: [round_up(K, 128),  round_up(ceil(M/16), 4)]
                # ETP shards M (dim 0 of the weight), so strip to the unpadded sizes.
                if metadata.get("rowwise_scale_inv") is not None:
                    m_rows = round_up_to_nearest_multiple(M, 128)
                    metadata["rowwise_scale_inv"] = metadata["rowwise_scale_inv"][:m_rows]
                if metadata.get("columnwise_scale_inv") is not None:
                    m_tiles = round_up_to_nearest_multiple(
                        math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4
                    )
                    metadata["columnwise_scale_inv"] = (
                        metadata["columnwise_scale_inv"][:, :m_tiles].contiguous()
                    )
            else:
                # MXFP8 scale_inv shapes (see MXFP8Quantizer.get_scale_shape):
                #   rowwise_scale_inv:    [round_up(M, 128),     round_up(K//32, 4)]
                #   columnwise_scale_inv: [round_up(M//32, 4),   round_up(K, 128)]
                # ETP shards M (dim 0 of the weight), so strip to the unpadded sizes.
                if metadata.get("rowwise_scale_inv") is not None:
                    m_rows = round_up_to_nearest_multiple(M, 128)
                    metadata["rowwise_scale_inv"] = metadata["rowwise_scale_inv"][:m_rows]
                if metadata.get("columnwise_scale_inv") is not None:
                    m_tiles = round_up_to_nearest_multiple(
                        M // MXFP8_BLOCK_SCALING_SIZE, 4
                    )
                    metadata["columnwise_scale_inv"] = (
                        metadata["columnwise_scale_inv"][:m_tiles]
                    )

            return type(tensor)(**metadata, shape=self._unsharded_shape, dtype=torch.bfloat16)
        else:
            return tensor[:-self.pad_length]

    def _all_gather_weight(self, async_op, skip_weight_cast, cast_noop_flag, fwd, nvtx_label=None):
        """Quantize (if needed) and all-gather weight. Returns (weight_total, handle)."""
        if nvtx_label is None:
            nvtx_label = (
                self._debug_name
                + (".fwd" if fwd else ".bwd")
                + (".async" if async_op else ".sync")
            )

        weights = self._weights

        # 1. Transition state for async gathers.
        if async_op:
            for w in weights:
                w._set_state(ETPWeightState.ASYNC_WAIT)
        else:
            for w in weights:
                w._set_state(ETPWeightState.DATA_READY_SYNC)

        # 2. Prepare: quantize, set usage direction.
        # Static eligibility (quantizer class, flags, amax group) is fixed
        # after model construction — compute once and cache on self so the
        # hot path only pays the cheap per-call skip_weight_cast check.
        if ETP_CONFIG.coalesce_amax_allreduce:
            static_ok = getattr(self, "_coalesced_amax_static", None)
            if static_ok is None:
                static_ok = _coalesced_amax_static_eligible(weights)
                self._coalesced_amax_static = static_ok
            # Per-call: match the skip_weight_cast gate in _quantize_if_needed
            # (fire when either skip_weight_cast is False or cast_noop_flag
            # was provided by the FP8/NVFP4 recipe).
            use_coalesced = static_ok and not (
                skip_weight_cast is True and cast_noop_flag is None
            )
        else:
            use_coalesced = False

        if use_coalesced:
            _quantize_with_coalesced_amax(weights, skip_weight_cast, cast_noop_flag)
        else:
            for w in weights:
                w._quantize_if_needed(skip_weight_cast, cast_noop_flag)
        for w in weights:
            if w.did_cast_to_low_precision:
                w._quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

        # 3. Build gather inputs.
        quantizers = [w._quantizer for w in weights]
        if weights[0].did_cast_to_low_precision:
            gather_weights = [w.quantized for w in weights]
        else:
            gather_weights = list(w.get_padded_shard() for w in weights)

        # 4. Cache checkout — use pooled buffers for both async and sync gathers
        #    to avoid allocating fresh memory each iteration.
        dtypes = [q.dtype if q is not None else w.dtype for q, w in zip(quantizers, weights)]
        out_buffers = []
        cache = get_global_ETP_cache()
        for p, dt in zip(weights, dtypes):
            if fwd:
                if p._ag_ticket_fwd is None:
                    p._ag_ticket_fwd = cache.reserve(p, dt, fwd=True)
                    cache.get(p._ag_ticket_fwd)
                    cache.release(p._ag_ticket_fwd)
                out_buffers.append(cache.get(p._ag_ticket_fwd))
            else:
                if p._ag_ticket_bwd is None:
                    p._ag_ticket_bwd = cache.reserve(p, dt, fwd=False)
                out_buffers.append(cache.get(p._ag_ticket_bwd))

        # 5. Communicate.
        etp_group = weights[0].group
        if out_buffers is not None and len(gather_weights) > 1:
            assert len(set(id(b) for b in out_buffers)) == len(out_buffers), \
                "Duplicate output buffers in batched all-gather — experts need distinct cache keys"

        if len(gather_weights) > 1:
            nvtx_range_push(f"{nvtx_label}.batched_etp_ag")
            results, handle = grouped_gather_along_first_dim(
                gather_weights, etp_group,
                async_op=async_op,
                quantizers=quantizers,
                output_tensors=out_buffers,
            )
            nvtx_range_pop(f"{nvtx_label}.batched_etp_ag")
        else:
            nvtx_range_push(f"{nvtx_label}.etp_ag")
            weight_total, handle = gather_along_first_dim(
                gather_weights[0], etp_group,
                quantizer=quantizers[0],
                async_op=async_op,
                output_tensor=out_buffers[0] if out_buffers is not None else None,
            )
            nvtx_range_pop(f"{nvtx_label}.etp_ag")
            results = [weight_total]

        result = results if self.is_routed_expert else results[0]

        # 6. Wrap handle.
        if async_op:
            handle = ETPShardHandle(handle, weights)
        else:
            handle = None

        return result, handle

    def _wait_param_gather(self):
        # Wait-site for the async AG. Issuer (all_gather_and_prefetch{,_bwd})
        # and wait-site both use the TARGET's ag_stream so the caller-stream
        # "preEvent" PyTorch records at issue time lives on an idle stream.
        # A busy issue-stream would queue the preEvent behind pending work,
        # delay NCCL start, and — even with the sync chain main ← ag_event ←
        # ag_stream handle.wait() ← NCCL endEvent — leave the consumer GEMM
        # reading a partial AG buffer. (NCCL kernel itself runs on PyTorch's
        # per-PG ncclStream, not ag_stream.) handle.wait() here inserts the
        # wait on NCCL's completion event into ag_stream; ag_event.record()
        # then marks ag_stream for consumers (main_stream via ag_event.wait
        # or MLM drains via main.wait_stream).
        with torch.cuda.stream(get_ag_stream(self.chain_id, self.group)):
            if self._prefetch_handle is not None:
                self._prefetch_handle.wait()
                self._prefetch_handle = None
                self.ag_event.record()

    def _all_gather_weight_on_demand(self, fwd, skip_weight_cast=False, cast_noop_flag=None):
        result, _ = self._all_gather_weight(
            async_op=False,
            skip_weight_cast=skip_weight_cast,
            cast_noop_flag=cast_noop_flag,
            fwd=fwd,
        )
        result = result if self.is_routed_expert else [result]
        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result,self._weights)]
        return result if self.is_routed_expert else result[0]

    def _get_prefetched_weight(self, fwd, skip_weight_cast=False, cast_noop_flag=None):
        # Stale-read guard: state must reflect an AG issued for this cycle;
        # otherwise cache.get() would return the prior iter's AG buffer.
        if ETP_CONFIG.check_param_states:
            for w in self._weights:
                assert w.state in (
                    ETPWeightState.ASYNC_WAIT,
                    ETPWeightState.DATA_READY,
                    ETPWeightState.DATA_READY_SYNC,
                ), (
                    f"[ETP] _get_prefetched_weight({'fwd' if fwd else 'bwd'}) on "
                    f"{self._debug_name} with state={w.state!r} — no AG issued; "
                    f"cache.get() would return stale data. Check the chain's "
                    f"_need_weight_prefetch flag and issuer's prefetch logic."
                )
        # Wait for async prefetch if in progress
        self._wait_param_gather()
        self.ag_event.wait()

        # Retrieve prefetched results from cache
        result = []
        cache = get_global_ETP_cache()
        for w in self._weights:
            ticket = w._ag_ticket_fwd if fwd else w._ag_ticket_bwd
            result.append(cache.get(ticket))

        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, self._weights)]
        return result if self.is_routed_expert else result[0]

    def all_gather_and_prefetch_bwd(self, nvtx_label=None):
        """
        Backward variant: get current weight (from cache if prefetched, else
        sync gather) and async-prefetch prev_w.

        Safe thanks to the coat-check cache: get() returns the current buffer
        to the pool, and the prefetch's checkout() will allocate a separate
        buffer if the pool is empty (i.e. the current buffer is still live
        via the caller's tensor reference).

        Returns:
            weight_total
        """

        if ETP_CONFIG.weight_prefetch and self.next_w is not None:
            result = self._get_prefetched_weight(False, skip_weight_cast=True)
        else:
            result = self._all_gather_weight_on_demand(False, skip_weight_cast=True)

        if (
            ETP_CONFIG.weight_prefetch
            and self.prev_w is not None
            and self.prev_w._need_weight_prefetch
            and self.prev_w._need_weight_prefetch_bwd
        ):
            # Issue on the target's ag_stream (see _wait_param_gather).
            target_stream = get_ag_stream(self.prev_w.chain_id, self.prev_w.group)
            with torch.cuda.stream(target_stream):
                _, handle = self.prev_w._all_gather_weight(
                    async_op=True, skip_weight_cast=True, cast_noop_flag=None,
                    fwd=False, nvtx_label=nvtx_label,
                )
            self.prev_w._prefetch_handle = handle

        # The unsharded tensor has been returned, no pending work so reset state to NONE
        for w in self._weights:
            w._set_state(ETPWeightState.NONE)

        if ETP_CONFIG.weight_prefetch and self.next_w is not None:
            cache = get_global_ETP_cache()
            for w in self._weights:
                cache.release(w._ag_ticket_bwd)

        return result

    def batched_all_gather_and_prefetch_bwd(self, nvtx_label=None):
        """Batched backward all-gather + prefetch. Wrapper around all_gather_and_prefetch_bwd."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch_bwd(nvtx_label=nvtx_label)

    def all_gather_and_prefetch(
        self,
        fwd: bool = True,
        skip_weight_cast: bool = False,
        cast_noop_flag: torch.Tensor = None,
        nvtx_label: str = None,
    ):
        """
        All-gather current weight and async-prefetch the next weight.

        Returns:
            weight_total
        """
        if ETP_CONFIG.weight_prefetch and self.prev_w is not None:
            result = self._get_prefetched_weight(True, skip_weight_cast, cast_noop_flag)
        else:
            result = self._all_gather_weight_on_demand(True, skip_weight_cast, cast_noop_flag)

        # Prefetch next weight
        if (
            ETP_CONFIG.weight_prefetch
            and self.next_w is not None
            and self.next_w._need_weight_prefetch
        ):
            # Issue on the target's ag_stream (see _wait_param_gather).
            target_stream = get_ag_stream(self.next_w.chain_id, self.next_w.group)
            with torch.cuda.stream(target_stream):
                _, handle = self.next_w._all_gather_weight(
                    async_op=True,
                    skip_weight_cast=skip_weight_cast,
                    cast_noop_flag=cast_noop_flag,
                    fwd=fwd, nvtx_label=nvtx_label,
                )
            self.next_w._prefetch_handle = handle

        # The unsharded tensor has been returned, no pending work so reset state to NONE
        for w in self._weights:
            w._set_state(ETPWeightState.NONE)

        # Lazy population of linked list: link previous weight to current weight
        # Uses per-chain state so dense and expert chains never cross-link.
        cls = type(self)
        chain = cls._get_chain_state(self.chain_id)
        if not self.prefetch_initialized:
            last_w = chain['last_weight']
            if last_w is not None and last_w.next_w is None:
                cls._buffer_link_table_row(last_w, self, chain)
                last_w.next_w = self
                self.prev_w = last_w

            cache = get_global_ETP_cache()

            # Set the fwd ag buffer
            quantizers = [w._quantizer for w in self._weights]
            dtypes = [q.dtype if q is not None else w.dtype for q, w in zip(quantizers, self._weights)]
            for w, dt in zip(self._weights, dtypes):
                w._ag_ticket_fwd = cache.reserve(w, dt, fwd=True)
                cache.get(w._ag_ticket_fwd)
                cache.release(w._ag_ticket_fwd)

            self.prefetch_initialized = True
            chain['last_weight'] = self
        elif not chain['link_table_flushed'] and chain['link_table_buffer']:
            # Second forward pass: flush the complete table atomically to avoid interleaving
            chain['link_table_flushed'] = True
            print_rank_0("\n".join(chain['link_table_buffer']) + "\n")

        return result

    def batched_all_gather_and_prefetch(self, **kwargs):
        """Batched all-gather + prefetch for expert weights. Wrapper around all_gather_and_prefetch."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch(**kwargs)

    def get_wgrad_tensor(self):
        return _wgrad_pool_get(self._unsharded_shape, self.main_grad.dtype, self.device)

    def register_grad_accum_hook(self, grad_accum_node, hook):
        """Register a DDP backward hook to be called from _finalize_wgrad.

        For ETP params, autograd may receive None (async RS) so the normal grad
        accumulator hook never fires. Instead, _finalize_wgrad calls the hook
        explicitly after RS wait + gradient accumulation, ensuring DDP's
        register_grad_ready fires at exactly the right time.
        """
        self._grad_accum_node = grad_accum_node
        self._grad_accum_hook = hook

    @staticmethod
    def _finalize_wgrad(param, wgrad_rs):
        """Post-RS per-param processing: strip padding, accumulate, call DDP hook.

        Accumulates the reduce-scattered wgrad into main_grad and triggers
        the DDP backward hook (register_grad_ready) so the DP reduce-scatter
        fires at the correct time during backward.
        """

        param._set_rs_state(ETPWeightState.NONE)

        # 1. Strip padding
        if param.is_padded_last_rank:
            wgrad_rs = param._strip_padding(wgrad_rs)

        # 2. Accumulation: accumulate wgrad into main_grad
        param.main_grad.add_(wgrad_rs)
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = True
        dummy_grad = get_dummy_wgrad(list(param.main_grad.shape), param.dtype)

        # 3. Trigger DDP backward hook (register_grad_ready).
        # ETP bypasses autograd's normal gradient flow (returns None for async RS,
        # accumulates directly into main_grad), so we must trigger the DDP hook
        # manually. Do NOT set param.grad before calling — the hook checks
        # param.grad and would accumulate it into main_grad if zero_out_wgrad
        # is True, corrupting the gradient with a non-zero dummy.
        if getattr(param, '_grad_accum_hook', None) is not None:
            param._grad_accum_hook()

        return dummy_grad


    def _wait_reduce_scatter(self):
        # Asymmetric wrt _wait_param_gather: RS is issued from main_stream
        # (not rs_stream) because main produced the RS input (wgrad) and
        # naturally holds the write→read ordering. Wait-site enters rs_stream
        # so it observes NCCL completion and rs_event marks it for consumers.
        with torch.cuda.stream(get_rs_stream(self.chain_id, self.group)):
            if self._wgrad_rs_handle is not None:
                self._wgrad_rs_handle.wait()
                self._wgrad_rs_handle = None
                self.rs_event.record()
        # Release stashed wgrad inputs: UNGRAPHED buffers go back to the pool;
        # GRAPHED just drops Python refs (addresses must stay stable for CG).
        if getattr(self, '_wgrad_input_bufs', None) is not None:
            if self.chain_id == ETPChain.UNGRAPHED.value:
                for buf in self._wgrad_input_bufs:
                    _wgrad_pool_put(buf)
            self._wgrad_input_bufs = None

    def _reduce_scatter(self, wgrads, async_op, nvtx_label=None):
        """Reduce-scatter one or more wgrads. Returns (outputs, handle).

        Single tensor: plain reduce-scatter (no coalescing).
        Multiple tensors: coalesced reduce-scatter.
        """
        if nvtx_label is None:
            nvtx_label = (
                self._debug_name
                + ".bwd"
                + (".async" if async_op else ".sync")
            )

        for w in self._weights:
            if async_op:
                w._set_rs_state(ETPWeightState.ASYNC_WAIT)
            else:
                w._set_rs_state(ETPWeightState.DATA_READY_SYNC)

        if self.pad_length > 0:
            wgrads = [torch.nn.functional.pad(w, (0, 0, 0, self.pad_length)) for w in wgrads]

        if async_op:
            dtypes = [w.dtype for w in wgrads]
            out_buffers = []
            cache = get_global_ETP_cache()
            for p, dt in zip(self._weights, dtypes):
                if p._rs_ticket is None:
                    p._rs_ticket = cache.reserve(p, dt, fwd=False, reduce_scatter=True)
                out_buffers.append(cache.get(p._rs_ticket))
        else:
            out_buffers = [None] * len(wgrads)

        if len(wgrads) == 1:
            nvtx_range_push(f"{nvtx_label}.etp_rs")
            out, handle = reduce_scatter_along_first_dim(
                wgrads[0], self.group, async_op=async_op, output=out_buffers[0]
            )
            nvtx_range_pop(f"{nvtx_label}.etp_rs")
            return [out], handle
        else:
            outputs = []
            nvtx_range_push(f"{nvtx_label}.batched_etp_rs")
            with torch.distributed._coalescing_manager(
                group=self.group,
                device=wgrads[0].device,
                async_ops=async_op,
            ) as cm:
                for out_buffer, tensor in zip(out_buffers, wgrads):
                    out, _ = reduce_scatter_along_first_dim(tensor, self.group, output=out_buffer)
                    outputs.append(out)
            nvtx_range_pop(f"{nvtx_label}.batched_etp_rs")

            return outputs, cm if async_op else None

    def wgrad_reduce_scatter(self, wgrad, nvtx_label=None):
        """Reduce-scatter wgrad(s). Sync for last weight, async+deferred for others.

        Accepts a single tensor (non-routed) or list of tensors (routed experts).

        Returns:
            Single tensor or list for sync (last weight) — backward should return this.
            None or tuple of Nones for async — backward should return this.
        """
        batched = isinstance(wgrad, (list, tuple))
        wgrads = list(wgrad) if batched else [wgrad]
        weights = self._weights

        # UNGRAPHED-chain wgrads are recycled via the standalone pool (_wgrad_pool_put).
        # GRAPHED-chain wgrads cannot pool-recycle because CUDA graphs require
        # stable buffer addresses across replay.
        poolable = self.chain_id == ETPChain.UNGRAPHED.value

        if ETP_CONFIG.weight_prefetch and self.prev_w is not None:
            # Async reduce-scatter (not last weight — deferred finish).  Issue on rs_stream to
            # match wait-site (issue-site invariant; see _wait_param_gather).  wgrad is produced
            # on outer stream by bwd GEMM, so sync outer → rs_stream first.
            outer_stream = torch.cuda.current_stream()
            rs_stream = get_rs_stream(self.chain_id, self.group)
            outer_sync_event = torch.cuda.Event()
            outer_sync_event.record(outer_stream)
            rs_stream.wait_event(outer_sync_event)
            with torch.cuda.stream(rs_stream):
                _, rs_handle = self._reduce_scatter(wgrads, async_op=True, nvtx_label=nvtx_label)
            self._wgrad_rs_handle = ETPShardHandle(rs_handle, weights, reduce_scatter=True)
            # Stash wgrad input buffers — cannot recycle yet because the async RS
            # kernel is still reading them on rs_stream.
            self._wgrad_input_bufs = wgrads
            ret = tuple([None] * len(wgrads)) if batched else None
        else:
            # Sync reduce-scatter (last weight in chain) — RS done, recycle immediately
            sharded, _ = self._reduce_scatter(wgrads, async_op=False, nvtx_label=nvtx_label)
            result = [self._finalize_wgrad(p, g) for p, g in zip(weights, sharded)]
            if poolable:
                for buf in wgrads:
                    _wgrad_pool_put(buf)
            ret = result if batched else result[0]

        # Wait for last reduce scatter if it was async
        # Currently only support reduce scattering in reverse order
        if ETP_CONFIG.weight_prefetch and self.next_w is not None:
            self.next_w._wait_reduce_scatter()
            self.next_w.rs_event.wait()

            cache = get_global_ETP_cache()
            for w in self.next_w._weights:
                self._finalize_wgrad(w, cache.get(w._rs_ticket))
                cache.release(w._rs_ticket)

        return ret

    def batched_wgrad_reduce_scatter(self, wgrad_list, nvtx_label=None):
        """Batched version of wgrad_reduce_scatter."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.wgrad_reduce_scatter(wgrad_list, nvtx_label=nvtx_label)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.detach:
            with torch._C.DisableTorchFunctionSubclass():
                # Perform the raw detach
                result = func(*args, **kwargs)
            # Re-wrap it in your subclass so PyTorch is happy
            return result.as_subclass(type(self))

        # 2. For everything else (add, mul, etc.), be transparent/decay.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

@dataclass
class _TicketSlot:
    """Internal slot backing a persistent ticket in the ETP buffer cache."""
    key: tuple                                   # cache key (shape, dtype, ...)
    param: 'ETPShardedParam'                     # for lazy allocation metadata
    dtype: object                                # torch.dtype or tex.DType
    reduce_scatter: bool
    fwd: bool
    chain_id: str = ETPChain.GRAPHED.value       # chain this slot belongs to
    buf: Optional[torch.Tensor] = field(default=None)  # None when released or after clear()


class ETPWeightCache:
    """
    Ticket-based buffer pool for ETP all-gather / reduce-scatter buffers.

    - ``reserve(param, dtype, fwd)`` → ``ticket``
      Assigns a persistent ticket (no buffer allocated yet).
    - ``get(ticket)`` → ``buffer``
      Returns the buffer, lazily allocating from pool or fresh if needed.
    - ``release(ticket)``
      Returns the buffer to the pool.  Ticket remains valid; next ``get()``
      will re-allocate from the pool.
    - ``clear()``
      Drops all buffers and pools.  Tickets remain valid; next ``get()``
      lazily allocates fresh buffers.
    """

    # Bytes per element for known dtypes (used for logging).
    _BYTES_PER_ELEMENT = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
        tex.DType.kFloat4E2M1: 0.5,
        tex.DType.kFloat8E4M3: 1,
    }

    def __init__(self):
        self._pool: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self._slots: Dict[int, _TicketSlot] = {}
        self._next_ticket: int = 0
        self._total_bytes: int = 0              # running total of allocated bytes
        self.key_to_allocate_func = {}

    @staticmethod
    def _buf_bytes(shape, dtype) -> int:
        """Estimate buffer size in bytes."""
        numel = 1
        for d in shape:
            numel *= d
        bpe = ETPWeightCache._BYTES_PER_ELEMENT.get(dtype, None)
        return numel * bpe

    def _allocate_buffer(self, param: 'ETPShardedParam', dtype, reduce_scatter, fwd) -> torch.Tensor:
        if reduce_scatter:
            out_shape = param._sharded_padded_shape
        else:
            out_shape = param._unsharded_shape_padded

        if not isinstance(dtype, torch.dtype):
            quantizer = param._quantizer
            assert quantizer is not None
            param._quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

            buf = param._quantizer.make_empty(
                out_shape, 
                dtype=torch.bfloat16, 
                device=torch.cuda.current_device(),
            )
        else:
            buf = torch.empty(
                out_shape, dtype=dtype, device=param.device, memory_format=torch.contiguous_format
            )

        buf_bytes = self._buf_bytes(out_shape, dtype)
        self._total_bytes += buf_bytes
        print_rank_0(
            f"[ETP Cache] +{buf_bytes / 1024**2:.1f} MB  (shape={out_shape}, dtype={dtype})  "
            f"total={self._total_bytes / 1024**2:.1f} MB id: {id(buf)} fwd: {fwd}"
        )
        return buf

    def reserve(self, param: 'ETPShardedParam', dtype, fwd: bool, reduce_scatter=False) -> int:
        """Assign a persistent ticket.  No buffer is allocated until ``get()``."""
        key = param._get_cache_key(dtype, fwd, reduce_scatter)
        ticket = self._next_ticket
        self._next_ticket += 1

        self._slots[ticket] = _TicketSlot(
            key=key, param=param, dtype=dtype, reduce_scatter=reduce_scatter, fwd=fwd,
            chain_id=getattr(param, 'chain_id', ETPChain.UNGRAPHED.value),
        )
        return ticket

    def get(self, ticket: int) -> torch.Tensor:
        """Return the buffer for *ticket*, lazily allocating if needed."""
        slot = self._slots[ticket]
        if slot.buf is None:
            pool = self._pool[slot.key]
            slot.buf = pool.pop() if pool else self._allocate_buffer(
                slot.param, slot.dtype, slot.reduce_scatter, fwd=slot.fwd
            )
            self.key_to_allocate_func[slot.key] = (slot.param, slot.dtype, slot.reduce_scatter, slot.fwd)
            
        return slot.buf

    def release(self, ticket: int):
        """Return the buffer to the pool.  Ticket remains valid.

        slot.buf is intentionally NOT cleared: get() must stay idempotent so that
        CUDA-graph-captured buffers keep their fixed address across replays, and
        reallocate_to_mempool() can find every dense-chain buffer.
        """
        slot = self._slots[ticket]
        if slot.buf is None:
            return
        # Use identity check — tensor == tensor returns a multi-element bool tensor
        # which crashes in a boolean context ("Boolean value of Tensor is ambiguous").
        if not any(b is slot.buf for b in self._pool.get(slot.key, [])):
            self._pool[slot.key].append(slot.buf)

    def clear(self):
        """Drop all buffers; tickets remain valid and lazily re-allocate on next get()."""
        for slot in self._slots.values():
            slot.buf = None
        self._pool.clear()
        self._total_bytes = 0

    def reallocate_to_mempool(self, device, mempool):
        """Re-allocate GRAPHED-chain ticket buffers into a CUDA graph memory pool.

        Call BEFORE graph capture so every GRAPHED-chain buffer lives in the capture
        pool and no allocations are recorded inside the graph. UNGRAPHED-chain
        buffers are left in regular memory (they are never referenced by any
        captured graph).
        """

        # Identify keys that belong to the GRAPHED chain
        graphed_keys = set()
        for slot in self._slots.values():
            if slot.chain_id == ETPChain.GRAPHED.value:
                graphed_keys.add(slot.key)

        # Clone only GRAPHED-chain pool buffers into the passed in mempool
        self._total_bytes = 0
        new_pool = defaultdict(list)
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mempool)
        for key, buffers in self._pool.items():
            if key not in graphed_keys:
                continue
            new_buffers = []
            for _ in range(len(buffers)):
                buf = self._allocate_buffer(*self.key_to_allocate_func[key])
                new_buffers.append(buf)
            new_pool[key] = new_buffers
        torch._C._cuda_endAllocateToPool(device, mempool)

        # Map each buffer in the old pool to its corresponding new one (GRAPHED only)
        old_to_new_buff = {}
        for key, old_pool in self._pool.items():
            if key not in graphed_keys:
                continue
            new = new_pool[key]
            for old_buf, new_buf in zip(old_pool, new):
                old_to_new_buff[old_buf] = new_buf

        # Replace each GRAPHED slot's reference; keep UNGRAPHED slots unchanged
        for slot in self._slots.values():
            if slot.chain_id == ETPChain.GRAPHED.value and slot.buf is not None and slot.buf in old_to_new_buff:
                slot.buf = old_to_new_buff[slot.buf]

        # Merge: GRAPHED keys get new buffers, UNGRAPHED keys keep old ones
        for key, buffers in self._pool.items():
            if key not in graphed_keys:
                new_pool[key] = buffers
        self._pool = new_pool

        # Remap quantized params into the CG mempool — but only for params on
        # the GRAPHED chain. UNGRAPHED-chain params (embedding, output_layer,
        # and MoE paths whose scope is not captured) run eagerly and don't
        # need their quantized storage in the CG mempool.
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mempool)
        for w in _ETP_PARAMS:
            if getattr(w, "chain_id", ETPChain.GRAPHED.value) != ETPChain.GRAPHED.value:
                continue
            if w.quantized is None:
                continue
            if isinstance(w.quantized, NVFP4TensorStorage):
                w.quantized._rowwise_data = torch.clone(w.quantized._rowwise_data)
                w.quantized._columnwise_data = torch.clone(w.quantized._columnwise_data)
                w.quantized._rowwise_scale_inv = torch.clone(w.quantized._rowwise_scale_inv)
                w.quantized._columnwise_scale_inv = torch.clone(w.quantized._columnwise_scale_inv)
                w.quantized._amax_columnwise = torch.clone(w.quantized._amax_columnwise)
                w.quantized._amax_rowwise = torch.clone(w.quantized._amax_rowwise)
            elif isinstance(w.quantized, MXFP8TensorStorage):
                w.quantized._rowwise_data = torch.clone(w.quantized._rowwise_data)
                w.quantized._columnwise_data = torch.clone(w.quantized._columnwise_data)
                w.quantized._rowwise_scale_inv = torch.clone(w.quantized._rowwise_scale_inv)
                w.quantized._columnwise_scale_inv = torch.clone(w.quantized._columnwise_scale_inv)
            else:
                assert False
        torch._C._cuda_endAllocateToPool(device, mempool)

        return

def get_global_ETP_cache() -> ETPWeightCache:
    """Get or lazily create the global cache instance."""
    global _ETP_CACHE
    if _ETP_CACHE is None:
        _ETP_CACHE = ETPWeightCache()
    return _ETP_CACHE


def reallocate_etp_cache_to_mempool(device, mempool):
    """Re-allocate all ETP cache buffers into a CUDA graph memory pool."""
    if _ETP_CACHE is not None:
        _ETP_CACHE.reallocate_to_mempool(device, mempool)


def wait_async_comms(chain_id: str = None):
    """Wait on in-flight ETP async communications (all-gathers + reduce-scatters).

    Args:
        chain_id: If specified, only drain params belonging to this chain
                  (ETPChain.GRAPHED.value or ETPChain.UNGRAPHED.value).
                  If None, drain all chains.
    """
    for param in list(_inflight_comm_params):
        if chain_id is not None and getattr(param, 'chain_id', ETPChain.UNGRAPHED.value) != chain_id:
            continue
        param._wait_param_gather()
        param._wait_reduce_scatter()


@dataclass
class BatchedNVFP4AllGatherAsyncHandle:
    """Handle for batched asynchronous NVFP4 all-gathers."""
    output_handles: List[_NVFP4AllGatherAsyncHandle]
    outer_async_handle: torch.distributed.Work
    _synchronized: bool = False

    def wait(self) -> None:
        """Wait for the async operation to complete and post-process the tensor."""
        if self._synchronized:
            return
        self.outer_async_handle.wait()
        # Fixes interleaved data for transposed tensor/scale inv and pads scale inv if needed.
        for output_handle in self.output_handles:
            if output_handle is not None:
                assert output_handle.async_handle is None
                output_handle.post_process_nvfp4_gather()
                # release any tensor references just in case
                output_handle.output = None
                output_handle.columnwise_data_interleaved = None
                output_handle.columnwise_scale_inv_interleaved = None

        self._synchronized = True


def grouped_gather_along_first_dim(
    weights: list,
    process_group,
    async_op: bool = False,
    quantizers: list = None,
    output_tensors: list = None,
):
    """
    All-gather multiple weights in a single coalesced operation.

    Handles NVFP4 post-processing for both sync and async paths.
    """
    # Determine device from first weight.
    inp = weights[0]
    if isinstance(inp, NVFP4TensorStorage):
        device = (
            inp._rowwise_data.device if inp._rowwise_data is not None
            else inp._columnwise_data.device
        )
    else:
        device = inp.device

    weights_all = []
    weight_handles = []
    with torch.distributed._coalescing_manager(
        group=process_group, device=device, async_ops=async_op,
    ) as gather_coalescing_manager:
        for i, weight in enumerate(weights):
            weight_all, weight_handle = gather_along_first_dim(
                weight, process_group,
                quantizer=quantizers[i],
                output_tensor=output_tensors[i] if output_tensors is not None else None,
                grouped=True,
            )
            weights_all.append(weight_all)
            weight_handles.append(weight_handle)

    if async_op:
        handle = gather_coalescing_manager
        if (
            quantizers is not None
            and getattr(quantizers[0], "columnwise_usage", False)
        ):
            handle = BatchedNVFP4AllGatherAsyncHandle(weight_handles, handle)
    else:
        for wh in weight_handles:
            if isinstance(wh, _NVFP4AllGatherAsyncHandle):
                wh.post_process_nvfp4_gather()
        handle = None

    return weights_all, handle
