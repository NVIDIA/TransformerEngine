# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections import defaultdict
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import re
import torch
from contextlib import nullcontext

from ..distributed import (
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
    _NVFP4AllGatherAsyncHandle
)
from ..quantized_tensor import QuantizedTensor
from ..tensor import NVFP4TensorStorage, MXFP8TensorStorage
from ..utils import nvtx_range_pop, nvtx_range_push
from .base import get_dummy_wgrad

import transformer_engine_torch as tex

DEBUG_TENSOR = None


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

# Global set of ETPShardedParam with in-flight async comms (AG or RS).
_inflight_comm_params: set = set()
AG_STREAM = None
RS_STREAM = None

def get_ag_stream():
    global AG_STREAM
    if AG_STREAM is None:
        AG_STREAM = torch.cuda.Stream()
    return AG_STREAM

def get_rs_stream():
    global RS_STREAM
    if RS_STREAM is None:
        RS_STREAM = torch.cuda.Stream()
    return RS_STREAM

@dataclass
class ETPConfig:
    """Global configuration for Extended Tensor Parallelism."""
    pad_for_alignment: int = 16
    check_param_states: bool = True
    weight_prefetch: bool = True

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
        etp_shard.group = etp_group
        etp_shard.ps_size = etp_size
        # register the newly sharded param back to the module
        module._parameters[name] = etp_shard

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
        for w in self.etp_shards:
            if self.reduce_scatter:
                w._set_rs_state(ETPWeightState.DATA_READY)
            else:
                w._set_state(ETPWeightState.DATA_READY)

        _inflight_comm_params.discard(self.etp_shards[0])


class ETPShardedParam(torch.nn.Parameter):

    _pending_rs_weight = None
    _first_weight_flag = True
    _last_weight = None
    _link_node_count = 0
    _link_table_buffer: List[str] = []
    _link_table_flushed: bool = False

    @classmethod
    def _buffer_link_table_row(cls, prev: "ETPShardedParam", curr: "ETPShardedParam") -> None:
        """Buffer one row of the prefetch-link table (flushed atomically on the second forward pass)."""
        _W = 70

        def _layer_id(name: str) -> str:
            m = re.search(r"\d+", name)
            return m.group() if m else "-"

        cls._link_node_count += 1
        if cls._link_node_count == 1:
            cls._link_table_buffer.append(
                f"\n{'node_id':>7} | {'layer_id':>8} | {'curr_weight_name':<{_W}} | prev_weight_name"
                f"\n{'-'*7}-+-{'-'*8}-+-{'-'*_W}-+-{'-'*_W}"
            )
            # Seed weight (first ETP param) as row 0
            cls._link_table_buffer.append(
                f"{'0':>7} | {_layer_id(prev._debug_name):>8} | {prev._debug_name:<{_W}} | -"
            )
        cls._link_table_buffer.append(
            f"{cls._link_node_count:>7} | {_layer_id(curr._debug_name):>8} | "
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
        self.ag_event = torch.cuda.Event(external=True)
        # Quantization
        self._quantizer = None
        self.did_cast_to_low_precision = False
        self.quantized = None
        # Prefetching linked list
        self.prefetch_initialized = False
        self.next_w = None
        self.prev_w = None
        # Grouped gemm
        self.is_routed_expert = False
        self.expert_idx = None
        self.group = None
        self.weight_list = None
        # Reduce-scatter state (set during wgrad_reduce_scatter)
        self.rs_state = ETPWeightState.NONE
        self.wgrad_rs = None
        self._wgrad_rs_handle = None
        self.fuse_wgrad_accumulation = False
        self._grad_accum_node = None
        self._grad_accum_hook = None
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
        for w in weights:
            w._quantize_if_needed(skip_weight_cast, cast_noop_flag)
            if w.did_cast_to_low_precision:
                w._quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

        # 3. Build gather inputs.
        quantizers = [w._quantizer for w in weights]
        if weights[0].did_cast_to_low_precision:
            gather_weights = [w.quantized for w in weights]
        else:
            gather_weights = list(w.get_padded_shard() for w in weights)

        # 4. Cache checkout (async only — sync gathers don't need pooled buffers).
        if async_op:
            dtypes = [q.dtype if q is not None else w.dtype for q, w in zip(quantizers, weights)]
            out_buffers = []
            cache = get_global_ETP_cache()
            for p, dt in zip(weights, dtypes):
                if fwd:
                    if p._ag_ticket_fwd is None:
                        p._ag_ticket_fwd = cache.reserve(p, dt, fwd=True)
                    out_buffers.append(cache.get(p._ag_ticket_fwd))
                else:
                    if p._ag_ticket_bwd is None:
                        p._ag_ticket_bwd = cache.reserve(p, dt, fwd=False)
                    out_buffers.append(cache.get(p._ag_ticket_bwd))
        else:
            out_buffers = None

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
            # Since wait() may sychronize against a different stream than the current stream,
            # an event is recorded and waited on when the data is retrieved, which ensures the
            # AG always finishes before returning the unsharded param
            with torch.cuda.stream(get_ag_stream()):
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

        if self.next_w is not None:
            result = self._get_prefetched_weight(False, skip_weight_cast=True)
        else:
            result = self._all_gather_weight_on_demand(False, skip_weight_cast=True)

        if (
            ETP_CONFIG.weight_prefetch 
            and self.prev_w is not None 
            and self.prev_w._need_weight_prefetch
        ):
            _, handle = self.prev_w._all_gather_weight(
                async_op=True, skip_weight_cast=True, cast_noop_flag=None,
                fwd=False, nvtx_label=nvtx_label,
            )
            self.prev_w._prefetch_handle = handle

        # The unsharded tensor has been returned, no pending work so reset state to NONE
        for w in self._weights:
            w._set_state(ETPWeightState.NONE)

        if self.next_w is not None:
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
        if self.prev_w is not None:
            result = self._get_prefetched_weight(True, skip_weight_cast, cast_noop_flag)
        else:
            result = self._all_gather_weight_on_demand(True, skip_weight_cast, cast_noop_flag)

        # Prefetch next weight
        if (
            ETP_CONFIG.weight_prefetch 
            and self.next_w is not None 
            and self.next_w._need_weight_prefetch
        ):
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

        if self.prev_w is not None:
            cache = get_global_ETP_cache()
            for w in self._weights:
                cache.release(w._ag_ticket_fwd)

        # Lazy population of linked list: link previous weight to current weight
        cls = type(self)
        if not self.prefetch_initialized:
            if cls._last_weight is not None and cls._last_weight.next_w is None:
                cls._buffer_link_table_row(cls._last_weight, self)
                cls._last_weight.next_w = self
                self.prev_w = cls._last_weight
            self.prefetch_initialized = True
        elif not cls._link_table_flushed and cls._link_table_buffer:
            # Second forward pass: flush the complete table atomically to avoid interleaving
            cls._link_table_flushed = True
            print_rank_0("\n".join(cls._link_table_buffer) + "\n")
        cls._last_weight = self

        return result

    def batched_all_gather_and_prefetch(self, **kwargs):
        """Batched all-gather + prefetch for expert weights. Wrapper around all_gather_and_prefetch."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch(**kwargs)

    def get_wgrad_tensor(self):
        return torch.empty(
            self._unsharded_shape,
            dtype=self.main_grad.dtype,
            device=self.device,
            requires_grad=False,
        )

    def register_grad_accum_hook(self, grad_accum_node, hook):
        self._grad_accum_node = grad_accum_node
        self._grad_accum_hook = hook

    @staticmethod
    def _finalize_wgrad(param, wgrad_rs, fuse_wgrad_accumulation):
        """Post-RS per-param processing: strip padding, accumulate, call hook.

        Returns None for fused (grad already accumulated into main_grad),
        or the stripped wgrad for unfused (to be returned to autograd).
        """

        param._set_rs_state(ETPWeightState.NONE)

        # 1. Strip padding
        if param.is_padded_last_rank:
            wgrad_rs = param._strip_padding(wgrad_rs)

        # 2. Accumulate
        if fuse_wgrad_accumulation:
            param.main_grad.add_(wgrad_rs)
            if hasattr(param, "grad_added_to_main_grad"):
                param.grad_added_to_main_grad = True
            dummy_grad = get_dummy_wgrad(list(param.main_grad.shape), param.dtype)

        # 3. Post hook
        if param._grad_accum_hook is not None:
            param.grad = dummy_grad if fuse_wgrad_accumulation else wgrad_rs
            param._grad_accum_hook(param)

        return None if fuse_wgrad_accumulation else wgrad_rs

    def _wait_reduce_scatter(self):
        # assert self._wgrad_rs_handle is not None or is_graph_capturing()
        with torch.cuda.stream(get_rs_stream()):
            if self._wgrad_rs_handle is not None:
                self._wgrad_rs_handle.wait()
                self._wgrad_rs_handle = None
                self.rs_event.record()

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

    def wgrad_reduce_scatter(self, wgrad, fuse_wgrad_accumulation, nvtx_label=None):
        """Reduce-scatter wgrad(s). Sync for last weight, async+deferred for others.

        Accepts a single tensor (non-routed) or list of tensors (routed experts).

        Returns:
            Single tensor or list for sync (last weight) — backward should return this.
            None or tuple of Nones for async — backward should return this.
        """
        batched = isinstance(wgrad, (list, tuple))
        wgrads = list(wgrad) if batched else [wgrad]
        weights = self._weights

        if ETP_CONFIG.weight_prefetch and self.prev_w is not None:
            # Async reduce-scatter (not last weight — deferred finish)
            self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            _, rs_handle = self._reduce_scatter(wgrads, async_op=True, nvtx_label=nvtx_label)
            self._wgrad_rs_handle = ETPShardHandle(rs_handle, weights, reduce_scatter=True)
            ret = tuple([None] * len(wgrads)) if batched else None
        else:
            # Sync reduce-scatter (last weight in chain)
            sharded, _ = self._reduce_scatter(wgrads, async_op=False, nvtx_label=nvtx_label)
            result = [self._finalize_wgrad(p, g, fuse_wgrad_accumulation)
                      for p, g in zip(weights, sharded)]
            ret = result if batched else result[0]

        # Wait for last reduce scatter if it was async
        # Currently only support reduce scattering in reverse order
        if self.next_w is not None:
            self.next_w._wait_reduce_scatter()
            self.next_w.rs_event.wait()

            cache = get_global_ETP_cache()
            fuse_wgrad_accumulation = self.next_w._weights[0].fuse_wgrad_accumulation
            for w in self.next_w._weights:
                self._finalize_wgrad(w, cache.get(w._rs_ticket), fuse_wgrad_accumulation)
                cache.release(w._rs_ticket)

        return ret

    def batched_wgrad_reduce_scatter(self, wgrad_list, fuse_wgrad_accumulation, nvtx_label=None):
        """Batched version of wgrad_reduce_scatter."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.wgrad_reduce_scatter(wgrad_list, fuse_wgrad_accumulation, nvtx_label=nvtx_label)

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
            key=key, param=param, dtype=dtype, reduce_scatter=reduce_scatter, fwd=fwd
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
        """Return the buffer to the pool.  Ticket remains valid."""
        slot = self._slots[ticket]
        assert slot.buf is not None
        if slot.buf not in self._pool[slot.key]:
            self._pool[slot.key].append(slot.buf)

    def clear(self):
        """Drop all buffers; tickets remain valid and lazily re-allocate on next get()."""
        for slot in self._slots.values():
            slot.buf = None
        self._pool.clear()
        self._total_bytes = 0

    def reallocate_to_mempool(self, device, mempool):
        """Re-allocate all ticket buffers into a CUDA graph memory pool.

        Call BEFORE graph capture so every buffer lives in the capture pool
        and no allocations are recorded inside the graph.
        """

        # Clone the current memory pool buffers but into the passed in mempool
        self._total_bytes = 0
        new_pool = defaultdict(list)
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mempool)
        for key, buffers in self._pool.items():
            new_buffers = []
            for _ in range(len(buffers)):
                buf = self._allocate_buffer(*self.key_to_allocate_func[key])
                new_buffers.append(buf)
            new_pool[key] = new_buffers
        torch._C._cuda_endAllocateToPool(device, mempool)

        # Map each buffer in the old pool to its corresponding new one
        old_to_new_buff = {}
        for key, old_pool in self._pool.items():
            new = new_pool[key]
            for old_buf, new_buf in zip(old_pool, new):
                old_to_new_buff[old_buf] = new_buf
        # Replace each slot's reference to its corresponding new one
        for slot in self._slots.values():
            if slot.buf is not None:
                slot.buf = old_to_new_buff[slot.buf]

        self._pool = new_pool
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


def wait_async_comms():
    """Wait on all in-flight ETP async communications (all-gathers + reduce-scatters).
    """
    for param in list(_inflight_comm_params):
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
