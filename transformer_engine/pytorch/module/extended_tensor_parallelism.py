# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections import defaultdict
from typing import Dict, List
from enum import Enum
from dataclasses import dataclass
import torch

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


class ETPWeightState(Enum):
    NONE = "NONE"              # Sharded, no pending operation
    ASYNC_WAIT = "ASYNC_WAIT"  # Async all-gather in progress
    ASYNC_DONE = "ASYNC_DONE"  # Async all-gather complete, result in cache

_STATE_TRANSITIONS = {
    ETPWeightState.NONE:       {ETPWeightState.ASYNC_WAIT},
    ETPWeightState.ASYNC_WAIT: {ETPWeightState.ASYNC_DONE},
    ETPWeightState.ASYNC_DONE: {ETPWeightState.NONE},
}


# Global AG Prefetching Buffer for ETP.
_ALL_GATHER_BUFFER = None


@dataclass
class ETPConfig:
    """Global configuration for Extended Tensor Parallelism."""
    pad_for_alignment: int = 16
    weight_prefetch: bool = True

ETP_CONFIG = ETPConfig()

def update_config(**kwargs):
    """Update the global ETP configuration."""
    for key, value in kwargs.items():
        if not hasattr(ETP_CONFIG, key):
            raise ValueError(f"Unknown ETP config option: {key}")
        setattr(ETP_CONFIG, key, value)


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

    def __init__(self, handle, etp_shards: list):
        self.handle = handle
        self.etp_shards = etp_shards

    def wait(self):
        if self.handle is not None:
            self.handle.wait()
        for w in self.etp_shards:
            w._set_state(ETPWeightState.ASYNC_DONE)


class ETPShardedParam(torch.nn.Parameter):

    _pending_rs_weight = None
    _first_weight_flag = True
    _last_weight = None

    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):
        requires_grad = kwargs.get('requires_grad', True)
        return super(ETPShardedParam, cls).__new__(cls, tensor, requires_grad=requires_grad)

    def __init__(self, x, *args, **kwargs):
        super().__init__()
        
        self.state = ETPWeightState.NONE
        self._cache_ticket = None
        self._prefetch_handle = None
        self._grad_accum_node = None
        self._grad_accum_hook = None
        # Quantization
        self._quantizer = None
        self.did_cast_to_low_precision = False
        self.quantized = None
        # Prefetching linked list
        self.is_first_weight = False
        self.next_w = None
        self.prev_w = None
        # Grouped gemm
        self.is_routed_expert = False
        self.expert_idx = None
        self.group = None
        self.weight_list = None
        # Reduce-scatter state (set during wgrad_reduce_scatter)
        self.wgrad_rs = None
        self.wgrad_rs_handle = None
        self.fuse_wgrad_accumulation = False
        # Padding
        self.is_padded_last_rank = False
        self.pad_length = 0

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

    def get_padded_shard(self):
        if self.pad_length > 0 and self.is_padded_last_rank:
            return torch.nn.functional.pad(self, (0, 0, 0, self.pad_length))  
        return self

    def _set_state(self, new_state: ETPWeightState):
        """Validate and update state machine transition."""
        assert new_state in _STATE_TRANSITIONS[self.state], \
            f"Invalid state transition: {self.state} -> {new_state}"
        self.state = new_state

    def _get_cache_key(self, dtype, fwd: bool) -> tuple:
        """Build cache key using output shape + dtype.

        Weights with matching gathered shape and dtype share a buffer.
        For expert weights gathered in parallel, self.expert_idx distinguishes them so
        each gets a distinct buffer, while same-indexed experts across layers share.
        """
        
        if not isinstance(dtype, torch.dtype):
            return (self._unsharded_shape_padded, dtype, fwd, not fwd, self.expert_idx)
        return (self._unsharded_shape_padded, dtype, self.expert_idx)

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
        weights = self._weights

        # 1. Transition state for async gathers.
        if async_op:
            for w in weights:
                w._set_state(ETPWeightState.ASYNC_WAIT)

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
            for p, dt in zip(weights, dtypes):
                assert p._cache_ticket is None, \
                    f"Cache ticket leak: weight {id(p)} still has unreturned ticket {p._cache_ticket}"
                buf, p._cache_ticket = get_global_ETP_cache().checkout(p, dt, fwd)
                out_buffers.append(buf)
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

    def _get_unsharded(self, fwd, skip_weight_cast=False, cast_noop_flag=None):
        """Get unsharded (all-gathered) weight tensor(s).

        Handles both routed experts (returns list) and single weights (returns tensor).
        Supports sync gather, async prefetch wait, and cache retrieval.
        """
        weights = self._weights

        # Wait for async prefetch if in progress
        if weights[0].state == ETPWeightState.ASYNC_WAIT:
            self._prefetch_handle.wait()
            self._prefetch_handle = None

        if weights[0].state == ETPWeightState.NONE:
            # Synchronous all-gather (no cache — buffers allocated inline)
            result, _ = self._all_gather_weight(
                async_op=False,
                skip_weight_cast=skip_weight_cast,
                cast_noop_flag=cast_noop_flag,
                fwd=fwd,
            )
            result = result if self.is_routed_expert else [result]

        elif weights[0].state == ETPWeightState.ASYNC_DONE:
            # Retrieve prefetched results from cache
            cache = get_global_ETP_cache()
            result = []
            for w in weights:
                buf = cache.get(w._cache_ticket)
                w._cache_ticket = None
                # Post-gather quantization safety net: weight was prefetched
                # before weight_quantizer was set
                if not w.did_cast_to_low_precision:
                    if w._quantizer is not None and not isinstance(buf, QuantizedTensor):
                        w._quantize_if_needed()
                        buf = w._quantizer.quantize(buf)
                w._set_state(ETPWeightState.NONE)
                result.append(buf)
        else:
            assert False, f"Unexpected state: {weights[0].state}"

        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, weights)]
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
        result = self._get_unsharded(fwd=False, skip_weight_cast=True)

        if ETP_CONFIG.weight_prefetch and self.prev_w is not None:
            _, handle = self.prev_w._all_gather_weight(
                async_op=True, skip_weight_cast=True, cast_noop_flag=None,
                fwd=False, nvtx_label=nvtx_label,
            )
            self.prev_w._prefetch_handle = handle
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
        # Lazy population of linked list: link previous weight to current weight
        cls = type(self)
        if cls._first_weight_flag:
            self.is_first_weight = True
            cls._first_weight_flag = False

        if self.is_first_weight:
            cls._last_weight = None

        if cls._last_weight is not None and cls._last_weight.next_w is None:
            print_rank_0(f"linking curr w: {id(self)} {self.is_routed_expert} prev_w: {id(cls._last_weight)}")
            cls._last_weight.next_w = self
            self.prev_w = cls._last_weight
        cls._last_weight = self

        result = self._get_unsharded(fwd, skip_weight_cast=skip_weight_cast, cast_noop_flag=cast_noop_flag)

        if ETP_CONFIG.weight_prefetch and self.next_w is not None:
            target = self.next_w
            _, handle = target._all_gather_weight(
                async_op=True, skip_weight_cast=skip_weight_cast,
                cast_noop_flag=cast_noop_flag, fwd=fwd, nvtx_label=nvtx_label,
            )
            target._prefetch_handle = handle
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

        return dummy_grad if fuse_wgrad_accumulation else wgrad_rs

    def _reduce_scatter(self, wgrads, async_op):
        """Reduce-scatter one or more wgrads. Returns (outputs, handle).

        Single tensor: plain reduce-scatter (no coalescing).
        Multiple tensors: coalesced reduce-scatter.
        """

        if self.pad_length > 0:
            wgrads = [torch.nn.functional.pad(w, (0, 0, 0, self.pad_length)) for w in wgrads]

        if len(wgrads) == 1:
            out, handle = reduce_scatter_along_first_dim(
                wgrads[0], self.group, async_op=async_op
            )
            return [out], handle
        else:
            outputs = []
            with torch.distributed._coalescing_manager(
                group=self.group,
                device=wgrads[0].device,
                async_ops=async_op,
            ) as cm:
                for tensor in wgrads:
                    out, _ = reduce_scatter_along_first_dim(tensor, self.group)
                    outputs.append(out)
            return outputs, cm if async_op else None

    def wgrad_reduce_scatter(self, wgrad, fuse_wgrad_accumulation):
        """Reduce-scatter wgrad(s). Sync for last weight, async+deferred for others.

        Accepts a single tensor (non-routed) or list of tensors (routed experts).

        Returns:
            Single tensor or list for sync (last weight) — backward should return this.
            None or tuple of Nones for async — backward should return this.
        """
        batched = isinstance(wgrad, (list, tuple))
        wgrads = list(wgrad) if batched else [wgrad]
        weights = self._weights

        # Wait for last reduce scatter if it was async
        if ETPShardedParam._pending_rs_weight is not None:
            param = ETPShardedParam._pending_rs_weight
            assert param is self.next_w
            param.wgrad_rs_handle.wait()
            param.wgrad_rs_handle = None

            for p, g in zip(param._weights, param.wgrad_rs):
                self._finalize_wgrad(p, g, param.fuse_wgrad_accumulation)
            ETPShardedParam._pending_rs_weight = None

        if self.prev_w is None:
            # Sync reduce-scatter (last weight in chain)
            sharded, _ = self._reduce_scatter(wgrads, async_op=False)
            result = [self._finalize_wgrad(p, g, fuse_wgrad_accumulation)
                      for p, g in zip(weights, sharded)]
            return result if batched else result[0]
        else:
            # Async reduce-scatter (not last weight — deferred finish)
            self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            self.wgrad_rs, self.wgrad_rs_handle = self._reduce_scatter(wgrads, async_op=True)
            ETPShardedParam._pending_rs_weight = self
            return tuple([None] * len(wgrads)) if batched else None

    def batched_wgrad_reduce_scatter(self, wgrad_list, fuse_wgrad_accumulation):
        """Batched version of wgrad_reduce_scatter."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.wgrad_reduce_scatter(wgrad_list, fuse_wgrad_accumulation)

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

class ETPWeightCache:
    """
    Buffers are pooled by cache key (shape + dtype).  Two operations:

    - ``checkout(param, dtype, fwd)`` → ``(buffer, ticket)``
      Takes a buffer from the pool (or allocates). Ticket is ``id(buf)``.
    - ``get(ticket, param, dtype, fwd)`` → ``buffer``
      Retrieves the buffer, asserts key matches, returns it to the pool,
      and invalidates the ticket.

    Every checkout is paired with exactly one get (1:1).
    Two weights sharing the same cache key get distinct buffers if one
    is still checked out, preventing aliasing.
    """

    # Bytes per element for known dtypes (used for logging).
    _BYTES_PER_ELEMENT = {
        torch.bfloat16: 2, torch.float16: 2, torch.float32: 4,
        tex.DType.kFloat4E2M1: 0.5,
        tex.DType.kFloat8E4M3: 1,
    }

    def __init__(self):
        self._pool: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self._tickets: Dict[int, tuple] = {}   # ticket → (key, buf)
        self._free_tickets: list[int] = []      # recycled ticket IDs
        self._max_ticket: int = 0               # high-water mark for ticket allocation
        self._total_bytes: int = 0              # running total of allocated bytes

    @staticmethod
    def _buf_bytes(shape, dtype) -> int:
        """Estimate buffer size in bytes."""
        numel = 1
        for d in shape:
            numel *= d
        bpe = ETPWeightCache._BYTES_PER_ELEMENT.get(dtype, None)
        return numel * bpe

    def _allocate_buffer(self, param: 'ETPShardedParam', dtype) -> torch.Tensor:
        out_shape = param._unsharded_shape_padded
        if not isinstance(dtype, torch.dtype):
            quantizer = param._quantizer
            assert quantizer is not None
            assert quantizer.rowwise_usage ^ quantizer.columnwise_usage

            device = torch.cuda.current_device()
            buf = param._quantizer.make_empty(out_shape, dtype=torch.bfloat16, device=device)
        else:
            buf = torch.empty(
                out_shape, dtype=dtype, device=param.device, memory_format=torch.contiguous_format
            )
        buf_bytes = self._buf_bytes(out_shape, dtype)
        self._total_bytes += buf_bytes
        print_rank_0(
            f"[ETP Cache] +{buf_bytes / 1024**2:.1f} MB  (shape={out_shape}, dtype={dtype})  "
            f"total={self._total_bytes / 1024**2:.1f} MB"
        )
        return buf

    def checkout(self, param: 'ETPShardedParam', dtype, fwd: bool):
        """Get a buffer for all-gather output.  Returns (buffer, ticket).

        Ticket IDs are recycled so they stay bounded.
        If all buffers for this key are checked out, allocates a new one.
        """
        key = param._get_cache_key(dtype, fwd)
        pool = self._pool[key]
        buf = pool.pop() if pool else self._allocate_buffer(param, dtype)

        if self._free_tickets:
            ticket = self._free_tickets.pop()
        else:
            ticket = self._max_ticket
            self._max_ticket += 1
        self._tickets[ticket] = (key, buf)
        return buf, ticket

    def get(self, ticket: int) -> torch.Tensor:
        """Retrieve buffer by ticket and return it to the pool.

        This combines the old get + ticket_return into a single call.
        After this call the ticket is invalidated and the buffer is
        available for future checkouts.
        """
        assert ticket in self._tickets, f"Invalid ticket: {ticket}"
        key, buf = self._tickets.pop(ticket)
        self._free_tickets.append(ticket)
        self._pool[key].append(buf)
        return buf


def get_global_ETP_cache() -> ETPWeightCache:
    """Get or lazily create the global cache instance."""
    global _ALL_GATHER_BUFFER
    if _ALL_GATHER_BUFFER is None:
        _ALL_GATHER_BUFFER = ETPWeightCache()
    return _ALL_GATHER_BUFFER


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
