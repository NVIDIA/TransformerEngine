# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared infrastructure for the cuDNN frontend linear-attention variants.

Every variant (Gated DeltaNet, Gated DeltaNet v2, ...) differs only in its gate
tensors and in the cuDNN frontend op it calls. Everything else -- the TE layout
contract, the Q/K/V validation, the THD conversion, the ``cu_seqlens`` handling,
the recurrent-state contract, and the TransformerEngine module lifecycle -- is
shared and lives here.

This module is **experimental** and subject to change.
"""

import importlib
import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.distributed import get_distributed_world_size, checkpoint
from transformer_engine.pytorch.utils import nvtx_range_pop, nvtx_range_push


@lru_cache(maxsize=None)
def _import_linear_attention_op(op_name: str, variant: str) -> Callable:
    """Import a cuDNN frontend linear-attention custom op lazily."""
    try:
        ops = importlib.import_module("cudnn.linear_attention.ops")
        return getattr(ops, op_name)
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            f"{variant} attention requires a nvidia-cudnn-frontend installation that provides "
            f"cudnn.linear_attention.ops.{op_name} and its kernel runtime. Install "
            "cuDNN frontend from the matching source revision with the 'cutedsl' extra."
        ) from exc


def _to_thd(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Convert a dense sequence tensor to the THD layout required by cuDNN."""
    if qkv_format == "thd":
        return tensor
    if qkv_format == "sbhd":
        tensor = tensor.transpose(0, 1)
    return tensor.reshape(-1, *tensor.shape[2:])


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> None:
    """Validate a cumulative sequence-length tensor."""
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"{name} must have shape [batch_size + 1].")
    if cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must contain at least a start and end offset.")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {cu_seqlens.dtype}.")
    if cu_seqlens.device != device:
        raise ValueError(f"{name} must be on {device}, got {cu_seqlens.device}.")


def _needs_eager_linear_attention(call: Dict[str, Any]) -> Optional[str]:
    """Why this linear-attention call has to run outside the graph, or None.

    `call` maps the module's `forward` parameter names to the arguments this
    call passed, including `self`.
    """
    if call.get("checkpoint_core_attention", False):
        return "activation checkpointing of the attention"
    return None


class LinearAttentionKernelAdapter(torch.nn.Module):
    """Adapter from TransformerEngine attention layouts to the cuDNN frontend.

    Subclasses name their variant, validate their gate tensors, and call their
    cuDNN frontend op on THD inputs; this class owns everything the variants
    share.

    The cuDNN frontend linear-attention ops are differentiated through
    ``torch.autograd`` (they register their own backward internally), so this
    module does not define an explicit backward.
    """

    # Short name of the attention variant, used in error messages.
    variant: str = ""
    # Name of the TransformerEngine module that owns the adapter.
    module_name: str = ""
    # Name of the cuDNN frontend custom op backing the variant.
    op_name: str = ""

    def __init__(
        self,
        scale: float,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_q_heads = num_q_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self._dense_cu_seqlens_key: Optional[Tuple[torch.device, int, int]] = None
        self._dense_cu_seqlens: Optional[torch.Tensor] = None

    def _validate_qkv_shapes(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_format: str,
    ) -> None:
        """Check the variant's Q/K/V layout against the module's head geometry.

        The default encodes GDN/GDN-2's layout: aligned Q/K/V token dimensions,
        identical Q/K shapes, equal Q/V head counts, and state heads derived from
        Q heads. That does not describe linear attention generally -- GDP carries
        ``num_householder`` K/V rows per Q row -- so variants whose layout differs
        override this.
        """
        expected_rank = 3 if qkv_format == "thd" else 4
        qkv = (query_layer, key_layer, value_layer)
        if any(tensor.dim() != expected_rank for tensor in qkv):
            raise ValueError(
                f"Q, K, and V must be {expected_rank}D tensors for qkv_format={qkv_format!r}."
            )
        if query_layer.shape != key_layer.shape:
            raise ValueError(
                f"{self.variant} requires Q and K to have the same shape; got "
                f"{tuple(query_layer.shape)} and {tuple(key_layer.shape)}."
            )
        if query_layer.shape[:-2] != value_layer.shape[:-2]:
            raise ValueError(
                f"{self.variant} requires Q, K, and V to have the same token dimensions."
            )
        self._validate_head_geometry(query_layer, value_layer)

    def _validate_head_geometry(
        self,
        query_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> None:
        """Check the Q/V head counts and head sizes against the output contract."""
        if query_layer.shape[-2] != self.num_q_heads:
            raise ValueError(
                f"{self.variant} Q and K must have {self.num_q_heads} heads, "
                f"got {query_layer.shape[-2]}."
            )
        # The underlying ops support grouped value heads, but the module's output
        # contract is fixed at construction time. Integrations that use more V
        # heads must expand Q/K before the module.
        # TODO(KshitijLakhani/cyanguwa): cuDNN's GDN-2 op accepts distinct Q/K/V head
        # counts; plumb that through the module's output contract so the wrapper stops
        # requiring them to be equal.
        if value_layer.shape[-2] != self.num_q_heads:
            raise ValueError(
                f"{self.variant} V must have {self.num_q_heads} heads, "
                f"got {value_layer.shape[-2]}. {self.module_name} requires its output "
                "width to match num_attention_heads * v_head_dim."
            )
        if query_layer.shape[-1] != self.qk_head_dim:
            raise ValueError(
                f"{self.variant} Q and K head dimension must match kv_channels; expected "
                f"{self.qk_head_dim}, got {query_layer.shape[-1]}."
            )
        if value_layer.shape[-1] != self.v_head_dim:
            raise ValueError(
                f"{self.variant} V head dimension must match kv_channels; expected "
                f"{self.v_head_dim}, got {value_layer.shape[-1]}."
            )

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        """Check the variant's gate tensors against the validated Q layout."""
        raise NotImplementedError

    def _call_op(
        self,
        op: Callable,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        gates: Dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        **kernel_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the variant's op on THD tensors, returning (output, final_state)."""
        raise NotImplementedError

    def _resolve_cu_seqlens(
        self,
        query_layer: torch.Tensor,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Return the kernel's cu_seqlens and, for dense inputs, the sequence length.

        Dense batches are packed batches of equal-length sequences, so they get a
        cached ``[0, s, 2s, ...]`` offset tensor rather than a fresh allocation
        per call.
        """
        device = query_layer.device
        if qkv_format == "thd":
            if cu_seqlens is None:
                raise ValueError(
                    f"cu_seqlens is required for {self.variant} with qkv_format='thd'."
                )
            _validate_cu_seqlens(cu_seqlens, device=device, name="cu_seqlens")
            return cu_seqlens, None

        if cu_seqlens is not None:
            raise ValueError(
                f"Dense {self.variant} inputs do not accept cu_seqlens. "
                "Use qkv_format='thd' for packed or ragged batches."
            )
        if qkv_format == "bshd":
            batch_size, sequence_length = query_layer.shape[:2]
        else:
            sequence_length, batch_size = query_layer.shape[:2]
        cache_key = (device, batch_size, sequence_length)
        if self._dense_cu_seqlens_key != cache_key:
            self._dense_cu_seqlens = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=device) * sequence_length
            )
            self._dense_cu_seqlens_key = cache_key
        return self._dense_cu_seqlens, sequence_length

    def _validate_initial_state(
        self,
        initial_state: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Check a recurrent state against this module's output contract."""
        if initial_state is None:
            return
        expected_state_shape = (
            batch_size,
            self.num_q_heads,
            self.v_head_dim,
            self.qk_head_dim,
        )
        if initial_state.device != device:
            raise ValueError(
                f"{self.variant} initial_state must be on {device}, got {initial_state.device}."
            )
        if initial_state.dtype != torch.float32:
            raise TypeError(
                f"{self.variant} initial_state must have dtype torch.float32, "
                f"got {initial_state.dtype}."
            )
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"{self.variant} initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}."
            )

    def _run(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        gates: Dict[str, torch.Tensor],
        initial_state: Optional[torch.Tensor] = None,
        *,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        **kernel_kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Validate the inputs, run the variant's op, and restore the TE layout."""
        if qkv_format not in {"thd", "bshd", "sbhd"}:
            raise ValueError(
                f"{self.variant} attention only supports qkv_format={{'thd', 'bshd', 'sbhd'}}, "
                f"got {qkv_format!r}."
            )

        qkv = (query_layer, key_layer, value_layer)
        gate_names = ", ".join(gates)
        if not all(isinstance(tensor, torch.Tensor) for tensor in (*qkv, *gates.values())):
            raise TypeError(
                f"{self.variant} Q, K, V, and the gates ({gate_names}) must be "
                "torch.Tensor instances."
            )
        if initial_state is not None and not isinstance(initial_state, torch.Tensor):
            raise TypeError(f"{self.variant} initial_state must be a torch.Tensor when provided.")

        self._validate_qkv_shapes(query_layer, key_layer, value_layer, qkv_format)

        device = query_layer.device
        if any(not tensor.is_cuda for tensor in qkv):
            raise ValueError(f"{self.variant} attention only supports CUDA tensors.")
        if any(tensor.device != device for tensor in (*qkv, *gates.values())):
            raise ValueError(
                f"Q, K, V, and the gates ({gate_names}) must be on the same CUDA device."
            )
        if query_layer.dtype != key_layer.dtype or query_layer.dtype != value_layer.dtype:
            raise TypeError(f"Q, K, and V must have the same dtype for {self.variant} attention.")
        if query_layer.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(
                f"{self.variant} Q, K, and V must have dtype float16 or bfloat16 (no cuDNN "
                f"frontend {self.variant} engine supports float32 inputs), "
                f"got {query_layer.dtype}."
            )
        self._validate_gates(gates, query_layer)

        cu_seqlens, sequence_length = self._resolve_cu_seqlens(query_layer, qkv_format, cu_seqlens)
        batch_size = cu_seqlens.shape[0] - 1
        self._validate_initial_state(initial_state, batch_size=batch_size, device=device)

        op = _import_linear_attention_op(self.op_name, self.variant)
        try:
            output, final_state = self._call_op(
                op,
                _to_thd(query_layer, qkv_format),
                _to_thd(key_layer, qkv_format),
                _to_thd(value_layer, qkv_format),
                {name: _to_thd(tensor, qkv_format) for name, tensor in gates.items()},
                cu_seqlens,
                initial_state=initial_state,
                output_final_state=output_final_state,
                **kernel_kwargs,
            )
        except ImportError as exc:
            raise ImportError(
                f"The cuDNN frontend {self.variant} kernel runtime is unavailable. Install cuDNN "
                "frontend from the matching source revision with the 'cutedsl' extra."
            ) from exc

        if qkv_format == "thd":
            output = output.reshape(output.shape[0], -1)
        else:
            output = output.reshape(batch_size, sequence_length, -1)
            if qkv_format == "sbhd":
                output = output.transpose(0, 1).contiguous()

        if output_final_state:
            return output, final_state
        return output


class LinearAttentionBase(torch.nn.Module):
    """Common construction and dispatch for the linear-attention modules.

    Linear attention holds no parameters and runs no TransformerEngine GEMMs -- the
    cuDNN frontend op owns its own matmuls -- so this deliberately does not derive
    from ``TransformerEngineBaseModule``: that class's quantization state (FP8 meta,
    quantizers, the FP8 extra state in the state dict) would be built on every
    forward only to go unused. These modules therefore ignore FP8 autocast and
    always run at the input precision. What they do need from the TE module
    contract is reimplemented here: the tensor-parallel group handshake and the
    forward lifecycle.

    Because this is a second module type participating in TE-wide lifecycle
    behavior, the minimal contract it implements is spelled out here, so
    framework-level code that today keys off ``TransformerEngineBaseModule``
    knows what to expect from these modules:

    * ``set_tensor_parallel_group(tp_group)`` -- accepts the TP group after
      construction and marks it initialized; ``prepare_forward`` raises if
      ``tp_size > 1`` and it was never called.
    * ``fast_setattr(name, value)`` -- the FSDP hook. ``prepare_te_modules_for_fsdp``
      injects ``fsdp_group`` through it on every module ``get_te_classes()`` matches,
      and these modules are in that set.
    * ``prepare_forward`` / ``end_forward`` (and the ``prepare_forward_ctx``
      context manager) -- input validation plus the module's NVTX range.
    * CUDA-graph discovery -- these modules are recognized by the graph capture
      machinery, and ``_needs_eager_linear_attention`` marks the calls that must
      run outside the graph.

    Deliberately *not* implemented, since there are no parameters or TE GEMMs:
    FP8/quantization state, ``fp8_init``, and the FP8 extra state in the state
    dict. Anything added to the TE module contract that a parameter-free module
    should honor needs mirroring here.

    Subclasses build a :class:`LinearAttentionKernelAdapter` from the head
    geometry resolved here and implement `forward`.

    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                head size for query/key and value tensors. If ``int``, the same size
                is used for both; if ``Tuple[int, int]``, the first element is the
                query/key head size and the second is the value head size.
    qkv_format : str, default = `sbhd`
               dimension format for query_layer, key_layer and value_layer,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length,
               `b` batch size, `h` the number of heads, `d` head size, and
               `t` the total number of tokens in a batch, with
               ``t = sum(s_i)`` for all sequences in the batch. Linear attention
               is inherently causal; padded batches must use `qkv_format='thd'`
               with `cu_seqlens` passed to `forward` to exclude padding tokens
               from the recurrence.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current module when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    scale : Optional[float], default = `None`
          scale for the recurrence. Defaults to ``1.0 / sqrt(kv_channels)``
          (or the query/key head size, if ``kv_channels`` is a tuple).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        qkv_format: str = "sbhd",
        tp_size: int = 1,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("TransformerEngine needs CUDA.")

        self.qkv_format = qkv_format

        self.tp_group = None
        self.tp_group_initialized = False
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}.")
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"tp_size ({self.tp_size})."
            )

        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads // self.tp_size
        self.layer_number = 1 if layer_number is None else layer_number

        self.qk_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        self.v_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[1]

        self.scale = 1.0 / math.sqrt(self.qk_head_dim) if scale is None else scale

    def fast_setattr(self, name: str, value: Any) -> None:
        """Set a regular attribute, bypassing ``nn.Module.__setattr__``.

        Part of the TE module contract: ``prepare_te_modules_for_fsdp`` injects
        ``fsdp_group`` through this on every module ``get_te_classes()`` matches,
        and these modules are in that set.

        Should be used for regular attributes, but not properties nor
        parameters/buffers.
        """
        self.__dict__[name] = value

    def set_tensor_parallel_group(self, tp_group: Optional[dist_group_type]) -> None:
        """Set the tensor parallel group for this module before the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group
        self.tp_group_initialized = True

    def prepare_forward(
        self,
        inp: torch.Tensor,
        allow_non_contiguous: bool = False,
    ) -> torch.Tensor:
        """Check the inputs and open this module's NVTX range."""
        if not inp.is_cuda:
            raise RuntimeError(f"TransformerEngine needs CUDA. Got input on device: {inp.device}")
        if self.tp_size > 1 and not self.tp_group_initialized:
            raise RuntimeError(
                "Tensor parallel group not initialized. Call set_tensor_parallel_group() "
                "before forward pass when tp_size > 1."
            )

        nvtx_range_push(self.__class__.__name__ + " forward")
        if not allow_non_contiguous and not inp.is_contiguous():
            inp = inp.contiguous()
        return inp

    def end_forward(self) -> None:
        """Close the NVTX range opened by `prepare_forward`."""
        nvtx_range_pop()

    @contextmanager
    def prepare_forward_ctx(
        self,
        inp: torch.Tensor,
        allow_non_contiguous: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """Check and prepare for forward execution, closing the NVTX range on exit."""
        inp = self.prepare_forward(inp, allow_non_contiguous=allow_non_contiguous)
        try:
            yield inp
        finally:
            self.end_forward()

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        return checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=None,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

    def _dispatch_attention(
        self,
        attention_func: Callable,
        forward_args: Tuple[Any, ...],
        forward_kwargs: Dict[str, Any],
        checkpoint_core_attention: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run the kernel adapter, optionally with activation checkpointing."""
        if checkpoint_core_attention:
            return self._checkpointed_attention_forward(
                attention_func, *forward_args, **forward_kwargs
            )
        return attention_func(*forward_args, **forward_kwargs)
