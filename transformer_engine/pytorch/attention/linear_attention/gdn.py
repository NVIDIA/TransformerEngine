# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaNet linear attention.

This module is **experimental** and subject to change.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.jit import no_torch_dynamo

from .base import (
    AlignedTimelineKernelAdapter,
    LinearAttentionBase,
    _needs_eager_linear_attention,
)


def _aligned(tensor):
    """Clone offset views when FE needs a 16-byte-aligned address."""
    tensor = tensor.contiguous()
    return tensor.clone() if tensor.data_ptr() % 16 else tensor


class _GDNKernelAdapter(AlignedTimelineKernelAdapter):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDN.

    GDN gates the recurrence with one scalar per token and head: a log decay
    ``g`` and a write strength ``beta``.
    """

    variant = "GDN"
    module_name = "GatedDeltaNetAttention"
    op_name = "gated_delta_net"

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        g, beta = gates["g"], gates["beta"]
        if g.dtype != torch.float32 or beta.dtype != torch.float32:
            raise TypeError(
                "GDN g and beta must have dtype torch.float32 (the kernel-native dtype)."
            )
        expected_gate_shape = (*query_layer.shape[:-2], self.num_q_heads)
        if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
            raise ValueError(
                "GDN g and beta must both have shape "
                f"{expected_gate_shape}; got {tuple(g.shape)} and {tuple(beta.shape)}."
            )

    def _call_op(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        gates: Dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._op(
            query_layer,
            key_layer,
            value_layer,
            gates["g"],
            gates["beta"],
            cu_seqlens,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def _prepare_cp_inputs(self, tensors, initial_state):
        """Pad partial feature tiles to keep FE's state gradients correct."""
        # Some FE engines mishandle irregular tiles, such as K=32, V=96.
        # Zero extension preserves the recurrence and the original scale.
        q, k, v, g, beta = (_aligned(t) for t in tensors)
        key_dim, value_dim = k.shape[-1], v.shape[-1]
        key_pad = max(64, 1 << (key_dim - 1).bit_length()) - key_dim
        value_pad = max(64, 1 << (value_dim - 1).bit_length()) - value_dim
        if key_pad:
            q, k = F.pad(q, (0, key_pad)), F.pad(k, (0, key_pad))
        if value_pad:
            v = F.pad(v, (0, value_pad))
        if initial_state is not None and (key_pad or value_pad):
            initial_state = F.pad(initial_state, (0, key_pad, 0, value_pad))
        return (q, k, v, g, beta), initial_state

    @staticmethod
    def _summary_op(name):
        """Resolve optional FE summary entry points without requiring them at import."""
        try:
            from cudnn.linear_attention import ops

            return getattr(ops, name)
        except (AttributeError, ImportError) as exc:
            raise ImportError(f"cuDNN frontend does not provide {name}.") from exc

    def _cp_forward_summary(self, query, key, value, gates, cu_seqlens, **kernel_kwargs):
        """Return H,T; use ordinary state passes if no summary engine supports this shape."""
        del query
        k, v, g, beta = (_aligned(t) for t in (key, value, gates["g"], gates["beta"]))
        try:
            return self._summary_op("gated_delta_net_summary")(
                k, v, g, beta, cu_seqlens, **kernel_kwargs
            )
        except ImportError:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if type(exc).__name__ != "cudnnGraphNotSupportedError":
                raise
        # A zero seed gives H; an identity seed with zero values gives T.
        additive = self._call_op(
            k,
            k,
            v,
            {"g": g, "beta": beta},
            cu_seqlens,
            initial_state=None,
            output_final_state=True,
            **kernel_kwargs,
        )[1]
        count, heads, _, key_dim = additive.shape
        identity = torch.eye(key_dim, device=k.device, dtype=torch.float32)
        identity = identity.expand(count, heads, key_dim, key_dim).contiguous()
        transition = self._call_op(
            k,
            k,
            torch.zeros_like(k),
            {"g": g, "beta": beta},
            cu_seqlens,
            initial_state=identity,
            output_final_state=True,
            **kernel_kwargs,
        )[1]
        return additive, transition

    def _cp_backward_summary(
        self, query, key, value, gates, grad_output, cu_seqlens, **kernel_kwargs
    ):
        """Return the reverse state-gradient recipe, including local output losses."""
        del value
        q, k, g, beta, grad_output = (
            _aligned(t) for t in (query, key, gates["g"], gates["beta"], grad_output)
        )
        try:
            return self._summary_op("gated_delta_net_summary_bwd")(
                q,
                k,
                g,
                beta,
                grad_output,
                cu_seqlens,
                scale=self.scale,
                output_transition=True,
                **kernel_kwargs,
            )
        except ImportError:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if type(exc).__name__ != "cudnnGraphNotSupportedError":
                raise
        values = torch.zeros_like(grad_output)
        with torch.enable_grad():
            seed = torch.zeros(
                (cu_seqlens.numel() - 1, q.shape[1], grad_output.shape[-1], k.shape[-1]),
                device=q.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            output = self._call_op(
                q.detach(),
                k.detach(),
                values,
                {"g": g.detach(), "beta": beta.detach()},
                cu_seqlens,
                initial_state=seed,
                output_final_state=False,
                **kernel_kwargs,
            )[0]
            (additive,) = torch.autograd.grad(output, seed, grad_output)
        _, transition = self._cp_forward_summary(
            q, k, values, {"g": g, "beta": beta}, cu_seqlens, **kernel_kwargs
        )
        return additive, transition.transpose(-1, -2).contiguous()

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        *,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cp_group: Optional[dist.ProcessGroup] = None,
        global_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run GDN and return a TE-layout output, optionally with the final state."""
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            from .context_parallel import context_parallel_attention

            return context_parallel_attention(
                self,
                query_layer,
                key_layer,
                value_layer,
                {"g": g, "beta": beta},
                initial_state,
                group=cp_group,
                qkv_format=qkv_format,
                cu_seqlens=cu_seqlens,
                global_cu_seqlens=global_cu_seqlens,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        if global_cu_seqlens is not None:
            if cp_group is None:
                raise ValueError("global_cu_seqlens requires CP.")
            if not isinstance(global_cu_seqlens, torch.Tensor):
                raise TypeError("global_cu_seqlens must be a torch.Tensor.")
            if global_cu_seqlens.dtype != torch.int32 or global_cu_seqlens.dim() != 1:
                raise ValueError("global_cu_seqlens must be a one-dimensional int32 tensor.")
            if qkv_format == "thd":
                if cu_seqlens is not None and cu_seqlens.tolist() != global_cu_seqlens.tolist():
                    raise ValueError("Local cu_seqlens must match the size-one global input.")
                cu_seqlens = global_cu_seqlens.to(query_layer.device)
            elif qkv_format in {"bshd", "sbhd"}:
                batch, length = (
                    query_layer.shape[:2]
                    if qkv_format == "bshd"
                    else (query_layer.shape[1], query_layer.shape[0])
                )
                if global_cu_seqlens.tolist() != [i * length for i in range(batch + 1)]:
                    raise ValueError("Dense global_cu_seqlens does not match the size-one input.")
        return self._run(
            query_layer,
            key_layer,
            value_layer,
            {"g": g, "beta": beta},
            initial_state,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


class GatedDeltaNetAttention(LinearAttentionBase):
    """Apply Gated DeltaNet linear attention through the cuDNN frontend.

    This module is **experimental** and subject to change.

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
               ``t = sum(s_i)`` for all sequences in the batch. Gated DeltaNet is
               inherently causal. Pack valid tokens from padded batches into
               `qkv_format='thd'` and pass their `cu_seqlens` to `forward`.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `GatedDeltaNetAttention` when multiple such
                 modules are concatenated, for instance in consecutive transformer blocks.
    scale : Optional[float], default = `None`
          scale for the Gated DeltaNet recurrence. Defaults to
          ``1.0 / sqrt(kv_channels)`` (or the query/key head size, if
          ``kv_channels`` is a tuple).
    cp_group : ProcessGroup, default = `None`
              Context parallel process group. ``None`` disables CP. Group rank
              order must follow document token order. THD uses contiguous pieces
              of the global packed buffer; BSHD/SBHD use contiguous pieces
              of each sequence. Q/K/V and gates must already be local tensors.
              Summary communication uses all-gather. No output gather is performed.
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
        *,
        cp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            num_attention_heads,
            kv_channels,
            qkv_format=qkv_format,
            tp_size=tp_size,
            tp_group=tp_group,
            layer_number=layer_number,
            scale=scale,
        )

        self.gdn_attention = _GDNKernelAdapter(
            self.scale,
            self.num_attention_heads_per_partition,
            self.qk_head_dim,
            self.v_head_dim,
        )
        self.set_context_parallel_group(cp_group)

    def set_context_parallel_group(self, cp_group: Optional[dist.ProcessGroup]) -> None:
        """Set or disable CP before forward, including after distributed initialization.

        A single process group is supported. This first implementation uses the
        current CUDA stream and contiguous token ordering, without SDPA's dual
        chunk layout or communication-stream configuration.
        """
        if cp_group is not None and not isinstance(cp_group, dist.ProcessGroup):
            raise TypeError("GDN cp_group must be one torch.distributed.ProcessGroup or None.")
        self.cp_group = cp_group
        self.gdn_attention._cp_partition_cache = None

    @no_torch_dynamo(when=_needs_eager_linear_attention)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        *,
        qkv_format: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        global_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Gated DeltaNet linear attention.

        Parameters
        ----------
        query_layer, key_layer, value_layer : torch.Tensor
            Query, key, and value tensors, in the layout given by `qkv_format`
            (or the module's configured `qkv_format` when omitted).
        g : torch.Tensor
            Per-head log-decay gate, of shape matching Q/K/V's token dimensions
            followed by the number of attention heads on this tensor-parallel
            rank (`num_attention_heads // tp_size`). Required.
        beta : torch.Tensor
            Per-head write-strength gate, of the same shape as `g`. Required.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured `qkv_format` for this call.
        cu_seqlens : Optional[torch.Tensor], default = `None`
            Cumulative sequence lengths for packed (`thd`) inputs, of shape
            `[batch_size + 1]` and dtype `torch.int32`.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations for this module are recomputed
            during the backward pass instead of saved.
        initial_state : Optional[torch.Tensor], default = `None`
            Recurrent state to seed the Gated DeltaNet recurrence with, of shape
            `[batch_size, num_attention_heads // tp_size, v_head_dim, qk_head_dim]`
            and dtype `torch.float32`.
        output_final_state : bool, default = `False`
            If true, also return the final recurrent state.
        use_qk_l2norm_in_kernel : bool, default = `False`
            If true, L2-normalize Q and K inside the kernel before the recurrence.
        global_cu_seqlens : Optional[torch.Tensor], default = `None`
            Required for THD CP: global cumulative document lengths, int32 on
            CPU or CUDA, identical on all ranks. Rank ``r`` supplies tokens
            ``[r*T//P:(r+1)*T//P]`` from each packed input, where ``T`` is the
            final global offset and ``P`` is CP size. No CP divisibility constraint
            applies; pack tokens without gaps. Local ``cu_seqlens`` is derived; if
            supplied, it must match the nonempty local document fragments.
            Optional for dense CP: supply ``[0, S, 2*S, ..., B*S]`` for global
            sequence length ``S`` when it is not divisible by CP size. Each
            rank then supplies the sequence slice ``[r*S//P:(r+1)*S//P]`` from
            every batch row. Without this metadata, dense CP assumes equal local
            sequence lengths and infers ``S = local_sequence_length * P``.

        Notes
        -----
        With CP, ``initial_state`` describes all global documents and must have
        identical values on all ranks. Each document's initial-state gradient is
        returned on its first owning rank, with zeros elsewhere; sum these
        contributions over CP to inspect a replicated leaf state's full gradient.
        This ownership also permits chaining final states into the next call.
        Final states are replicated for all global documents. Their backward
        sums gradients from every replica; divide a replicated final-state loss
        by CP size to count it once. All ranks must call forward and backward with matching
        options, even when their local loss or token count is zero. Only
        first-order gradients are supported, matching the underlying FE path.
        CP executes eagerly, including inside a ``torch.compile`` model.
        Irregular feature sizes are internally extended with zeros and cropped
        back, preserving the recurrence and original attention scale. No token
        padding is introduced.
        """
        if g is None or beta is None:
            raise ValueError(
                "GatedDeltaNetAttention requires both g and beta; "
                f"got g={'set' if g is not None else 'None'} and "
                f"beta={'set' if beta is not None else 'None'}."
            )

        gdn_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens": cu_seqlens,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "cp_group": self.cp_group,
            "global_cu_seqlens": global_cu_seqlens,
        }
        with self.prepare_forward_ctx(query_layer, allow_non_contiguous=True) as query_layer:
            return self._dispatch_attention(
                self.gdn_attention,
                (query_layer, key_layer, value_layer, g, beta, initial_state),
                gdn_kwargs,
                checkpoint_core_attention,
            )
