# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoEBlock for TransformerEngine JAX.

This module exposes :class:`MoEBlock`, a self-contained Flax Linen MoE layer
that wires together TE's fused router, a selectable token-dispatch backend
(pure-JAX ``unfused_*`` or fused Triton), TE's ``grouped_dense``, and an
optional ragged-all-to-all (A2A / A2Av) expert-parallelism strategy.

Architecture
------------

The MoEBlock is decomposed into orthogonal stages so the EP wrapper can
inject collectives between them:

* ``_route``:           gate logits -> top-k routing decisions (+ aux loss).
* ``_global_permute``:  scatter tokens to experts; produces
                        ``[num_tokens*topk + maybe_padding, hidden]`` and
                        per-expert ``group_sizes`` of length ``num_experts``.
* ``_expert_ffn``:      three ``grouped_dense`` calls + activation. Operates
                        on whatever ``(rows, group_sizes, n_groups)`` it is
                        handed -- agnostic to whether ``n_groups`` is the
                        global expert count (no-EP) or the local expert
                        count (A2A-EP).
* ``_global_combine``:  inverse of ``_global_permute`` -- gather + weighted
                        sum across top-k experts.

Two top-level forward variants compose those stages:

* ``_forward_no_ep``:   route -> permute -> ffn -> combine. Each TE
                        primitive's ``custom_partitioning`` rule handles
                        DP / FSDP / TP automatically.
* ``_forward_a2a_ep``:  wraps the body in :func:`jax.shard_map` and inserts
                        ``all_gather(group_sizes)`` + forward
                        ``ragged_all_to_all`` + local permute around the
                        FFN, plus their inverses afterwards. This is the
                        only place ``shard_map`` is used; A2A is the
                        canonical EP strategy because the in-flight NCCL
                        EP component will require this same data layout.

Note on ``align_size > 0``
--------------------------

Both permutation backends pad each expert's group to a multiple of
``align_size`` when requested, which is what CUBLASLt's grouped GEMM wants
for FP8 shape selection. The pure-JAX backend additionally appends a
zero-input padding tail to keep the buffer statically sized for JIT, so
``sum(group_sizes) <= sorted_inputs.shape[0]`` strictly. TE's
``grouped_dense`` FFI today asserts ``m == sum(group_sizes)`` at
``transformer_engine/jax/csrc/extensions/gemm.cpp:1029``; relaxing that
check to ``m >= sum(group_sizes)`` (the kernel itself only iterates over
``sum(group_sizes)`` rows via ``nvte_multi_tensor_gemm``) is the cleanest
way to support ``align_size > 0`` end-to-end. Until that lands the
``align_size > 0`` tests stay xfail.
"""

from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from ..dense import grouped_dense
from ..permutation import (
    _routing_map_to_selected_experts,
    compute_ragged_all_to_all_params,
    compute_reverse_ragged_all_to_all_params,
    local_permute_after_a2a,
    local_unpermute_before_a2a,
    token_combine,
    token_dispatch,
    unfused_token_combine,
    unfused_token_dispatch,
)
from ..quantize import noop_quantizer_set
from ..router import ScoreFunction, fused_moe_aux_loss, fused_topk_with_score_function
from ..sharding import with_sharding_constraint_by_logical_axes
from .module import TransformerEngineBase, _convert_to_activation_function

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
Initializer = Callable[[PRNGKey, Shape, DType], Array]


__all__ = ["MoEBlock"]


# =============================================================================
# MoEBlock
# =============================================================================


class MoEBlock(TransformerEngineBase):
    """Mixture-of-Experts Flax Linen block.

    Encapsulates the full MoE forward pass: gate projection, fused top-k
    routing, optional auxiliary load-balancing loss, token dispatch,
    per-expert two-layer FFN via grouped GEMMs, activation, token combine,
    and optional ragged-all-to-all expert parallelism.

    Two permutation backends are pluggable via ``permutation_backend``:

    * ``"pure_jax"`` (default) -- argsort-based
      :func:`~transformer_engine.jax.permutation.unfused_token_dispatch` /
      :func:`~transformer_engine.jax.permutation.unfused_token_combine`.
      Faster than Triton in profiling for DeepSeek-style configs.
    * ``"triton"`` -- TE's fused
      :func:`~transformer_engine.jax.permutation.token_dispatch` /
      :func:`~transformer_engine.jax.permutation.token_combine` Triton
      kernels.

    Expert parallelism (``expert_parallelism_axis is not None``) uses the
    **ragged-all-to-all** EP strategy (a.k.a. A2Av): each shard routes its
    own tokens globally over all experts, then a forward
    ``ragged_all_to_all`` exchanges per-expert chunks so each shard ends up
    holding only the tokens for its local experts; after the FFN a reverse
    ``ragged_all_to_all`` returns each shard's outputs to it. This matches
    the layout the in-flight NCCL EP component expects.

    Parameters
    ----------
    num_experts : int
        Total number of experts.
    num_experts_per_tok : int
        Top-k value (number of experts each token is routed to).
    intermediate_size : int
        Per-expert FFN hidden dim.

    activation_type : str
        FFN activation applied to the gate projection. Paired with the up
        projection in the SwiGLU-style ``act(wi_0) * wi_1`` product.
        Resolved via :func:`flax.linen.<name>` (``"silu"``, ``"gelu"``,
        ``"relu"``, ``"swish"``, ...) plus ``"linear"`` for identity.

    score_function : str or ScoreFunction
        ``"softmax"`` (default) or ``"sigmoid"`` for
        :func:`fused_topk_with_score_function`.
    use_pre_softmax : bool
        Apply softmax before top-k when ``score_function="softmax"``.
    num_groups : int
        Number of routing groups for grouped top-k (DeepSeek). ``<=0``
        disables.
    group_topk : int
        Top-k at the group level. ``<=0`` disables.
    scaling_factor : float
        Scaling factor applied to output probs.
    use_expert_bias : bool
        If ``True``, registers a learnable ``expert_bias`` parameter of
        shape ``[num_experts]`` and passes it to the fused router. The
        router primitive validates that this is paired with
        ``score_function="sigmoid"``.
    aux_loss_coeff : float
        If ``> 0``, compute and return the MoE auxiliary load-balancing
        loss scalar via :func:`fused_moe_aux_loss`. ``0`` disables.

    gate_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the gate kernel of shape
        ``[hidden, num_experts]``.
    wi_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the ``wi_0`` and ``wi_1`` kernels of
        shape ``[num_experts, hidden, intermediate]``. Default
        ``("exp", "embed", "mlp")``.
    wo_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the ``wo`` kernel of shape
        ``[num_experts, intermediate, hidden]``. Default
        ``("exp", "mlp", "embed")``.
    input_axes : tuple[str, ...]
        Logical axes used to constrain the input activation sharding at the
        block boundary. ``()`` (default) means no constraint.

    expert_parallelism_axis : Optional[str]
        Mesh axis along which experts are split. When set, the forward
        pass is wrapped in :func:`jax.shard_map` that implements the
        ragged-all-to-all EP strategy. When ``None`` (default), no
        ``shard_map`` wrapper is used; each TE primitive's
        ``custom_partitioning`` rule handles DP / FSDP / TP automatically.
    tensor_parallelism_axis : Optional[str]
        Mesh axis for tensor parallelism on the FFN intermediate dim. When
        set, the output of the ``wo`` grouped GEMM is ``psum_scatter`` ed
        along this axis.

    permutation_backend : str
        ``"pure_jax"`` (default) or ``"triton"``.
    align_size : int
        Alignment for per-expert group sizes after padding. ``0`` disables
        padding (the only supported configuration end-to-end today). ``>0``
        is required for quantized TE grouped GEMM whose recipe-specific
        alignment must divide ``align_size``; see the module docstring for
        the FFI assertion that currently blocks ``>0`` for both backends.

    dtype : jnp.dtype
        Compute and parameter dtype.
    kernel_init : Initializer
        Initializer for all kernels (gate + per-expert FFN). Defaults to
        ``variance_scaling(1.0, 'fan_in', 'truncated_normal')`` (Flax
        convention).
    use_bias : bool
        If ``True``, registers per-expert FFN biases ``wi_0_bias``,
        ``wi_1_bias``, ``wo_bias``.
    """

    # Architecture
    num_experts: int = 8
    num_experts_per_tok: int = 2
    intermediate_size: int = 2048
    activation_type: str = "silu"

    # Routing
    score_function: Union[str, ScoreFunction] = "softmax"
    use_pre_softmax: bool = False
    num_groups: int = -1
    group_topk: int = -1
    scaling_factor: float = 1.0
    use_expert_bias: bool = False
    aux_loss_coeff: float = 0.0

    # Sharding
    gate_kernel_axes: Tuple[Optional[str], ...] = ()
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp")
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed")
    input_axes: Tuple[Optional[str], ...] = ()

    # Parallelism
    expert_parallelism_axis: Optional[str] = None
    tensor_parallelism_axis: Optional[str] = None
    # ``jax.sharding.Mesh`` to use when ``expert_parallelism_axis`` is set.
    # Required for the ``shard_map`` wrapper; ignored otherwise.
    mesh: Optional[Any] = None

    # Permutation
    permutation_backend: str = "pure_jax"
    align_size: int = 0

    # Dtypes / init / misc
    dtype: DType = jnp.float32
    kernel_init: Optional[Initializer] = None
    bias_init: Initializer = nn.initializers.zeros
    expert_bias_init: Initializer = nn.initializers.zeros
    use_bias: bool = False

    def __post_init__(self):
        if self.kernel_init is None:
            object.__setattr__(
                self,
                "kernel_init",
                nn.initializers.variance_scaling(
                    1.0, "fan_in", "truncated_normal", dtype=self.dtype
                ),
            )
        if self.permutation_backend not in ("pure_jax", "triton"):
            raise ValueError(
                "permutation_backend must be 'pure_jax' or 'triton',"
                f" got {self.permutation_backend!r}"
            )
        super().__post_init__()

    # ------------------------------------------------------------------
    # Parameter registration
    # ------------------------------------------------------------------

    def _make_params(self, hidden_size: int) -> dict:
        """Register module parameters and return them as a dict."""
        gate_kernel = self.param(
            "gate_kernel",
            nn.with_logical_partitioning(self.kernel_init, self.gate_kernel_axes),
            (hidden_size, self.num_experts),
            self.dtype,
        )
        wi_0 = self.param(
            "wi_0",
            nn.with_logical_partitioning(self.kernel_init, self.wi_kernel_axes),
            (self.num_experts, hidden_size, self.intermediate_size),
            self.dtype,
        )
        wi_1 = self.param(
            "wi_1",
            nn.with_logical_partitioning(self.kernel_init, self.wi_kernel_axes),
            (self.num_experts, hidden_size, self.intermediate_size),
            self.dtype,
        )
        wo = self.param(
            "wo",
            nn.with_logical_partitioning(self.kernel_init, self.wo_kernel_axes),
            (self.num_experts, self.intermediate_size, hidden_size),
            self.dtype,
        )
        params: dict = {
            "gate_kernel": gate_kernel,
            "wi_0": wi_0,
            "wi_1": wi_1,
            "wo": wo,
        }
        if self.use_bias:
            params["wi_0_bias"] = self.param(
                "wi_0_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "mlp")),
                (self.num_experts, self.intermediate_size),
                self.dtype,
            )
            params["wi_1_bias"] = self.param(
                "wi_1_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "mlp")),
                (self.num_experts, self.intermediate_size),
                self.dtype,
            )
            params["wo_bias"] = self.param(
                "wo_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "embed")),
                (self.num_experts, hidden_size),
                self.dtype,
            )
        if self.use_expert_bias:
            params["expert_bias"] = self.param(
                "expert_bias",
                nn.with_logical_partitioning(self.expert_bias_init, ("exp",)),
                (self.num_experts,),
                self.dtype,
            )
        return params

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @nn.compact
    def __call__(self, inputs: Array) -> Tuple[Array, Optional[Array]]:
        """Run the MoE forward pass.

        Parameters
        ----------
        inputs : jnp.ndarray
            Input tensor of shape ``[batch, sequence, hidden]``.

        Returns
        -------
        output : jnp.ndarray
            Output tensor of shape ``[batch, sequence, hidden]``.
        aux_loss : Optional[jnp.ndarray]
            Scalar auxiliary load-balancing loss when
            ``aux_loss_coeff > 0``, else ``None``.
        """
        assert inputs.ndim == 3, (
            f"MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        )
        inputs = with_sharding_constraint_by_logical_axes(inputs, self.input_axes)

        _, _, hidden_size = inputs.shape
        params = self._make_params(hidden_size)

        # The gate runs OUTSIDE any EP shard_map: under EP each shard
        # projects only its local slice of tokens, producing local gate
        # logits with the same per-shard layout as ``inputs``.
        gate_logits = self._gate(inputs, params["gate_kernel"])

        if self.expert_parallelism_axis is None:
            output, aux_loss = self._forward_no_ep(inputs, gate_logits, params)
        else:
            output, aux_loss = self._forward_a2a_ep(inputs, gate_logits, params)

        if self.aux_loss_coeff <= 0.0:
            aux_loss = None
        return output, aux_loss

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _gate(self, inputs: jnp.ndarray, gate_kernel: jnp.ndarray) -> jnp.ndarray:
        """Linear gate projection ``inputs @ gate_kernel``.

        Kept as a plain ``einsum`` (not ``DenseGeneral``) so it composes
        cleanly with the EP shard_map: the gate runs in the outer
        (pre-shard_map) scope and its output passes through the
        ``shard_map`` boundary unchanged.
        """
        kernel = gate_kernel.astype(inputs.dtype)
        return jnp.einsum("bsh,he->bse", inputs, kernel)

    # ------------------------------------------------------------------
    # Route
    # ------------------------------------------------------------------
    #
    # The router is split into two pieces so the EP path can compute
    # aux_loss over global (cross-shard) statistics without re-running
    # the main top-k path. ``_route_topk`` returns the per-token routing
    # decisions (used by ``_global_permute``) and ``_compute_aux_loss``
    # returns the scalar load-balancing loss given the (possibly
    # gathered) logits.

    def _route_topk(
        self,
        logits_2d: jnp.ndarray,
        expert_bias: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run the fused router top-k selection."""
        sparse_probs, routing_map = fused_topk_with_score_function(
            logits_2d,
            topk=self.num_experts_per_tok,
            use_pre_softmax=self.use_pre_softmax,
            num_groups=self.num_groups,
            group_topk=self.group_topk,
            scaling_factor=self.scaling_factor,
            score_function=self.score_function,
            expert_bias=expert_bias,
        )
        sparse_probs = sparse_probs.astype(self.dtype)
        return sparse_probs, routing_map

    def _compute_aux_loss(
        self,
        logits_2d: jnp.ndarray,
    ) -> Optional[jnp.ndarray]:
        """Compute the MoE auxiliary load-balancing loss.

        The score-for-aux kernel has no data dependency on the main
        routing kernel, so XLA can overlap them on the GPU.

        ``logits_2d`` should be the *full* logits tensor over the global
        token batch -- under EP the caller is responsible for
        :func:`jax.lax.all_gather` ing the logits before calling this so
        the aux_loss formula
        ``loss = (E * coeff / (k * T^2)) * sum_i(sum_t(probs[t,i]) * tokens[i])``
        sees the global ``T`` and the global ``tokens_per_expert``.
        """
        if self.aux_loss_coeff <= 0.0:
            return None
        aux_scores, aux_routing_map = fused_topk_with_score_function(
            logits_2d.astype(jnp.float32),
            topk=self.num_experts_per_tok,
            score_function=self.score_function,
            compute_aux_scores=True,
        )
        aux_tokens_per_expert = jnp.sum(
            aux_routing_map.astype(jnp.int32), axis=0
        )
        return fused_moe_aux_loss(
            aux_scores.astype(jnp.float32),
            aux_tokens_per_expert,
            topk=self.num_experts_per_tok,
            coeff=self.aux_loss_coeff,
        )

    # ------------------------------------------------------------------
    # Global permute (route -> token dispatch)
    # ------------------------------------------------------------------

    def _global_permute(
        self,
        inputs_2d: jnp.ndarray,
        sparse_probs: jnp.ndarray,
        routing_map: jnp.ndarray,
    ) -> dict:
        """Dispatch tokens to the global expert axis.

        Returns a permutation-result dict suitable both for the no-EP
        forward (where the same buffer feeds ``_expert_ffn`` directly) and
        for the A2A-EP path (where the buffer is sliced + sent over the EP
        axis before the FFN). The dict carries the per-backend opaque
        state needed to invert the dispatch in :meth:`_global_combine`.

        The output dict layout is::

            {
                "backend":         "pure_jax" | "triton",
                "sorted_inputs":   [buffer_size, hidden],
                "group_sizes":     [num_experts],     # per-expert,
                                                       # length == E always.
                "perm_state":      UnfusedPermState | None,   # pure_jax
                "row_id_map":      jnp.ndarray | None,        # triton
                "pad_offsets":     jnp.ndarray | None,        # triton
                "routing_weights": jnp.ndarray | None,        # pure_jax
                "merging_probs":   jnp.ndarray | None,        # triton
            }
        """
        num_tokens = inputs_2d.shape[0]
        topk = self.num_experts_per_tok

        if self.permutation_backend == "pure_jax":
            selected_experts, routing_weights = _routing_map_to_selected_experts(
                sparse_probs, routing_map, topk
            )
            sorted_inputs, perm_state, group_sizes = unfused_token_dispatch(
                inputs_2d,
                selected_experts,
                num_experts=self.num_experts,
                num_experts_per_tok=topk,
                align_size=self.align_size,
            )
            return {
                "backend": "pure_jax",
                "sorted_inputs": sorted_inputs,
                "group_sizes": group_sizes,
                "perm_state": perm_state,
                "routing_weights": routing_weights,
            }

        # triton
        num_out_tokens = num_tokens * topk
        align_size_arg = self.align_size if self.align_size > 0 else None
        (
            sorted_inputs,
            _permuted_probs,
            row_id_map,
            pad_offsets,
            group_sizes,
        ) = token_dispatch(
            inputs_2d,
            routing_map,
            num_out_tokens=num_out_tokens,
            probs=sparse_probs,
            align_size=align_size_arg,
        )
        return {
            "backend": "triton",
            "sorted_inputs": sorted_inputs,
            "group_sizes": group_sizes,
            "row_id_map": row_id_map,
            "pad_offsets": pad_offsets,
            "merging_probs": sparse_probs,
        }

    # ------------------------------------------------------------------
    # Expert FFN (three grouped_dense calls + activation)
    # ------------------------------------------------------------------

    def _expert_ffn(
        self,
        sorted_inputs: jnp.ndarray,
        group_sizes: jnp.ndarray,
        params: dict,
        n_groups: int,
    ) -> jnp.ndarray:
        """Run the per-expert SwiGLU-style FFN over a permuted buffer.

        Parameters
        ----------
        sorted_inputs : jnp.ndarray
            Permuted tokens of shape ``[buffer_size, hidden]`` (rows
            grouped by expert).
        group_sizes : jnp.ndarray
            Per-group token counts of shape ``[n_groups]``.
            ``sum(group_sizes)`` must equal ``buffer_size`` (TE
            ``grouped_dense`` FFI assertion at
            ``transformer_engine/jax/csrc/extensions/gemm.cpp:1029``).
        params : dict
            Block parameters from :meth:`_make_params`. Reads ``wi_0``,
            ``wi_1``, ``wo``, and the optional bias entries.
        n_groups : int
            Number of expert groups. Equals ``self.num_experts`` for the
            no-EP path and ``num_experts // num_ep`` for the A2A-EP path.
            Used to size the per-call quantizer set so the FP8 metadata
            tensors match ``group_sizes``.

        Returns
        -------
        expert_outputs : jnp.ndarray
            ``[buffer_size, hidden]``.
        """
        wi_0 = params["wi_0"]
        wi_1 = params["wi_1"]
        wo = params["wo"]

        # Each grouped_dense call gets its own quantizer_set with
        # n_groups matching ``group_sizes``; this keeps the FP8 meta
        # tensors correctly sized in both no-EP and A2A-EP cases.
        q_set_w0 = self.generate_quantizer_set(postfix="_w0", n_groups=n_groups)
        q_set_w1 = self.generate_quantizer_set(postfix="_w1", n_groups=n_groups)
        q_set_wo = self.generate_quantizer_set(postfix="_wo", n_groups=n_groups)

        # Cast kernels to the activation dtype when no FP8 quantization
        # is active (mirrors DenseGeneral).
        if q_set_w0 == noop_quantizer_set:
            wi_0 = wi_0.astype(sorted_inputs.dtype)
        if q_set_w1 == noop_quantizer_set:
            wi_1 = wi_1.astype(sorted_inputs.dtype)
        if q_set_wo == noop_quantizer_set:
            wo = wo.astype(sorted_inputs.dtype)

        # ``grouped_dense`` accepts per-expert bias of shape (G, N); it
        # adds ``bias[i]`` to the ``group_sizes[i]`` rows belonging to
        # expert ``i`` in the permuted layout.
        wi_0_bias = params.get("wi_0_bias") if self.use_bias else None
        wi_1_bias = params.get("wi_1_bias") if self.use_bias else None
        wo_bias = params.get("wo_bias") if self.use_bias else None

        layer_w0 = grouped_dense(
            sorted_inputs,
            wi_0,
            group_sizes,
            contracting_dims=((1,), (1,)),
            bias=wi_0_bias,
            quantizer_set=q_set_w0,
        )
        layer_w1 = grouped_dense(
            sorted_inputs,
            wi_1,
            group_sizes,
            contracting_dims=((1,), (1,)),
            bias=wi_1_bias,
            quantizer_set=q_set_w1,
        )

        act_fn = _convert_to_activation_function(self.activation_type)
        intermediate = act_fn(layer_w0) * layer_w1

        expert_outputs = grouped_dense(
            intermediate,
            wo,
            group_sizes,
            contracting_dims=((1,), (1,)),
            bias=wo_bias,
            quantizer_set=q_set_wo,
        )
        return expert_outputs

    # ------------------------------------------------------------------
    # Global combine (token combine -> back to [B, S, H])
    # ------------------------------------------------------------------

    def _global_combine(
        self,
        expert_outputs: jnp.ndarray,
        perm_result: dict,
        batch_size: int,
        sequence_length: int,
    ) -> jnp.ndarray:
        """Inverse of :meth:`_global_permute`.

        Gathers per-expert outputs back into ``[batch, sequence, hidden]``
        and applies the per-token weighted sum across the top-k experts.
        """
        backend = perm_result["backend"]
        if backend == "pure_jax":
            return unfused_token_combine(
                expert_outputs,
                perm_result["perm_state"],
                perm_result["routing_weights"],
                num_experts_per_tok=self.num_experts_per_tok,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        # triton
        out_2d = token_combine(
            expert_outputs,
            perm_result["row_id_map"],
            merging_probs=perm_result["merging_probs"],
            pad_offsets=perm_result["pad_offsets"],
        )
        hidden_size = out_2d.shape[-1]
        return out_2d.reshape(batch_size, sequence_length, hidden_size).astype(
            self.dtype
        )

    # ------------------------------------------------------------------
    # No-EP forward
    # ------------------------------------------------------------------

    def _forward_no_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Single-shard or DP/FSDP/TP forward (no shard_map wrapper).

        DP / FSDP / TP all flow through each TE primitive's
        ``custom_partitioning`` rule -- there is no cross-primitive
        collective that the rules cannot express on their own, so a
        ``shard_map`` is unnecessary here.
        """
        batch_size, sequence_length, hidden_size = inputs.shape
        inputs_2d = inputs.reshape(-1, hidden_size)
        logits_2d = gate_logits.reshape(-1, self.num_experts)

        sparse_probs, routing_map = self._route_topk(
            logits_2d, params.get("expert_bias")
        )
        aux_loss = self._compute_aux_loss(logits_2d)
        perm = self._global_permute(inputs_2d, sparse_probs, routing_map)
        expert_outputs = self._expert_ffn(
            perm["sorted_inputs"],
            perm["group_sizes"],
            params,
            n_groups=self.num_experts,
        )
        output = self._global_combine(
            expert_outputs, perm, batch_size, sequence_length
        )

        if self.tensor_parallelism_axis is not None:
            output = jax.lax.psum_scatter(
                output,
                self.tensor_parallelism_axis,
                scatter_dimension=2,
                tiled=True,
            )
        return output, aux_loss

    # ------------------------------------------------------------------
    # A2A (ragged-all-to-all) EP forward
    # ------------------------------------------------------------------

    def _forward_a2a_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Wrap the body in a ``shard_map`` that runs a forward
        ``ragged_all_to_all`` (A2A / A2Av) around the FFN.

        For each EP shard the wrapper:

        1. Routes the shard's local tokens **globally** over all
           ``num_experts`` experts (no roll, no local-mask -- every shard
           sees the full expert axis).
        2. ``all_gather`` s its per-expert ``group_sizes`` so all shards
           know the complete ``[num_ep, num_experts]`` token-count matrix.
        3. Forward ``ragged_all_to_all`` over the EP axis: each shard
           sends per-expert chunks to the shard that owns those experts,
           and receives chunks for its own ``num_experts // num_ep``
           local experts from every other shard.
        4. Reorders the received buffer from ``(source_shard, expert)``
           to ``(expert, source_shard)`` ordering so each local expert's
           tokens are contiguous.
        5. Runs the three ``grouped_dense`` calls + activation over the
           ``E_local``-group buffer.
        6. Reverses the local reorder.
        7. Reverse ``ragged_all_to_all`` over EP returns each shard's
           token outputs to it.
        8. Inverts the global permute and applies the top-k weighted sum.
        """
        from jax.experimental.shard_map import shard_map

        ep_axis = self.expert_parallelism_axis
        if self.mesh is None:
            raise ValueError(
                "MoEBlock.expert_parallelism_axis is set; `mesh` must also"
                " be provided so the EP shard_map can be built."
            )
        mesh = self.mesh
        num_ep = mesh.shape[ep_axis]
        assert self.num_experts % num_ep == 0, (
            f"num_experts={self.num_experts} must be divisible by EP"
            f" size={num_ep}"
        )
        num_experts_local = self.num_experts // num_ep

        # Pre-compute the worst-case A2A receive buffer size (compile-time
        # constant). Each shard contributes ``b_l*S*topk = B*S*topk/num_ep``
        # token-expert pairs across all experts; the worst case for one
        # shard is "every global pair lands on this shard's local
        # experts" -- ``num_ep * (B*S*topk/num_ep) = B*S*topk`` rows. JIT
        # needs this static, so we use the global ``batch_size`` from the
        # outer scope (sharded layouts don't change it).
        global_batch_size, sequence_length, _hidden = inputs.shape
        topk = self.num_experts_per_tok
        recv_buffer_rows = global_batch_size * sequence_length * topk

        # Pack everything that crosses the shard_map boundary into a dict
        # pytree. shard_map fully supports pytrees: ``in_specs`` must
        # structurally match ``captured`` and we build them in lockstep
        # so adding/removing an optional bias is one ``dict[name] = ...``.
        captured: dict = {
            "inputs": inputs,
            "gate_logits": gate_logits,
            "wi_0": params["wi_0"],
            "wi_1": params["wi_1"],
            "wo": params["wo"],
        }
        in_specs: dict = {
            "inputs": P(ep_axis, None, None),
            "gate_logits": P(ep_axis, None, None),
            "wi_0": P(ep_axis, None, None),
            "wi_1": P(ep_axis, None, None),
            "wo": P(ep_axis, None, None),
        }
        if "expert_bias" in params:
            captured["expert_bias"] = params["expert_bias"]
            in_specs["expert_bias"] = P(ep_axis)
        if "wi_0_bias" in params:
            for name in ("wi_0_bias", "wi_1_bias", "wo_bias"):
                captured[name] = params[name]
                in_specs[name] = P(ep_axis, None)

        def _a2a_fn(local: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
            shard_id = jax.lax.axis_index(ep_axis)

            # -- Stage 1: per-shard route + global permute over all E --
            # Inside the shard_map body each input has its EP axis already
            # consumed, so ``local_inputs.shape == [B/num_ep, S, H]``.
            local_inputs = local["inputs"]
            local_logits = local["gate_logits"]
            local_b, local_s, local_h = local_inputs.shape
            inputs_2d = local_inputs.reshape(-1, local_h)
            logits_2d = local_logits.reshape(-1, self.num_experts)

            # The router operates over the full expert axis, so the
            # EP-sharded ``expert_bias`` (in_spec ``P(ep_axis)``) must be
            # all-gathered before being passed in.
            if "expert_bias" in local:
                full_expert_bias = jax.lax.all_gather(
                    local["expert_bias"], axis_name=ep_axis, tiled=True
                )
            else:
                full_expert_bias = None
            sparse_probs, routing_map = self._route_topk(
                logits_2d, full_expert_bias
            )

            # aux_loss must see the global token batch and the global
            # tokens_per_expert: its formula ``E*coeff/(k*T^2) * sum_i(
            # sum_t(probs[t,i]) * tokens[i])`` is not shard-decomposable
            # (the sum_t * tokens product is data-dependent across
            # shards). Cheapest fix: gather logits along the EP axis and
            # run the aux-loss kernel on the global tensor. The aux
            # branch has no data dependency on the main routing path so
            # XLA can overlap the two on the GPU.
            if self.aux_loss_coeff > 0.0:
                global_logits_2d = jax.lax.all_gather(
                    logits_2d, axis_name=ep_axis, axis=0, tiled=True
                )
                aux_loss = self._compute_aux_loss(global_logits_2d)
            else:
                aux_loss = None

            perm = self._global_permute(inputs_2d, sparse_probs, routing_map)
            global_group_sizes = perm["group_sizes"]  # [E]

            # -- Stage 2: gather per-expert counts across the EP axis --
            all_shards_tokens_per_expert = jax.lax.all_gather(
                global_group_sizes[None, :],
                axis_name=ep_axis,
                axis=0,
                tiled=True,
            )  # [num_ep, num_experts]

            # -- Stage 3: forward ragged_all_to_all over EP --
            in_off, send_sz, out_off, recv_sz = compute_ragged_all_to_all_params(
                all_shards_tokens_per_expert, shard_id, num_ep
            )
            recv_buf = jnp.zeros(
                (recv_buffer_rows, local_h),
                dtype=perm["sorted_inputs"].dtype,
            )
            x_recv = jax.lax.ragged_all_to_all(
                perm["sorted_inputs"],
                recv_buf,
                in_off,
                send_sz,
                out_off,
                recv_sz,
                axis_name=ep_axis,
            )

            # -- Stage 4: local permute (source_shard, expert) -> (expert, shard)
            sorted_x, local_group_sizes, local_perm_state = (
                local_permute_after_a2a(
                    x_recv,
                    all_shards_tokens_per_expert,
                    shard_id,
                    num_ep,
                )
            )

            # -- Stage 5: per-expert FFN (E_local groups) --
            local_params: dict = {
                "wi_0": local["wi_0"],
                "wi_1": local["wi_1"],
                "wo": local["wo"],
            }
            if "wi_0_bias" in local:
                local_params["wi_0_bias"] = local["wi_0_bias"]
                local_params["wi_1_bias"] = local["wi_1_bias"]
                local_params["wo_bias"] = local["wo_bias"]
            expert_outputs = self._expert_ffn(
                sorted_x,
                local_group_sizes,
                local_params,
                n_groups=num_experts_local,
            )

            # -- Stage 6: invert local permute --
            x_send_back = local_unpermute_before_a2a(
                expert_outputs, local_perm_state
            )

            # -- Stage 7: reverse ragged_all_to_all over EP --
            in_off_r, send_sz_r, out_off_r, recv_sz_r = (
                compute_reverse_ragged_all_to_all_params(
                    all_shards_tokens_per_expert, shard_id, num_ep
                )
            )
            send_back_buf = jnp.zeros_like(perm["sorted_inputs"])
            y_back = jax.lax.ragged_all_to_all(
                x_send_back,
                send_back_buf,
                in_off_r,
                send_sz_r,
                out_off_r,
                recv_sz_r,
                axis_name=ep_axis,
            )

            # -- Stage 8: invert global permute, weighted sum over top-k --
            output = self._global_combine(
                y_back, perm, batch_size=local_b, sequence_length=local_s
            )

            if self.tensor_parallelism_axis is not None:
                output = jax.lax.psum_scatter(
                    output,
                    self.tensor_parallelism_axis,
                    scatter_dimension=2,
                    tiled=True,
                )

            # ``out_specs`` must match the returned pytree structurally,
            # so always emit a real scalar for aux_loss; the outer
            # ``__call__`` re-strips it to None when aux_loss_coeff <= 0.
            if aux_loss is None:
                aux_loss = jnp.zeros((), dtype=self.dtype)
            return output, aux_loss

        # ``check_rep=False`` disables shard_map's invariant that any
        # output declared as ``P()`` is replicated across ``ep_axis``.
        # We use ``axis_index(ep_axis)`` inside ``_a2a_fn`` so the body
        # is genuinely non-replicated, which would otherwise (correctly)
        # fail the check. ``ragged_all_to_all`` already produces the
        # right cross-shard semantics; this is the standard JAX escape
        # hatch when collectives + per-shard logic coexist.
        return shard_map(
            _a2a_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(P(ep_axis, None, None), P()),
            check_rep=False,
        )(captured)
