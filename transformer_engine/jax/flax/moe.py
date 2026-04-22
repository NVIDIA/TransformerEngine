# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoEBlock for TransformerEngine JAX.

This module exposes :class:`MoEBlock`, a self-contained Flax Linen MoE layer
that wires together TE's fused router, a selectable token-dispatch backend
(pure-JAX ``unfused_*`` or fused Triton), TE's ``grouped_dense``, and optional
ring-of-experts Expert Parallelism.
"""

from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from ..dense import grouped_dense
from ..permutation import (
    _routing_map_to_selected_experts,
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
    routing, optional auxiliary load-balancing loss, token dispatch, per-expert
    two-layer FFN via grouped GEMMs, activation, token combine, and optional
    ring-of-experts expert parallelism.

    The permutation step is pluggable via ``permutation_backend``:
    ``"pure_jax"`` (default) uses the pure-JAX argsort-based
    ``unfused_token_dispatch`` / ``unfused_token_combine`` in
    :mod:`transformer_engine.jax.permutation`; ``"triton"`` uses TE's fused
    ``token_dispatch`` / ``token_combine`` kernels.

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
        projection in the SwiGLU-style ``act(wi_0) * wi_1`` product. Resolved
        via :func:`flax.linen.<name>` (``"silu"``, ``"gelu"``, ``"relu"``,
        ``"swish"``, ...) plus ``"linear"`` for identity.

    score_function : str or ScoreFunction
        ``"softmax"`` (default) or ``"sigmoid"`` for :func:`fused_topk_with_score_function`.
    use_pre_softmax : bool
        Apply softmax before top-k when ``score_function="softmax"``.
    num_groups : int
        Number of routing groups for grouped top-k (DeepSeek). ``<=0`` disables.
    group_topk : int
        Top-k at the group level. ``<=0`` disables.
    scaling_factor : float
        Scaling factor applied to output probs.
    use_expert_bias : bool
        If ``True``, registers a learnable ``expert_bias`` parameter of shape
        ``[num_experts]`` and passes it to the fused router. The router
        primitive validates that this is paired with ``score_function="sigmoid"``.
    aux_loss_coeff : float
        If ``> 0``, compute and return the MoE auxiliary load-balancing loss
        scalar via :func:`fused_moe_aux_loss`. ``0`` disables.

    gate_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the gate kernel of shape
        ``[hidden, num_experts]``.
    wi_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the ``wi_0`` and ``wi_1`` kernels of
        shape ``[num_experts, hidden, intermediate]``. Default:
        ``("exp", "embed", "mlp")``.
    wo_kernel_axes : tuple[str, ...]
        Logical partitioning axes for the ``wo`` kernel of shape
        ``[num_experts, intermediate, hidden]``. Default:
        ``("exp", "mlp", "embed")``.
    input_axes : tuple[str, ...]
        Logical axes used to constrain the input activation sharding at the
        block boundary. ``()`` (default) means no constraint.

    expert_parallelism_axis : Optional[str]
        Mesh axis along which experts are split. When set, the forward pass
        is wrapped in :func:`jax.experimental.shard_map.shard_map` that
        implements the ring-of-experts EP strategy: ``all_gather`` on inputs
        and gate logits, local routing + dispatch + FFN + combine, then
        ``psum_scatter`` on the output. When ``None`` (default), no
        ``shard_map`` wrapper is used; each primitive's ``custom_partitioning``
        rule handles DP/FSDP/TP automatically.
    tensor_parallelism_axis : Optional[str]
        Mesh axis for tensor parallelism on the FFN intermediate dim. When
        set, the output of the ``wo`` grouped GEMM is ``psum_scatter`` ed
        along this axis (inside the ``shard_map`` when EP is enabled, else at
        the end of the forward pass).

    permutation_backend : str
        ``"pure_jax"`` (default) or ``"triton"``.
    align_size : int
        Alignment for per-expert group sizes after padding. ``0`` disables
        padding (faster for the unquantized path). ``>0`` is required for
        quantized TE grouped GEMM whose recipe-specific alignment must divide
        ``align_size``.

    dtype : jnp.dtype
        Compute and parameter dtype.
    kernel_init : Initializer
        Initializer for all kernels (gate + per-expert FFN). Defaults to
        ``variance_scaling(1.0, 'fan_in', 'truncated_normal')`` (Flax convention).
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

    def _make_params(self, hidden_size: int):
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
        params = {
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
            Scalar auxiliary load-balancing loss when ``aux_loss_coeff > 0``,
            else ``None``.
        """
        assert inputs.ndim == 3, (
            f"MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        )
        inputs = with_sharding_constraint_by_logical_axes(inputs, self.input_axes)

        _, _, hidden_size = inputs.shape
        params = self._make_params(hidden_size)

        # Gate runs OUTSIDE the EP shard_map below, so each EP shard projects
        # its own local slice of tokens and we later all-gather only the
        # smaller logits tensor instead of the full inputs.
        gate_logits = self._gate(inputs, params["gate_kernel"])

        if self.expert_parallelism_axis is None:
            # No EP: each primitive's own ``custom_partitioning`` rule handles
            # DP / FSDP / TP across the mesh - no shard_map needed.
            output, aux_loss = self._forward_body(
                inputs,
                gate_logits,
                params,
                num_experts_local=self.num_experts,
                roll_to_expert_id=None,
            )
        else:
            # Ring-EP: ``_forward_body`` is wrapped in a shard_map that
            # orchestrates the cross-primitive collectives (all_gather inputs
            # / logits before, psum_scatter output after) which per-primitive
            # ``custom_partitioning`` cannot express on its own.
            output, aux_loss = self._forward_ring_ep(inputs, gate_logits, params)

        if self.aux_loss_coeff <= 0.0:
            aux_loss = None
        return output, aux_loss

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _gate(self, inputs: jnp.ndarray, gate_kernel: jnp.ndarray) -> jnp.ndarray:
        """Linear gate projection ``inputs @ gate_kernel``.

        Kept as a plain matmul (not ``DenseGeneral``) so it integrates cleanly
        with the EP shard_map: the gate matmul runs in the outer (pre-shard_map)
        scope and its output is all-gathered along the EP axis inside.
        """
        # Cast kernel to input dtype outside FP8 scope (gate is typically BF16/FP32).
        kernel = gate_kernel.astype(inputs.dtype)
        return jnp.einsum("bsh,he->bse", inputs, kernel)

    # ------------------------------------------------------------------
    # Forward body (shared between no-EP and ring-EP paths)
    # ------------------------------------------------------------------

    def _forward_body(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
        num_experts_local: int,
        roll_to_expert_id: Optional[int],
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Routing + dispatch + per-expert FFN + combine.

        Used both bare (no EP) and inside the ring-EP shard_map. In the
        ring-EP case ``inputs`` and ``gate_logits`` are the post-all_gather
        global tensors, ``num_experts_local == num_experts // num_ep``, and
        ``roll_to_expert_id`` is the offset that brings this shard's experts
        into slots ``[0, num_experts_local)``.
        """
        batch_size, sequence_length, hidden_size = inputs.shape
        inputs_2d = inputs.reshape(-1, hidden_size)
        logits_2d = gate_logits.reshape(-1, self.num_experts)

        sparse_probs, routing_map, aux_loss = self._route(
            logits_2d, params.get("expert_bias")
        )

        if roll_to_expert_id is not None:
            # Rotate expert columns so this shard's experts come first.
            routing_map = jnp.roll(routing_map, -roll_to_expert_id, axis=-1)
            sparse_probs = jnp.roll(sparse_probs, -roll_to_expert_id, axis=-1)
            if self.permutation_backend == "triton":
                # Triton path: zero out remote-expert columns so the fused
                # ``token_dispatch`` never writes tokens routed off-shard.
                # The pure-JAX path zeroes garbage *output* rows below
                # instead, since masking the routing_map directly would
                # break the argsort-based permutation.
                local_mask = (
                    jnp.arange(self.num_experts) < num_experts_local
                )
                routing_map = routing_map * local_mask
                sparse_probs = sparse_probs * local_mask.astype(sparse_probs.dtype)

        expert_outputs, combine_state = self._dispatch_and_expert_ffn(
            inputs_2d,
            sparse_probs,
            routing_map,
            params,
            num_experts_local=num_experts_local,
            # The roll is already baked into ``routing_map``/``sparse_probs``
            # above, so the unfused dispatch must not roll again.
            roll_to_expert_id=0 if roll_to_expert_id is not None else None,
        )

        if (
            roll_to_expert_id is not None
            and self.permutation_backend == "pure_jax"
        ):
            # Zero the rows of ``expert_outputs`` past the real local-expert
            # token count: ``grouped_dense`` leaves them as garbage because
            # ``group_sizes`` was truncated to the local slice. Without this
            # the unsort + weighted-sum in combine would mix garbage into
            # every token's output (mirrors Maxtext's moe.py).
            real_mask = (
                jnp.arange(expert_outputs.shape[0])
                < combine_state["local_real_size"]
            )
            expert_outputs = jnp.where(real_mask[:, None], expert_outputs, 0)

        output = self._combine(
            expert_outputs,
            combine_state,
            batch_size=batch_size,
            sequence_length=sequence_length,
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
    # Ring-of-Experts EP wrapper
    # ------------------------------------------------------------------

    def _forward_ring_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Wrap :meth:`_forward_body` in a ring-of-experts ``shard_map``.

        For each EP shard the wrapper:
          1. ``all_gather`` s the local inputs / logits / expert_bias along
             the EP axis so the routing sees every token globally.
          2. Calls ``_forward_body`` with ``roll_to_expert_id =
             num_experts_per_shard * shard_id`` and the EP-local weight slice.
          3. ``psum_scatter`` s the resulting ``[B*num_ep, S, H]`` output back
             to the EP-sharded ``[B, S, H]`` layout.
        """
        from jax.experimental.shard_map import shard_map

        ep_axis = self.expert_parallelism_axis
        if self.mesh is None:
            raise ValueError(
                "MoEBlock.expert_parallelism_axis is set; `mesh` must also be"
                " provided so the ring-of-experts shard_map can be built."
            )
        mesh = self.mesh
        num_ep = mesh.shape[ep_axis]
        assert self.num_experts % num_ep == 0, (
            f"num_experts={self.num_experts} must be divisible by EP size={num_ep}"
        )
        num_experts_per_shard = self.num_experts // num_ep

        # Pack everything that crosses the shard_map boundary into a dict
        # pytree. shard_map fully supports pytrees: ``in_specs`` must
        # structurally match ``captured``, and we build them in lockstep so
        # adding/removing an optional bias is a single ``dict[name] = ...``.
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

        def _ring_fn(local: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
            shard_id = jax.lax.axis_index(ep_axis)

            gathered_inputs = jax.lax.all_gather(
                local["inputs"], axis_name=ep_axis, tiled=True
            )
            gathered_logits = jax.lax.all_gather(
                local["gate_logits"], axis_name=ep_axis, tiled=True
            )

            local_params: dict = {
                "wi_0": local["wi_0"],
                "wi_1": local["wi_1"],
                "wo": local["wo"],
            }
            if "expert_bias" in local:
                # The router operates over the full expert axis, so the
                # EP-sharded bias must be all-gathered.
                local_params["expert_bias"] = jax.lax.all_gather(
                    local["expert_bias"], axis_name=ep_axis, tiled=True
                )
            if "wi_0_bias" in local:
                local_params["wi_0_bias"] = local["wi_0_bias"]
                local_params["wi_1_bias"] = local["wi_1_bias"]
                local_params["wo_bias"] = local["wo_bias"]

            output, aux_loss = self._forward_body(
                gathered_inputs,
                gathered_logits,
                local_params,
                num_experts_local=num_experts_per_shard,
                roll_to_expert_id=num_experts_per_shard * shard_id,
            )

            # ``output`` is [B*num_ep, S, H] (global batch after all_gather);
            # psum_scatter along EP returns the local [B, S, H] slice.
            output = jax.lax.psum_scatter(
                output, ep_axis, scatter_dimension=0, tiled=True
            )

            # ``out_specs`` must match the returned pytree structurally, so
            # always emit a real scalar for aux_loss; the outer ``__call__``
            # re-strips it to None when ``aux_loss_coeff <= 0``.
            if aux_loss is None:
                aux_loss = jnp.zeros((), dtype=self.dtype)
            return output, aux_loss

        # ``check_rep=False`` disables shard_map's invariant that any output
        # declared as ``P()`` is replicated across ``ep_axis``. We use
        # ``axis_index(ep_axis)`` inside ``_ring_fn`` to compute a per-shard
        # roll, which makes the body genuinely non-replicated and would
        # otherwise (correctly) fail the check. The ``psum_scatter`` of the
        # output already produces the right cross-shard semantics; this is
        # the standard JAX escape hatch when collectives + per-shard logic
        # coexist.
        return shard_map(
            _ring_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(P(ep_axis, None, None), P()),
            check_rep=False,
        )(captured)

    # ------------------------------------------------------------------
    # Route
    # ------------------------------------------------------------------

    def _route(
        self,
        logits_2d: jnp.ndarray,
        expert_bias: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """Run the fused router and optional aux-loss."""
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

        aux_loss = None
        if self.aux_loss_coeff > 0.0:
            # The score-for-aux kernel runs independently (no data dependency
            # on the main kernel), so XLA can overlap them on the GPU.
            aux_scores, aux_routing_map = fused_topk_with_score_function(
                logits_2d,
                topk=self.num_experts_per_tok,
                score_function=self.score_function,
                compute_aux_scores=True,
            )
            aux_tokens_per_expert = jnp.sum(
                aux_routing_map.astype(jnp.int32), axis=0
            )
            aux_loss = fused_moe_aux_loss(
                aux_scores,
                aux_tokens_per_expert,
                topk=self.num_experts_per_tok,
                coeff=self.aux_loss_coeff,
            )

        return sparse_probs, routing_map, aux_loss

    # ------------------------------------------------------------------
    # Dispatch + expert FFN
    # ------------------------------------------------------------------

    def _dispatch_and_expert_ffn(
        self,
        inputs_2d: jnp.ndarray,
        sparse_probs: jnp.ndarray,
        routing_map: jnp.ndarray,
        params: dict,
        num_experts_local: int,
        roll_to_expert_id: Optional[int],
    ) -> Tuple[jnp.ndarray, dict]:
        """Dispatch tokens, run the three grouped GEMMs + activation, return expert outputs.

        Returns a tuple ``(expert_outputs, combine_state)`` where
        ``combine_state`` carries the per-backend state needed to rebuild the
        original token ordering in :meth:`_combine`.
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
                roll_to_expert_id=roll_to_expert_id,
            )
            # Slice group_sizes to just this shard's experts. When not using
            # EP, ``num_experts_local == self.num_experts`` so this is a no-op.
            group_sizes = group_sizes[:num_experts_local]
            # ``local_real_size = sum(group_sizes)`` is the number of permuted
            # rows that actually correspond to tokens routed to this shard's
            # experts. Used by the ring-EP caller to zero out garbage rows
            # before combine.
            combine_state = {
                "backend": "pure_jax",
                "perm_state": perm_state,
                "routing_weights": routing_weights,
                "local_real_size": jnp.sum(group_sizes),
            }
        else:  # "triton"
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
            group_sizes = group_sizes[:num_experts_local]
            combine_state = {
                "backend": "triton",
                "row_id_map": row_id_map,
                "pad_offsets": pad_offsets,
                "merging_probs": sparse_probs,
                "group_sizes": group_sizes,
            }

        # ------------------------------------------------------------------
        # Expert FFN: grouped GEMMs w0, w1 + activation + w_o.
        # ------------------------------------------------------------------
        wi_0 = params["wi_0"]
        wi_1 = params["wi_1"]
        wo = params["wo"]

        # Each grouped_dense call gets its own quantizer_set with
        # ``n_groups=num_experts_local``; this matches the shape of
        # ``group_sizes`` passed in and keeps the quantizer FP8 meta correctly
        # sized per shard.
        q_set_w0 = self.generate_quantizer_set(
            postfix="_w0", n_groups=num_experts_local
        )
        q_set_w1 = self.generate_quantizer_set(
            postfix="_w1", n_groups=num_experts_local
        )
        q_set_wo = self.generate_quantizer_set(
            postfix="_wo", n_groups=num_experts_local
        )

        # Cast kernels to the sort dtype when no FP8 quantization is active
        # (mirrors DenseGeneral).
        if q_set_w0 == noop_quantizer_set:
            wi_0 = wi_0.astype(sorted_inputs.dtype)
        if q_set_w1 == noop_quantizer_set:
            wi_1 = wi_1.astype(sorted_inputs.dtype)
        if q_set_wo == noop_quantizer_set:
            wo = wo.astype(sorted_inputs.dtype)

        # ``grouped_dense`` accepts per-expert bias of shape (G, N); it adds
        # ``bias[i]`` to the ``group_sizes[i]`` rows belonging to expert ``i``
        # in the permuted layout.
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

        return expert_outputs, combine_state

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    def _combine(
        self,
        expert_outputs: jnp.ndarray,
        combine_state: dict,
        batch_size: int,
        sequence_length: int,
    ) -> jnp.ndarray:
        if combine_state["backend"] == "pure_jax":
            return unfused_token_combine(
                expert_outputs,
                combine_state["perm_state"],
                combine_state["routing_weights"],
                num_experts_per_tok=self.num_experts_per_tok,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        # triton
        out_2d = token_combine(
            expert_outputs,
            combine_state["row_id_map"],
            merging_probs=combine_state["merging_probs"],
            pad_offsets=combine_state["pad_offsets"],
        )
        hidden_size = out_2d.shape[-1]
        return out_2d.reshape(batch_size, sequence_length, hidden_size).astype(
            self.dtype
        )
