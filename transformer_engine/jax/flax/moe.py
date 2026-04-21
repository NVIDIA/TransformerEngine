# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoEBlock for TransformerEngine JAX.

This module exposes :class:`MoEBlock`, a self-contained Flax Linen MoE layer
that wires together TE's fused router, a selectable token-dispatch backend
(pure-JAX MaxText-style or Triton), TE's ``grouped_dense``, and optional
ring-of-experts Expert Parallelism.

See ``plans/te_jax_moeblock_926b7994.plan.md`` for the full design rationale
and the mapping to Maxtext's ``RoutedMoE``.
"""

from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from ..dense import grouped_dense
from ..mt_permutation import mt_token_combine, mt_token_dispatch
from ..permutation import token_combine, token_dispatch
from ..quantize import noop_quantizer_set
from ..router import ScoreFunction, fused_moe_aux_loss, fused_topk_with_score_function
from ..sharding import with_sharding_constraint_by_logical_axes
from .module import TransformerEngineBase

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
Initializer = Callable[[PRNGKey, Shape, DType], Array]


__all__ = ["MoEBlock"]


# =============================================================================
# Helpers
# =============================================================================


_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "identity": lambda x: x,
    "linear": lambda x: x,
}


def _get_activation_fn(name: str) -> Callable:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation_type={name!r}; supported: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]


def _extract_topk_from_routing_map(
    sparse_probs: jnp.ndarray,
    routing_map: jnp.ndarray,
    topk: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert TE's ``(sparse_probs, routing_map)`` to ``(selected_experts, weights)``.

    ``routing_map`` is a boolean mask of shape ``[num_tokens, num_experts]``
    with exactly ``topk`` ``True`` positions per row. ``sparse_probs`` is the
    same-shape float tensor whose non-zero entries are the routing weights.

    The per-token top-k expert IDs are recovered as the last ``topk`` indices
    of ``argsort(routing_map)`` (``False < True``), and the corresponding
    weights are gathered from ``sparse_probs`` along the expert axis.

    The within-row expert ordering does not have to match the router's
    top-k ordering: :func:`mt_token_dispatch` and :func:`mt_token_combine`
    only require that ``selected_experts`` and ``weights`` are consistent with
    each other.
    """
    # Cast to int32 so argsort has a well-defined ordering. (Ascending argsort
    # on 0/1 puts the ``True`` positions last; we then slice the last ``topk``.)
    selected_experts = jnp.argsort(routing_map.astype(jnp.int32), axis=-1)[:, -topk:]
    weights = jnp.take_along_axis(sparse_probs, selected_experts, axis=-1)
    return selected_experts, weights


# =============================================================================
# MoEBlock
# =============================================================================


class MoEBlock(TransformerEngineBase):
    """Mixture-of-Experts Flax Linen block.

    Encapsulates the full MoE forward pass: gate projection, fused top-k
    routing, optional auxiliary load-balancing loss, token dispatch, per-expert
    two-layer FFN via grouped GEMMs, activation, token combine, and optional
    ring-of-experts expert parallelism.

    The permutation step is pluggable: the default ``permutation_backend="pure_jax"``
    uses the MaxText-style argsort-based dispatch/combine in
    :mod:`transformer_engine.jax.mt_permutation`, which empirically outperforms
    the Triton kernels on several E2E workloads. ``permutation_backend="triton"``
    uses TE's ``token_dispatch`` / ``token_combine`` kernels.

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
        projection in the SwiGLU-style ``act(wi_0) * wi_1`` product. Supported:
        ``"silu"``/``"swish"`` (default), ``"gelu"``, ``"relu"``,
        ``"identity"``/``"linear"``.

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
        ``[num_experts]`` and passes it to the fused router. Only valid with
        ``score_function="sigmoid"`` (DeepSeek V3 loss-free load balancing).
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
        ``"pure_jax"`` (default; faster on many E2E workloads) or ``"triton"``.
    align_size : int
        Alignment for per-expert group sizes after padding. ``0`` disables
        padding (faster for the unquantized path). ``>0`` is required for
        quantized TE grouped GEMM whose recipe-specific alignment must divide
        ``align_size``. Passed through to both permutation backends.
    use_custom_sort_vjp : bool
        Only used when ``permutation_backend="pure_jax"``. If ``True``, uses
        a custom VJP for the argsort-based gather (faster in most cases).

    dtype : jnp.dtype
        Compute and parameter dtype.
    kernel_init : Initializer
        Initializer for all kernels. Defaults to ``variance_scaling(1.0,
        'fan_in', 'truncated_normal')``.
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
    use_custom_sort_vjp: bool = True

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
        if self.use_expert_bias:
            # ``fused_topk_with_score_function`` only accepts ``expert_bias``
            # under the sigmoid score function. Raise early to surface the
            # misconfiguration instead of failing deep inside the kernel.
            score_func = (
                self.score_function.name.lower()
                if isinstance(self.score_function, ScoreFunction)
                else str(self.score_function).lower()
            )
            if score_func != "sigmoid":
                raise ValueError(
                    "use_expert_bias=True requires score_function='sigmoid';"
                    f" got {self.score_function!r}."
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
    def __call__(
        self,
        inputs: Array,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        """Run the MoE forward pass.

        Parameters
        ----------
        inputs : jnp.ndarray
            Input tensor of shape ``[batch, sequence, hidden]``.
        deterministic : bool
            Reserved for future dropout-based routing; currently unused.

        Returns
        -------
        output : jnp.ndarray
            Output tensor of shape ``[batch, sequence, hidden]``.
        aux_loss : Optional[jnp.ndarray]
            Scalar auxiliary load-balancing loss when ``aux_loss_coeff > 0``,
            else ``None``.
        """
        del deterministic  # unused for now

        assert inputs.ndim == 3, (
            f"MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        )
        inputs = with_sharding_constraint_by_logical_axes(inputs, self.input_axes)

        batch_size, sequence_length, hidden_size = inputs.shape
        params = self._make_params(hidden_size)

        # Gate projection runs OUTSIDE the EP shard_map (mirroring Maxtext),
        # so that each EP shard projects its own local slice of tokens and we
        # later all-gather only the logits, not the full inputs.
        gate_logits = self._gate(inputs, params["gate_kernel"])

        if self.expert_parallelism_axis is not None:
            return self._forward_ring_ep(inputs, gate_logits, params)
        return self._forward_single_shard(inputs, gate_logits, params)

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _gate(self, inputs: jnp.ndarray, gate_kernel: jnp.ndarray) -> jnp.ndarray:
        """Linear gate projection ``inputs @ gate_kernel``.

        Kept as a plain matmul (not ``DenseGeneral``) so it integrates cleanly
        with the EP shard_map below: the gate matmul runs in the outer
        (pre-shard_map) scope and its output is all-gathered along the EP axis
        inside the shard_map.
        """
        # Cast kernel to input dtype outside FP8 scope (gate is typically BF16/FP32).
        kernel = gate_kernel.astype(inputs.dtype)
        return jnp.einsum("bsh,he->bse", inputs, kernel)

    # ------------------------------------------------------------------
    # Single-shard (no EP) forward
    # ------------------------------------------------------------------

    def _forward_single_shard(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        batch_size, sequence_length, hidden_size = inputs.shape

        inputs_2d = inputs.reshape(-1, hidden_size)
        logits_2d = gate_logits.reshape(-1, self.num_experts)

        sparse_probs, routing_map, aux_loss = self._route(
            logits_2d, params.get("expert_bias")
        )

        expert_outputs, combine_state = self._dispatch_and_expert_ffn(
            inputs_2d,
            sparse_probs,
            routing_map,
            params,
            num_experts_local=self.num_experts,
            roll_to_expert_id=None,
            local_tokens_per_expert_count=self.num_experts,
        )

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
    # Ring-of-Experts EP forward
    # ------------------------------------------------------------------

    def _forward_ring_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Wrap the dispatch / FFN / combine pipeline in a ring-of-experts
        ``shard_map``.

        Inside the shard_map each EP shard:
          1. ``all_gather`` s the inputs and logits along the EP axis so it
             sees every token globally.
          2. Routes with ``roll_to_expert_id = num_experts_per_shard * shard_id``
             so its local experts are in slots ``[0, num_experts_per_shard)``.
          3. Dispatches tokens, slicing ``group_sizes`` to the first
             ``num_experts_per_shard`` entries (the rest correspond to remote
             experts and should be zero after the roll/mask).
          4. Runs the per-expert FFN on its local expert slice of
             ``wi_0`` / ``wi_1`` / ``wo``.
          5. Combines at the expanded-batch shape ``[B * num_ep, S, H]`` then
             ``psum_scatter`` s along the EP axis to return the local slice.
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

        # in_specs / out_specs use PartitionSpec over the EP axis for inputs/
        # outputs (leading batch dim is split across EP) and ``P("exp", ...)``
        # for the expert weights, where we require the user's logical axis
        # rules to map ``"exp"`` to the EP mesh axis. The expert bias is
        # similarly sharded along the expert axis.
        inputs_spec = P(ep_axis, None, None)
        logits_spec = P(ep_axis, None, None)
        wi_spec = P(ep_axis, None, None)
        wo_spec = P(ep_axis, None, None)
        output_spec = P(ep_axis, None, None)
        scalar_spec = P()
        bias_1d_spec = P(ep_axis)
        bias_2d_spec = P(ep_axis, None)

        expert_bias_value = params.get("expert_bias")
        wi_0_bias_value = params.get("wi_0_bias")
        wi_1_bias_value = params.get("wi_1_bias")
        wo_bias_value = params.get("wo_bias")

        in_specs = [
            inputs_spec,
            logits_spec,
            wi_spec,
            wi_spec,
            wo_spec,
        ]
        captured = [
            inputs,
            gate_logits,
            params["wi_0"],
            params["wi_1"],
            params["wo"],
        ]
        if expert_bias_value is not None:
            in_specs.append(bias_1d_spec)
            captured.append(expert_bias_value)
        if wi_0_bias_value is not None:
            in_specs.extend([bias_2d_spec, bias_2d_spec, bias_2d_spec])
            captured.extend([wi_0_bias_value, wi_1_bias_value, wo_bias_value])

        out_specs = (output_spec, scalar_spec)

        use_expert_bias = expert_bias_value is not None
        use_bias = wi_0_bias_value is not None

        def _ring_fn(*args):
            idx = 0
            local_inputs = args[idx]; idx += 1
            local_gate_logits = args[idx]; idx += 1
            local_wi_0 = args[idx]; idx += 1
            local_wi_1 = args[idx]; idx += 1
            local_wo = args[idx]; idx += 1
            local_expert_bias = None
            if use_expert_bias:
                local_expert_bias = args[idx]; idx += 1
            local_wi_0_bias = local_wi_1_bias = local_wo_bias = None
            if use_bias:
                local_wi_0_bias = args[idx]; idx += 1
                local_wi_1_bias = args[idx]; idx += 1
                local_wo_bias = args[idx]; idx += 1

            shard_id = jax.lax.axis_index(ep_axis)

            # All-gather inputs and logits along the EP axis so each shard
            # sees the global tokens.
            gathered_inputs = jax.lax.all_gather(
                local_inputs, axis_name=ep_axis, tiled=True
            )
            gathered_logits = jax.lax.all_gather(
                local_gate_logits, axis_name=ep_axis, tiled=True
            )

            # If the user also sharded by EP on the expert_bias, ``local_expert_bias``
            # is already the local slice; the router operates over the full
            # expert axis, so all-gather to reconstruct.
            global_expert_bias = None
            if local_expert_bias is not None:
                global_expert_bias = jax.lax.all_gather(
                    local_expert_bias, axis_name=ep_axis, tiled=True
                )

            batch_size = gathered_inputs.shape[0]
            sequence_length = gathered_inputs.shape[1]
            hidden_size = gathered_inputs.shape[2]

            inputs_2d = gathered_inputs.reshape(-1, hidden_size)
            logits_2d = gathered_logits.reshape(-1, self.num_experts)

            sparse_probs, routing_map, aux_loss = self._route(
                logits_2d, global_expert_bias
            )

            # Ring-of-experts roll: after rolling expert columns by
            # ``-num_experts_per_shard * shard_id``, this shard's experts
            # occupy slots ``[0, num_experts_per_shard)`` in ``routing_map``
            # and ``sparse_probs``.
            #
            # For the Triton backend we additionally mask the remote-expert
            # columns to False/0 so ``token_dispatch`` never writes those
            # tokens into the local permuted buffer. For the pure-JAX backend
            # we leave the routing_map untouched (mirroring Maxtext): the roll
            # passed to ``mt_token_dispatch`` sorts remote-expert tokens past
            # the local slots, and we later zero out those garbage rows of
            # ``expert_outputs`` before the combine.
            roll = num_experts_per_shard * shard_id
            routing_map = jnp.roll(routing_map, -roll, axis=-1)
            sparse_probs = jnp.roll(sparse_probs, -roll, axis=-1)
            if self.permutation_backend == "triton":
                local_expert_mask = (
                    jnp.arange(self.num_experts) < num_experts_per_shard
                )
                routing_map = routing_map * local_expert_mask[None, :]
                sparse_probs = sparse_probs * local_expert_mask[None, :].astype(
                    sparse_probs.dtype
                )

            # Build a reduced-expert view of the weights: the outer ``shard_map``
            # has already sliced the leading expert axis down to
            # ``num_experts_per_shard`` per shard. Pass it through as-is to the
            # dispatch / expert-FFN path with ``num_experts_local = num_experts_per_shard``.
            local_params = {
                "gate_kernel": None,  # unused past gate
                "wi_0": local_wi_0,
                "wi_1": local_wi_1,
                "wo": local_wo,
            }
            if use_bias:
                local_params["wi_0_bias"] = local_wi_0_bias
                local_params["wi_1_bias"] = local_wi_1_bias
                local_params["wo_bias"] = local_wo_bias

            expert_outputs, combine_state = self._dispatch_and_expert_ffn(
                inputs_2d,
                sparse_probs,
                routing_map,
                local_params,
                num_experts_local=num_experts_per_shard,
                roll_to_expert_id=0,  # roll is already applied on routing_map
                local_tokens_per_expert_count=num_experts_per_shard,
            )

            # For the pure-JAX backend in ring-EP mode, zero out expert-output
            # rows that correspond to remote experts (which ``grouped_dense``
            # leaves as garbage since ``group_sizes`` was truncated to the
            # local slice). Without this, the unsort + weighted-sum in
            # combine would mix garbage into every token's output. Matches
            # ``moe.py:1731-1733`` in Maxtext.
            if self.permutation_backend == "pure_jax":
                real_mask = (
                    jnp.arange(expert_outputs.shape[0])
                    < combine_state["local_real_size"]
                )
                expert_outputs = jnp.where(
                    real_mask[:, None], expert_outputs, 0
                )

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

            # ``output`` is [B*num_ep, S, H] (global batch after all-gather);
            # psum_scatter along EP returns the local [B, S, H] slice.
            output = jax.lax.psum_scatter(
                output,
                ep_axis,
                scatter_dimension=0,
                tiled=True,
            )

            if aux_loss is None:
                aux_loss = jnp.zeros((), dtype=self.dtype)
            return output, aux_loss

        output, aux_loss = shard_map(
            _ring_fn,
            mesh=mesh,
            in_specs=tuple(in_specs),
            out_specs=out_specs,
            check_rep=False,
        )(*captured)

        if self.aux_loss_coeff <= 0.0:
            aux_loss = None
        return output, aux_loss

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
        local_tokens_per_expert_count: int,
    ) -> Tuple[jnp.ndarray, dict]:
        """Dispatch tokens, run the three grouped GEMMs + activation, return expert outputs.

        Returns a tuple ``(expert_outputs, combine_state)`` where
        ``combine_state`` carries the per-backend state needed to rebuild the
        original token ordering in :meth:`_combine`.
        """
        num_tokens = inputs_2d.shape[0]
        topk = self.num_experts_per_tok

        if self.permutation_backend == "pure_jax":
            selected_experts, routing_weights = _extract_topk_from_routing_map(
                sparse_probs, routing_map, topk
            )
            sorted_inputs, perm_state, group_sizes = mt_token_dispatch(
                inputs_2d,
                selected_experts,
                num_experts=self.num_experts,
                num_experts_per_tok=topk,
                align_size=self.align_size,
                roll_to_expert_id=roll_to_expert_id,
                use_custom_sort_vjp=self.use_custom_sort_vjp,
            )
            # Slice group_sizes to just this shard's experts. When not using
            # EP, ``num_experts_local == self.num_experts`` so this is a no-op.
            group_sizes = group_sizes[:local_tokens_per_expert_count]
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
            group_sizes = group_sizes[:local_tokens_per_expert_count]
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

        act_fn = _get_activation_fn(self.activation_type)
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
            return mt_token_combine(
                expert_outputs,
                combine_state["perm_state"],
                combine_state["routing_weights"],
                num_experts_per_tok=self.num_experts_per_tok,
                batch_size=batch_size,
                sequence_length=sequence_length,
                use_custom_sort_vjp=self.use_custom_sort_vjp,
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
