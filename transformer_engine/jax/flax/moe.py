# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoE block for TransformerEngine JAX.

This module exposes :class:`_MoEBlock`, an **experimental** self-contained
Flax Linen MoE layer. It is intentionally prefixed with an underscore
while TE's NCCL-backed EP component (and the recipe-driven alignment
follow-up) stabilises; the public ``MoEBlock`` alias will be introduced
once those dependencies are ready (target: the TE release following the
2.16 code freeze). Until then please treat the class, its parameters,
and :class:`GlobalPermuteResult` as unstable.

See the class docstring for the architecture, the EP / FSDP strategies,
and the ``_align_size > 0`` contract.
"""

from enum import Enum
from functools import partial
from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn, struct as flax_struct
from jax.sharding import PartitionSpec as P

from ..dense import grouped_dense
from ..permutation import (
    routing_map_to_selected_experts,
    compute_ragged_all_to_all_params,
    compute_reverse_ragged_all_to_all_params,
    local_permute_after_a2a,
    local_unpermute_before_a2a,
    PureJaxPermState,
    pure_jax_token_combine,
    pure_jax_token_dispatch,
    token_combine,
    token_dispatch,
)
from ..quantize import noop_quantizer_set
from ..router import ScoreFunction, fused_moe_aux_loss, fused_topk_with_score_function
from ..sharding import (
    _get_mesh,
    get_active_resource_axis,
    with_sharding_constraint_by_logical_axes,
)
from .module import TransformerEngineBase, _convert_to_activation_function

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
Initializer = Callable[[PRNGKey, Shape, DType], Array]


__all__ = ["GlobalPermuteResult", "PermutationBackend", "_MoEBlock"]


# =============================================================================
# PermutationBackend
# =============================================================================


class PermutationBackend(Enum):
    """Token-dispatch / combine backend used by :class:`_MoEBlock`.

    * ``PURE_JAX``: ``jnp.argsort`` + gather paths compiled as plain XLA;
      typically faster than ``TRITON`` in current testing because XLA can
      fuse the ops with surrounding work.
    * ``TRITON``: TE's fused Triton kernels.
    """

    PURE_JAX = "pure_jax"
    TRITON = "triton"


# =============================================================================
# GlobalPermuteResult
# =============================================================================
#
# Output of :meth:`_MoEBlock._global_permute`. Carried as a pytree (so it
# crosses ``jax.shard_map`` / ``jax.value_and_grad`` boundaries
# transparently) and consumed by :meth:`_MoEBlock._global_combine`. The
# fields populated depend on the permutation backend; the unused fields
# stay ``None``.
#
# Per-backend payloads (anything else is ``None``):
#   pure_jax: ``perm_state``, ``routing_weights``
#   triton:   ``row_id_map``, ``pad_offsets``, ``merging_probs``


@flax_struct.dataclass
class GlobalPermuteResult:
    """Result of :meth:`_MoEBlock._global_permute`."""

    sorted_inputs: jnp.ndarray
    group_sizes: jnp.ndarray
    perm_state: Optional[PureJaxPermState] = None
    routing_weights: Optional[jnp.ndarray] = None
    row_id_map: Optional[jnp.ndarray] = None
    pad_offsets: Optional[jnp.ndarray] = None
    merging_probs: Optional[jnp.ndarray] = None
    backend: PermutationBackend = flax_struct.field(
        pytree_node=False, default=PermutationBackend.PURE_JAX
    )


# =============================================================================
# _MoEBlock
# =============================================================================


class _MoEBlock(TransformerEngineBase):
    """Mixture-of-Experts Flax Linen block (**experimental**).

    .. warning::

       This class is exposed as ``_MoEBlock`` (leading underscore) on
       purpose: it is not part of the stable public API yet. The TE
       NCCL-backed EP component and the recipe-driven ``_align_size``
       follow-up both need to land before this is promoted to a public
       ``MoEBlock``. Until then, expect signature changes, including
       to :class:`GlobalPermuteResult` and :class:`PermutationBackend`.
       Target promotion: the TE release after the 2.16 code freeze.

    Encapsulates the full MoE forward pass: gate projection, fused top-k
    routing, optional auxiliary load-balancing loss, token dispatch,
    per-expert two-layer FFN via grouped GEMMs, activation, token combine,
    and optional ragged-all-to-all expert parallelism.

    Architecture
    ------------

    The block is decomposed into orthogonal stages so the EP wrapper can
    inject collectives between them:

    * ``_route``: gate logits -> top-k routing decisions (+ aux loss).
    * ``_global_permute``: scatter tokens to experts; produces
      ``[num_tokens*topk + maybe_padding, hidden]`` and per-expert
      ``group_sizes`` of length ``num_experts``.
    * ``_expert_ffn``: three ``grouped_dense`` calls + activation.
      Operates on whatever ``(rows, group_sizes, n_groups)`` it is
      handed -- agnostic to whether ``n_groups`` is the global expert
      count (no-EP) or the local expert count (A2A-EP).
    * ``_global_combine``: inverse of ``_global_permute`` -- gather +
      weighted sum across top-k experts.

    Two top-level forward variants compose those stages:

    * ``_forward_no_ep``: route -> permute -> ffn -> combine. Each TE
      primitive's ``custom_partitioning`` rule handles DP / FSDP
      automatically.
    * ``_forward_a2a_ep``: wraps the body in :func:`jax.shard_map` and
      inserts ``all_gather(group_sizes)`` + forward
      ``ragged_all_to_all`` + local permute around the FFN, plus their
      inverses afterwards. This is the only place ``shard_map`` is
      used; A2A is the canonical EP strategy because the in-flight
      NCCL EP component will require this same data layout.

    Note on ``_align_size > 0``
    ---------------------------

    Both permutation backends pad each expert's group to a multiple of
    ``_align_size`` when requested, which is what cuBLASLt's grouped
    GEMM wants for FP8 shape selection. The pure-JAX backend
    additionally appends a zero-input padding tail to keep the buffer
    statically sized for JIT, so ``sum(group_sizes) <=
    sorted_inputs.shape[0]`` strictly. The V1 grouped GEMM FFI asserts
    strict equality ``m == sum(group_sizes)`` and is therefore
    incompatible with ``_align_size > 0``; the V2 cuBLASLt-backed
    grouped GEMM relaxes this to ``m >= sum(group_sizes)`` and only
    iterates over the populated ragged region. The ``_align_size > 0``
    tests therefore force ``NVTE_JAX_ENFORCE_V2_GROUPED_GEMM=1`` and
    ``skip`` if V2 is not supported on the target hardware / dtype.

    Two permutation backends are pluggable via ``permutation_backend``:

    * :attr:`PermutationBackend.PURE_JAX` (default) -- argsort-based
      :func:`~transformer_engine.jax.permutation.pure_jax_token_dispatch` /
      :func:`~transformer_engine.jax.permutation.pure_jax_token_combine`.
      Faster than Triton in profiling for DeepSeek-style configs.
    * :attr:`PermutationBackend.TRITON` -- TE's fused
      :func:`~transformer_engine.jax.permutation.token_dispatch` /
      :func:`~transformer_engine.jax.permutation.token_combine` Triton
      kernels.

    Expert parallelism is configured via :class:`MeshResource`'s
    ``ep_resource`` axis. When that axis is set on the active
    :func:`~transformer_engine.jax.global_mesh_resource` and has more
    than one device, ``_MoEBlock`` dispatches to the
    **ragged-all-to-all** EP strategy (a.k.a. A2Av): each shard routes
    its own tokens globally over all experts, then a forward
    ``ragged_all_to_all`` exchanges per-expert chunks so each shard
    ends up holding only the tokens for its local experts; after the
    FFN a reverse ``ragged_all_to_all`` returns each shard's outputs
    to it. This matches the layout the in-flight NCCL EP component
    expects.

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
    num_groups : Optional[int]
        Number of routing groups for grouped top-k (DeepSeek). ``None``
        (default) disables.
    group_topk : Optional[int]
        Top-k at the group level. ``None`` (default) disables.
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

    data_parallelism_axes : tuple[str, ...]
        Additional mesh axes that the input *batch* dim is sharded over
        IN ADDITION to ``MeshResource.ep_resource``. Setting this to
        e.g. ``("fsdp",)`` makes the ``shard_map`` ``in_specs`` for the
        batch dim become ``P(("ep", "fsdp"), None, None)`` -- giving
        each device a unique slice of the batch (true FSDP) instead of
        replicating the per-ep-shard batch across fsdp peers.
        Routing is unaffected: ``axis_index("ep")`` still controls the
        ragged-all-to-all; the extra fsdp peers within an ep group send
        and receive their own batch slices in lockstep. Default ``()``
        preserves legacy ZeRO-1-style behavior (activations replicated
        on fsdp within an ep group).

    permutation_backend : PermutationBackend
        :attr:`PermutationBackend.PURE_JAX` (default) or
        :attr:`PermutationBackend.TRITON`.

    dtype : jnp.dtype
        Compute and parameter dtype.
    kernel_init : Initializer
        Initializer for all kernels (gate + per-expert FFN). Defaults to
        ``variance_scaling(1.0, 'fan_in', 'truncated_normal')`` (Flax
        convention).
    use_bias : bool
        If ``True``, registers per-expert FFN biases ``wi_0_bias``,
        ``wi_1_bias``, ``wo_bias``.

    TODO:
    -----
    ``_align_size`` is an internal, non-public knob (alignment for
    per-expert group sizes after padding). A follow-up PR will infer it
    from the active quantization recipe, after which it will become a
    fully-internal implementation detail. Until then it stays
    intentionally underscored to discourage callers from depending on
    it.
    """

    # Architecture
    num_experts: int = 8
    num_experts_per_tok: int = 2
    intermediate_size: int = 2048
    activation_type: str = "silu"

    # Routing
    score_function: Union[str, ScoreFunction] = "softmax"
    use_pre_softmax: bool = False
    num_groups: Optional[int] = None
    group_topk: Optional[int] = None
    scaling_factor: float = 1.0
    use_expert_bias: bool = False
    aux_loss_coeff: float = 0.0

    # Sharding
    gate_kernel_axes: Tuple[Optional[str], ...] = ()
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp")
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed")
    input_axes: Tuple[Optional[str], ...] = ()

    # Parallelism
    #
    # The EP axis is resolved from ``global_mesh_resource().ep_resource``
    # and the active mesh, not configured per-instance. ``_MoEBlock``
    # uses ``_forward_a2a_ep`` when that axis exists on the mesh and
    # has > 1 device; otherwise it uses ``_forward_no_ep``.
    data_parallelism_axes: Tuple[str, ...] = ()

    # Permutation
    permutation_backend: PermutationBackend = PermutationBackend.PURE_JAX
    # See class docstring "Notes": internal, will be inferred from the
    # quantization recipe in a follow-up PR.
    _align_size: int = 0

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
        if not isinstance(self.permutation_backend, PermutationBackend):
            raise TypeError(
                "permutation_backend must be a PermutationBackend,"
                f" got {self.permutation_backend!r}"
            )
        super().__post_init__()

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
        assert (
            inputs.ndim == 3
        ), f"_MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        inputs = with_sharding_constraint_by_logical_axes(inputs, self.input_axes)

        _, _, hidden_size = inputs.shape

        # Param registrations are inlined here (not in a helper) so each
        # ``self.param`` lives close to the rest of the entry point.
        # Note: under EP the FFN weights and ``expert_bias`` are
        # consumed *inside* a ``shard_map`` body. Flax's ``self.param``
        # must run OUTSIDE any JAX transform that would alter the
        # variable scope (``shard_map`` does), so the registrations stay
        # here in ``__call__`` and the values are passed down explicitly
        # via ``in_specs``. ``_gate`` is called outside ``shard_map`` in
        # both paths, so its kernel is registered inline inside
        # ``_gate`` itself rather than here.

        gate_logits = self._gate(inputs)

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
        wi_0_bias = wi_1_bias = wo_bias = None
        if self.use_bias:
            wi_0_bias = self.param(
                "wi_0_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "mlp")),
                (self.num_experts, self.intermediate_size),
                self.dtype,
            )
            wi_1_bias = self.param(
                "wi_1_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "mlp")),
                (self.num_experts, self.intermediate_size),
                self.dtype,
            )
            wo_bias = self.param(
                "wo_bias",
                nn.with_logical_partitioning(self.bias_init, ("exp", "embed")),
                (self.num_experts, hidden_size),
                self.dtype,
            )
        expert_bias = None
        if self.use_expert_bias:
            expert_bias = self.param(
                "expert_bias",
                nn.with_logical_partitioning(self.expert_bias_init, ("exp",)),
                (self.num_experts,),
                self.dtype,
            )

        ep_axis = get_active_resource_axis("ep_resource")
        if ep_axis is None:
            output, aux_loss = self._forward_no_ep(
                inputs,
                gate_logits,
                wi_0=wi_0,
                wi_1=wi_1,
                wo=wo,
                wi_0_bias=wi_0_bias,
                wi_1_bias=wi_1_bias,
                wo_bias=wo_bias,
                expert_bias=expert_bias,
            )
        else:
            output, aux_loss = self._forward_a2a_ep(
                inputs,
                gate_logits,
                ep_axis=ep_axis,
                wi_0=wi_0,
                wi_1=wi_1,
                wo=wo,
                wi_0_bias=wi_0_bias,
                wi_1_bias=wi_1_bias,
                wo_bias=wo_bias,
                expert_bias=expert_bias,
            )

        if self.aux_loss_coeff <= 0.0:
            aux_loss = None
        return output, aux_loss

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _gate(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Linear gate projection ``inputs @ gate_kernel``.

        Kept as a plain ``einsum`` (not ``DenseGeneral``) so it composes
        cleanly with the EP shard_map: the gate runs in the outer
        (pre-shard_map) scope and its output passes through the
        ``shard_map`` boundary unchanged. Because the gate runs outside
        any ``shard_map`` body in both EP and no-EP forwards, the
        ``gate_kernel`` parameter is registered inline here.

        The gating GEMM is intentionally kept in ``self.dtype`` (typically
        ``bfloat16``) and is **not** autocast to FP8 even when the caller
        wraps the block in :func:`transformer_engine.jax.autocast`. Two
        reasons: (1) the GEMM is tiny (``H * E`` with ``E`` small) and
        contributes well under 1% of the block's compute, so quantization
        savings are marginal; (2) the resulting logits feed a top-k +
        softmax (or sigmoid) routing decision that is sensitive to
        quantization noise -- routing flips at low-confidence tokens
        could materially hurt model quality. To override, wrap the call
        site in your own ``autocast`` and manually replace this method.
        """
        hidden_size = inputs.shape[-1]
        gate_kernel = self.param(
            "gate_kernel",
            nn.with_logical_partitioning(self.kernel_init, self.gate_kernel_axes),
            (hidden_size, self.num_experts),
            self.dtype,
        )
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
        # ``fused_topk_with_score_function`` uses ``-1`` as the
        # "disabled" sentinel for the grouped-routing knobs; translate
        # our ``None`` user-facing default to that sentinel here.
        sparse_probs, routing_map = fused_topk_with_score_function(
            logits_2d,
            topk=self.num_experts_per_tok,
            use_pre_softmax=self.use_pre_softmax,
            num_groups=-1 if self.num_groups is None else self.num_groups,
            group_topk=-1 if self.group_topk is None else self.group_topk,
            scaling_factor=self.scaling_factor,
            score_function=self.score_function,
            expert_bias=expert_bias,
        )
        sparse_probs = sparse_probs.astype(self.dtype)
        return sparse_probs, routing_map

    def _compute_aux_loss(
        self,
        logits_2d: jnp.ndarray,
        tokens_per_expert: jnp.ndarray,
    ) -> Optional[jnp.ndarray]:
        """Compute the MoE auxiliary load-balancing loss.

        The score-for-aux kernel reads only ``logits_2d`` and the final
        reduction reads only the (already-computed) ``tokens_per_expert``,
        so the aux scores can run concurrently with the main routing
        path on the GPU.

        ``logits_2d`` should be the *full* logits tensor over the global
        token batch -- under EP the caller is responsible for
        :func:`jax.lax.all_gather` ing the logits before calling this so
        the aux_loss formula
        ``loss = (E * coeff / (k * T^2)) * sum_i(sum_t(probs[t,i]) * tokens[i])``
        sees the global ``T``.

        ``tokens_per_expert`` must be the per-expert token-assignment
        count from the *actual* routing decision -- i.e. derived from
        ``_route_topk``'s ``routing_map``, not recomputed from a clean
        top-k. This matters under DeepSeek-style routing
        (``num_groups > 0`` / ``group_topk > 0``) where the
        post-grouping routing differs from a plain top-k. Under EP the
        caller is responsible for summing over all (ep + dp) shards
        first so the count is global.
        """
        if self.aux_loss_coeff <= 0.0:
            return None
        # The "compute_aux_scores=True" kernel intentionally ignores
        # num_groups/group_topk/expert_bias and returns the dense
        # post-score-function scores over all experts. Those scores are
        # what the aux-loss formula expects (raw scoring, no grouping
        # bias); the routing decisions used for ``tokens_per_expert``
        # come from the caller-supplied real ``routing_map``.
        aux_scores, _ = fused_topk_with_score_function(
            logits_2d.astype(jnp.float32),
            topk=self.num_experts_per_tok,
            score_function=self.score_function,
            compute_aux_scores=True,
        )
        return fused_moe_aux_loss(
            aux_scores.astype(jnp.float32),
            tokens_per_expert.astype(jnp.int32),
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
    ) -> GlobalPermuteResult:
        """Dispatch tokens to the global expert axis.

        Returns a :class:`GlobalPermuteResult` suitable both for the
        no-EP forward (where the same buffer feeds ``_expert_ffn``
        directly) and for the A2A-EP path (where the buffer is sliced +
        sent over the EP axis before the FFN). The result carries the
        per-backend opaque state needed to invert the dispatch in
        :meth:`_global_combine`.
        """
        num_tokens = inputs_2d.shape[0]
        topk = self.num_experts_per_tok

        if self.permutation_backend is PermutationBackend.PURE_JAX:
            selected_experts, routing_weights = routing_map_to_selected_experts(
                sparse_probs, routing_map, topk
            )
            sorted_inputs, perm_state, group_sizes = pure_jax_token_dispatch(
                inputs_2d,
                selected_experts,
                num_experts=self.num_experts,
                num_experts_per_tok=topk,
                align_size=self._align_size,
            )
            return GlobalPermuteResult(
                backend=PermutationBackend.PURE_JAX,
                sorted_inputs=sorted_inputs,
                group_sizes=group_sizes,
                perm_state=perm_state,
                routing_weights=routing_weights,
            )

        # triton
        num_out_tokens = num_tokens * topk
        align_size_arg = self._align_size if self._align_size > 0 else None
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
        return GlobalPermuteResult(
            backend=PermutationBackend.TRITON,
            sorted_inputs=sorted_inputs,
            group_sizes=group_sizes,
            row_id_map=row_id_map,
            pad_offsets=pad_offsets,
            merging_probs=sparse_probs,
        )

    # ------------------------------------------------------------------
    # Expert FFN (three grouped_dense calls + activation)
    # ------------------------------------------------------------------

    def _expert_ffn(
        self,
        sorted_inputs: jnp.ndarray,
        group_sizes: jnp.ndarray,
        n_groups: int,
        wi_0: jnp.ndarray,
        wi_1: jnp.ndarray,
        wo: jnp.ndarray,
        wi_0_bias: Optional[jnp.ndarray] = None,
        wi_1_bias: Optional[jnp.ndarray] = None,
        wo_bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Run the per-expert SwiGLU-style FFN over a permuted buffer.

        All ``wi_*`` / ``wo`` weights and the optional biases are passed
        in as explicit args (rather than registered inline here) because
        in the EP path this method runs *inside* a ``shard_map`` body
        and Flax param registration must happen outside that scope.

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
        n_groups : int
            Number of expert groups. Equals ``self.num_experts`` for the
            no-EP path and ``num_experts // num_ep`` for the A2A-EP path.
            Used to size the per-call quantizer set so the FP8 metadata
            tensors match ``group_sizes``.
        wi_0, wi_1, wo : jnp.ndarray
            Expert weight tensors. Shapes (no-EP):
            ``(num_experts, hidden, intermediate)`` for wi_*,
            ``(num_experts, intermediate, hidden)`` for wo. Under EP
            the leading expert dim is sliced to ``num_experts // num_ep``.
        wi_0_bias, wi_1_bias, wo_bias : Optional[jnp.ndarray]
            Optional per-expert biases (shape ``(num_experts, N)``);
            ``grouped_dense`` adds ``bias[i]`` to the rows belonging to
            expert ``i`` in the permuted layout.

        Returns
        -------
        expert_outputs : jnp.ndarray
            ``[buffer_size, hidden]``.
        """
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
        perm_result: GlobalPermuteResult,
        batch_size: int,
        sequence_length: int,
    ) -> jnp.ndarray:
        """Inverse of :meth:`_global_permute`.

        Gathers per-expert outputs back into ``[batch, sequence, hidden]``
        and applies the per-token weighted sum across the top-k experts.
        """
        if perm_result.backend is PermutationBackend.PURE_JAX:
            return pure_jax_token_combine(
                expert_outputs,
                perm_result.perm_state,
                perm_result.routing_weights,
                num_experts_per_tok=self.num_experts_per_tok,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        # triton
        out_2d = token_combine(
            expert_outputs,
            perm_result.row_id_map,
            merging_probs=perm_result.merging_probs,
            pad_offsets=perm_result.pad_offsets,
        )
        hidden_size = out_2d.shape[-1]
        return out_2d.reshape(batch_size, sequence_length, hidden_size).astype(self.dtype)

    # ------------------------------------------------------------------
    # No-EP forward
    # ------------------------------------------------------------------

    def _forward_no_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        *,
        wi_0: jnp.ndarray,
        wi_1: jnp.ndarray,
        wo: jnp.ndarray,
        wi_0_bias: Optional[jnp.ndarray] = None,
        wi_1_bias: Optional[jnp.ndarray] = None,
        wo_bias: Optional[jnp.ndarray] = None,
        expert_bias: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Single-shard or DP/FSDP forward (no shard_map wrapper).

        DP / FSDP both flow through each TE primitive's
        ``custom_partitioning`` rule -- there is no cross-primitive
        collective that the rules cannot express on their own, so a
        ``shard_map`` is unnecessary here.

        Sharding contract for callers
        -----------------------------

        On this no-EP path the grouped quantize and grouped GEMMs run
        in the caller's outer SPMD context (no ``shard_map`` boundary).
        Their custom_partitioning rules read sharding from each input's
        ``NamedSharding`` and propagate consistent shardings on outputs.
        Concretely:

        * ``inputs`` should be FSDP/DP-sharded on the batch dim
          (``input_axes`` in :class:`_MoEBlock` enforces this via a
          logical ``with_sharding_constraint``).
        * ``wi_*`` / ``wo`` weights should carry the logical axes
          ``wi_kernel_axes`` / ``wo_kernel_axes`` so FSDP shards a
          weight non-contracting dim, gathered inside ``grouped_dense``
          before the GEMM.
        * The wgrad reduce-scatter (when FSDP is active) is emitted by
          ``grouped_dense_bwd``'s partitioning rule; no explicit
          collective is needed here.

        Without those shardings the grouped GEMM falls back to
        replicated-everywhere semantics (legal but defeats FSDP/DP).
        Tested in ``tests/jax/test_distributed_moe_block.py`` for the
        EP=2 + FSDP=2 case; the no-EP + FSDP-only case shares the same
        infra and is covered when ``ep_resource`` is unset on the
        active ``MeshResource``.
        """
        batch_size, sequence_length, hidden_size = inputs.shape
        inputs_2d = inputs.reshape(-1, hidden_size)
        logits_2d = gate_logits.reshape(-1, self.num_experts)

        sparse_probs, routing_map = self._route_topk(logits_2d, expert_bias)
        # ``tokens_per_expert`` MUST come from the real routing_map so the
        # aux-loss objective matches actual routing decisions under
        # DeepSeek-style num_groups/group_topk routing.
        tokens_per_expert = jnp.sum(routing_map.astype(jnp.int32), axis=0)
        aux_loss = self._compute_aux_loss(logits_2d, tokens_per_expert)
        perm = self._global_permute(inputs_2d, sparse_probs, routing_map)
        expert_outputs = self._expert_ffn(
            perm.sorted_inputs,
            perm.group_sizes,
            n_groups=self.num_experts,
            wi_0=wi_0,
            wi_1=wi_1,
            wo=wo,
            wi_0_bias=wi_0_bias,
            wi_1_bias=wi_1_bias,
            wo_bias=wo_bias,
        )
        output = self._global_combine(expert_outputs, perm, batch_size, sequence_length)
        return output, aux_loss

    # ------------------------------------------------------------------
    # A2A (ragged-all-to-all) EP forward
    # ------------------------------------------------------------------

    def _forward_a2a_ep(
        self,
        inputs: jnp.ndarray,
        gate_logits: jnp.ndarray,
        *,
        ep_axis: str,
        wi_0: jnp.ndarray,
        wi_1: jnp.ndarray,
        wo: jnp.ndarray,
        wi_0_bias: Optional[jnp.ndarray] = None,
        wi_1_bias: Optional[jnp.ndarray] = None,
        wo_bias: Optional[jnp.ndarray] = None,
        expert_bias: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
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

        mesh = _get_mesh()
        if mesh is None or mesh.empty:
            raise ValueError(
                "_MoEBlock requires an active jax.sharding.Mesh (either via"
                " `with mesh:` or `jax.set_mesh`) when EP is configured on"
                " the active MeshResource."
            )
        num_ep = mesh.shape[ep_axis]
        assert (
            self.num_experts % num_ep == 0
        ), f"num_experts={self.num_experts} must be divisible by EP size={num_ep}"
        num_experts_local = self.num_experts // num_ep

        # Compose the BATCH sharding axis tuple. ``ep`` is always part of
        # the batch axis (so ragged_all_to_all has data to route); any
        # ``data_parallelism_axes`` are added on top so the per-device
        # batch slice is genuinely unique (true FSDP / DP).
        # Examples:
        #   data_parallelism_axes=()              -> P('ep', None, None)
        #   data_parallelism_axes=('fsdp',)       -> P(('ep','fsdp'), None, None)
        #   data_parallelism_axes=('fsdp','data') -> P(('ep','fsdp','data'), ...)
        for ax in self.data_parallelism_axes:
            if ax not in mesh.shape:
                raise ValueError(
                    f"data_parallelism_axes contains {ax!r} but mesh has"
                    f" axes {tuple(mesh.shape.keys())}"
                )
        if len(self.data_parallelism_axes) == 0:
            batch_pspec_axis: Any = ep_axis
        else:
            batch_pspec_axis = (ep_axis, *self.data_parallelism_axes)
        # The size by which the per-device batch is divided BEYOND ep.
        # Used to tighten the worst-case ragged_all_to_all recv buffer:
        # at most ``num_ep`` peers each send their entire local
        # ``B/(num_ep*dp_size)*S*topk`` token-expert pairs, so the worst
        # recv per device is ``num_ep * B/(num_ep*dp_size)*S*topk
        # = B/dp_size * S * topk``.
        dp_size = 1
        for ax in self.data_parallelism_axes:
            dp_size *= mesh.shape[ax]

        global_batch_size, sequence_length, _hidden = inputs.shape
        topk = self.num_experts_per_tok
        # The shard_map's ``in_specs=P((ep, *dp_axes), ...)`` requires the
        # batch dim to be divisible by ``num_ep * dp_size``; check upfront
        # here for a clearer error than the one shard_map would raise at
        # trace time.
        batch_divisor = num_ep * dp_size
        if global_batch_size % batch_divisor != 0:
            raise ValueError(
                f"batch={global_batch_size} not divisible by prod(data_parallelism_axes)={dp_size}"
            )
        # Worst-case A2A receive count per shard: every peer can send its
        # full per-expert-aligned local buffer. With ``_align_size > 0``
        # each per-expert group can be padded by up to ``_align_size - 1``
        # rows, so per shard the receive can overshoot the unpadded count
        # by up to ``num_experts * (_align_size - 1)``. Skipping this
        # extra slack would let ``ragged_all_to_all`` write past
        # ``recv_buf`` when EP and padding are combined.
        recv_buffer_rows = (global_batch_size // dp_size) * sequence_length * topk
        if self._align_size > 0:
            recv_buffer_rows += self.num_experts * (self._align_size - 1)

        # Pack everything that crosses the shard_map boundary into a dict
        # pytree. shard_map fully supports pytrees: ``in_specs`` must
        # structurally match ``captured`` and we build them in lockstep
        # so adding/removing an optional bias is one ``dict[name] = ...``.
        # Params must be packed here (rather than passed inline by
        # ``self.param`` inside the body) because Flax variable scopes
        # must not be entered from inside a JAX transform's body.
        captured: dict = {
            "inputs": inputs,
            "gate_logits": gate_logits,
            "wi_0": wi_0,
            "wi_1": wi_1,
            "wo": wo,
        }
        in_specs: dict = {
            "inputs": P(batch_pspec_axis, None, None),
            "gate_logits": P(batch_pspec_axis, None, None),
            "wi_0": P(ep_axis, None, None),
            "wi_1": P(ep_axis, None, None),
            "wo": P(ep_axis, None, None),
        }
        if expert_bias is not None:
            captured["expert_bias"] = expert_bias
            in_specs["expert_bias"] = P(ep_axis)
        if wi_0_bias is not None:
            captured["wi_0_bias"] = wi_0_bias
            captured["wi_1_bias"] = wi_1_bias
            captured["wo_bias"] = wo_bias
            for name in ("wi_0_bias", "wi_1_bias", "wo_bias"):
                in_specs[name] = P(ep_axis, None)

        a2a_body = partial(
            self._a2a_body,
            ep_axis=ep_axis,
            num_ep=num_ep,
            num_experts_local=num_experts_local,
            recv_buffer_rows=recv_buffer_rows,
        )

        # ``check_rep=False`` disables shard_map's invariant that any
        # output declared as ``P()`` is replicated across ``ep_axis``.
        # We use ``axis_index(ep_axis)`` inside ``_a2a_body`` so the
        # body is genuinely non-replicated, which would otherwise
        # (correctly) fail the check. ``ragged_all_to_all`` already
        # produces the right cross-shard semantics; this is the standard
        # JAX escape hatch when collectives + per-shard logic coexist.
        return shard_map(
            a2a_body,
            mesh=mesh,
            in_specs=(in_specs,),
            out_specs=(P(batch_pspec_axis, None, None), P()),
            check_rep=False,
        )(captured)

    # ------------------------------------------------------------------
    # Body of the per-shard A2A-EP forward (extracted from
    # :meth:`_forward_a2a_ep` for readability). Runs *inside* the
    # ``shard_map`` and is therefore in EP-manual mode: collectives over
    # ``ep_axis`` are explicit, the rest of the mesh stays in auto mode.
    # ------------------------------------------------------------------

    def _a2a_body(
        self,
        local: dict,
        *,
        ep_axis: str,
        num_ep: int,
        num_experts_local: int,
        recv_buffer_rows: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        sparse_probs, routing_map = self._route_topk(logits_2d, full_expert_bias)

        # aux_loss must see the global token batch and the global
        # tokens_per_expert: its formula ``E*coeff/(k*T^2) * sum_i(
        # sum_t(probs[t,i]) * tokens[i])`` is not shard-decomposable
        # (the sum_t * tokens product is data-dependent across
        # shards). We need a *single* collective:
        #   * ``all_gather`` logits over (ep + any DP axes) so both
        #     (a) the score-for-aux kernel and (b) a re-run of
        #     ``_route_topk`` see the full token batch. The re-run
        #     gives us the global per-expert token count directly,
        #     avoiding a separate ``psum``. Two consecutive global
        #     collectives over the same replica group at the very
        #     start of the program have been observed to deadlock
        #     under FP8 autocast on some XLA + NCCL combinations,
        #     so we keep this branch to one collective.
        # The aux branch has no data dependency on the main routing
        # path beyond what is already gathered, so XLA can overlap
        # the two routings on the GPU.
        if self.aux_loss_coeff > 0.0:
            # ``axis_name`` accepts a tuple ⇒ a single collective
            # over the cartesian product of axes; XLA may lower
            # this to one multi-axis op or split it.
            if len(self.data_parallelism_axes) == 0:
                aux_collective_axes: Any = ep_axis
            else:
                aux_collective_axes = (ep_axis, *self.data_parallelism_axes)
            global_logits_2d = jax.lax.all_gather(
                logits_2d, axis_name=aux_collective_axes, axis=0, tiled=True
            )
            # Re-run topk on the gathered logits to obtain the
            # *global* routing_map post-grouping (respects
            # num_groups/group_topk/expert_bias just like the local
            # routing). Summing over the global token dim gives the
            # exact same counts as ``psum(local_tokens_per_expert)``
            # without an extra collective. The duplicate topk
            # compute is small relative to the FFNs.
            _, global_routing_map = self._route_topk(global_logits_2d, full_expert_bias)
            global_tokens_per_expert = jnp.sum(global_routing_map.astype(jnp.int32), axis=0)
            aux_loss = self._compute_aux_loss(global_logits_2d, global_tokens_per_expert)
        else:
            aux_loss = None

        perm = self._global_permute(inputs_2d, sparse_probs, routing_map)
        global_group_sizes = perm.group_sizes  # [E]

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
            dtype=perm.sorted_inputs.dtype,
        )
        x_recv = jax.lax.ragged_all_to_all(
            perm.sorted_inputs,
            recv_buf,
            in_off,
            send_sz,
            out_off,
            recv_sz,
            axis_name=ep_axis,
        )

        # -- Stage 4: local permute (source_shard, expert) -> (expert, shard)
        sorted_x, local_group_sizes, local_perm_state = local_permute_after_a2a(
            x_recv,
            all_shards_tokens_per_expert,
            shard_id,
            num_ep,
        )

        # -- Stage 5: per-expert FFN (E_local groups) --
        expert_outputs = self._expert_ffn(
            sorted_x,
            local_group_sizes,
            n_groups=num_experts_local,
            wi_0=local["wi_0"],
            wi_1=local["wi_1"],
            wo=local["wo"],
            wi_0_bias=local.get("wi_0_bias"),
            wi_1_bias=local.get("wi_1_bias"),
            wo_bias=local.get("wo_bias"),
        )

        # -- Stage 6: invert local permute --
        x_send_back = local_unpermute_before_a2a(expert_outputs, local_perm_state)

        # -- Stage 7: reverse ragged_all_to_all over EP --
        in_off_r, send_sz_r, out_off_r, recv_sz_r = compute_reverse_ragged_all_to_all_params(
            all_shards_tokens_per_expert, shard_id, num_ep
        )
        send_back_buf = jnp.zeros_like(perm.sorted_inputs)
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
        output = self._global_combine(y_back, perm, batch_size=local_b, sequence_length=local_s)

        # ``out_specs`` must match the returned pytree structurally,
        # so always emit a real scalar for aux_loss; the outer
        # ``__call__`` re-strips it to None when aux_loss_coeff <= 0.
        if aux_loss is None:
            aux_loss = jnp.zeros((), dtype=self.dtype)
        return output, aux_loss
