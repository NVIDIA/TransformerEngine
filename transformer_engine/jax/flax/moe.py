# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoE block for TransformerEngine JAX.

This module exposes :class:`_MoEBlock`, an experimental Flax Linen layer
that is a thin wrapper around the framework-agnostic functional MoE entry
point :func:`transformer_engine.jax.moe.moe`. The wrapper's only job is
to:

1. Register the gate kernel, per-expert FFN kernels, and optional biases
   as ``self.param`` slots (with the right
   :func:`flax.linen.with_logical_partitioning` annotations so JAX's
   sharding layer FSDPs the params correctly).
2. Resolve the EP axis name from the active
   :class:`transformer_engine.jax.sharding.MeshResource`.
3. Forward all knobs to :func:`moe`.

All routing, dispatch, FFN, combine, and aux-loss logic lives in
``moe.py`` under a *single* ``jax.custom_vjp`` so future fusions
(FP8-on-the-wire EP, fused ``ragged_all_to_all + grouped_gemm``, gate +
route + dispatch fusion) can land without touching this wrapper.

The class is intentionally underscore-prefixed; the public ``MoEBlock``
alias will be introduced once TE's NCCL-backed EP component (and the
recipe-driven alignment follow-up) stabilises (target: the TE release
following the 2.16 code freeze).
"""

from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P  # noqa: F401  (re-exported for convenience)

from ..moe import PermutationBackend, moe
from ..quantize import noop_quantizer_set
from ..router import ScoreFunction
from ..sharding import get_active_resource_axis
from .module import TransformerEngineBase

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
Initializer = Callable[[PRNGKey, Shape, DType], Array]


__all__ = ["PermutationBackend", "_MoEBlock"]


class _MoEBlock(TransformerEngineBase):
    """Experimental Flax MoE layer over TransformerEngine.

    See module docstring for the design (this class is a thin Flax
    wrapper around :func:`transformer_engine.jax.moe.moe`). Constructor
    knob set kept compatible with the previous bespoke implementation so
    existing call sites need no changes.

    Parameters
    ----------
    num_experts : int
        Total number of experts. Under EP this must be divisible by the
        EP mesh axis size.
    num_experts_per_tok : int
        Top-k value for routing.
    intermediate_size : int
        Hidden dim of the per-expert FFN (the inner ``mlp`` axis).
    activation_type : str
        Activation between ``layer_w0 @ wi_0`` and the elementwise
        product with ``layer_w0 @ wi_1``. Default ``"silu"``.

    score_function : Union[str, ScoreFunction]
        ``"softmax"`` (default) or ``"sigmoid"`` for the routing scores.
    use_pre_softmax : bool
        Apply softmax before topk (vs. after).
    num_groups, group_topk : Optional[int]
        Grouped top-k knobs (DeepSeek-style). ``None`` disables grouping.
    scaling_factor : float
        Multiplier on the routing weights.
    use_expert_bias : bool
        If ``True``, registers a per-expert routing bias (shape ``[E]``).
        Only meaningful with ``score_function="sigmoid"``; the underlying
        primitive validates the pairing.
    aux_loss_coeff : float
        If ``> 0``, return the MoE auxiliary load-balancing loss scalar
        in addition to the main output.

    gate_kernel_axes, wi_kernel_axes, wo_kernel_axes, input_axes :
        Logical sharding axis tuples (consumed by Flax's
        :func:`with_logical_partitioning` and our internal
        :func:`with_sharding_constraint_by_logical_axes`).
    data_parallelism_axes : tuple[str, ...]
        FSDP axes over which the input *batch* dim is sharded IN
        ADDITION to the EP axis. Empty (default) means activations are
        replicated across non-EP axes within an EP group; set e.g.
        ``("fsdp",)`` for true FSDP-of-batch where each device owns a
        unique slice of the batch.
    permutation_backend : PermutationBackend
        ``PURE_JAX`` (default) or ``TRITON``.
    _align_size : int
        Per-expert group-size alignment (``0`` disables; required > 0
        for quantized grouped GEMM). Internal knob; will be inferred
        from the active quantization recipe in a follow-up PR.

    dtype : jnp.dtype
        Compute / parameter dtype.
    kernel_init, bias_init, expert_bias_init : Initializers.
    use_bias : bool
        Register per-expert FFN biases.

    Quantization is currently configured via the standard TE autocast
    context (``fp8_autocast``/``with_quantizer_set``); per-call
    quantizer sets can also be passed through ``__call__``'s
    ``quantizer_sets`` keyword once we stabilise the recipe pipeline.
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

    # Sharding (logical axes)
    gate_kernel_axes: Tuple[Optional[str], ...] = ()
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp")
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed")
    input_axes: Tuple[Optional[str], ...] = ()

    # Parallelism
    data_parallelism_axes: Tuple[str, ...] = ()

    # Permutation
    permutation_backend: PermutationBackend = PermutationBackend.PURE_JAX
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
                "permutation_backend must be a PermutationBackend, got"
                f" {self.permutation_backend!r}"
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Array) -> Tuple[Array, Optional[Array]]:
        """Run the MoE forward pass.

        Parameters
        ----------
        inputs : jnp.ndarray
            ``[batch, sequence, hidden]``.

        Returns
        -------
        output : jnp.ndarray
            ``[batch, sequence, hidden]``.
        aux_loss : Optional[jnp.ndarray]
            Scalar load-balancing loss when ``aux_loss_coeff > 0``,
            else ``None``.
        """
        assert (
            inputs.ndim == 3
        ), f"_MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        _, _, hidden_size = inputs.shape

        # Param registrations -- must run OUTSIDE any JAX transform that
        # alters the variable scope (e.g. shard_map). The functional
        # ``moe(...)`` opens its own shard_map internally for the EP
        # path, so registering params here is correct.
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

        return moe(
            inputs,
            gate_kernel,
            wi_0,
            wi_1,
            wo,
            wi_0_bias,
            wi_1_bias,
            wo_bias,
            expert_bias,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            activation_type=self.activation_type,
            score_function=self.score_function,
            use_pre_softmax=self.use_pre_softmax,
            num_groups=self.num_groups,
            group_topk=self.group_topk,
            scaling_factor=self.scaling_factor,
            aux_loss_coeff=self.aux_loss_coeff,
            permutation_backend=self.permutation_backend,
            align_size=self._align_size,
            gate_inside_vjp=True,
            ep_axis=ep_axis,
            data_parallelism_axes=self.data_parallelism_axes,
            input_axes=self.input_axes,
            gate_kernel_axes=self.gate_kernel_axes,
            wi_kernel_axes=self.wi_kernel_axes,
            wo_kernel_axes=self.wo_kernel_axes,
            quantizer_sets=(noop_quantizer_set, noop_quantizer_set, noop_quantizer_set),
            dtype=self.dtype,
        )
