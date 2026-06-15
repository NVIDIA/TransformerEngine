# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Flax Linen MoE block for TransformerEngine JAX.

This module exposes :class:`_MoEBlock`, an experimental Flax Linen layer
that wraps the framework-agnostic functional MoE entry point
:func:`transformer_engine.jax.moe.moe`. The wrapper's only job is to:

1. Register the gate kernel, per-expert FFN kernels, and optional FFN
   biases as ``self.param`` slots (with the right
   :func:`flax.linen.with_logical_partitioning` annotations so JAX's
   sharding layer FSDPs the params correctly).
2. Resolve the EP axis name from the active
   :class:`transformer_engine.jax.sharding.MeshResource`.
3. Forward all knobs to :func:`moe`.

The class is intentionally underscore-prefixed; the public ``MoEBlock``
alias will be introduced once TE's NCCL-backed EP component (and the
recipe-driven alignment follow-up) stabilises.
"""

from dataclasses import field
from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

# Re-exported so downstream users can ``from transformer_engine.jax.flax.moe
# import P`` without a second jax.sharding import.
from jax.sharding import PartitionSpec as P  # noqa: F401  # pylint: disable=unused-import

from ..moe import moe
from ..quantize import QuantizerSet, noop_quantizer_set
from ..router import ScoreFunction
from ..sharding import get_active_resource_axis
from .module import TransformerEngineBase

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
Initializer = Callable[[PRNGKey, Shape, DType], Array]


__all__ = ["_MoEBlock"]


class _MoEBlock(TransformerEngineBase):
    """Experimental Flax MoE layer over TransformerEngine.

    See module docstring for the design (this class is a thin Flax
    wrapper around :func:`transformer_engine.jax.moe.moe`).

    Parameters
    ----------
    num_experts : int
        Total number of experts. Must be divisible by the EP mesh axis size.
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

    apply_topk_weights_early : bool
        When True, fold per-token top-k weights into the FFN intermediate
        (next to ``act(gate) * up``) instead of into the post-down-projection
        combine. Both placements are mathematically equivalent (the down
        projection is linear); the early placement gives XLA a chance to
        fuse the multiply with the activation.

    gate_kernel_axes, wi_kernel_axes, wo_kernel_axes, input_axes :
        Logical sharding axis tuples (consumed by Flax's
        :func:`with_logical_partitioning` and our internal
        :func:`with_sharding_constraint_by_logical_axes`).
    data_parallelism_axes : tuple[str, ...]
        DP/FSDP axes over which the input *batch* dim is sharded IN
        ADDITION to the EP axis. Empty (default) means activations are
        replicated across non-EP axes within an EP group; set e.g.
        ``("fsdp",)`` or ``("dp", "fsdp")`` for outer data axes where
        each EP group owns a unique slice of the batch.

    dtype : jnp.dtype
        Compute / parameter dtype.
    kernel_init, bias_init : Initializers.
    use_bias : bool
        Register per-expert FFN biases.
    ffn_quantizer_set : QuantizerSet
        Quantizer set for the grouped-GEMM FFN body. Defaults to no
        quantization; MXFP8 callers should also set ``align_size=128``.
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

    # Sharding (logical axes)
    gate_kernel_axes: Tuple[Optional[str], ...] = ()
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp")
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed")
    input_axes: Tuple[Optional[str], ...] = ()

    # Parallelism
    data_parallelism_axes: Tuple[str, ...] = ()

    # Aux loss (global expert-load balancing). 0.0 disables; non-zero
    # enables the second return value and routes its gradient back to
    # the gate.
    aux_loss_coeff: float = 0.0

    # Fusion knob
    apply_topk_weights_early: bool = False

    # Minimum per-expert slot alignment fed to ``tex.ep_prepare``. Default 0
    # uses the natural slot count; set to e.g. 128 to satisfy FP8 grouped-GEMM
    # tile alignment.
    align_size: int = 0

    # Dtypes / init / misc
    dtype: DType = jnp.float32
    kernel_init: Optional[Initializer] = None
    bias_init: Initializer = nn.initializers.zeros
    use_bias: bool = False
    # Per-expert router bias added before the top-k. Only meaningful when
    # score_function='sigmoid'.
    use_expert_bias: bool = False
    ffn_quantizer_set: QuantizerSet = field(default_factory=lambda: noop_quantizer_set)

    def __post_init__(self):
        if self.kernel_init is None:
            object.__setattr__(
                self,
                "kernel_init",
                nn.initializers.variance_scaling(
                    1.0, "fan_in", "truncated_normal", dtype=self.dtype
                ),
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
            0-d scalar when ``aux_loss_coeff > 0``, ``None`` otherwise.
        """
        assert (
            inputs.ndim == 3
        ), f"_MoEBlock expects [batch, sequence, hidden] input, got shape {inputs.shape}"
        _, _, hidden_size = inputs.shape

        # Param registrations stay in the Flax module scope. The functional
        # ``moe(...)`` calls custom-partitioned EP and grouped-GEMM primitives
        # directly, so the FFN body no longer wraps the layer params in a
        # separate mapping transform.
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
                nn.with_logical_partitioning(self.bias_init, ("exp",)),
                (self.num_experts,),
                jnp.float32,
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
            self.ffn_quantizer_set,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            activation_type=self.activation_type,
            score_function=self.score_function,
            use_pre_softmax=self.use_pre_softmax,
            num_groups=self.num_groups,
            group_topk=self.group_topk,
            scaling_factor=self.scaling_factor,
            aux_loss_coeff=self.aux_loss_coeff,
            apply_topk_weights_early=self.apply_topk_weights_early,
            align_size=self.align_size,
            ep_axis=ep_axis,
            data_parallelism_axes=self.data_parallelism_axes,
            input_axes=self.input_axes,
            gate_kernel_axes=self.gate_kernel_axes,
            wi_kernel_axes=self.wi_kernel_axes,
            wo_kernel_axes=self.wo_kernel_axes,
            dtype=self.dtype,
        )
