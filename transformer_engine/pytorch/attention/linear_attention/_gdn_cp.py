# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Local cuDNN GDN operations for an external context-parallel orchestrator.

This experimental interface owns neither collectives nor process groups. Inputs
use packed THD layout with expanded Q/K heads, normalized Q/K, FP32 gates, and
physical int32 sequence boundaries. State matrices use [sequence, head, value,
key] order. The caller validates sequence boundaries and supplies causal states.
"""

import torch


class GDNContextParallelBackend:
    """Stateless local operations; head geometry is determined by each input."""

    def __init__(self):
        from cudnn.linear_attention import ops

        required = ("gated_delta_net", "gated_delta_net_summary", "gated_delta_net_summary_bwd")
        if not all(callable(getattr(ops, name, None)) for name in required) or not hasattr(
            torch.ops.cudnn, "gated_delta_net_bwd"
        ):
            raise RuntimeError(
                "GDN chunkwise CP requires cuDNN forward/backward state summary APIs"
            )
        self._ops = ops

    def forward_summary(self, k, v, g, beta, cu_seqlens):
        """Return (update, transition), with S_out = S_in @ transition + update."""
        return self._ops.gated_delta_net_summary(k, v, g, beta, cu_seqlens)

    def forward(self, q, k, v, g, beta, cu_seqlens, *, scale, initial_state):
        """Return local output for the supplied incoming states."""
        output, _ = self._ops.gated_delta_net(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=False,
        )
        return output

    def backward_summary(self, q, k, g, beta, output_grad, cu_seqlens, *, scale):
        """Return (gamma, transition), with dS_in = dS_out @ transition + gamma."""
        return self._ops.gated_delta_net_summary_bwd(
            q,
            k,
            g,
            beta,
            output_grad.contiguous(),
            cu_seqlens,
            scale=scale,
            output_transition=True,
        )

    def backward(
        self,
        output_grad,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        *,
        scale,
        initial_state,
        final_state_grad,
    ):
        """Return dQ, dK, dV, dG and dBeta, including the outgoing state gradient."""
        return torch.ops.cudnn.gated_delta_net_bwd(
            output_grad.contiguous(),
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            d_final_state=final_state_grad,
        )[:5]
