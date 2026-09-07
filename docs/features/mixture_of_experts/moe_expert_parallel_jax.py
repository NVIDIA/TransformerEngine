# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_EXPERT_PARALLEL_JAX
import jax

from transformer_engine.jax.ep import ep_bootstrap
from transformer_engine.jax.moe import (
    get_moe_recv_capacity_per_rank,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

num_experts = 8
top_k = 2
ep_axis = "ep"
ep_size = mesh.shape[ep_axis]
max_tokens_per_rank = x.shape[0] * x.shape[1] // jax.process_count()
recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
    num_experts=num_experts,
    num_experts_per_tok=top_k,
    max_tokens_per_rank=max_tokens_per_rank,
    ep_size=ep_size,
)

# Initialize EP eagerly once per process. The mesh has one device per process.
mesh_resource = MeshResource(ep_resource=ep_axis)
with mesh, global_shard_guard(mesh_resource):
    ep_bootstrap(
        world_size=jax.process_count(),
        rank=jax.process_index(),
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=x.shape[-1],
        max_token_dtype=x.dtype,
    )
record_ep_bootstrap_signature_for_moe(
    num_experts=num_experts,
    max_tokens_per_rank=max_tokens_per_rank,
    recv_capacity_per_rank=recv_capacity_per_rank,
    hidden_dim=x.shape[-1],
    ep_size=ep_size,
)

# mesh:        jax.sharding.Mesh with an "ep" axis and one device per process
# x:           [batch, seq, hidden_size], BF16 and sharded over the mesh
# gate_kernel: [hidden_size, num_experts]          router projection
# wi:          [num_experts, hidden_size, 2 * ffn]  gated FC1 (gate and value)
# wo:          [num_experts, ffn, hidden_size]      FC2
with mesh, global_shard_guard(mesh_resource):
    output, aux_loss, total_recv_tokens = moe(
        x,
        gate_kernel,
        wi,
        wo,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        activation_type="silu",
        score_function="softmax",
        aux_loss_coeff=1e-2,  # load-balancing loss; 0 disables it
        ep_axis=ep_axis,
        dtype=x.dtype,
        recv_capacity_per_rank=recv_capacity_per_rank,
    )
# output:   [batch, seq, hidden_size]
# aux_loss: scalar load-balancing loss (None when aux_loss_coeff == 0)
# total_recv_tokens: receive count before any capacity-based token dropping
# END_MOE_EXPERT_PARALLEL_JAX
