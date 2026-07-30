# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single-process check for the EP dispatch custom_partitioning rule.

The EP FFI needs a NCCL bootstrap to compile or run (see test_multi_process_ep.py
for functional coverage), so this drives the partition rule directly to confirm
it admits a tp-sharded hidden dim and propagates it to recv_tokens.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from transformer_engine.jax.cpp_extensions.ep import EpDispatchPrimitive
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

# partition only reads arg_infos[i].sharding.
_ArgInfo = namedtuple("_ArgInfo", ["sharding"])


def test_ep_dispatch_partition_hidden_tp():
    """Dispatch accepts a tp-sharded hidden dim and recv_tokens inherits it."""
    if len(jax.devices()) < 4:
        pytest.skip("needs >= 4 devices for a (ep=2, tp=2) mesh")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), axis_names=("ep", "tp"))
    # tokens [B, S, H] with H sharded on tp; routing tensors share the ep leading axis.
    arg_infos = (
        _ArgInfo(NamedSharding(mesh, P("ep"))),  # handle_mem
        _ArgInfo(NamedSharding(mesh, P("ep", None, None))),  # topk_idx
        _ArgInfo(NamedSharding(mesh, P("ep", None, "tp"))),  # tokens
        _ArgInfo(NamedSharding(mesh, P("ep", None, None))),  # topk_weights
    )
    with mesh, global_shard_guard(MeshResource(ep_resource="ep", tp_resource="tp")):
        _, _, out_shardings, _ = EpDispatchPrimitive.partition(
            8, 0, 128, True, mesh, arg_infos, None
        )

    recv_tokens_sharding, recv_topk_weights_sharding = out_shardings
    assert recv_tokens_sharding.spec == P("ep", None, "tp")
    assert recv_topk_weights_sharding.spec == P("ep")
