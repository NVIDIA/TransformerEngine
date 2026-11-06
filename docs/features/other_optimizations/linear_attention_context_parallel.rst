..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

GDN context parallelism
=======================

PyTorch :class:`~transformer_engine.pytorch.GatedDeltaNetAttention` supports
contiguous context parallelism (CP). Each rank receives local Q, K, V and gate
tensors. TE exchanges recurrent-state summaries so local attention produces
the same result as processing each complete document. Outputs stay on the
input token owners; there is no output all-gather.

Set ``cp_group`` at construction, or call ``set_context_parallel_group`` before
forward. ``None`` disables CP. A group of size one follows the local path.
Group rank order defines token order within a document. All ranks must use
matching options and participate in forward and backward, including ranks
with no tokens or no local loss.

Input ownership
---------------

For CP size ``P`` and group rank ``r``:

* **THD:** rank ``r`` owns ``[r*T//P:(r+1)*T//P]`` of the global packed token
  buffer. Pass the same global ``int32`` cumulative document lengths as
  ``global_cu_seqlens`` on every rank. This metadata can be on CPU or CUDA;
  local fragment boundaries are derived automatically.
* **BSHD/SBHD:** rank ``r`` owns ``[r*S//P:(r+1)*S//P]`` of every batch row.
  With equal local sequence lengths, TE infers ``S``. For uneven slices,
  also pass global offsets ``[0,S,2*S,...,B*S]``.

Q, K, V, ``g`` and ``beta`` must follow the same ownership. TE does not
redistribute inputs or interpret SDPA's dual-chunk layout. A local conversion
between BSHD/SBHD and THD does not change which tokens a rank owns.

Document lengths need not be divisible by CP size. THD sequences are packed
without gaps; the caller removes physical padding before calling LA. Empty
documents are allowed and retain their initial state.

For example, lengths ``[8,4,12]`` with CP size four give:

.. code-block:: text

    rank 0: A[0:6]
    rank 1: A[6:8], B[0:4]
    rank 2: C[0:6]
    rank 3: C[6:12]

Only A and C need cross-rank state propagation. Independent documents never
share memory. The initial collective can still broadcast unused summary slots.

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    # cp is an initialized torch.distributed process group. The input pipeline
    # supplies this rank's token interval for Q/K/V and both gates.
    attn = te.GatedDeltaNetAttention(
        num_attention_heads=32,
        kv_channels=128,
        qkv_format="thd",
        cp_group=cp,
    )
    global_cu = torch.tensor([0, 8, 12, 24], dtype=torch.int32)
    output = attn(
        q_local, k_local, v_local, g=g_local, beta=beta_local,
        global_cu_seqlens=global_cu,
    )

Recurrent states and gradients
------------------------------

The optional ``initial_state`` is FP32 with shape
``[global_documents, local_heads, value_dim, key_dim]`` and identical values
on every CP rank. Requested final states have the same global shape and are
replicated using a separate collective. Empty documents return their seed.

Each initial-state gradient is returned on its document's first owning rank,
with zeros elsewhere. Sum these contributions over CP when inspecting a
replicated leaf state's full gradient. This ownership also permits state
chaining between CP calls.

Final-state backward sums gradients from all replicas. Divide a loss applied
identically to replicated final states by CP size to count it once. A loss
over each rank's local attention outputs needs no such adjustment.

Execution
---------

For a fragment, cuDNN produces the affine map ``S_out = S_in @ T + H``.
TE all-gathers the required summaries and composes them in document order to
obtain local starting states. Local cuDNN attention then runs on every rank.
Backward uses a reverse summary exchange to propagate state gradients.
Composition uses FP32 arithmetic in the common library.

GDN CP supports FP16/BF16 inputs, FP32 gates and states, Q/K normalization,
activation checkpointing and tensor parallelism. Head dimensions and device
support follow the available cuDNN frontend engines. Where a dedicated summary
engine is unavailable, ordinary GDN state passes provide a slower fallback.

CP runs eagerly, including inside ``torch.compile`` models, and supports
first-order gradients. Full-graph compilation and CUDA graph capture are
outside this implementation's supported contract. Other LA flavors, direct
dual-chunk ownership, headwise CP and whole-document redistribution are
separate features.
