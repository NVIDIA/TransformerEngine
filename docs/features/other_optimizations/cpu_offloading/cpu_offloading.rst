..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

CPU Offloading
===================================

.. note::

    CPU Offloading in Transformer Engine is currently available only for **PyTorch**.
    It supports all PyTorch modules, not just TE layers.

CPU offloading moves activation tensors from GPU to CPU memory during the
forward pass and reloads them during backward. Transfers are **asynchronous**,
enabling significant GPU memory savings with minimal overhead.

Unlike activation checkpointing, offloading avoids recomputation — activations
are stored on CPU instead of being recalculated, making it faster when
CPU-GPU bandwidth is sufficient.


Hardware Support
----------------

CPU offloading benefits greatly from fast CPU-GPU interconnects.
The faster the link, the more effectively transfer time can be hidden
behind computation.

.. raw:: html
   :file: img/pcie_vs_nvlink.svg

*Figure 1. Traditional PCIe system vs GH200 Superchip with NVLink-C2C.*

Traditional **PCIe Gen5 x16** systems offer **128 GB/s** bidirectional bandwidth
between CPU and GPU, which limits offloading benefits.

With **NVLink-C2C** (GH200), bandwidth jumps to **900 GB/s** bidirectional,
making offloading increasingly attractive on modern NVIDIA superchips.

Note that offloading/reloading consumes HBM bandwidth, which may compete with
computation — even when transfers are asynchronous. At full speed, this takes
up to **900 GB/s** of HBM bandwidth. However, GH200's HBM3e provides **~4.9 TB/s**,
so offloading/reloading uses less than 20%, making the impact on compute minimal.

CPU Offloading in Transformer Engine
------------------------------------

Transformer Engine supports activations CPU offloading for sequences of layers, where each layer
consumes the output of the previous one. Note that these layers don't need to be TE layers —
they can be arbitrary PyTorch modules. Let's look at the API:

.. code-block:: python

    def get_cpu_offload_context(
        enabled: bool = False,
        num_layers: int = 1,
        model_layers: int = 1,
        manual_synchronization: bool = False,
        retain_pinned_cpu_buffers: bool = False,
        offload_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[ContextManager, Callable, Optional[ManualOffloadSynchronizer]]:
        ...

You need to specify the total number of layers in the model by setting the ``model_layers`` argument.
Then, you can specify how many layers to offload by setting the ``num_layers`` argument.
Due to the scheduling algorithm, you only need to specify a low enough number of layers to offload to
enable full overlap of computation and offload/reload.

**Default scheduling algorithm**

For ``num_layers`` layers offloaded of ``model_layers`` layers:

- First ``num_layers`` layers are offloaded to CPU.
- Offloading starts as tensors are saved for backward.
- At most ``(model_layers - num_layers)`` sets of activations are on GPU at any time;
  compute may be stalled to enforce this limit.
- Reloading of the tensor must end by the time the tensor is needed for the backward pass of the layer.

Below we present two example scenarios — one with full overlap of computation and offload/reload, and one with stalls.
Let's see the first scenario:

.. raw:: html
   :file: img/scheduling.svg

*Figure 2. With* ``num_layers=2`` *, at most 3 sets of activations are on GPU. Offloading fully overlaps with forward, reloading fully overlaps with backward.*

When ``num_layers`` is too high, the GPU memory limit forces stalls. Let's see an example:

.. raw:: html
   :file: img/scheduling_stall.svg

*Figure 3. With* ``num_layers=3`` *and* ``model_layers=5`` *, at most 2 sets of activations can be on GPU (5−3=2), which causes stalls.*

In this case:

- **Forward**: Layer 4 cannot start until Layer 2 is offloaded, otherwise there would be
  3 sets of activations on GPU (Layers 2, 3, 4).
- **Backward**: Layer 3 backward cannot start immediately — its activations are still
  on CPU and must be reloaded first. Note that some tensors may finish reloading earlier,
  allowing parts of the layer (e.g., a sublayer) to run while the rest waits.
  The same applies to Layers 2 and 1.

Example
-------

The :func:`transformer_engine.pytorch.get_cpu_offload_context` function returns:

- **context manager** — wrap each layer's forward pass to enable activation capture.
- **sync function** — call after each layer on the output tensor as shown in the example below.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: basic_offload_example.py
         :language: python
         :start-after: # START_BASIC_EXAMPLE
         :end-before: # END_BASIC_EXAMPLE


Manual Synchronization
----------------------

For custom scheduling (e.g. in pipeline parallelism), set ``manual_synchronization=True``
and pass your own ``offload_stream``. This gives you a ``ManualOffloadSynchronizer``
with explicit control over transfers, and lets you synchronize via stream operations.

The ``ManualOffloadSynchronizer`` object provides the following methods:

- ``start_offload_layer(layer_id)`` — begin async copy to CPU.
- ``release_activation_forward_gpu_memory(layer_id)`` — free GPU memory after offload completes.
- ``start_reload_layer(layer_id)`` — begin async copy back to GPU.

.. warning::

   Never call ``release_activation_forward_gpu_memory()`` before the offload completes.
   Always synchronize the offload stream first, otherwise data may be corrupted.

.. tabs::

   .. tab:: PyTorch

      The example demonstrates:
      
      1. **Forward pass**: After each layer, call ``start_offload_layer(i)`` to begin
         async copy of layer ``i``'s activations to CPU.
      2. **Release GPU memory**: Call ``offload_stream.synchronize()`` to wait for all
         offloads to finish, then ``release_activation_forward_gpu_memory(i)`` to free
         the GPU tensors.
      3. **Before backward**: Call ``start_reload_layer(i)`` to begin async reload.
         The compute stream will automatically wait for each tensor to be reloaded
         before it's accessed in backward.

      .. literalinclude:: manual_offload_example.py
         :language: python
         :start-after: # START_MANUAL_EXAMPLE
         :end-before: # END_MANUAL_EXAMPLE


CPU Offloading and CUDA Graphs
------------------------------

CPU offloading works with CUDA graphs — async copies and stream synchronization
are GPU operations that can be captured and replayed, even when accessing
pinned CPU memory (via PCIe DMA, without CPU involvement).

.. note::

   The entire forward and backward pass must be captured in a single graph.
   Per-layer graph capture is not supported due to cross-layer synchronization.

.. note::

   Allocating pinned CPU memory is currently not graphable. Use
   ``retain_pinned_cpu_buffers=True`` and run a warm-up iteration before
   capture to pre-allocate buffers that are reused during replay.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: cuda_graphs_example.py
         :language: python
         :start-after: # START_CUDA_GRAPHS_EXAMPLE
         :end-before: # END_CUDA_GRAPHS_EXAMPLE

Caveats
-------

.. warning::

   **Memory layout changes**:

   Offloading/reloading can change tensor memory layout and relations:

   - Adjacent tensors (e.g., ``a`` and ``b``) may not be adjacent after reload.
   - Views of the same storage may be restored as separate allocations.

   To mitigate this, we skip offloading non-trivial views (except for TE
   attention kernels, which are tested and supported). Custom kernels
   relying on memory layout may still fail.

