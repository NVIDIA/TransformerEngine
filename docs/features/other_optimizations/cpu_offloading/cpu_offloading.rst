..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

*Figure 1. Traditional PCIe system vs GB200 Superchip with NVLink-C2C.*

Traditional **PCIe Gen5 x16** systems offer **128 GB/s** bidirectional bandwidth
between CPU and GPU, which limits offloading benefits.

With **NVLink-C2C** (GB200), bandwidth jumps to **900 GB/s** bidirectional per link,
making offloading increasingly attractive on modern NVIDIA superchips.
The GB200 pairs a Grace CPU with 480 GB LPDDR5X memory and two Blackwell GPUs,
each with 192 GB HBM3e (384 GB total), providing ample CPU memory for offloading
activations.

Offloading/reloading consumes HBM bandwidth, which may compete with
other GPU operations — even when transfers are asynchronous.
This is unlikely to affect compute-bound operations like GEMMs, but the impact on
memory-bound operations like quantization may be noticeable.


CPU Offloading in Transformer Engine
------------------------------------

Transformer Engine supports CPU offloading of activations for **sequential models**. By sequential, we mean that the model is a sequence of layers, where each layer
consumes the output of the previous one — which is the case for most LLM architectures. These layers may be any PyTorch modules and not just TE layers.

.. raw:: html
   :file: img/layer_sequence.svg

*Figure 2. CPU offloading supports sequential layer pipelines (top), but not graphs with branching or merging (bottom). Note that inside the layer, arbitrary control flow is allowed.*

The example below shows how to offload activations for a sequence of ``torch.nn.Linear`` layers using the default scheduling algorithm:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_basic_offload_example.py
         :language: python
         :start-after: # START_BASIC_EXAMPLE
         :end-before: # END_BASIC_EXAMPLE



Let's take a look at the API in detail:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

          def get_cpu_offload_context(
              enabled: bool = False,
              num_layers: Optional[int] = 1,
              model_layers: int = 1,
              manual_synchronization: bool = False,
              offload_stream: Optional[torch.cuda.Stream] = None,
              # ... (legacy parameters omitted, see API reference)
          ) -> Union[Tuple[ContextManager, Callable], Tuple[ContextManager, Callable, ManualOffloadSynchronizer]]:
              ...

The ``model_layers`` parameter must always be set to the total number of layers in the model.
There are two modes of operation:

1. **Default scheduling** — set ``num_layers`` to the number of layers to offload.
   The algorithm automatically schedules offload/reload operations to overlap with computation.

2. **Manual synchronization** — set ``manual_synchronization=True`` (``num_layers`` is ignored in this mode).
   This mode provides explicit control over when to start offload/reload using the returned ``ManualOffloadSynchronizer``.

The :func:`transformer_engine.pytorch.get_cpu_offload_context` function returns:

- **context manager** — wrap each layer's forward pass with it to enable activation capture.
- **sync function** — call on the output tensor after each layer, as shown in the example below.


Default Offloading Scheduling
-----------------------------

Default scheduling is enabled when ``manual_synchronization=False`` (the default).
The ``num_layers`` parameter must be specified to set the number of layers to offload.
The algorithm then automatically determines when to offload and reload activations
to maximize overlap with computation.

For ``num_layers`` layers offloaded of ``model_layers`` layers:

- First ``num_layers`` layers are offloaded to CPU.
- Offloading starts as soon as tensors are saved for backward — it does not wait
  for the layer's forward pass to complete.
- At most ``(model_layers - num_layers)`` sets of activations are on GPU at any time;
  both compute and reload may be stalled to enforce this limit.
- Reloading must complete by the time the tensor is needed for the layer's backward pass.
- ``num_layers`` must be at most ``model_layers - 1`` (setting it to ``model_layers``
  raises an assertion error). However, ``model_layers - 1`` leaves only 1 activation set
  on GPU at a time — compute and transfers cannot overlap, and a warning is raised.
  For full overlap, use ``model_layers - 2`` or less.

Specifying a low enough ``num_layers`` enables full overlap of computation
and offload/reload. The following two scenarios illustrate this — one with full overlap, and one with stalls.

.. raw:: html
   :file: img/scheduling.svg

*Figure 3. With* ``num_layers=2`` *and* ``model_layers=5`` *, at most 3 sets of activations are on GPU. Layer 1 offloading starts during its forward pass (when the first tensor is saved for backward). Offloading fully overlaps with forward, reloading fully overlaps with backward.*

When ``num_layers`` is too high, the GPU memory limit forces stalls:

.. raw:: html
   :file: img/scheduling_stall.svg

*Figure 4. With* ``num_layers=3`` *and* ``model_layers=5`` *, at most 2 sets of activations can be on GPU (5-3=2), which causes stalls. In forward, Layer 4 cannot start until Layer 2 is offloaded, otherwise there would be 3 sets of activations on GPU (Layers 2, 3, 4). In backward, Layer 3 cannot start immediately — its activations are still on CPU and must be reloaded first. Some tensors may finish reloading earlier, allowing parts of the layer (e.g., a sublayer) to run while the rest waits. The same applies to Layers 2 and 1.*


Manual Synchronization
----------------------

For custom scheduling, set ``manual_synchronization=True``
and pass a custom ``offload_stream``. This returns a ``ManualOffloadSynchronizer``
with explicit control over transfers and allows synchronization via stream operations.

This mode is useful when training does not follow the standard "all forwards then all backwards"
pattern — for example, in pipeline parallelism. Having access to the ``offload_stream`` enables
custom synchronization logic (e.g., waiting, recording events) tailored to the specific workload.

The ``ManualOffloadSynchronizer`` object provides the following methods:

- ``start_offload_layer(layer_id)`` — queue async GPU→CPU copies on the offload stream.
  Before each copy, the offload stream waits for an event recorded when that tensor
  was saved for backward.
- ``release_activation_forward_gpu_memory(layer_id)`` — wait for the offload to complete
  and release GPU memory.
- ``start_reload_layer(layer_id)`` — queue async CPU→GPU copies on the offload stream.
  When tensors are accessed in backward, compute stream waits for each tensor's reload
  to complete.

To skip offloading for a specific layer, simply do not call any of these methods for that layer.

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

      .. literalinclude:: pytorch_manual_offload_example.py
         :language: python
         :start-after: # START_MANUAL_EXAMPLE
         :end-before: # END_MANUAL_EXAMPLE


CPU Offloading and CUDA Graphs
------------------------------

CPU offloading works with CUDA graphs — async copies and stream synchronization
are GPU operations that can be captured and replayed, even when accessing
pinned CPU memory (via PCIe DMA, without CPU involvement).

.. note::

   We recommend capturing the entire forward and backward pass in a single graph.
   Async copy operations (offload/reload) must complete within the same graph where
   they started. If the graph ends before copies finish, PyTorch will block waiting
   for them, defeating the purpose of graph capture.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_cuda_graphs_example.py
         :language: python
         :start-after: # START_CUDA_GRAPHS_EXAMPLE
         :end-before: # END_CUDA_GRAPHS_EXAMPLE

Caveats
-------

.. warning::

   **Heuristic activation detection**:

   CPU Offloading is implemented using
   `PyTorch saved tensors hooks <https://docs.pytorch.org/docs/stable/notes/autograd.html#saved-tensors-hooks-doc>`_.
   PyTorch saves various tensors for backward — not just activations, but also weights and other data.

   Activation detection is heuristic: all CUDA tensors that are not ``torch.nn.Parameter`` are offloaded.
   For TE layers, tensors that should not be offloaded are manually excluded.
   For non-TE layers, no such exclusions exist, so some tensors may remain pinned in GPU memory
   even after being copied to CPU (e.g., if the layer stores references in ``ctx``),
   resulting in wasted bandwidth with no memory savings.

   To exclude specific tensors from offloading, use :func:`mark_not_offload`:

   .. code-block:: python

      from transformer_engine.pytorch import mark_not_offload
      mark_not_offload(tensor)

.. warning::

   **Memory layout changes**:

   Offloading/reloading can change tensor memory layout and relations:

   1. Views of the same storage may be restored as separate allocations.
   2. Adjacent tensors may not be adjacent after reload.

   CUDA kernels that rely on specific memory layout may produce unexpected results.
   To mitigate (1), non-trivial views are excluded from offloading by default.
   TE attention kernels are an exception — they use internal handling that is tested and supported.
   Issue (2) is not mitigated — custom kernels that assume adjacent tensors share
   contiguous memory may still fail.

   If you encounter layout-related issues, use :func:`mark_not_offload` to exclude
   problematic tensors from offloading.
