Project and Architecture
========================

This section describes where Transformer Engine fits in the software stack,
how its major pieces work together, and how those boundaries shape the code.

Role in the software stack
--------------------------

Transformer Engine provides optimized building blocks for Transformer models
on NVIDIA GPUs. It is not a complete LLM training toolkit. Higher-level
frameworks and training toolkits compose its operations and modules into full
models and own the orchestration of the training system.

At a high level, Transformer Engine has two pieces:

.. list-table::
   :header-rows: 1
   :widths: 16 24 44 16

   * - Piece
     - Location
     - Responsibilities
     - API surface
   * - Common layer
     - ``transformer_engine/common/``
     - Implements framework-independent functionality. Performance-critical
       operations are written in C++ and CUDA and use CUDA libraries such as
       cuBLASLt and cuDNN where appropriate. Shared Python code provides
       functionality used by both framework integrations. The layer exposes
       focused operations with well-defined inputs and outputs, such as
       quantization, GEMM, and RMSNorm forward and backward passes.
     - C API and shared Python API
   * - Framework-aware layer
     - ``transformer_engine/pytorch/`` and ``transformer_engine/jax/``
     - Integrates the common operations with PyTorch and JAX. It adapts
       framework tensors and execution models, manages framework-visible state,
       and composes common operations into larger components. This layer also
       presents features such as quantization recipes.
     - Framework-specific Python APIs

The binding code sits at the boundary between the two pieces. It translates
framework objects and execution conventions into calls to the C API while
keeping the common layer independent of PyTorch and JAX.

Scope boundary
~~~~~~~~~~~~~~

Most Transformer Engine functionality operates at or below the level of an
individual Transformer layer. The project provides optimized operations and
modules that a larger training system can combine, rather than owning the
entire model or training architecture.

This boundary is especially important for distributed execution. Transformer
Engine may implement or integrate parallel techniques that affect its
operations and layers, but model-level orchestration belongs to higher-level
toolkits. Pipeline parallelism, for example, is outside Transformer Engine's
scope and is the responsibility of a toolkit such as Megatron.

Choosing where a change belongs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This division provides a useful starting point when locating or designing a
change:

* Framework-independent computation and GPU execution belong in the common
  layer and should be exposed through the C API when needed by a frontend.
  Framework-independent Python functionality shared by both frontends also
  belongs in the common layer.
* Framework tensor handling, execution semantics, state management, recipes,
  and composition into user-facing modules belong in the framework-aware
  layer.
* Features that span both pieces should keep the reusable computation in the
  common layer and place framework-specific policy and integration in the
  corresponding frontend.

Repository structure
--------------------

The following map focuses on the durable areas that help contributors decide
where to look or make a change. Detailed subsystem maps belong in the
implementation chapters.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Location
     - Purpose
   * - ``transformer_engine/common/``
     - Framework-independent C++, CUDA, and shared Python functionality, as
       described above.
   * - ``transformer_engine/pytorch/``
     - PyTorch integration and framework-specific Python API.
   * - ``transformer_engine/jax/``
     - JAX integration and framework-specific Python API.
   * - ``transformer_engine/debug/``
     - Numerical debugging tools. The design is intended to be
       cross-framework, although the current implementation supports PyTorch
       only.
   * - ``tests/``
     - The test implementations, organized into C++, distributed C++, PyTorch,
       and JAX areas.
   * - ``qa/``
     - Launcher scripts that select, configure, and run validation in the CI
       test classes. L0 jobs, such as ``L0_pytorch_unittest``, run whenever CI
       is launched. L1, L2, and L3 classes are reserved for less frequent
       nightly or weekly validation.
   * - ``benchmarks/``
     - Performance benchmarks and profiling utilities, kept separate from
       correctness tests.
   * - ``docs/``
     - Sphinx documentation sources, including the user guide, API reference,
       developer documentation, and the tutorials and notebooks under ``docs/examples/``.
   * - ``examples/``
     - Runnable standalone PyTorch and JAX examples. These focus on executable
       code with little accompanying explanation, unlike the tutorials under
       ``docs/examples/``.
   * - ``build_tools/`` and top-level packaging files
     - Helpers for building the common library and framework extensions,
       version handling, and wheel creation. ``setup.py``, ``pyproject.toml``,
       and ``MANIFEST.in`` provide the top-level build and packaging entry
       points.
   * - ``3rdparty/``
     - External source dependencies managed as Git submodules, including cuDNN
       Frontend, CUTLASS, GoogleTest, and NCCL.

Architecture topics
-------------------

.. toctree::
   :maxdepth: 1

   layered_architecture
   pytorch_linear_walkthrough

Working architectural principles
--------------------------------

.. important::

   The statements below are working proposals, not settled project policy.
   They summarize the architecture described so far and must be revisited as
   the remaining subsystem documentation is developed. Before they become
   normative, each statement should be checked against multiple subsystems and
   any necessary exceptions should be documented.

Candidate invariants
~~~~~~~~~~~~~~~~~~~~

* **The common layer is framework-independent.** It may contain C++, CUDA,
  shared Python, Triton, and other kernel technologies, but it must not depend
  on PyTorch or JAX.
* **Framework-aware layers do not contain GPU kernels.** In particular, they
  should not require NVCC. Kernel implementations belong in the common layer.
* **Framework integrations use the public C API for compiled common
  functionality.** The same API can be used without PyTorch or JAX.
* **Common operations are focused and caller-directed.** The caller selects an
  operation and supplies policy such as the quantization type; the common layer
  selects the concrete kernel implementation.
* **Composite operations belong in the framework-aware layer.** Modules such
  as ``Linear`` interpret recipes and options, then coordinate framework-native
  operations, common operations, communication, and state.
* **Tensor memory is normally owned by the framework.** Outputs, workspaces,
  scales, amax values, and saved state should be regular framework tensors
  passed to common operations.
* **Common-layer device allocation is exceptional.** It is appropriate only
  when the required allocation cannot be obtained through a framework API,
  such as certain symmetric-memory allocations.
* **Execution plans and library handles belong to the common layer.** They are
  implementation details of the kernels and libraries managed there and may
  be cached by that layer.
* **Public APIs are compatibility boundaries.** Existing C APIs remain
  available until the next major version when a replacement is necessary.
  Public Python APIs evolve compatibly, normally through optional keyword
  arguments.
* **Private framework bindings are not compatibility boundaries.** PyTorch C++
  extensions, JAX FFI bindings, and their internal callers may evolve together.

Candidate design guidance
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Expose equivalent intent through framework-native constructs.** PyTorch
  and JAX should provide broadly symmetrical functionality without being
  forced into identical APIs.
* **Place shared concepts in the common layer.** A concept such as a
  quantization recipe should have one framework-independent definition, while
  each frontend determines how to realize it during execution.
* **Separate policy from mechanism.** User-visible choices and multi-operation
  composition belong in the framework-aware layer; low-level dispatch belongs
  in the common layer.
* **Prefer framework-native functionality when it is sufficient.** Common
  implementations are most appropriate when specialized kernels are needed or
  when communication and computation must be designed together.
* **Optimize representations across the full operation lifecycle.** Forward
  execution should account for what backward will need while avoiding
  unnecessary storage, communication, transposes, and requantization.
* **Keep Transformer Engine focused on composable building blocks.**
  Functionality generally belongs at or below the Transformer-layer level
  rather than implementing a complete model-training system.
* **Distributed ownership follows the computation boundary.** Framework-native
  communication stays in the frontend; the common layer participates when
  communication is inseparable from kernel execution.
* **External facilities may still require explicit integration.** Transformer
  Engine does not implement FSDP, for example, but its modules must understand
  enough about FSDP to behave correctly with it.
