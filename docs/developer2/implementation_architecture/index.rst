Implementation Architecture
===========================

Detailed guides to the major subsystems, their interfaces, execution flows,
invariants, common change points, validation, and pitfalls.

Planned coverage
----------------

* C and C++ core, type system, build pipeline, and kernels.
* PyTorch frontend, autograd integration, and native bindings.
* JAX frontend, module system, primitives, and FFI.
* Quantization types, recipes, layouts, and scaling modes.
* Attention dispatch, backends, kernels, and context parallelism.
* Distributed training and communication overlap.
* Operation fusing.
* Other major subsystems identified during migration.

.. TODO: Migrate the strongest material from the existing developer guide and
   split this section into subsystem pages as part of that migration.
