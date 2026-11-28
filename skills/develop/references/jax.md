# JAX conventions

JAX code is traced and compiled: avoid retracing, host synchronization, and layout changes XLA
cannot fuse.

- **Primitive**: a `BasePrimitive` subclass in `jax/cpp_extensions/` with `abstract`, `lowering`
  (via `ffi.ffi_lowering`), `impl`, `batcher`, `partition`, and `shardy_sharding_rule`, finished
  with `register_primitive`. Never omit the batching and sharding rules; reuse `get_padded_spec`
  and `check_valid_batch_dims`. Partition with `custom_partitioning` and Shardy (implement
  `shardy_sharding_rule`) unless the code owner specifies otherwise; `shard_map` is accepted but
  less preferred. Copy the closest primitive, such as `softmax.py`.
- **FFI handler** in `jax/csrc/extensions/`: `XLA_FFI_DEFINE_HANDLER_SYMBOL` with `FFI::Bind()`,
  `FFI_CudaGraph_Traits` when the op is CUDA graph capturable, registered in `pybind.cpp`, ending
  with `ffi_with_cuda_error_check()`. Put one-time setup in an initialize handler. XLA owns all
  buffers, so request scratch space as a workspace output instead of allocating. Alias an output to
  an input where possible with `operand_output_aliases` in `ffi.ffi_lowering`, as the amax buffers
  do.
- **No host reads of array values** in API code (for example `.item()`, `.tolist()`, `np.asarray`,
  Python branching on them): they sync eagerly and fail under `jit`. If truly unavoidable, get the
  code owner's confirmation first and comment why. Check implicit transfers with
  `jax.transfer_guard("disallow")`. Pass static configuration as hashable primitive attributes.
- **No needless transposes, reshapes, casts, or copies** around a custom call.
- **API**: expose functions through `jax.custom_vjp` with static configuration in
  `nondiff_argnums`, as `dense.py` does (non-array values only; pass arrays as inputs and return
  `None` as their cotangent). Flax modules call these functions instead of reimplementing them.
  Keep functions pure: no in-place mutation, no new global state.
- **Sharding and quantization**: use `jax/sharding.py` (`MeshResource`,
  `with_sharding_constraint_by_logical_axes`) in public API code instead of raw mesh and
  `PartitionSpec` handling (primitive `partition` rules are the exception). Quantize through
  `jax/quantize/` (`Quantizer`, `ScaledTensor`, `ScalingMode`) and dispatch on the scaling mode,
  not on tensor types.
