# PyTorch conventions

The steady-state path is usually bound by CPU launch overhead: every avoidable Python statement,
ATen call, allocation, and copy costs time.

- **No host-device sync on the hot path**: any operation that makes the host wait for or read
  device data, for example `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `int()` or `bool()` of a
  device tensor, `torch.cuda.synchronize()`, boolean-mask indexing, `nonzero`, or Python branching
  on device values (not an exhaustive list). Keep data such as split sizes on the device, or take
  it from the caller as host integers. If a sync is truly unavoidable, stop and get the code
  owner's confirmation first, and comment why.
- **Verify syncs, transfer carefully**: check with `torch.cuda.set_sync_debug_mode("error")` (a
  prototype that can miss some syncs). Host copies use preallocated, reused pinned buffers with
  `non_blocking=True` (pinning per call blocks), and the destination is not read before an event or
  sync.
- **No needless copies**: `.clone()` and `.to(copy=True)` always copy, while `.contiguous()` and
  `.to()` return the input when nothing changes but still cost a call. Prefer views, call them only
  when the layout or dtype can differ, and never add a copy "to be safe". Reject an unsupported
  layout when the copy would be silent or costly.
- **Allocate less**: write into caller-provided buffers (`out=`). Prefer `torch.empty`, and use
  `zeros` only where the contents must start at zero.
- **Do per-configuration work once**: compute metadata at construction or first use and cache it,
  as the fuser caches fused operation lists and `functools.lru_cache` backs device queries. Return
  early when a step does not apply.
- **Fewer calls**: replace a Python loop over tensors, or a chain of small ATen ops, with one
  C++ binding or fused kernel (use the multi-tensor bindings for lists). Keep
  `torch.autograd.Function` argument lists short by grouping non-tensor arguments into a tuple, as
  `module/fp8_padding.py` does. In internal hot paths use the quantized storage classes when
  tensor-subclass behavior is not needed.
- **Stay compatible** with CUDA graphs (stable pointers and shapes, no sync) and `torch.compile`
  (no graph breaks). For a steady-state change, measure launch overhead on a small input with
  `benchmarks/`, not only kernel time.
- **`torch.compile` integration**: pick the lightest route that fits the op, keep eager on the
  direct path (`torch.compiler.is_compiling()`), and provide fake metadata matching real outputs
  without aliasing inputs:
  - **Reuse existing registered ops** when an external library already registers `torch.library`
    custom ops (as cuDNN Frontend linear attention in
    `pytorch/attention/linear_attention/base.py` or FlashAttention does): do not wrap it in another
    custom op; resolve the callable in `__init__` and keep surrounding Python traceable.
  - **Direct `torch.library.custom_op`** for C++/CUDA bindings with only plain `torch.Tensor` and
    primitive arguments (as `pytorch/attention/dot_product_attention/softmax.py` and
    `pytorch/permutation.py` do): register the op (`mutates_args=()` when functional), its
    `register_fake` metadata function, and `register_autograd` with `setup_context` (wrap a native
    backward kernel in a second custom op).
  - **TE dynamo adapter** (`pytorch/dynamo/`: `register_custom_op`,
    `register_custom_op_with_autograd`, as `pytorch/module/linear.py` does) when inputs or outputs
    include quantized tensors (`TensorOrQuantized`), `Quantizer` objects (value-opaque via
    `register_value_opaque_quantizer`), or `ProcessGroup`: define typed `@dataclass` forward and
    backward arg containers, caller-provided `fake_impl` / `fwd_fake_impl` + `bwd_fake_impl` using
    `TensorSpec` (or `pytorch/dynamo/fake.py`) matching all `None` slots and `ctx_attrs`, and
    `setup_context` / `setup_saved_tensors`; fall back cleanly when registration returns `None`.
  - **Fusible / `Sequential` ops** (`BasicOperation` / `FusedOperation`, as
    `pytorch/ops/basic/basic_linear.py` and `pytorch/ops/basic/activation.py` do): set
    `fwd_args_type` and `bwd_args_type` dataclasses (auto-registered per direction in
    `FusibleOperation.__init_subclass__`), implement `pack_*_args`, `*_compute` (avoiding input
    mutation when `in_custom_op=True`), `*_compute_fake` with `TensorSpec`,
    `forward_setup_context`, and `compile_unsupported_reason` for eager fallback.
- **Compose**: new behavior is a `BasicOperation` in `pytorch/ops/basic/`; a performance fusion is
  a `FusedOperation` in `pytorch/ops/fused/` registered with `register_forward_fusion` or
  `register_backward_fusion`. A basic operation does not depend on its neighbors; the fuser decides.
  Share code between modules through `pytorch/module/_common.py`, `pytorch/utils.py`, and
  `pytorch/ops/_common.py` instead of copying blocks.
- **Quantizers**: use the `Quantizer` / `QuantizedTensorStorage` interface and recipes, not
  `isinstance` chains over tensor types. For a per-type decision, add a method to the base class
  and override it, as `supports_only_rowwise_all_gather` does.
