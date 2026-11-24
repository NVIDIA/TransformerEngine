---
name: develop
description: >-
  Write or change Transformer Engine code (common C++/CUDA, PyTorch, JAX) so it reuses existing
  infrastructure, sits in the layer that owns it, and reads like the surrounding code. Use before
  adding an operation, module, binding, kernel, primitive, option, or test.
license: Apache-2.0
---

# Develop Transformer Engine code

Find what already exists before writing code, and let the nearest sibling set the structure.
`AGENTS.md` stays the authority for architecture, build, test, and contribution rules.

## Rules

- Search by behavior (verbs, shapes, dtypes), not only by the name you would give the code. Follow
  the closest existing operation, module, binding, kernel, or test. If something new is added, say
  in the PR why nothing existing fit.
- Extend before adding: a new argument, case, or dtype on an existing function beats a sibling
  function. Add a helper only when none fits, next to its siblings (C++: `common/common.h`,
  `common/util/`).
- Decide dtype, recipe, and architecture behavior in one place (type-switch macro, capability
  function, or registry), not at every call site. Derive shape-specific behavior from the inputs.
- Reuse an existing `NVTE_*` variable or argument before adding another; a new variable needs an
  entry in `docs/envvars.rst`.
- Validate inputs at the boundary: `NVTE_CHECK(cond, "message ", value, ".")` in C++;
  `ValueError`, `TypeError`, or `RuntimeError` naming the offending value in Python. Reject
  unsupported input instead of silently falling back to another path.
- Keep functions focused, with no unused options or commented-out code. Prefer deleting code to
  adding it.

## References

Read only what the task touches.

- `transformer_engine/pytorch/` or its bindings: [references/pytorch.md](references/pytorch.md)
  (CPU overhead, host syncs, copies, fusible operations, quantizers).
- `transformer_engine/jax/` or its bindings: [references/jax.md](references/jax.md) (primitives,
  FFI handlers, `custom_vjp`, sharding, quantization).
- Adding or changing any unit test: [references/tests.md](references/tests.md). Add tests for new
  behavior, few and consolidated.
- Reviewing instead of writing: use the `review` skill.
