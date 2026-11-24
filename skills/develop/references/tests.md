# Unit test conventions

Add the fewest tests that pin the change. CI time is a shared budget, and every added case runs on
every pipeline.

## Whether and where

- Add a test for each new behavior, failure mode, or fixed bug, and only for that. Do not test
  unchanged paths or repeat existing coverage; a pure refactor needs no new test.
- Extend the existing file and class for the area (for example `tests/pytorch/test_fusible_ops.py`,
  `tests/pytorch/test_numerics.py`, `tests/jax/test_custom_call_compute.py`,
  `tests/cpp/operator/`). A new file is for a new component only, and must be registered or CI
  never runs it: PyTorch and distributed JAX files are listed one by one in the `qa/L0_*` and
  `qa/L1_*` launchers, C++ files in `tests/cpp/operator/CMakeLists.txt`; other JAX tests run from
  the whole `tests/jax` directory. Add a distributed test only when a partition rule or a
  communication path changes.
- A fixed bug gets one regression test, at the lowest level that reproduces it.

## Keep the suite small

- Fold new cases into an existing parametrized test by adding a value or parameter, not a sibling
  test.
- Cover what the change can break: boundary shapes (empty, one, not a multiple of the tile or block
  size), the affected dtypes and recipes, and forward plus backward when both change. Pick the axes
  the change touches instead of the full dtype x recipe x shape x layout product.
- Use small shapes, and share setup and reference computation across cases. Prefer one multi-case
  test over several that each repeat an expensive setup (distributed launch, CUDA graph capture,
  large module build).

## Writing the test

- Reuse helpers instead of local copies: `tests/pytorch/utils.py` (`dtype_tols`, `assert_close`,
  `assert_close_grads`, `reset_rng_states`) and the file's `make_reference_and_test_tensors`;
  `tests/jax/utils.py` (`assert_allclose`, `dtype_tols`); `tests/cpp/test_common.h`.
- Follow "Numerical and Backend Validation" in `AGENTS.md` for references, tolerances, quantized
  operands, and skip reasons. Check gradients when the backward changes.
- Make the test able to fail: prefill outputs with a sentinel such as NaN, compare so a NaN cannot
  pass, and give each new invalid-input path a rejection test.
- Seed deterministically and do not depend on test order or global state.
- Name each test for the behavior it checks, small enough to diagnose from its name.
