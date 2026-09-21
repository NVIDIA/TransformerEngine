# Transformer Engine Agent Guide

This file provides repository-specific guidance for working on Transformer
Engine. `README.rst`, `CONTRIBUTING.rst`, `docs/`, `qa/` remain the authoritative sources for
user documentation, contribution policy, and executable CI behavior.

## Project Scope

Transformer Engine provides optimized building blocks for Transformer models
on NVIDIA GPUs. It is not a complete model-training system. Higher-level
frameworks and toolkits compose its operations and modules into models and own
model-level orchestration.

Most functionality is at or below the level of an individual Transformer
layer. Tensor, sequence, expert, and context parallel techniques may be in
scope when they affect Transformer Engine operations. Pipeline parallelism,
distributed optimizer orchestration, and complete training loops generally
belong to higher-level systems such as Megatron-LM.

Do not copy version tables into this file. Current hardware, CUDA, cuDNN,
compiler, Python, and framework requirements are maintained in `README.rst`
and `docs/installation.rst`.

## Repository Map

- `transformer_engine/common/`: framework-independent C++, CUDA, C API,
  shared Python definitions, kernel dispatch, and CUDA-library integrations.
- `transformer_engine/pytorch/`: PyTorch public API, modules, quantized tensor
  types, recipes, attention, optimizers, state, and private C++ bindings.
- `transformer_engine/jax/`: JAX and Flax public APIs, primitives, custom calls,
  quantization, state, and private C++ bindings.
- `transformer_engine/debug/`: numerical debugging and inspection features.
  Verify current framework support before extending or documenting them.
- `tests/cpp/` and `tests/cpp_distributed/`: common-library tests.
- `tests/pytorch/` and `tests/jax/`: framework-specific unit, numerical,
  integration, attention, quantization, and distributed tests.
- `qa/`: launchers for Transformer Engine's authoritative CI test suites. The
  GitHub Actions workflows expose only a subset of the full CI; use the tests
  and launchers under `qa/` to determine the coverage expected by the project's
  internal CI. `L0_*` contains the basic contribution checks, while higher
  levels contain distributed, integration, and specialized coverage.
- `benchmarks/`: performance benchmarks and profiling utilities, separate
  from correctness tests.
- `docs/`: Sphinx sources, API references, tutorials, and notebooks.
- `examples/`: runnable PyTorch and JAX examples.
- `build_tools/`, `setup.py`, `pyproject.toml`, and `MANIFEST.in`: build,
  packaging, version, and wheel infrastructure.
- `3rdparty/`: Git submodules pinned by the parent repository.

## Architecture and Ownership

Transformer Engine has a framework-independent common layer and
framework-aware PyTorch and JAX layers.

The common layer owns reusable computation, the public C API, low-level
dispatch, CUDA kernels, calls to CUDA libraries, runtime-compiled kernels, and
shared concepts used by both frontends. The framework layers adapt tensors and
execution models, manage framework-visible state, interpret recipes and
options, and compose focused operations into user-facing modules.

Place framework-independent computation in `transformer_engine/common/`. When
a frontend needs compiled common functionality that is not already readily
available in both supported frameworks, expose it through the C API or a
Python-based DSL such as Triton and adapt it in that frontend's private binding.
Prefer designs that do not re-enter Python on every steady-state invocation.
CuTeDSL kernels dispatched from common C++ may be JIT-compiled and registered
through TVM FFI. Python may participate in initial registration and compilation,
but steady-state dispatch should use the cached C++/TVM FFI function.
Framework-specific tensor handling, autograd or transformation semantics, state,
and user-facing composition belong in the corresponding frontend.

### Working architecture rules

These rules describe the intended architecture. Check them against the
subsystem being changed and document necessary exceptions.

- The common layer must not depend on PyTorch or JAX.
- GPU kernels belong in the common layer; private framework bindings translate
  framework tensor and execution conventions into common API calls.
- User-visible policy and multi-operation composition belong in a frontend;
  low-level kernel selection belongs in the common layer.
- Device memory is owned by the framework and passed to common
  operations. Common-layer device allocation is exceptional.
- Execution plans and CUDA-library handles may be owned and cached by the
  common layer.
- Framework-native communication generally stays in the frontend unless communication
  is inseparable from common computation, as in some expert-parallel and
  communication/GEMM-overlap paths.
- Public C and Python APIs are compatibility boundaries. Private framework
  bindings may evolve together with their callers.
- PyTorch and JAX should expose equivalent intent where practical, while
  retaining framework-native APIs.
- Feature parity between the PyTorch and JAX frontends is the goal, but is not a requirement.

For a cross-layer change, trace the affected path:

```text
Public framework API or module
  -> framework Python implementation
  -> private framework C++/FFI binding
  -> public common C API
  -> common dispatch and kernel implementation
```

A shared C or CUDA change requires focused common-layer validation where
applicable, plus validation through each affected frontend. Validate both
PyTorch and JAX when both use the changed behavior or dispatch path. A
frontend-only change does not require validation of the other frontend, though
shared behavior, documentation, serialization, and feature parity may still be
affected.

## Build and Environment

Use a compatible NGC PyTorch or JAX development container when possible. For a
host build, follow `docs/installation.rst`. The selected framework must be
installed before Transformer Engine because the root build imports it while
configuring the framework extension. Development builds therefore use
`--no-build-isolation`.

Initialize the pinned source dependencies before the first build:

```bash
git submodule update --init --recursive
```

The root editable build is the normal development build. Run it from the
repository root. The frontend is detected automatically but can also be selected explicitly
via NVTE_FRAMEWORK environment variable during build:

```bash
NVTE_FRAMEWORK=pytorch python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=jax python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=pytorch,jax python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=none python -m pip install -e . -v --no-build-isolation
```

Common build controls are:

- `NVTE_CUDA_ARCHS`: semicolon-separated target compute capabilities. For
  development, set this to the exact architecture being tested to avoid
  compiling unused variants. On Blackwell, use the architecture-specific
  target such as `100a` rather than the `100` family selector, which expands
  to multiple variants. Use a family selector or explicit list when building
  for multiple GPU architectures or producing portable artifacts.
- `MAX_JOBS` or `NVTE_BUILD_MAX_JOBS`: concurrent compilation jobs.
- `NVTE_BUILD_THREADS_PER_JOB`: threads within a build job.
- `NVTE_USE_CCACHE=1` and `NVTE_CCACHE_BIN`: compiler caching.
- `NVTE_CMAKE_BUILD_DIR`: alternate common-library build directory.
- `NVTE_BUILD_DEBUG=1`: debug build.
- `CUDA_HOME`, `CUDA_PATH`, `CUDNN_PATH`, and `CXX`: toolchain selection.

Before limiting compilation to a small fixed number of jobs or threads, check
the number of CPU cores available on the system and choose concurrency that
fits the current machine.

Optional native components add discovery and linking requirements:

- `NVTE_WITH_NCCL_EP`: NCCL expert parallelism; enabled by default for
  applicable Hopper-or-newer targets and requires compatible NCCL headers and
  libraries.
- `NVTE_UB_WITH_MPI=1`: MPI userbuffers bootstrap; requires `MPI_HOME`.
- `NVTE_ENABLE_NVSHMEM=1`: NVSHMEM; requires `NVSHMEM_HOME`.
- `NVTE_WITH_CUBLASMP=1`: cuBLASMp; may require `CUBLASMP_HOME`.
- `NVTE_WITH_CUSOLVERMP=1`: cuSolverMp; may require `CUSOLVERMP_HOME`.

See `docs/envvars.rst`, `setup.py`, `build_tools/`, and
`transformer_engine/common/CMakeLists.txt` for less common settings.

### Rebuild scope

- Python-only change: no rebuild after an editable install; restart processes
  that already imported the module.
- C++ change, either in common code or a framework binding: rerun the same build
  command and configuration used to build Transformer Engine before the change.
  This preserves the selected frontends, architecture targets, and optional
  features.
- Build logic, compiler flags, optional features, or framework selection:
  rerun the full editable build; use a clean build tree if its generated state
  is incompatible.
- Submodule source or revision: rebuild native targets and validate the
  dependent component.

CMake normally reuses `build/cmake`. Before removing a build tree, check
`NVTE_CMAKE_BUILD_DIR` and the verbose build output. A clean tree is commonly
needed after changing toolchains, CMake generators, architecture targets, or
optional native dependencies.

### Verify the imported build

NGC images may already contain Transformer Engine. After building, confirm that
the framework sees a GPU and that TE resolves to the current worktree:

Sandboxing may block access to GPU devices. If a GPU visibility check fails or
reports no devices, do not conclude that the system has no GPUs until the check
has also been attempted outside the sandbox with the required permission.

```bash
python -c 'import torch, transformer_engine, transformer_engine.pytorch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformer_engine.__file__)'
```

```bash
python -c 'import jax, transformer_engine, transformer_engine.jax; print(jax.__version__, jax.devices()); print(transformer_engine.__file__)'
```

## Testing

### Focused framework tests

```bash
# PyTorch
python -m pytest -v tests/pytorch/path/to/test_file.py -k 'specific_case'
python -m pytest -v tests/pytorch/test_sanity.py

# JAX: use the repository pytest configuration
python -m pytest -c tests/jax/pytest.ini -v \
    tests/jax/path/to/test_file.py -k 'specific_case'
python -m pytest -c tests/jax/pytest.ini -v tests/jax/test_layer.py
```

### Common C++ and CUDA tests

Separately built after the main build.

```bash
cmake -GNinja -S tests/cpp -B tests/cpp/build \
    -DCMAKE_CUDA_ARCHITECTURES="$NVTE_CUDA_ARCHS"
cmake --build tests/cpp/build
ctest --test-dir tests/cpp/build --output-on-failure
```

For a focused GoogleTest run:

```bash
tests/cpp/build/operator/test_operator --gtest_list_tests
tests/cpp/build/operator/test_operator --gtest_filter='*RelevantPattern*'
```

### QA launchers

`CONTRIBUTING.rst` requires the `L0_*` checks. Their scripts default to
`TE_PATH=/opt/transformerengine` and `XML_LOG_DIR=/logs`; override both for a
local checkout:

```bash
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_pytorch_unittest/test.sh
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_jax_unittest/test.sh
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_cppunittest/test.sh
```

Inspect a QA script before running it: some install dependencies or require a
particular container, GPU architecture, GPU count, or external repository.
Use the higher-level launcher for distributed, Megatron Core, FSDP, ONNX, and
attention-backend coverage rather than reconstructing its environment.

### Validation by changed area

- Shared Python: focused behavior through each affected frontend.
- PyTorch or JAX API: focused frontend test plus related integration coverage.
- Private framework binding: loaded-extension check and a focused numerical or
  integration test.
- Common C/CUDA: focused C++ operator test plus each affected frontend.
- Build or packaging: build in an appropriate clean environment, verify
  artifact paths, then use `qa/L0_pytorch_wheel/` or `qa/L0_jax_wheel/`.
- Distributed component: smallest supported multi-process case, then its QA
  launcher; record the GPU count and topology.
- Public API: tests, documentation, and compatibility across supported callers.

## Numerical and Backend Validation

For numerical or low-precision changes:

- Compare outputs and applicable gradients with an independent native-framework
  or higher-precision reference, not only shapes or successful execution.
- Cover the affected dtypes, recipes, layouts, boundary shapes, forward and
  backward paths, and GPU architectures.
- Base tolerances on the operation, dtype, accumulation, and reference. Do not
  loosen them solely to make a failing case pass.
- Record flags such as `NVTE_ALLOW_NONDETERMINISTIC_ALGO` that change expected
  numerical behavior.
- Exercise the intended backend and fallback behavior when the operation can
  use cuDNN, FlashAttention, unfused attention, Triton, NVRTC, or an
  architecture-specific kernel.
- Give skips and expected failures a capability-based or tracked-defect reason;
  do not remove a failing configuration before identifying its cause.

Choose tolerances for the quantity being compared, not merely for the lowest
precision used internally. For example, an MXFP8 GEMM that returns BF16 should
use BF16-appropriate output tolerances rather than FP8-appropriate tolerances
when its operands are chosen to be exactly representable in MXFP8. One way to
construct such operands is to generate random values, quantize and dequantize
them, and use the resulting values in both the reference and operation under
test. Apply this construction to every quantized operand, including parameters
held by a module such as weights, not only to the module's explicit inputs.

When reporting a GPU or distributed failure, include the exact command, input
shape/dtype/layout, relevant `NVTE_*` variables, GPU model and count, compute
capability, topology, and framework/CUDA/cuDNN/NCCL/driver versions. Also verify
which backend was selected and that the imported extension came from the
current worktree. Reduce failures across ranks, shapes, recipes, and backends
while preserving the observed behavior.

## Performance Changes

Use the relevant program under `benchmarks/` and report its exact command. A
performance result must identify the selected TE backend or kernel, GPU model
and count, software versions, shapes, layouts, dtypes, recipes, warm-up,
synchronization, input distribution, and measurement statistic. Separate
compile or autotune time, Python launch overhead, communication, and
steady-state kernel time where they materially affect the result.

Compare on equivalent hardware and software configurations and confirm
numerical correctness before accepting a faster result. Do not place one-off
performance measurements in correctness tests.

## Formatting and Linting

- C++ follows the Google C++ Style Guide and local conventions.
- Python is formatted with Black at line length 100.
- C++, CUDA, and headers under `transformer_engine/` use the repository
  clang-format configuration.
- CI linting uses pylint and cpplint through the framework L0 scripts.

Run pre-commit on explicit changed paths:

```bash
pre-commit run --files path/to/changed.py path/to/changed.cu
```

Run the framework lint that covers the changed code:

```bash
TE_PATH="$PWD" bash qa/L0_pytorch_lint/test.sh
TE_PATH="$PWD" bash qa/L0_jax_lint/test.sh
```

Both lint scripts also inspect common code. Use `CPP_ONLY=1` or `PYTHON_ONLY=1`
to select one half during iteration.

## Documentation

Update documentation for changes to public APIs, supported behavior, defaults,
environment variables, actionable error messages, installation/build
requirements, or performance-relevant behavior.

- Environment variables are documented in `docs/envvars.rst`.
- Standalone examples live under `examples/pytorch/` and `examples/jax/`.
- Tutorials and notebooks live under `docs/examples/`.

## Contribution Requirements

Follow `CONTRIBUTING.rst` and `.github/PULL_REQUEST_TEMPLATE.md`.

- Every PR must be reviewed by its human author before it is submitted for
  maintainer review. If an agent prepares a PR fully autonomously, the PR must
  remain a draft, and the agent must notify the user working with it that they
  need to review the PR before marking it ready for maintainer review.
- PR and commit titles use imperative mood.
- Every contributed commit requires a DCO sign-off (`git commit -s`).
- New files must satisfy `qa/L0_license/test.sh`. If a contributor retains
  separate copyright, use the documented `exclude_copyright` mechanism.
- Contributions require applicable tests and documentation, no new warnings,
  and the `L0_*` checks described in `CONTRIBUTING.rst`.

```bash
TE_PATH="$PWD" bash qa/L0_license/test.sh
```
