#!/usr/bin/env bash
# Build TransformerEngine and C++ tests in Docker, then run tests.
# Usage:
#   ./docker_build_and_test.sh                    # build, run all operator tests
#   ./docker_build_and_test.sh --clean            # clean + build + run all tests
#   ./docker_build_and_test.sh --gtest_filter="*Swizzle*"
#   ./docker_build_and_test.sh --clean --gtest_filter="OperatorTest/SwizzleTestSuite*"
#
# Lint only specific files (paths relative to repo root, space-separated):
#   LINT_FILES="transformer_engine/common/swizzle/swizzle.cu transformer_engine/common/include/transformer_engine/swizzle.h" ./docker_build_and_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="nvcr.io/nvidia/pytorch:25.02-py3"
MOUNT="${SCRIPT_DIR}:/workspace/TransformerEngine"

DO_CLEAN="0"
TEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      DO_CLEAN="1"
      shift
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

docker run --gpus all -it --rm \
  -v "${MOUNT}" \
  -e DO_CLEAN="${DO_CLEAN}" \
  -e LINT_FILES="${LINT_FILES:-}" \
  "${IMAGE}" \
  bash -c '
    set -e
    cd /workspace/TransformerEngine

    if [ "${DO_CLEAN}" = "1" ]; then
      echo "=== Cleaning build artifacts ==="
      rm -rf build/ tests/cpp/build/ *.so libtransformer_engine.so *.egg-info
      find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
      find . -name "*.so" -type f -delete 2>/dev/null || true
      pip uninstall -y transformer-engine 2>/dev/null || true
    fi

    echo "=== Building TransformerEngine ==="
    NVTE_CUDA_ARCHS="89" MAX_JOBS=4 pip install --no-build-isolation -v -e .

    echo "=== Building C++ tests ==="
    cd tests/cpp
    cmake -GNinja -Bbuild .
    cmake --build build

    echo "=== Lint checks (C++ and Python) ==="
    cd /workspace/TransformerEngine
    if [ -n "${LINT_FILES}" ]; then
      pip3 install cpplint==1.6.0 pylint==3.3.1 -q
      for f in ${LINT_FILES}; do
        [ -f "$f" ] || continue
        case "$f" in
          *.cu|*.cuh|*.c|*.cpp|*.h|*.hpp|*.cc|*.cxx) echo "cpplint $f"; python3 -m cpplint --root=transformer_engine/common/include "$f" ;;
          *.py) echo "pylint $f"; python3 -m pylint "$f" ;;
          *) echo "skip (unknown type) $f" ;;
        esac
      done
    else
      TE_PATH=/workspace/TransformerEngine bash qa/L0_pytorch_lint/test.sh
      TE_PATH=/workspace/TransformerEngine bash qa/L0_jax_lint/test.sh
    fi

    echo "=== L0_* tests ==="
    # for d in qa/L0_*/; do
    #   echo "--- $d ---"
    #   (cd /workspace/TransformerEngine && TE_PATH=/workspace/TransformerEngine bash "$d/test.sh")
    # done

    echo "=== Running operator tests ==="
    cd /workspace/TransformerEngine/tests/cpp
    ./build/operator/test_operator "$@"
  ' _ "${TEST_ARGS[@]}"