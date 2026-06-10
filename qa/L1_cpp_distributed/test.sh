# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

# Find TE
: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

TE_LIB_PATH=$(pip3 show transformer-engine | grep -E "Location:|Editable project location:" | tail -n 1 | awk '{print $NF}')
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

if [[ $(nvidia-smi --list-gpus | wc -l) -ge 4 ]]; then
    cd $TE_PATH/tests/cpp_distributed
    cmake -GNinja -S. -Bbuild || test_fail "configure"
    cmake --build build || test_fail "build"

    # test_comm_gemm: per-rank XML to avoid a write race on a shared path.
    mpirun --allow-run-as-root --np 4 --oversubscribe bash -c \
        "exec ./build/test_comm_gemm --gtest_output=xml:$XML_LOG_DIR/cpp_distributed_test_comm_gemm.rank\${OMPI_COMM_WORLD_RANK}.xml" \
        || test_fail "test_comm_gemm"

    # EP suites; runner self-skips on pre-Hopper GPUs.
    GTEST_XML_PREFIX="$XML_LOG_DIR/cpp_distributed_test_ep" \
        bash ./run_test_ep.sh 4 ./build || test_fail "test_ep"
fi

if [ "$RET" -ne 0 ]; then
    echo "FAILED sub-tests:$FAILED_CASES"
fi
exit $RET
