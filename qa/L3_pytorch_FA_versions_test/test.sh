# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

# Limit parallel build jobs to avoid overwhelming system resources
export MAX_JOBS=32

# Iterate over Flash Attention versions
sm_arch=`python3 -c "import torch; sm = torch.cuda.get_device_capability(0); print(sm[0]*10+sm[1])"`
export FLASH_ATTN_CUDA_ARCHS=$sm_arch
if [ $sm_arch -gt 90 ]
then
  FA_versions=(2.8.3 4.0.0b11)
elif [ $sm_arch -eq 90 ]
then
  FA_versions=(2.8.3 3.0.0b1 4.0.0b11)
fi

# CP tests are expensive and run only once per arch:
#   - sm90 (H100): FA3 (3.0.0b1) - context_parallel.py only supports FA3 on Hopper
#   - sm>90 (B200): latest FA4 - FA3 is not built/installed for sm>90
# Non-CP tests still run for every FA version in the array.
if [ $sm_arch -eq 90 ]; then
    CP_FA_VERSION="3.0.0b1"
else
    CP_FA_VERSION="${FA_versions[-1]}"
fi

for fa_version in "${FA_versions[@]}"
do

  # Build Flash Attention
  if [ "${fa_version}" \< "3.0.0" ]
  then
    pip3 install flash-attn==${fa_version} --no-build-isolation
  elif [[ "${fa_version}" == 4.* ]]
  then
    pip3 install flash-attn-4==${fa_version} nvidia-cutlass-dsl[cu13]==4.4.2 --no-build-isolation
  else
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/hopper && python setup.py install
    cd ../../
  fi

  # Ensure local test utils is found before nvidia-cutlass-dsl's utils package
  export PYTHONPATH=$TE_PATH/tests/pytorch:${PYTHONPATH:-}

  # Run tests
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  echo "Detected $NUM_GPUS GPU(s)"
  if [ "$fa_version" = "$CP_FA_VERSION" ]; then
    echo "Running CP tests with FA $fa_version (CP version for sm$sm_arch)"
    if [ "$NUM_GPUS" -ge 5 ]; then
      CP_NUM_GPUS=$(( NUM_GPUS - 1 > 4 ? 4 : NUM_GPUS - 1 ))
      CP_GPUS=$(seq -s, 1 $CP_NUM_GPUS)
      echo "Running tests in parallel: test_attention.py on GPU 0, test_attention_with_cp.py on GPUs $CP_GPUS ($CP_NUM_GPUS GPUs)"

      CUDA_VISIBLE_DEVICES=0 NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s \
        --junitxml=$XML_LOG_DIR/pytest.xml \
        $TE_PATH/tests/pytorch/attention/test_attention.py &
      PID_ATTN=$!

      CUDA_VISIBLE_DEVICES=$CP_GPUS NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s \
        --junitxml=$XML_LOG_DIR/pytest_test_attention_with_cp.xml \
        $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py &
      PID_CP=$!

      wait $PID_ATTN || test_fail "test_attention.py"
      wait $PID_CP || test_fail "test_attention_with_cp.py"
    else
      echo "Running tests sequentially: need >=5 GPUs for parallel execution (1 for test_attention + 4 for test_attention_with_cp)"
      NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py"
      NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_attention_with_cp.xml $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py || test_fail "test_attention_with_cp.py"
    fi
  else
    echo "Skipping CP tests for FA $fa_version (CP only runs with FA $CP_FA_VERSION on sm$sm_arch)"
    NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py (FA $fa_version)"
  fi
done

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
