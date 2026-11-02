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

# Checkpoint for FP8 delayed scaling uses pickle
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1

# Iterate over Flash Attention versions
sm_arch=`python3 -c "import torch; sm = torch.cuda.get_device_capability(0); print(sm[0]*10+sm[1])"`
export FLASH_ATTN_CUDA_ARCHS=$sm_arch
# FA2: routine tests are in L0 and L1; sm90 and sm100+; non-CP and CP
#      targeted tests are here; e.g. 2.8.4 pinned commit, non-CP
# FA3: routine tests are here; sm90 only; non-CP and CP
# FA4: routine tests are here; sm100+ only; non-CP and CP
FA2_SOURCE_VERSION="2.8.4"
FA2_SOURCE_COMMIT="1fc0221183e799363001394a12dd9e78e15f5dda"
FA2_SOURCE_GENCODE_ARCHS="80 90 100 110 120 121"
# FA2 names only family base archs, and a base build covers the rest of its major
# family, so sm103/sm107 run the compute_100f cubin that "100" produces.
fa2_arch=$(( sm_arch / 10 * 10 ))
case " $FA2_SOURCE_GENCODE_ARCHS " in
  *" $sm_arch "*) fa2_arch=$sm_arch ;;
esac
FA2_versions=()
case " $FA2_SOURCE_GENCODE_ARCHS " in
  *" $fa2_arch "*)
    FA2_versions=("$FA2_SOURCE_VERSION")
    export FLASH_ATTN_CUDA_ARCHS=$fa2_arch
    [ "$fa2_arch" = "$sm_arch" ] \
      || echo "Building FA2 for sm${fa2_arch} on sm${sm_arch} (FA2 has no --gencode for sm${sm_arch})"
    ;;
  *) echo "Skipping FA2 on sm${sm_arch} as FA2 has no --gencode for this arch" ;;
esac
CP_FA_VERSION=""
if [ $sm_arch -gt 90 ]
then
  FA_versions=("${FA2_versions[@]}" 4.0.0b33)
  CP_FA_VERSION="4.0.0b33"
elif [ $sm_arch -eq 90 ]
then
  FA_versions=("${FA2_versions[@]}" 3.0.0b1)
  CP_FA_VERSION="3.0.0b1"
else
  error_exit "No L3 FlashAttention generation is defined for sm${sm_arch}"
fi

for fa_version in "${FA_versions[@]}"
do

  # The FA distributions share the flash_attn namespace. Keep exactly one
  # installed so import-time discovery and the iteration label cannot disagree.
  pip3 uninstall -y flash-attn flash-attn-3 flash-attn-4 \
    || error_exit "Failed to isolate Flash Attention $fa_version"
  export NVTE_FLASH_ATTN_V2=0
  export NVTE_FLASH_ATTN_V3=0
  export NVTE_FLASH_ATTN_V4=0

  # Build Flash Attention
  if [ "${fa_version}" \< "3.0.0" ]
  then
    export NVTE_FLASH_ATTN_V2=1
    if [ "${fa_version}" = "$FA2_SOURCE_VERSION" ]
    then
      pip3 install --no-build-isolation \
        "git+https://github.com/Dao-AILab/flash-attention.git@${FA2_SOURCE_COMMIT}" \
        || error_exit "Failed to install Flash Attention $fa_version from ${FA2_SOURCE_COMMIT}"
    else
      pip3 install flash-attn==${fa_version} --no-build-isolation \
        || error_exit "Failed to install Flash Attention $fa_version"
    fi
    installed=$(python3 -c "import flash_attn; print(flash_attn.__version__)") \
      || error_exit "Failed to import Flash Attention $fa_version"
    if [ "$installed" != "${fa_version}" ]
    then
      error_exit "Flash Attention installed is $installed but this iteration expects ${fa_version}"
    fi
  elif [[ "${fa_version}" == 4.* ]]
  then
    export NVTE_FLASH_ATTN_V4=1
    # FA4 is intentionally last in every version array. CUTLASS DSL 4.8.0 supports
    # SM107, so replace the image-matched stack only for this final
    # iteration; later iterations would otherwise need that stack restored.
    pip3 uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
      nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 \
      || error_exit "Failed to isolate CUTLASS DSL for Flash Attention $fa_version"
    pip3 install flash-attn-4==${fa_version} nvidia-cutlass-dsl[cu13]==4.8.0 \
      --no-build-isolation || error_exit "Failed to install Flash Attention $fa_version"
  else
    export NVTE_FLASH_ATTN_V3=1
    # FA3 source build (~20 min). Skip if FA3 is already installed.
    if python3 -c "import flash_attn_3" 2>/dev/null; then
      echo "FA3 already installed (from base image); skipping source build"
    else
      git clone https://github.com/Dao-AILab/flash-attention.git
      cd flash-attention/hopper && python setup.py install \
        || error_exit "Failed to install Flash Attention $fa_version"
      cd ../../
    fi
  fi

  # Ensure local test utils is found before nvidia-cutlass-dsl's utils package
  export PYTHONPATH=$TE_PATH/tests/pytorch:${PYTHONPATH:-}

  # Run tests
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  echo "Detected $NUM_GPUS GPU(s)"

  # Suffix junit XMLs with the FA version so per-iteration results are preserved
  # (otherwise pytest.xml is overwritten on each loop iteration and we lose timing
  # data for all but the last FA version).
  fa_tag="${fa_version//./_}"
  XML_ATTN="$XML_LOG_DIR/pytest_test_attention_fa${fa_tag}.xml"
  XML_CP="$XML_LOG_DIR/pytest_test_attention_with_cp_fa${fa_tag}.xml"

  # test_attention.py reloads its own trusted delayed-scaling FP8 checkpoint,
  # whose legacy extra state requires an explicit pickle opt-in.
  if [ -n "$CP_FA_VERSION" ] && [ "$fa_version" = "$CP_FA_VERSION" ]; then
    echo "Running CP tests with FA $fa_version (CP version for sm$sm_arch)"
    if [ "$NUM_GPUS" -ge 5 ]; then
      CP_NUM_GPUS=$(( NUM_GPUS - 1 > 4 ? 4 : NUM_GPUS - 1 ))
      CP_GPUS=$(seq -s, 1 $CP_NUM_GPUS)
      echo "Running tests in parallel: test_attention.py on GPU 0, test_attention_with_cp.py on GPUs $CP_GPUS ($CP_NUM_GPUS GPUs)"

      CUDA_VISIBLE_DEVICES=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 python3 -m pytest -v -s \
        --junitxml=$XML_ATTN \
        $TE_PATH/tests/pytorch/attention/test_attention.py &
      PID_ATTN=$!

      CUDA_VISIBLE_DEVICES=$CP_GPUS NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s \
        --junitxml=$XML_CP \
        $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py &
      PID_CP=$!

      wait $PID_ATTN || test_fail "test_attention.py (FA $fa_version)"
      wait $PID_CP || test_fail "test_attention_with_cp.py (FA $fa_version)"
    else
      echo "Running tests sequentially: need >=5 GPUs for parallel execution (1 for test_attention + 4 for test_attention_with_cp)"
      NVTE_TORCH_COMPILE=0 NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 python3 -m pytest -v -s --junitxml=$XML_ATTN $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py (FA $fa_version)"
      NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_CP $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py || test_fail "test_attention_with_cp.py (FA $fa_version)"
    fi
  else
    if [ -n "$CP_FA_VERSION" ]; then
      echo "Skipping CP tests for FA $fa_version (CP uses FA $CP_FA_VERSION on sm$sm_arch)"
    else
      echo "CP tests are not scheduled for the FA generation on sm$sm_arch"
    fi
    NVTE_TORCH_COMPILE=0 NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 python3 -m pytest -v -s --junitxml=$XML_ATTN $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py (FA $fa_version)"
  fi

  if [[ "${fa_version}" == 4.* ]]; then
    XML_BACKEND="$XML_LOG_DIR/pytest_test_attention_backend_selection_fa${fa_tag}.xml"
    NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml="$XML_BACKEND" \
      "$TE_PATH/tests/pytorch/attention/test_attention_backend_selection.py" \
      || test_fail "FA4 backend-selection tests (FA $fa_version)"
  fi
done

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
