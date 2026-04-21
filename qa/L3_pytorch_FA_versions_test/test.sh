# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.2.1

# Limit parallel build jobs to avoid overwhelming system resources
export MAX_JOBS=32

# Iterate over Flash Attention versions
sm_arch=`python3 -c "import torch; sm = torch.cuda.get_device_capability(0); print(sm[0]*10+sm[1])"`
export FLASH_ATTN_CUDA_ARCHS=$sm_arch
if [ $sm_arch -gt 90 ]
then
  FA_versions=(2.8.3 4.0.0b8)
elif [ $sm_arch -eq 90 ]
then
  FA_versions=(2.7.3 2.8.3 3.0.0b1 4.0.0b8)
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

  # Run tests
  NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/pytorch/attention/test_attention.py

done
