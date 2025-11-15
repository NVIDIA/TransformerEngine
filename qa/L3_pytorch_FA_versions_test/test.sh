# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  FA_versions=(2.8.3)
elif [ $sm_arch -eq 90 ]
then
  FA_versions=(2.7.3 2.8.3 3.0.0b1)
fi

for fa_version in "${FA_versions[@]}"
do

  # Build Flash Attention
  if [ "${fa_version}" \< "3.0.0" ]
  then
    pip3 install flash-attn==${fa_version} --no-build-isolation
  else
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/hopper && python setup.py install
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    mkdir -p $python_path/flash_attn_3
    cp flash_attn_interface.py $python_path/flash_attn_3/
    cd ../../
  fi

  # Run tests
  NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/pytorch/attention/test_attention.py

done
