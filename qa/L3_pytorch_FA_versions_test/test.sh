# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1

# Limit parallel build jobs to avoid overwhelming system resources
export MAX_JOBS=4

# Iterate over Flash Attention versions
sm_arch=`python -c "import torch; sm = torch.cuda.get_device_capability(0); print(sm[0]*10+sm[1])"`
if [ $sm_arch -gt 90 ]
then
  FA_versions=(2.7.3)
else
  FA_versions=(2.1.1 2.3.0 2.4.1 2.5.7 2.7.3 3.0.0b1)
fi

for fa_version in "${FA_versions[@]}"
do

  # Build Flash Attention
  if [ "${fa_version}" \< "3.0.0" ]
  then
    pip install flash-attn==${fa_version}
  else
    pip install "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    mkdir -p $python_path/flashattn_hopper
    wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py
  fi

  # Run tests
  NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py

done
