# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1 onnxruntime==1.13.1
FA_versions=(2.0.6 2.0.6.post2 2.0.7 2.0.8 2.1.1 2.1.2.post3 2.2.0 2.2.1 2.2.2 2.2.3.post2 2.2.4 2.2.4.post1 2.2.5 2.3.0 2.3.1.post1 2.3.2 2.3.3 2.3.4 2.3.5 2.3.6 2.4.0.post1 2.4.1 2.4.2 2.4.3.post1 2.5.0 2.5.1.post1 2.5.2 2.5.3 2.5.4 2.5.5 2.5.6 2.5.7 2.5.8 2.5.9.post1 2.6.0.post1 2.6.1 2.6.2 2.6.3)
for fa_version in "${FA_versions[@]}"
do
pip install flash-attn==${fa_version}
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py
done
