# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

docker build -t "wheel" -f build_tools/wheel_utils/Dockerfile .
docker run --runtime=nvidia --gpus=all --ipc=host "wheel"
rm -rf wheelhouse
docker cp $(docker ps -aq | head -1):/wheelhouse .

