# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

docker build --no-cache -t "x86_wheel" -f build_tools/wheel_utils/Dockerfile.x86 .
docker run --runtime=nvidia --gpus=all --ipc=host "x86_wheel"
rm -rf x86_wheelhouse
docker cp $(docker ps -aq | head -1):/wheelhouse x86_wheelhouse
