# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

docker build --no-cache -t "aarch_wheel" -f build_tools/wheel_utils/Dockerfile.aarch .
docker run --runtime=nvidia --gpus=all --ipc=host "aarch_wheel"
rm -rf aarch_wheelhouse
docker cp $(docker ps -aq | head -1):/wheelhouse/ aarch_wheelhouse
