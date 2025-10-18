# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Remove leftovers.
rm -rf x86_wheelhouse_cu12 x86_wheelhouse_cu13

# CUDA 12.
docker build --no-cache \
  --build-arg CUDA_MAJOR=12 \
  --build-arg CUDA_MINOR=3 \
  --build-arg BUILD_METAPACKAGE=true \
  --build-arg BUILD_COMMON=true \
  --build-arg BUILD_PYTORCH=true \
  --build-arg BUILD_JAX=true \
  -t "x86_wheel" -f build_tools/wheel_utils/Dockerfile.x86 .
docker run --runtime=nvidia --gpus=all --ipc=host "x86_wheel"
docker cp $(docker ps -aq | head -1):/wheelhouse x86_wheelhouse_cu12

# CUDA 13.
docker build --no-cache \
  --build-arg CUDA_MAJOR=13 \
  --build-arg CUDA_MINOR=0 \
  --build-arg BUILD_METAPACKAGE=false \
  --build-arg BUILD_COMMON=true \
  --build-arg BUILD_PYTORCH=false \
  --build-arg BUILD_JAX=false \
  -t "x86_wheel" -f build_tools/wheel_utils/Dockerfile.x86 .
docker run --runtime=nvidia --gpus=all --ipc=host "x86_wheel"
docker cp $(docker ps -aq | head -1):/wheelhouse x86_wheelhouse_cu13
