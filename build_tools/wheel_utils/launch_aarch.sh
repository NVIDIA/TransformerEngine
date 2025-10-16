# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Remove leftovers.
rm -rf aarch_wheelhouse_cu12 aarch_wheelhouse_cu13

# CUDA 12.
docker build --no-cache \
  --build-arg CUDA_MAJOR=12 \
  --build-arg CUDA_MINOR=3 \
  --build-arg BUILD_METAPACKAGE=false \
  --build-arg BUILD_COMMON=true \
  --build-arg BUILD_PYTORCH=false \
  --build-arg BUILD_JAX=false \
  -t "aarch_wheel" -f build_tools/wheel_utils/Dockerfile.aarch .
docker run --runtime=nvidia --gpus=all --ipc=host "aarch_wheel"
docker cp $(docker ps -aq | head -1):/wheelhouse aarch_wheelhouse_cu12

# CUDA 13.
docker build --no-cache \
  --build-arg CUDA_MAJOR=13 \
  --build-arg CUDA_MINOR=0 \
  --build-arg BUILD_METAPACKAGE=false \
  --build-arg BUILD_COMMON=true \
  --build-arg BUILD_PYTORCH=false \
  --build-arg BUILD_JAX=false \
  -t "aarch_wheel" -f build_tools/wheel_utils/Dockerfile.aarch .
docker run --runtime=nvidia --gpus=all --ipc=host "aarch_wheel"
docker cp $(docker ps -aq | head -1):/wheelhouse aarch_wheelhouse_cu13
