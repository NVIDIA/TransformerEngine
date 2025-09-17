# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -ex

# Paths
TMP_DIR=$(mktemp --directory)
rm -rf x86_wheelhouse
mkdir x86_wheelhouse

# CUDA 12 wheels
docker build \
       --no-cache \
       --tag x86_cu12_wheel \
       --file build_tools/wheel_utils/Dockerfile.x86 \
       --build-arg CUDA_VERSION_MAJOR=12 \
       --build-arg CUDA_VERSION_MINOR=3 \
       --build-arg BUILD_METAPACKAGE=true \
       --build-arg BUILD_COMMON=true \
       --build-arg BUILD_PYTORCH=true \
       --build-arg BUILD_JAX=true \
       .
docker run --runtime=nvidia --gpus=all --ipc=host x86_cu12_wheel
docker cp $(docker ps -aq | head -1):/wheelhouse ${TMP_DIR}
cp -r ${TMP_DIR}/wheelhouse/* x86_wheelhouse
rm -rf ${TMP_DIR}/wheelhouse

# CUDA 13 wheels
docker build \
       --no-cache \
       --tag x86_cu13_wheel \
       --file build_tools/wheel_utils/Dockerfile.x86 \
       --build-arg CUDA_VERSION_MAJOR=13 \
       --build-arg CUDA_VERSION_MINOR=0 \
       --build-arg BUILD_METAPACKAGE=false \
       --build-arg BUILD_COMMON=true \
       --build-arg BUILD_PYTORCH=false \
       --build-arg BUILD_JAX=false \
       .
docker run --runtime=nvidia --gpus=all --ipc=host x86_cu13_wheel
docker cp $(docker ps -aq | head -1):/wheelhouse ${TMP_DIR}
cp -r ${TMP_DIR}/wheelhouse/* x86_wheelhouse
rm -rf ${TMP_DIR}/wheelhouse

# Clean up
rm -rf ${TMP_DIR}
