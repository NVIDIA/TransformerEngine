# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

docker build -t "wheel" .
docker run -it --runtime=nvidia --gpus=all -v $(pwd)/../../:/TransformerEngine --ipc=host "wheel"
