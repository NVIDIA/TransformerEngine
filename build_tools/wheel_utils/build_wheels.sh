# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

export NVTE_RELEASE_BUILD=1
export TARGET_BRANCH=${TARGET_BRANCH:-wheels}

git clone https://github.com/ksivaman/TransformerEngine-1.git
cd TransformerEngine-1
git checkout $TARGET_BRANCH
git submodule update --init --recursive && \
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=manylinux_2_28_x86_64

# Set target name for build distribution.
whl_name=$(basename dist/*)
IFS='-' read -ra whl_parts <<< "$whl_name"
whl_name_target="${whl_parts[0]}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
mv dist/"$whl_name" dist/"$whl_name_target"
