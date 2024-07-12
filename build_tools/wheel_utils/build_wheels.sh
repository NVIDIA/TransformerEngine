# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

export NVTE_RELEASE_BUILD=1
export TARGET_BRANCH=${TARGET_BRANCH:-wheels}

git config --global --add safe.directory /TransformerEngine
cd /TransformerEngine
git checkout $TARGET_BRANCH
git submodule update --init --recursive && \
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=manylinux_2_28_x86_64

# Set target name for build distribution.
mkdir /wheelhouse
whl_name=$(basename dist/*)
IFS='-' read -ra whl_parts <<< "$whl_name"
whl_name_target="${whl_parts[0]}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
mv dist/"$whl_name" /wheelhouse/"$whl_name_target"

# PyTorch sdist
cd transformer_engine/pytorch
/opt/python/cp38-cp38/bin/pip install torch
/opt/python/cp38-cp38/bin/python setup.py sdist
cp dist/* /wheelhouse/

# JAX sdist
cd ../jax
/opt/python/cp38-cp38/bin/pip install jax jaxlib
/opt/python/cp38-cp38/bin/python setup.py sdist
cp dist/* /wheelhouse/

