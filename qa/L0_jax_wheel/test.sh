# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip install wheel

cd $TE_PATH
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-jax

VERSION=`cat $TE_PATH/build_tools/VERSION.txt`
WHL_BASE="transformer_engine-${VERSION}"

# Core wheel.
NVTE_RELEASE_BUILD=1 python setup.py bdist_wheel
wheel unpack dist/*
sed -i "s/Name: transformer-engine/Name: transformer-engine-cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
sed -i "s/Name: transformer_engine/Name: transformer_engine_cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu12-${VERSION}.dist-info"
wheel pack ${WHL_BASE}
rm dist/*.whl
mv *.whl dist/
NVTE_RELEASE_BUILD=1 NVTE_BUILD_METAPACKAGE=1 python setup.py bdist_wheel

cd transformer_engine/jax
NVTE_RELEASE_BUILD=1 python setup.py sdist

pip install dist/*
cd $TE_PATH
pip install dist/*.whl --no-deps

python $TE_PATH/tests/jax/test_sanity_import.py
