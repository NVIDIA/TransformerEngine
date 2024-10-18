# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

# Install dependencies
# Note: Need to install wheel locally since PaddlePaddle container
# already contains APT install.
pip install pydantic
pip install --user wheel==0.44.0

cd $TE_PATH
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-paddle

VERSION=`cat $TE_PATH/build_tools/VERSION.txt`
WHL_BASE="transformer_engine-${VERSION}"

# Core wheel.
NVTE_RELEASE_BUILD=1 python setup.py bdist_wheel
python -m wheel unpack dist/*
sed -i "s/Name: transformer-engine/Name: transformer-engine-cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
sed -i "s/Name: transformer_engine/Name: transformer_engine_cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu12-${VERSION}.dist-info"
python -m wheel pack ${WHL_BASE}
rm dist/*.whl
mv *.whl dist/
NVTE_RELEASE_BUILD=1 NVTE_BUILD_METAPACKAGE=1 python setup.py bdist_wheel
pip install dist/*.whl --no-deps

cd transformer_engine/paddle
NVTE_RELEASE_BUILD=1 python setup.py bdist_wheel
pip install dist/*

python $TE_PATH/tests/paddle/test_sanity_import.py
