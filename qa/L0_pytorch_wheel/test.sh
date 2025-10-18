# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

: "${TE_PATH:=/opt/transformerengine}"

pip3 install wheel || error_exit "Failed to install wheel"

cd $TE_PATH
pip3 uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch || error_exit "Failed to uninstall transformer-engine transformer-engine-cu12 transformer-engine-torch"

VERSION=`cat $TE_PATH/build_tools/VERSION.txt`
WHL_BASE="transformer_engine-${VERSION}"

# Core wheel.
NVTE_RELEASE_BUILD=1 pip3 wheel --no-build-isolation -vvv --wheel-dir ./dist . || error_exit "Failed to setup bdist_wheel"
wheel unpack dist/${WHL_BASE}-* || error_exit "Failed to unpack dist/${WHL_BASE}-*.whl"
sed -i "s/Name: transformer-engine/Name: transformer-engine-cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
sed -i "s/Name: transformer_engine/Name: transformer_engine_cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu12-${VERSION}.dist-info" || error_exit "Failed to move ${WHL_BASE}.dist-info to transformer_engine_cu12-${VERSION}.dist-info"
wheel pack ${WHL_BASE} || error_exit "Failed to pack ${WHL_BASE}"
rm dist/*.whl || error_exit "Failed to remove dist/*.whl"
mv *.whl dist/ || error_exit "Failed to move *.whl to dist/"
NVTE_RELEASE_BUILD=1 NVTE_BUILD_METAPACKAGE=1 pip3 wheel --no-build-isolation --no-deps -vvv --wheel-dir ./dist . || error_exit "Failed to setup metapackage"

cd transformer_engine/pytorch
NVTE_RELEASE_BUILD=1 pip3 wheel --no-build-isolation --no-deps -vvv --wheel-dir ./dist . || error_exit "Failed to setup sdist"

pip3 install --no-build-isolation --no-deps -vvv dist/* || error_exit "Failed to install dist/*"
cd $TE_PATH
pip3 install --no-build-isolation --no-deps -vvv dist/*.whl || error_exit "Failed to install dist/*.whl --no-deps"

python3 $TE_PATH/tests/pytorch/test_sanity_import.py || test_fail "test_sanity_import.py"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
