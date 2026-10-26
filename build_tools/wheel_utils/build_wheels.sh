# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

PLATFORM=${1:-manylinux_2_28_x86_64}
BUILD_METAPACKAGE=${2:-true}
BUILD_COMMON=${3:-true}
BUILD_PYTORCH=${4:-true}
BUILD_JAX=${5:-true}
CUDA_MAJOR=${6:-12}

export NVTE_RELEASE_BUILD=1
export NVTE_WITH_CUTEDSL=${NVTE_WITH_CUTEDSL:-0}
export PIP_CONSTRAINT=""
export TARGET_BRANCH=${TARGET_BRANCH:-}
mkdir -p /wheelhouse/logs

# Generate wheels for common library.
git config --global --add safe.directory /TransformerEngine
cd /TransformerEngine
git checkout $TARGET_BRANCH
git submodule update --init --recursive

# Install deps
/opt/python/cp310-cp310/bin/pip install cmake pybind11[global] ninja setuptools wheel 'nvidia-cudnn-frontend>=1.25.0' 'apache-tvm-ffi>=0.1.12'

if $BUILD_METAPACKAGE ; then
        cd /TransformerEngine
        NVTE_BUILD_METAPACKAGE=1 /opt/python/cp310-cp310/bin/python setup.py bdist_wheel 2>&1 | tee /wheelhouse/logs/metapackage.txt
        mv dist/* /wheelhouse/
fi

if $BUILD_COMMON ; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine-${VERSION}"

        if [[ "$NVTE_WITH_CUTEDSL" != "0" ]]; then
                WHEEL_PYTHON_TAG=cp310
                WHEEL_ABI_TAG=abi3
        else
                WHEEL_PYTHON_TAG=py3
                WHEEL_ABI_TAG=none
        fi

        # Create the wheel.
        /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --verbose --python-tag=$WHEEL_PYTHON_TAG --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt

        # Repack the wheel for specific cuda version.
        /opt/python/cp310-cp310/bin/wheel unpack dist/*
        # From python 3.10 to 3.11, the package name delimiter in metadata got changed from - (hyphen) to _ (underscore).
        sed -i "s/Name: transformer-engine/Name: transformer-engine-cu${CUDA_MAJOR}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        sed -i "s/Name: transformer_engine/Name: transformer_engine_cu${CUDA_MAJOR}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu${CUDA_MAJOR}-${VERSION}.dist-info"
        if [[ "$NVTE_WITH_CUTEDSL" != "0" ]]; then
                # Build with CuTeDSL support: uses the CPython 3.10+ Stable ABI.
                sed -i "s/Tag: cp310-cp310/Tag: cp310-abi3/g" "${WHL_BASE}/transformer_engine_cu${CUDA_MAJOR}-${VERSION}.dist-info/WHEEL"
        else
                # Build without CuTeDSL support: remain python version agnostic.
                sed -i "s/Tag: cp310-cp310/Tag: py3-none/g" "${WHL_BASE}/transformer_engine_cu${CUDA_MAJOR}-${VERSION}.dist-info/WHEEL"
        fi
        /opt/python/cp310-cp310/bin/wheel pack ${WHL_BASE}

        # Rename the wheel to match the tag written above.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}_cu${CUDA_MAJOR}-${whl_parts[1]}-${WHEEL_PYTHON_TAG}-${WHEEL_ABI_TAG}-${whl_parts[4]}"
        rm -rf $WHL_BASE dist
        mv *.whl /wheelhouse/"$whl_name_target"
fi

if $BUILD_PYTORCH ; then
	cd /TransformerEngine/transformer_engine/pytorch
	/opt/python/cp310-cp310/bin/pip install torch
	/opt/python/cp310-cp310/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/torch.txt
	cp dist/* /wheelhouse/
fi

if $BUILD_JAX ; then
	cd /TransformerEngine/transformer_engine/jax
	/opt/python/cp310-cp310/bin/pip install "jax[cuda${CUDA_MAJOR}_local]" jaxlib
	/opt/python/cp310-cp310/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
	cp dist/* /wheelhouse/
fi
