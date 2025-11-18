#!/bin/bash

set -eoxu pipefail

export NVTE_PYTORCH_FORCE_BUILD=TRUE
export NVTE_NO_LOCAL_VERSION=1
export NVTE_PYTORCH_FORCE_CXX11_ABI=$CXX11_ABI
export PIP_CONSTRAINT=

pip install wheel packaging nvidia-mathdx ninja pybind11

# 5h timeout since GH allows max 6h and we want some buffer
EXIT_CODE=0
timeout 5h python setup.py bdist_wheel --dist-dir=dist || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    wheel_name=$(python -c "import setup; print(setup.get_wheel_url()[1])" | tail -n 1)
    ls dist/*whl |xargs -I {} mv {} dist/${wheel_name}
    echo "wheel_name=${wheel_name}" | tee -a "$GITHUB_OUTPUT"
fi

echo $EXIT_CODE
