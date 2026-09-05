#!/bin/bash

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -eoxu pipefail

export NVTE_PYTORCH_FORCE_BUILD=TRUE
export NVTE_NO_LOCAL_VERSION=1
export NVTE_PYTORCH_FORCE_CXX11_ABI=$CXX11_ABI
export PIP_CONSTRAINT=

pip install wheel packaging nvidia-mathdx ninja pybind11

# 5h timeout since GH allows max 6h and we want some buffer
timeout 5h python setup.py bdist_wheel --dist-dir=dist

mapfile -t wheel_paths < <(find dist -maxdepth 1 -type f -name '*.whl' -print)
if [[ ${#wheel_paths[@]} -ne 1 ]]; then
    echo "Expected one built wheel, found ${#wheel_paths[@]}" >&2
    exit 1
fi
built_wheel=${wheel_paths[0]}

if [[ -n "${AARCH:-}" ]]; then
    case "$AARCH" in
        x86_64) platform_tag=linux_x86_64 ;;
        sbsa) platform_tag=linux_aarch64 ;;
        *) echo "Unsupported wheel architecture: $AARCH" >&2; exit 1 ;;
    esac

    python_tag="cp${PYTHON_VERSION//./}"
    expected_tag="${python_tag}-${python_tag}-${platform_tag}"
    python - "$built_wheel" "$expected_tag" <<'PY'
import sys
import zipfile

wheel_path, expected_tag = sys.argv[1:]
with zipfile.ZipFile(wheel_path) as wheel:
    wheel_metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tags = {
        line.removeprefix("Tag:").strip()
        for line in wheel.read(wheel_metadata).decode().splitlines()
        if line.startswith("Tag:")
    }
if expected_tag not in tags:
    raise SystemExit(f"Expected wheel tag {expected_tag}, found {sorted(tags)}")
PY

    wheel_name=$(python -c "import setup; print(setup.get_wheel_url()[1])" | tail -n 1)
    expected_suffix="cxx11abi${CXX11_ABI}-${expected_tag}.whl"
    if [[ "$wheel_name" != *"$expected_suffix" ]]; then
        echo "Expected wheel filename suffix $expected_suffix, found $wheel_name" >&2
        exit 1
    fi
else
    wheel_name=$(python -c "import setup; print(setup.get_wheel_url()[1])" | tail -n 1)
fi

if [[ "${SMOKE_TEST:-false}" == "true" ]]; then
    release_version=${RELEASE_VERSION:-}
    release_version=${release_version#v}
    if [[ -z "$release_version" ]]; then
        echo "Release version is required for the install smoke test" >&2
        exit 1
    fi
    pip install --no-cache-dir "transformer-engine==${release_version}" "$built_wheel"

    cuda_stub_dir=$(mktemp -d)
    test -f /usr/local/cuda/lib64/stubs/libcuda.so
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so "$cuda_stub_dir/libcuda.so.1"
    (
        cd /tmp
        LD_LIBRARY_PATH="$cuda_stub_dir:${LD_LIBRARY_PATH:-}" python -c \
            "import transformer_engine.pytorch; print('Transformer Engine import OK')"
    )
fi

mv "$built_wheel" "dist/$wheel_name"
echo "wheel_name=${wheel_name}" | tee -a "$GITHUB_OUTPUT"
