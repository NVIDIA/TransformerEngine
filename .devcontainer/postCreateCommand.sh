#!/usr/bin/env bash
set -euo pipefail

# pip install ninja
NVTE_USE_CCACHE=1 NVTE_CCACHE_BIN=sccache NVTE_FRAMEWORK=pytorch NVTE_BUILD_DEBUG=1 pip install -v --no-build-isolation -e .
