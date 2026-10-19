# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Qualified SM100 target: provision compatible FE/DSL and build TE with CuTeDSL.
set -euo pipefail
ulimit -c 0
: "${TE_PATH:=/opt/transformerengine}"
: "${XML_LOG_DIR:=/logs}"
mkdir -p "$XML_LOG_DIR"
export NVTE_ENABLE_CUTEDSL_BACKEND=1
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
export NVTE_FUSED_ATTN_FROST=2
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

python3 - <<'PREFLIGHT'
from importlib.metadata import version
from packaging.version import Version
import torch
import cudnn
import tvm_ffi
from cutlass.base_dsl.version_info import CUDA_VERSION
import transformer_engine
import transformer_engine.pytorch
assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0), "FROST QA requires SM100"
assert Version(version("nvidia-cutlass-dsl")) >= Version("4.8.0"), "FROST QA requires CuTeDSL >= 4.8"
assert CUDA_VERSION.major >= 13, "FROST QA requires the CUDA 13 CuTeDSL compiler flavor"
print("FROST QA imports:", transformer_engine.__file__, cudnn.__file__, tvm_ffi.__file__)
PREFLIGHT

python3 -m pytest -v --tb=auto \
    --junitxml="$XML_LOG_DIR/pytest_test_frost_attn.xml" \
    "$TE_PATH/tests/pytorch/attention/test_frost_attn.py" \
    "$TE_PATH/tests/pytorch/attention/test_frost_attn_fwd.py" \
    "$TE_PATH/tests/pytorch/attention/test_frost_attn_fwd_contract.py"

# Reject a green all-skipped run or selection that omitted native-route guards.
python3 - "$XML_LOG_DIR/pytest_test_frost_attn.xml" <<'RESULTS'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
assert len(cases) >= 104, "FROST QA did not execute the complete regression suite"
assert not any(child.tag in ("failure", "error", "skipped") for case in cases for child in case), "FROST QA requires zero failures, errors or skips"
names = [case.get("name", "") for case in cases]
for kind in ("q_length", "kv_length", "q_origin", "kv_origin", "graph_terminal"):
    assert f"test_frost_d128_nozero_metadata_guard[{kind}]" in names, "Missing isolated metadata-guard test: " + kind
assert "test_frost_d256_bottom_right_q_longer_fallback" in names
for kind in ("q_origin", "kv_origin", "q_limit", "kv_limit", "q_decrease", "q_base", "graph_origin"):
    assert f"test_frost_forward_metadata_guard[{kind}]" in names, "Missing isolated forward guard: " + kind
assert "test_frost_forward_compact_decline" in names
for prefix, count in (("test_frost_forward_known_contract", 3),
                      ("test_frost_forward_declines_optional_features", 10),
                      ("test_frost_forward_declines_unknown_contract", 5),
                      ("test_frost_forward_retains_feature_guards", 7)):
    assert sum(name.startswith(prefix + "[") for name in names) == count, "Missing forward contract tests: " + prefix
print("FROST QA passed:", len(cases), "tests, zero skips, all twelve isolated guards")
RESULTS
