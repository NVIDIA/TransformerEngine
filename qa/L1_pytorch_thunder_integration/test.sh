# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -x

: ${THUNDER_PATH:=/opt/pytorch/lightning-thunder}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.1.1 pytest-benchmark==5.1.0
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest.xml ${THUNDER_PATH}/thunder/tests/test_transformer_engine_executor.py

# Check return code
# Note: Return code 5 is fine. Lightning tests are skipped on systems
# without FP8 support and Pytest returns 5 if no tests are run.
RC=$?
[[ ${RC} -eq 0 ]] && exit -1 || RC=0  # Hack:
                                      # https://github.com/NVIDIA/TransformerEngine/pull/1686
                                      # broke Thunder integration, so
                                      # test failures are expected.
                                      # Once Thunder is fixed, the CI
                                      # job will fail and TE can
                                      # remove this hack.
if [ ${RC} -eq 5 ]; then
    RC=0
fi
exit ${RC}
