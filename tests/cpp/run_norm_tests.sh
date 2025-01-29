# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


if [ -z "$OUTPUT_FILE" ]; then
  OUTPUT_FILE="output_norms.txt"
fi

mkdir -p outputs
OUT="outputs/$OUTPUT_FILE"

echo "NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X0" >> $OUT
NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X0 >> $OUT

echo "NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X1" >> $OUT
NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X1 >> $OUT

echo "NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X0" >> $OUT
NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X0 >> $OUT

echo "NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X1" >> $OUT
NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*LN*.*X1 >> $OUT

echo "NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X0" >> $OUT
NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X0 >> $OUT

echo "NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X1" >> $OUT
NVTE_NORM_FWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X1 >> $OUT

echo "NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X0" >> $OUT
NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X0 >> $OUT

echo "NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X1" >> $OUT
NVTE_NORM_BWD_USE_CUDNN=1 ./build/operator/test_operator --gtest_filter=*RMS*.*X1 >> $OUT
