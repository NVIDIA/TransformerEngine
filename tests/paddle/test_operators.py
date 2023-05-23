# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE operators"""

import pytest
import paddle

from utils import assert_allclose, create_fp8_meta
import transformer_engine_paddle as tex
from transformer_engine.paddle.cpp_extensions import cast_to_fp8, cast_from_fp8, gemm, fp8_gemm
from transformer_engine.paddle.fp8 import is_fp8_available

paddle.seed(10)
GEMM_CASES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816), (16384, 2816, 1024),
              (16384, 1024, 1024)]
is_fp8_supported, reason = is_fp8_available()


def test_quantize_dequantize():
    """
    Test cast_to_fp8 and cast_from_fp8
    """
    a = paddle.rand(shape=(32, 32), dtype='float32')
    # Init fp8_meta
    fp8_meta = create_fp8_meta(num_fp8_tensors=3, amax_history_len=10)
    for fp8_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        a_fp8 = cast_to_fp8(a, fp8_meta, tex.FP8FwdTensors.GEMM1_OUTPUT, otype=fp8_dtype)
        b = cast_from_fp8(a_fp8,
                          fp8_meta,
                          tex.FP8FwdTensors.GEMM1_OUTPUT,
                          itype=fp8_dtype,
                          otype=tex.DType.kFloat32)
        assert_allclose(a.numpy(), b.numpy(), rtol=5e-2, atol=5e-2)


class TestGemm:
    """
    Tests for gemm(cuBLASLt) operator
    """

    @staticmethod
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_bf16(m, n, k):
        """
        Test "TN" BF16 GEMM
        """
        a = paddle.rand(shape=(m, k), dtype='bfloat16')
        b = paddle.rand(shape=(n, k), dtype='bfloat16')

        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = paddle.matmul(a, b.T)
        # CublasLt inside tex.te_gemm assumes inputs are column major.
        # Mathematically, A@B=C is equivalent to B^T@A^T=C^T, where X^T is the
        # transpose of X.
        # Here we perform "TN" GEMM in column major, i.e., b@a^T = C^T,
        # which is equivalent to a@b^T = C in row major.
        actual_out, _, _ = gemm(b, a, paddle.bfloat16, workspace, False, None, False, False, "TN",
                                None, None, False)

        assert_allclose(actual_out.numpy(), ref_out.numpy())

    @staticmethod
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_bf16_inplace(m, n, k):
        """
        Test "TN" BF16 GEMM, with accumulate=True
        """
        a = paddle.rand(shape=(m, k), dtype='bfloat16')
        b = paddle.rand(shape=(n, k), dtype='bfloat16')
        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = paddle.matmul(a, b.T)

        actual_out = paddle.zeros(shape=(m, n), dtype='bfloat16')
        _, _, _ = gemm(b, a, paddle.bfloat16, workspace, False, None, False, True, "TN", actual_out,
                       None, False)

        assert_allclose(actual_out.numpy(), ref_out.numpy())

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_fp8_randint(m, n, k):
        """
        Test "TN" FP8 GEMM
        """
        min_val = -8
        max_val = 8
        fp8_dtype = tex.DType.kFloat8E4M3
        out_dtype = paddle.float32
        fp8_meta = create_fp8_meta(num_fp8_tensors=3, amax_history_len=10)

        a = paddle.cast(paddle.randint(min_val, max_val, shape=(m, k)), 'float32')

        a_casted = cast_to_fp8(a, fp8_meta, tex.FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)
        b = paddle.cast(paddle.randint(min_val, max_val, shape=(n, k)), 'float32')
        b_casted = cast_to_fp8(b, fp8_meta, tex.FP8FwdTensors.GEMM1_WEIGHT, otype=fp8_dtype)
        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = paddle.matmul(a, b.T)
        actual_out = fp8_gemm(b_casted, fp8_meta.scale_inv, tex.FP8FwdTensors.GEMM1_WEIGHT,
                              fp8_dtype, a_casted, fp8_meta.scale_inv,
                              tex.FP8FwdTensors.GEMM1_INPUT, fp8_dtype, out_dtype, workspace)

        assert_allclose(actual_out.numpy(), ref_out.numpy())
