# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE operators"""

import struct

import numpy as np
import paddle
import paddle.nn.functional as F
import pytest

from utils import (
    assert_allclose,
    create_fp8_meta,
    get_fused_attention_backend,
    is_fused_attention_supported,
)

from transformer_engine import transformer_engine_paddle as tex
from transformer_engine.paddle.cpp_extensions import (
    cast_to_fp8,
    cast_from_fp8,
    gemm,
    fp8_gemm,
    transpose,
    cast_transpose,
    cast_transpose_bgrad,
    te_gelu,
    gelu_fp8,
    swiglu,
    swiglu_fp8,
    swiglu_pd,
    dswiglu,
    dgelu_cast_transpose_bgrad_fp8,
    layernorm_fwd_fp8,
    layernorm_fwd,
    layernorm_bwd,
    rmsnorm_fwd_fp8,
    rmsnorm_fwd,
    rmsnorm_bwd,
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_kvpacked,
    fused_attn_bwd_kvpacked,
    fused_attn_fwd,
    fused_attn_bwd,
    scaled_softmax_forward,
    scaled_softmax_backward,
    scaled_masked_softmax_forward,
    scaled_masked_softmax_backward,
    scaled_upper_triang_masked_softmax_forward,
    scaled_upper_triang_masked_softmax_backward,
)
from transformer_engine.paddle.fp8 import is_fp8_available
from transformer_engine.paddle.constants import FP8FwdTensors
from transformer_engine.common.recipe import DelayedScaling

GEMM_CASES = [
    (256, 256, 512),
    (32, 32, 32),
    (16384, 1024, 2816),
    (16384, 2816, 1024),
    (16384, 1024, 1024),
]
is_fp8_supported, reason = is_fp8_available()

SELF_ATTN_CASES = [(2, 512, 12, 64)]
CROSS_ATTN_CASES = [(2, 128, 512, 12, 64)]
FLASH_ATTN_CASES = [(2, 1024, 16, 64), (2, 2048, 16, 128)]
ATTN_DTYPES = [tex.DType.kFloat16, tex.DType.kBFloat16]


@pytest.fixture(autouse=True)
def setup():
    """Setup random seed before each test"""
    np.random.seed(10)
    paddle.seed(11)
    yield


@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("inplace", [True, False])
def test_quantize_dequantize(fp8_dtype, inplace):
    """
    Test cast_to_fp8 and cast_from_fp8
    """
    a = paddle.rand(shape=(32, 32), dtype="float32")
    # Init fp8_meta
    fp8_meta = create_fp8_meta()
    a_fp8 = paddle.zeros(shape=a.shape, dtype=paddle.uint8) if inplace else None
    a_fp8 = cast_to_fp8(a, fp8_meta, FP8FwdTensors.GEMM1_OUTPUT, otype=fp8_dtype, out=a_fp8)
    b = cast_from_fp8(
        a_fp8,
        fp8_meta,
        FP8FwdTensors.GEMM1_OUTPUT,
        itype=fp8_dtype,
        otype=tex.DType.kFloat32,
    )
    assert_allclose(a, b, rtol=5e-2, atol=5e-2)


def copy_bits_from_float_to_uint16(f):
    """
    Copy bits
    """
    return struct.unpack("<I", struct.pack("<f", f))[0] >> 16


def convert_float_to_uint16(float_list):
    """
    convert float to uint16
    """
    new_output = []
    for x in np.nditer(float_list):
        new_output.append(np.uint16(copy_bits_from_float_to_uint16(x)))
    new_output = np.reshape(new_output, float_list.shape).view(np.uint16)

    return new_output


class TestTranspose:
    """
    Test transpose operators
    """

    @staticmethod
    def test_transpose_bf16():
        """
        Test BF16 transpose
        """
        a = paddle.rand(shape=(16, 32), dtype="bfloat16")
        a_transposed = transpose(a, otype=tex.DType.kBFloat16)
        assert_allclose(a_transposed, a.T)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    def test_transpose_fp8(fp8_dtype):
        """
        Test FP8 transpose
        """
        min_val = -8
        max_val = 8
        a = paddle.cast(paddle.randint(min_val, max_val, shape=(16, 32)), "float32")
        fp8_meta = create_fp8_meta()
        a_fp8 = cast_to_fp8(a, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)
        a_fp8_transposed = transpose(a_fp8, otype=fp8_dtype)
        a_transposed = cast_from_fp8(
            a_fp8_transposed,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )
        assert_allclose(a_transposed, a.T)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_cast_transpose(fp8_dtype, inplace):
        """
        Test cast_transpose
        """
        min_val = -8
        max_val = 8
        a = paddle.cast(paddle.randint(min_val, max_val, shape=(16, 32)), "float32")
        fp8_meta = create_fp8_meta()
        a_fp8_casted, a_fp8_transposed = None, None
        if inplace:
            a_fp8_casted = paddle.zeros(shape=a.shape, dtype=paddle.uint8)
            a_fp8_transposed = paddle.zeros(shape=a.T.shape, dtype=paddle.uint8)
        a_fp8_casted, a_fp8_transposed = cast_transpose(
            a,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            otype=fp8_dtype,
            cast_out=a_fp8_casted,
            transpose_out=a_fp8_transposed,
        )

        a_transposed = cast_from_fp8(
            a_fp8_transposed,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        a_casted = cast_from_fp8(
            a_fp8_casted,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        assert_allclose(a_casted, a)
        assert_allclose(a_transposed, a.T)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    def test_cast_transpose_bgrad(fp8_dtype):
        """
        Test cast_transpose_bgrad
        """
        min_val = -8
        max_val = 8
        a = paddle.cast(paddle.randint(min_val, max_val, shape=(16, 32)), "float32")
        fp8_meta = create_fp8_meta()
        bgrad, a_fp8_casted, a_fp8_transposed = cast_transpose_bgrad(
            a, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype
        )

        a_transposed = cast_from_fp8(
            a_fp8_transposed,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        a_casted = cast_from_fp8(
            a_fp8_casted,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        assert_allclose(a_casted, a)
        assert_allclose(a_transposed, a.T)
        assert_allclose(bgrad, a.sum(axis=0))


class TestActivation:
    """
    Test activation operators
    """

    @staticmethod
    def test_gelu_bf16():
        """
        Test BF16 GELU Forward
        """
        a = paddle.rand(shape=(16, 32), dtype="bfloat16") * 2 - 1
        gelu_out = te_gelu(a, otype=tex.DType.kBFloat16)
        gelu_ref = paddle.nn.GELU()(a)

        assert_allclose(gelu_out, gelu_ref, rtol=1e-2)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    def test_gelu_fp8(fp8_dtype):
        """
        Test FP8 GELU Forward
        """
        a = paddle.rand(shape=(16, 32), dtype="float32") * 2 - 1
        fp8_meta = create_fp8_meta()

        gelu_out_fp8 = gelu_fp8(a, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)

        gelu_out = cast_from_fp8(
            gelu_out_fp8,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        gelu_ref = paddle.nn.GELU()(a)

        assert_allclose(gelu_out, gelu_ref, rtol=0.1, atol=0.01)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    def test_gelu_bwd_fp8(fp8_dtype):
        """
        Test FP8 GELU Backward
        """
        # y = GELU(x), calculate ref
        x = paddle.rand(shape=(16, 32), dtype="float32") * 2 - 1
        x.stop_gradient = False
        y = paddle.nn.GELU()(x)
        y_grad = paddle.rand(shape=(16, 32), dtype="float32") * 2 - 1
        paddle.autograd.backward([y], [y_grad], True)
        # calculate fp8
        fp8_meta = create_fp8_meta()
        x_grad_fp8, x_grad_t_fp8, dbias = dgelu_cast_transpose_bgrad_fp8(
            y_grad, x, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype
        )

        x_grad = cast_from_fp8(
            x_grad_fp8,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        x_grad_t = cast_from_fp8(
            x_grad_t_fp8,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        assert_allclose(x_grad, x.grad, rtol=0.1, atol=0.01)
        assert_allclose(x_grad_t, x.grad.T, rtol=0.1, atol=0.01)
        assert_allclose(dbias, x.grad.sum(axis=0), rtol=0.1, atol=0.01)

    @staticmethod
    def test_swiglu_bf16():
        """
        Test BF16 SwiGLU Forward
        """
        a = paddle.rand(shape=(16, 32), dtype="bfloat16") * 2 - 1
        swiglu_out = swiglu(a, otype=tex.DType.kBFloat16)
        swiglu_ref = swiglu_pd(a)

        assert_allclose(swiglu_out, swiglu_ref, rtol=1e-2)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
    def test_swiglu_fp8(fp8_dtype):
        """
        Test FP8 SwiGLU Forward
        """
        a = paddle.rand(shape=(16, 32), dtype="float32") * 2 - 1
        fp8_meta = create_fp8_meta()

        swiglu_out_fp8 = swiglu_fp8(a, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)

        swiglu_out = cast_from_fp8(
            swiglu_out_fp8,
            fp8_meta,
            FP8FwdTensors.GEMM1_INPUT,
            itype=fp8_dtype,
            otype=tex.DType.kFloat32,
        )

        swiglu_ref = swiglu_pd(a)

        assert_allclose(swiglu_out, swiglu_ref, rtol=0.1, atol=0.01)

    @staticmethod
    def test_swiglu_bwd():
        """
        Test SwiGLU Backward
        """
        # y = SwiGLU(x), calculate ref
        x = paddle.rand(shape=(16, 32), dtype="bfloat16") * 2 - 1
        x.stop_gradient = False
        y = swiglu_pd(x)
        y_grad = paddle.rand(shape=(16, 16), dtype="bfloat16") * 2 - 1
        paddle.autograd.backward([y], [y_grad], True)
        # calculate fp8
        x_grad = dswiglu(y_grad, x, otype=tex.DType.kBFloat16)

        assert_allclose(x_grad, x.grad, rtol=0.1, atol=0.01)


class TestGemm:
    """
    Tests for gemm(cuBLASLt) operator
    """

    @staticmethod
    @pytest.mark.skipif(
        paddle.device.cuda.get_device_capability() < (8, 0), reason="BF16 GEMM requires Ampere+ GPU"
    )
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_bf16(m, n, k):
        """
        Test "TN" BF16 GEMM
        """
        a = paddle.rand(shape=(m, k), dtype="bfloat16")
        b = paddle.rand(shape=(n, k), dtype="bfloat16")

        workspace = paddle.zeros(shape=[33_554_432], dtype="uint8")

        ref_out = paddle.matmul(a, b.T)
        # CublasLt inside tex.te_gemm assumes inputs are column major.
        # Mathematically, A@B=C is equivalent to B^T@A^T=C^T, where X^T is the
        # transpose of X.
        # Here we perform "TN" GEMM in column major, i.e., b@a^T = C^T,
        # which is equivalent to a@b^T = C in row major.
        actual_out, _, _ = gemm(
            b, a, paddle.bfloat16, workspace, False, None, False, False, "TN", None, None, False
        )

        assert_allclose(actual_out, ref_out, rtol=1.6e-2, atol=1e-5)

    @staticmethod
    @pytest.mark.skipif(
        paddle.device.cuda.get_device_capability() < (8, 0), reason="BF16 GEMM requires Ampere+ GPU"
    )
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_bf16_inplace(m, n, k):
        """
        Test "TN" BF16 GEMM, with accumulate=True
        """
        min_val = -16
        max_val = 16
        a = paddle.rand(shape=(m, k), dtype="bfloat16")
        b = paddle.rand(shape=(n, k), dtype="bfloat16")
        c = paddle.cast(paddle.randint(min_val, max_val, shape=(m, n)), "bfloat16")
        workspace = paddle.zeros(shape=[33_554_432], dtype="uint8")

        ref_out = c + paddle.matmul(a, b.T)

        actual_out = paddle.clone(c)
        _, _, _ = gemm(
            b,
            a,
            paddle.bfloat16,
            workspace,
            False,
            None,
            False,
            True,
            "TN",
            actual_out,
            None,
            False,
        )

        assert_allclose(actual_out, ref_out, rtol=5e-2, atol=5e-2)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_fp8_randint(m, n, k):
        """
        Test "TN" FP8 GEMM
        """
        min_val = -4
        max_val = 4
        fp8_dtype = tex.DType.kFloat8E4M3
        out_dtype = paddle.float32
        fp8_meta = create_fp8_meta(num_gemms=1)

        a = paddle.cast(paddle.randint(min_val, max_val, shape=(m, k)), "float32")

        a_casted = cast_to_fp8(a, fp8_meta, FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)
        b = paddle.cast(paddle.randint(min_val, max_val, shape=(n, k)), "float32")
        b_casted = cast_to_fp8(b, fp8_meta, FP8FwdTensors.GEMM1_WEIGHT, otype=fp8_dtype)
        workspace = paddle.zeros(shape=[33_554_432], dtype="uint8")

        ref_out = paddle.matmul(a, b.T)
        actual_out, _ = fp8_gemm(
            b_casted,
            fp8_meta.scale_inv,
            FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype,
            a_casted,
            fp8_meta.scale_inv,
            FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype,
            out_dtype,
            workspace,
        )

        assert_allclose(actual_out, ref_out)


class TestLayerNorm:
    """
    Test layernorm operators
    """

    @staticmethod
    def calc_fwd_ref(x, eps, gamma, beta):
        """
        Calculate reference using paddle layer_norm op
        """
        y = paddle.nn.functional.layer_norm(
            x=x, normalized_shape=x.shape[1:], weight=gamma, bias=beta, epsilon=eps
        )
        mean = paddle.mean(x, axis=-1)
        var = paddle.var(x, axis=-1)
        inv_var = paddle.sqrt(1.0 / var)
        return y, mean, inv_var

    @staticmethod
    def calc_bwd_ref(x, eps, gamma, beta, dy):
        """
        Calculate reference using paddle layer_norm op
        """
        x.stop_gradient = False
        gamma.stop_gradient = False
        beta.stop_gradient = False

        y = paddle.nn.functional.layer_norm(
            x=x, normalized_shape=x.shape[1:], weight=gamma, bias=beta, epsilon=eps
        )

        paddle.autograd.backward([y], [dy], True)

        return x.grad, gamma.grad, beta.grad

    def test_layernorm_fwd(self):
        """
        Test BF16 LayerNorm Forward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype="bfloat16")
        gamma = paddle.uniform(shape=(H,), dtype="bfloat16")
        beta = paddle.uniform(shape=(H,), dtype="bfloat16")

        y, mu, rsigma = layernorm_fwd(x, gamma, beta, eps, tex.DType.kBFloat16)

        y_ref, mu_ref, rsigma_ref = self.calc_fwd_ref(x, eps, gamma, beta)

        assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
        assert_allclose(mu, mu_ref, rtol=1e-3, atol=1e-3)
        assert_allclose(rsigma, rsigma_ref, rtol=5e-2, atol=5e-2)

    @staticmethod
    def test_layernorm_fwd_fp8():
        """
        Test FP8 LayerNorm Forward
        """
        fp8_dtype = tex.DType.kFloat8E4M3
        N, H = (16, 32)
        eps = 1e-3

        x = paddle.uniform(shape=(N, H), dtype="float32")
        gamma = paddle.uniform(shape=(H,), dtype="float32")
        beta = paddle.uniform(shape=(H,), dtype="float32")

        fp8_tensor = FP8FwdTensors.GEMM1_INPUT
        fp8_meta = create_fp8_meta()

        y_ref, mu_ref, rsigma_ref = layernorm_fwd(x, gamma, beta, eps, tex.DType.kFloat32)

        y_fp8, mu, rsigma = layernorm_fwd_fp8(x, gamma, beta, eps, fp8_meta, fp8_tensor, fp8_dtype)

        y = cast_from_fp8(y_fp8, fp8_meta, fp8_tensor, itype=fp8_dtype, otype=tex.DType.kFloat32)

        assert_allclose(y, y_ref, rtol=0.1, atol=0.01)
        assert_allclose(mu, mu_ref)
        assert_allclose(rsigma, rsigma_ref)

    def test_layernorm_bwd(self):
        """
        Test BF16 LayerNorm Backward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype="bfloat16")
        dy = paddle.uniform(shape=(N, H), dtype="bfloat16")
        gamma = paddle.uniform(shape=(H,), dtype="bfloat16")
        beta = paddle.uniform(shape=(H,), dtype="bfloat16")

        dx_ref, dgamma_ref, dbeta_ref = self.calc_bwd_ref(x, eps, gamma, beta, dy)

        _, mu, rsigma = layernorm_fwd(x, gamma, beta, eps, tex.DType.kBFloat16)
        dx, dgamma, dbeta = layernorm_bwd(dy, x, mu, rsigma, gamma)

        assert_allclose(dx, dx_ref, rtol=1e-5, atol=1e-5)
        assert_allclose(dgamma, dgamma_ref, rtol=1e-5, atol=1e-5)
        assert_allclose(dbeta, dbeta_ref, rtol=1e-5, atol=1e-5)


class TestRMSNorm:
    """
    Test rmsnorm operators
    """

    @staticmethod
    def calc_fwd_ref(x, eps, gamma):
        """
        Calculate rmsnorm reference using paddle op
        """

        norm = paddle.rsqrt(paddle.mean(x**2, axis=-1, keepdim=True) + eps)
        y = x * norm * gamma

        return y

    def calc_bwd_ref(self, x, eps, gamma, dy):
        """
        Calculate rmsnorm bwd reference using paddle op
        """
        x.stop_gradient = False
        gamma.stop_gradient = False

        y = self.calc_fwd_ref(x, eps, gamma)

        paddle.autograd.backward([y], [dy], True)

        return x.grad, gamma.grad

    def test_rmsnorm_fwd(self):
        """
        Test BF16 RMSNorm Forward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype="bfloat16")
        gamma = paddle.uniform(shape=(H,), dtype="bfloat16")

        y, _ = rmsnorm_fwd(x, gamma, eps, tex.DType.kBFloat16)

        y_ref = self.calc_fwd_ref(x, eps, gamma)

        assert_allclose(y, y_ref, rtol=1e-2, atol=1e-2)

    @staticmethod
    def test_rmsnorm_fwd_fp8():
        """
        Test FP8 RMSNorm Forward
        """
        fp8_dtype = tex.DType.kFloat8E4M3
        N, H = (16, 32)
        eps = 1e-3

        x = paddle.uniform(shape=(N, H), dtype="float32")
        gamma = paddle.uniform(shape=(H,), dtype="float32")

        fp8_tensor = FP8FwdTensors.GEMM1_INPUT
        fp8_meta = create_fp8_meta()

        y_ref, rsigma_ref = rmsnorm_fwd(x, gamma, eps, tex.DType.kFloat32)

        y_fp8, rsigma = rmsnorm_fwd_fp8(x, gamma, eps, fp8_meta, fp8_tensor, fp8_dtype)

        y = cast_from_fp8(y_fp8, fp8_meta, fp8_tensor, itype=fp8_dtype, otype=tex.DType.kFloat32)

        assert_allclose(y, y_ref, rtol=0.1, atol=0.01)
        assert_allclose(rsigma, rsigma_ref)

    def test_rmsnorm_bwd(self):
        """
        Test BF16 RMSNorm Backward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype="bfloat16")
        dy = paddle.uniform(shape=(N, H), dtype="bfloat16")
        gamma = paddle.uniform(shape=(H,), dtype="bfloat16")

        dx_ref, dgamma_ref = self.calc_bwd_ref(x, eps, gamma, dy)

        _, rsigma = rmsnorm_fwd(x, gamma, eps, tex.DType.kBFloat16)
        dx, dgamma = rmsnorm_bwd(dy, x, rsigma, gamma)

        assert_allclose(dx, dx_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(dgamma, dgamma_ref, rtol=1e-2, atol=5e-2)


class TestFusedAttn:
    """
    Test fused attention operators
    """

    def set_input(self, b, s_q, s_kv, h, d, dtype, attn_mode="self_attn", is_causal_masking=False):
        """
        set test input
        """

        def _random(shape):
            if self.dtype == "bfloat16":
                data = np.random.normal(loc=0.0, scale=0.02, size=shape).astype("float32")
                return convert_float_to_uint16(data)
            return np.random.normal(loc=0.0, scale=0.02, size=shape).astype(self.dtype)

        self.batch_size = b
        self.q_seqlen = s_q
        self.kv_seqlen = s_kv
        self.num_heads = h
        self.head_size = d
        self.dropout_prob = 0.0
        self.scaling_factor = 1.0 / np.sqrt(d)
        self.q_shape = (b, s_q, h, d)
        self.kv_shape = (b, s_kv, h, d)
        self.fuse_qkv_shape = (b, s_q, 3, h, d)
        self.fuse_kv_shape = (b, s_kv, 2, h, d)
        self.bias_shape = (1, h, s_q, s_kv)
        self.attn_mode = attn_mode
        self.dtype = dtype
        self.is_causal_masking = is_causal_masking

        self.q = _random(self.q_shape)
        if self.attn_mode == "self_attn":
            assert self.q_seqlen == self.kv_seqlen, "self attention requires q_seqlen == kv_seqlen"
            self.kv = self.q
        else:
            self.kv = _random(self.kv_shape)

        self.q_actual_seqlen = None
        if self.is_causal_masking:
            self.q_actual_seqlen = np.full(
                self.batch_size,
                self.q_seqlen,
                dtype=np.int32,
            )
        else:
            self.q_actual_seqlen = np.random.randint(
                low=20,
                high=self.q_seqlen,
                size=(self.batch_size,),
                dtype=np.int32,
            )
        self.kv_actual_seqlen = self.q_actual_seqlen

        self.q_cu_seqlen = np.cumsum(self.q_actual_seqlen)
        self.q_cu_seqlen = np.insert(self.q_cu_seqlen, 0, 0)
        self.kv_cu_seqlen = np.cumsum(self.kv_actual_seqlen)
        self.kv_cu_seqlen = np.insert(self.kv_cu_seqlen, 0, 0)
        self.attn_mask = np.ones(
            shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
            dtype=np.int32,
        )
        if self.is_causal_masking:
            assert attn_mode == "self_attn", "only support causal masking for self attention"
            for i in range(0, self.batch_size):
                for j in range(self.q_actual_seqlen[i]):
                    self.attn_mask[i, :, j, : j + 1] = 0
        else:
            for i in range(0, self.batch_size):
                self.attn_mask[i, :, : self.q_actual_seqlen[i], : self.kv_actual_seqlen[i]] = 0

        dout = _random((self.batch_size, self.q_seqlen, self.num_heads, self.head_size))
        self.dout = paddle.to_tensor(dout, dtype=self.dtype)

    def _get_reference_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.kv, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.kv, stop_gradient=False)

        q_out = paddle.transpose(x=q_tensor, perm=[0, 2, 1, 3])  # [b, s, h, d] -> [b, h, s, d]
        k_out = paddle.transpose(x=k_tensor, perm=[0, 2, 1, 3])  # [b, s, h, d] -> [b, h, s, d]
        v_out = paddle.transpose(x=v_tensor, perm=[0, 2, 1, 3])  # [b, s, h, d] -> [b, h, s, d]

        qk_out = paddle.matmul(
            x=q_out * self.scaling_factor,
            y=k_out,
            transpose_x=False,
            transpose_y=True,
        )

        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=True).cast("bool")
        attn_mask_vals = paddle.full(qk_out.shape, -1e4, qk_out.dtype)
        attn_mask_out = paddle.where(attn_mask, attn_mask_vals, qk_out)
        attn_mask_out = paddle.cast(attn_mask_out, "float32")
        softmax_out = F.softmax(attn_mask_out)
        softmax_out = paddle.cast(softmax_out, self.dtype)

        if self.dropout_prob:
            dropout_out = F.dropout(
                softmax_out,
                self.dropout_prob,
                training=self.training,
                mode="upscale_in_train",
            )
            qkv_out = paddle.matmul(dropout_out, v_out)
        else:
            qkv_out = paddle.matmul(softmax_out, v_out)

        out = paddle.transpose(qkv_out, perm=[0, 2, 1, 3])  # [b, h, s, d] -> [b, s, h, d]

        paddle.autograd.backward(
            [out],
            [self.dout],
            retain_graph=True,
        )
        return out, q_tensor.grad, k_tensor.grad, v_tensor.grad

    def _get_fused_attention_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        if self.attn_mode == "self_attn":
            qkv = np.stack([self.q, self.kv, self.kv], axis=2)  # [b, s, 3, h, d]
            qkv_tensor = paddle.to_tensor(qkv, stop_gradient=False)
        else:
            q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
            kv = np.stack([self.kv, self.kv], axis=2)  # [b, s, 2, h, d]
            kv_tensor = paddle.to_tensor(kv, stop_gradient=False)

        q_cu_seqlen_tensor = paddle.to_tensor(self.q_cu_seqlen, dtype="int32", stop_gradient=True)
        kv_cu_seqlen_tensor = paddle.to_tensor(self.kv_cu_seqlen, dtype="int32", stop_gradient=True)

        qkv_layout = "bs3hd" if self.attn_mode == "self_attn" else "bshd_bs2hd"
        fused_attention_backend = get_fused_attention_backend(
            num_heads=self.num_heads,
            num_gqa_groups=self.num_heads,
            q_seqlen=self.q_seqlen,
            kv_seqlen=self.kv_seqlen,
            head_size=self.head_size,
            dtype=self.dtype,
            dropout=self.dropout_prob,
            qkv_layout=qkv_layout,
            bias_type="no_bias",
            mask_type="causal" if self.is_causal_masking else "padding",
        )

        qkv_dtype = tex.DType.kBFloat16 if self.dtype == "bfloat16" else tex.DType.kFloat16
        out, softmax_aux_tensor, q_grad, k_grad, v_grad = None, None, None, None, None
        if self.attn_mode == "self_attn":
            out, softmax_aux_tensor, rng_state = fused_attn_fwd_qkvpacked(
                qkv_tensor,
                q_cu_seqlen_tensor,
                is_training=True,
                max_seqlen=self.q_seqlen,
                qkv_dtype=qkv_dtype,
                fused_attention_backend=fused_attention_backend,
                Bias=None,
                attn_scale=self.scaling_factor,
                dropout=self.dropout_prob,
                set_zero=False,
                attn_mask_type="causal" if self.is_causal_masking else "padding",
            )
            dqkv, _ = fused_attn_bwd_qkvpacked(
                qkv_tensor,
                q_cu_seqlen_tensor,
                rng_state,
                out,
                self.dout,
                softmax_aux_tensor,
                max_seqlen=self.q_seqlen,
                qkv_dtype=qkv_dtype,
                fused_attention_backend=fused_attention_backend,
                attn_scale=self.scaling_factor,
                dropout=self.dropout_prob,
                set_zero=False,
                attn_mask_type="causal" if self.is_causal_masking else "padding",
            )
            q_grad = dqkv[:, :, 0, :, :]
            k_grad = dqkv[:, :, 1, :, :]
            v_grad = dqkv[:, :, 2, :, :]
        else:  # attn_mode == 'cross_attn'
            out, softmax_aux_tensor, rng_state = fused_attn_fwd_kvpacked(
                q_tensor,
                kv_tensor,
                q_cu_seqlen_tensor,
                kv_cu_seqlen_tensor,
                is_training=True,
                max_seqlen_q=self.q_seqlen,
                max_seqlen_kv=self.kv_seqlen,
                qkv_dtype=qkv_dtype,
                fused_attention_backend=fused_attention_backend,
                Bias=None,
                attn_scale=self.scaling_factor,
                dropout=self.dropout_prob,
                set_zero=False,
            )
            dq, dkv, _ = fused_attn_bwd_kvpacked(
                q_tensor,
                kv_tensor,
                q_cu_seqlen_tensor,
                kv_cu_seqlen_tensor,
                rng_state,
                out,
                self.dout,
                softmax_aux_tensor,
                fused_attention_backend=fused_attention_backend,
                max_seqlen_q=self.q_seqlen,
                max_seqlen_kv=self.kv_seqlen,
                qkv_dtype=qkv_dtype,
                attn_scale=self.scaling_factor,
                dropout=self.dropout_prob,
                set_zero=False,
            )
            q_grad = dq
            k_grad = dkv[:, :, 0, :, :]
            v_grad = dkv[:, :, 1, :, :]

        return out, q_grad, k_grad, v_grad

    def _get_fused_attention_with_separate_qkv(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.kv, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.kv, stop_gradient=False)

        q_cu_seqlen_tensor = paddle.to_tensor(self.q_cu_seqlen, dtype="int32", stop_gradient=True)
        kv_cu_seqlen_tensor = paddle.to_tensor(self.kv_cu_seqlen, dtype="int32", stop_gradient=True)

        qkv_layout = "bshd_bshd_bshd"
        fused_attention_backend = get_fused_attention_backend(
            num_heads=self.num_heads,
            num_gqa_groups=self.num_heads,
            q_seqlen=self.q_seqlen,
            kv_seqlen=self.kv_seqlen,
            head_size=self.head_size,
            dtype=self.dtype,
            dropout=self.dropout_prob,
            qkv_layout=qkv_layout,
            bias_type="no_bias",
            mask_type="causal" if self.is_causal_masking else "padding",
        )

        qkv_dtype = tex.DType.kBFloat16 if self.dtype == "bfloat16" else tex.DType.kFloat16
        out, softmax_aux_tensor, rng_state = fused_attn_fwd(
            q_tensor,
            k_tensor,
            v_tensor,
            q_cu_seqlen_tensor,
            kv_cu_seqlen_tensor,
            is_training=True,
            max_seqlen_q=self.q_seqlen,
            max_seqlen_kv=self.kv_seqlen,
            qkv_dtype=qkv_dtype,
            fused_attention_backend=fused_attention_backend,
            Bias=None,
            attn_scale=self.scaling_factor,
            dropout=self.dropout_prob,
            set_zero=False,
            qkv_layout=qkv_layout,
            attn_mask_type="causal" if self.is_causal_masking else "padding",
        )
        dq, dk, dv, _ = fused_attn_bwd(
            q_tensor,
            k_tensor,
            v_tensor,
            q_cu_seqlen_tensor,
            kv_cu_seqlen_tensor,
            rng_state,
            out,
            self.dout,
            softmax_aux_tensor,
            fused_attention_backend=fused_attention_backend,
            max_seqlen_q=self.q_seqlen,
            max_seqlen_kv=self.kv_seqlen,
            qkv_dtype=qkv_dtype,
            attn_scale=self.scaling_factor,
            dropout=self.dropout_prob,
            set_zero=False,
            qkv_layout=qkv_layout,
            attn_mask_type="causal" if self.is_causal_masking else "padding",
        )

        return out, dq, dk, dv

    @pytest.mark.parametrize("b, s, h, d", SELF_ATTN_CASES)
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    @pytest.mark.parametrize("is_causal_masking", [True, False])
    def test_self_attn_forward_backward(self, b, s, h, d, dtype, is_causal_masking):
        """
        test self attention forward + backward
        """
        if not is_fused_attention_supported(
            num_heads=h,
            num_gqa_groups=h,
            q_seqlen=s,
            kv_seqlen=s,
            head_size=d,
            dtype=dtype,
            dropout=0.0,
            qkv_layout="bs3hd",
            bias_type="no_bias",
            mask_type="causal" if is_causal_masking else "padding",
        ):
            pytest.skip("cuDNN fused attention is not supported")
        self.set_input(b, s, s, h, d, dtype, "self_attn", is_causal_masking)
        reference_out, q_grad_ref, k_grad_ref, v_grad_ref = self._get_reference_out()
        fused_attention_out, q_grad, k_grad, v_grad = self._get_fused_attention_out()
        assert_allclose(reference_out, fused_attention_out, rtol=1e-3, atol=1e-2)
        assert_allclose(q_grad_ref, q_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(k_grad_ref, k_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(v_grad_ref, v_grad, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("b, s_q, s_kv, h, d", CROSS_ATTN_CASES)
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_cross_attn_forward_backward(self, b, s_q, s_kv, h, d, dtype):
        """
        test cross attention forward + backward
        """
        if not is_fused_attention_supported(
            num_heads=h,
            num_gqa_groups=h,
            q_seqlen=s_q,
            kv_seqlen=s_kv,
            head_size=d,
            dtype=dtype,
            dropout=0.0,
            qkv_layout="bshd_bs2hd",
            bias_type="no_bias",
            mask_type="padding",
        ):
            pytest.skip("cuDNN fused attention is not supported")
        self.set_input(b, s_q, s_kv, h, d, dtype, "cross_attn")
        reference_out, q_grad_ref, k_grad_ref, v_grad_ref = self._get_reference_out()
        fused_attention_out, q_grad, k_grad, v_grad = self._get_fused_attention_out()
        assert_allclose(reference_out, fused_attention_out, rtol=1e-3, atol=1e-2)
        assert_allclose(q_grad_ref, q_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(k_grad_ref, k_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(v_grad_ref, v_grad, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("b, s, h, d", FLASH_ATTN_CASES)
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    @pytest.mark.parametrize("is_causal_masking", [True])
    def test_flash_attn_forward_backward(self, b, s, h, d, dtype, is_causal_masking):
        """
        test flash attention forward + backward
        """
        if not is_fused_attention_supported(
            num_heads=h,
            num_gqa_groups=h,
            q_seqlen=s,
            kv_seqlen=s,
            head_size=d,
            dtype=dtype,
            dropout=0.0,
            qkv_layout="bs3hd",
            bias_type="no_bias",
            mask_type="causal" if is_causal_masking else "padding",
        ):
            pytest.skip("cuDNN fused attention is not supported")
        self.set_input(b, s, s, h, d, dtype, "self_attn", is_causal_masking)
        reference_out, q_grad_ref, k_grad_ref, v_grad_ref = self._get_reference_out()
        fused_attention_out, q_grad, k_grad, v_grad = self._get_fused_attention_out()
        assert_allclose(reference_out, fused_attention_out, rtol=1e-3, atol=1e-2)
        assert_allclose(q_grad_ref, q_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(k_grad_ref, k_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(v_grad_ref, v_grad, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("b, s, h, d", FLASH_ATTN_CASES)
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    @pytest.mark.parametrize("is_causal_masking", [False, True])
    def test_fused_attn_with_separate_qkv_forward_backward(
        self, b, s, h, d, dtype, is_causal_masking
    ):
        """
        test flash attention forward + backward with separate qkv inputs
        """
        if not is_fused_attention_supported(
            num_heads=h,
            num_gqa_groups=h,
            q_seqlen=s,
            kv_seqlen=s,
            head_size=d,
            dtype=dtype,
            dropout=0.0,
            qkv_layout="bshd_bshd_bshd",
            bias_type="no_bias",
            mask_type="causal" if is_causal_masking else "padding",
        ):
            pytest.skip("cuDNN fused attention is not supported")
        self.set_input(b, s, s, h, d, dtype, "self_attn", is_causal_masking)
        reference_out, q_grad_ref, k_grad_ref, v_grad_ref = self._get_reference_out()
        fused_attention_out, q_grad, k_grad, v_grad = self._get_fused_attention_with_separate_qkv()
        assert_allclose(reference_out, fused_attention_out, rtol=1e-3, atol=1e-2)
        assert_allclose(q_grad_ref, q_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(k_grad_ref, k_grad, rtol=1e-3, atol=1e-2)
        assert_allclose(v_grad_ref, v_grad, rtol=1e-3, atol=1e-2)


class TestSoftmax:
    """
    Test softmax operators
    """

    @staticmethod
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_scaled_softmax_fwd_bwd(dtype):
        """test scaled softmax"""
        B, H, S = (16, 4, 32)
        scale = 0.8

        x = paddle.uniform(shape=(B, H, S, S), dtype=dtype)
        x.stop_gradient = False
        dy = paddle.uniform(shape=(B, H, S, S), dtype=dtype)

        y_ref = F.softmax(scale * x)
        y = scaled_softmax_forward(x, scale)

        paddle.autograd.backward([y_ref], [dy], True)
        dx_ref = x.grad
        dx = scaled_softmax_backward(dy, y, scale)

        assert_allclose(y_ref, y, rtol=1e-4, atol=1e-3)
        assert_allclose(dx_ref, dx, rtol=1e-4, atol=1e-3)

    @staticmethod
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_scaled_masked_softmax_fwd_bwd(dtype):
        """test scaled masked softmax"""
        B, H, S = (16, 4, 32)
        scale = 0.8

        x = paddle.uniform(shape=(B, H, S, S), dtype=dtype)
        x.stop_gradient = False
        dy = paddle.uniform(shape=(B, H, S, S), dtype=dtype)
        mask = paddle.reshape(x[0, 0] > 0.3, shape=(1, 1, S, S))
        mask_flipped = x[0, 0] <= 0.3
        mask_ref = (mask_flipped.astype(dtype) - 1.0) * 1e4

        y_ref = F.softmax(scale * x + mask_ref)
        y = scaled_masked_softmax_forward(x, mask, scale)

        paddle.autograd.backward([y_ref], [dy], True)
        dx_ref = x.grad
        dx = scaled_masked_softmax_backward(dy, y, scale)

        assert_allclose(y_ref, y, rtol=1e-4, atol=1e-3)
        assert_allclose(dx_ref, dx, rtol=1e-4, atol=1e-3)

    @staticmethod
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_scaled_upper_triang_masked_softmax_fwd_bwd(dtype):
        """test scaled upper triang masked softmax"""
        B, S = (16, 32)
        scale = 0.8

        x = paddle.uniform(shape=(B, S, S), dtype=dtype)
        x.stop_gradient = False
        dy = paddle.uniform(shape=(B, S, S), dtype=dtype)

        mask = paddle.ones((S, S), dtype="int32")
        col_beg, col_end = 1, S
        for row in range(0, S):
            mask[row, col_beg:col_end] = 0
            col_beg += 1

        mask_ref = (mask.astype(dtype) - 1.0) * 1e4

        y_ref = F.softmax(scale * x + mask_ref)
        y = scaled_upper_triang_masked_softmax_forward(x, scale)

        paddle.autograd.backward([y_ref], [dy], True)
        dx_ref = x.grad
        dx = scaled_upper_triang_masked_softmax_backward(dy, y, scale)

        assert_allclose(y_ref, y, rtol=1e-4, atol=5e-3)
        assert_allclose(dx_ref, dx, rtol=1e-4, atol=5e-3)


@pytest.mark.parametrize("update_weight_scale_inv", [True, False])
def test_amax_and_scale_update(update_weight_scale_inv):
    """Test update_scale"""
    num_gemm = 6
    history_len = 1024
    recipe = DelayedScaling()
    fp8_dtype = tex.DType.kFloat8E4M3
    fp8_max = recipe.fp8_format.value.max_fwd
    non_weight_mask = paddle.to_tensor([True, False] * (num_gemm // 2))

    amax_history_tensor = paddle.rand(shape=[history_len, num_gemm], dtype="float32")
    rolled_history_ref = paddle.roll(amax_history_tensor, -1, axis=0)
    rolled_history_ref[0] = 0.0
    amax_tensor = paddle.max(amax_history_tensor, axis=0)
    scale_tensor = paddle.ones(shape=[num_gemm], dtype="float32")

    def calc_ref(amax, scale, fp8_max, margin=0):
        """Calculate reference scale"""
        sf = (fp8_max / amax) / (2**margin)
        sf = paddle.where(amax > 0.0, sf, scale)
        sf = paddle.where(paddle.isfinite(amax), sf, scale)
        return sf

    scale_ref = calc_ref(amax_tensor, scale_tensor, fp8_max, 0.0)
    if update_weight_scale_inv:
        scale_inv_ref = 1.0 / scale_ref
    else:
        scale_inv_ref = paddle.zeros_like(scale_tensor)
        scale_inv_ref = paddle.where(non_weight_mask, 1.0 / scale_ref, scale_inv_ref)

    # Placeholder
    scale_actual = paddle.zeros_like(scale_tensor)
    scale_inv_actual = paddle.zeros_like(scale_tensor)

    if update_weight_scale_inv:
        non_weight_mask = paddle.empty([0])
    tex.amax_and_scale_update_inplace(
        _amax_history=amax_history_tensor,
        _scale=scale_actual,
        _scale_inv=scale_inv_actual,
        non_weight_mask=non_weight_mask,
        fp8_dtype=int(fp8_dtype),
        margin=0.0,
        amax_compute="max",
    )

    assert_allclose(scale_actual, scale_ref, rtol=1e-7, atol=1e-7)
    assert_allclose(scale_inv_actual, scale_inv_ref, rtol=1e-7, atol=1e-7)
    assert_allclose(amax_history_tensor, rolled_history_ref, rtol=1e-7, atol=1e-7)


def test_update_latest_history():
    """Test update_latest_history"""
    num_gemm = 6
    history_len = 1024

    amax_history_tensor = paddle.rand(shape=[history_len, num_gemm], dtype="float32")
    amax = paddle.rand(shape=[num_gemm], dtype="float32")

    tex.update_latest_amax_history_inplace(_history=amax_history_tensor, amax=amax)

    assert_allclose(amax_history_tensor[0], amax, rtol=1e-7, atol=1e-7)
