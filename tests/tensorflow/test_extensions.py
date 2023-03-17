# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the cpp extensions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import transformer_engine_tensorflow as tex
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

from transformer_engine.tensorflow import TE_DType
from transformer_engine.tensorflow import get_stream_id


class ExtensionsTest(test.TestCase):
    @test_util.run_gpu_only
    def testCastFp8(self):
        input_shape = (16, 32)
        x = tf.random.uniform(input_shape)
        scale, amax, scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])
        offset = 0
        for fp8_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
            stream_id = get_stream_id()
            x_fp8 = tex.cast_to_fp8(
                x, scale, fp8_dtype, amax, scale_inv, offset, stream_id)
            y = tex.cast_from_fp8(
                x_fp8, scale_inv, fp8_dtype, TE_DType[x.dtype],
                offset, stream_id)
            self.assertAllClose(y, x, rtol=0.1, atol=0.01)

    @test_util.run_gpu_only
    def testTransposeFp8(self):
        stream_id = get_stream_id()

        x = tf.constant(np.random.uniform(-128, 127, (16, 32)), dtype=tf.int8)
        y = tex.fp8_transpose(x, tex.DType.kFloat8E4M3, stream_id)

        y_ref = tf.transpose(x, [1, 0])

        self.assertAllEqual(y, y_ref)

    @test_util.run_gpu_only
    def testMatmulFp8(self):
        stream_id = get_stream_id()
        fp8_dtype = tex.DType.kFloat8E4M3
        out_dtype = tex.DType.kFloat32

        a = tf.random.uniform([32, 16])
        a_scale, a_amax, a_scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])
        a_offset = 0
        a_casted = tex.cast_to_fp8(a, a_scale, fp8_dtype, a_amax, a_scale_inv,
                                   a_offset, stream_id)

        b = tf.random.uniform([16, 16])
        b_scale, b_amax, b_scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])
        b_offset = 0
        b_casted = tex.cast_to_fp8(b, b_scale, fp8_dtype, b_amax, b_scale_inv,
                                   b_offset, stream_id)

        use_bias = False
        bias = tf.zeros(())
        workspace = tf.zeros([33_554_432], dtype=tf.int8)

        # CublasLt inside tex.te_gemm assumes inputs are column major.
        # Mathematically, A@B=C is equivalent to B^T@A^T=C^T, where X^T is the
        # transpose of X. Actually, if we view X^T is the column major of X, we
        # don't need any explict transpose.
        # Note, for fp8 matmul, the first matrix has to be in transposed format.
        d = tex.te_gemm(b_casted, b_scale_inv, fp8_dtype, b_offset, a_casted,
                        a_scale_inv, fp8_dtype, a_offset, workspace, use_bias,
                        bias, False, None, True, False, False, False, False,
                        out_dtype, stream_id)

        # We assume b is in transposed format (see above). So we transpose it
        # back to apply the ordinary row-major matmul.
        bt = tf.transpose(b)
        d_ref = tf.matmul(a, bt)

        self.assertAllClose(d, d_ref, rtol=0.1, atol=0.01)

    @test_util.run_gpu_only
    def testLayerNormFwdFp8(self):
        stream_id = get_stream_id()
        fp8_dtype = tex.DType.kFloat8E4M3
        N, H = (16, 32)
        eps = 1e-3

        x = tf.random.uniform((N, H))
        gamma = tf.random.uniform((H,))
        beta = tf.random.uniform((H,))

        offset = 0
        scale, amax, scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])

        y_ref, mu_ref, rsigma_ref = tex.layernorm_fwd(
            x, gamma, beta, eps, stream_id)

        y_fp8, mu, rsigma = tex.layernorm_fwd_fp8(
            x, gamma, beta, eps, scale, fp8_dtype, amax, scale_inv, offset,
            stream_id)
        y = tex.cast_from_fp8(y_fp8, scale_inv, fp8_dtype, TE_DType[x.dtype],
                              offset, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.1, atol=0.01)
        self.assertAllClose(mu, mu_ref)
        self.assertAllClose(rsigma, rsigma_ref)

    @test_util.run_gpu_only
    def testGeluForwardFp8(self):
        stream_id = get_stream_id()
        fp8_dtype = tex.DType.kFloat8E4M3
        M, N = (16, 32)

        x = tf.random.uniform((M, N))

        offset = 0
        scale, amax, scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])

        y_ref = tf.nn.gelu(x, approximate=True)

        y_fp8 = tex.te_gelu(x, scale, fp8_dtype, amax,
                            scale_inv, offset, stream_id)
        y = tex.cast_from_fp8(y_fp8, scale_inv, fp8_dtype, TE_DType[x.dtype],
                              offset, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.1, atol=0.01)

    @test_util.run_gpu_only
    def testGeluForward(self):
        stream_id = get_stream_id()
        M, N = (16, 32)

        x = tf.random.uniform((M, N))

        y_ref = tf.nn.gelu(x, approximate=True)
        y = tex.te_gelu(x, None, TE_DType[x.dtype], None, None, 0, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.00001, atol=0.00001)

    @test_util.run_gpu_only
    def testGeluBackwardFp8(self):
        stream_id = get_stream_id()
        fp8_dtype = tex.DType.kFloat8E5M2
        M, K, N = (16, 32, 32)

        x = tf.random.uniform((M, K))
        bias = tf.random.uniform((K, ))
        dy = tf.random.uniform((M, K))

        offset = 0
        scale, amax, scale_inv = tf.ones([]), tf.zeros([]), tf.ones([])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, bias])
            x_gelu = tf.nn.bias_add(x, bias)
            y = tf.nn.gelu(x_gelu, approximate=True)
            loss = y * dy
        dgelu_ref, dbias_ref = tape.gradient(loss, [x_gelu, bias])

        dbias, dgelu_c, dgelu_t = tex.fp8_fused_cast_transpose_bgrad_dgelu(
            dy, x_gelu, scale, fp8_dtype, amax, scale_inv, offset, stream_id)
        dgelu = tex.cast_from_fp8(
            dgelu_c, scale_inv, fp8_dtype, TE_DType[x.dtype], offset, stream_id)

        self.assertAllClose(dgelu, dgelu_ref, rtol=0.1, atol=0.01)
        self.assertAllClose(dbias, dbias_ref)
        self.assertAllEqual(dgelu_c, tf.transpose(dgelu_t, [1, 0]))

    @test_util.run_gpu_only
    def testScaledUpperTriangMaskedSoftmaxFwd(self):
        stream_id = get_stream_id()
        B, F = (16, 32)
        scale = 0.8

        x = tf.random.uniform((B, F, F), dtype=tf.half)

        mask_operator = tf.linalg.LinearOperatorLowerTriangular(
            tf.ones((F, F), dtype=tf.bool))
        mask = mask_operator.to_dense()
        mask_output = tf.where(mask, scale * x, -10000.0)
        y_ref = tf.nn.softmax(mask_output, axis=-1)

        y = tex.scaled_upper_triang_masked_softmax_forward(x, scale, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.001, atol=0.001)

    @test_util.run_gpu_only
    def testScaledUpperTriangMaskedSoftmaxBwd(self):
        stream_id = get_stream_id()
        B, F = (16, 32)
        scale = 0.8

        x = tf.random.uniform((B, F, F), dtype=tf.half)
        dy = tf.random.uniform((B, F, F), dtype=tf.half)

        mask_operator = tf.linalg.LinearOperatorLowerTriangular(
            tf.ones((F, F), dtype=tf.bool))
        mask = mask_operator.to_dense()

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            mask_output = tf.where(mask, scale * x, -10000.0)
            y = tf.nn.softmax(mask_output, axis=-1)
            y = tf.cast(y, dtype=tf.half)
            loss = y * dy
        dx_ref = tape.gradient(loss, x)

        dx = tex.scaled_upper_triang_masked_softmax_backward(
            dy, y, scale, stream_id)

        self.assertAllClose(dx, dx_ref, rtol=0.001, atol=0.001)

    @test_util.run_gpu_only
    def testScaledMaskedSoftmaxFwd(self):
        stream_id = get_stream_id()
        B, N, F = (16, 4, 32)
        scale = 0.8

        x = tf.random.uniform((B, N, F, F), dtype=tf.half)
        # In NVTE, if the mask is true, the corresponding value is zero.
        # Whereas, TF does the opposite. In addition, NVTE requires the mask has
        # the same num of dims as the input.
        mask = tf.reshape(x[0, 0] > 0.3, shape=(1, 1, F, F))
        flipped_mask = x[0, 0] <= 0.3

        y_ref = tf.keras.layers.Softmax(axis=-1)(scale * x, flipped_mask)

        y = tex.scaled_masked_softmax_forward(x, mask, scale, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.001, atol=0.001)

    @test_util.run_gpu_only
    def testScaledMaskedSoftmaxBwd(self):
        stream_id = get_stream_id()
        B, N, F = (16, 4, 32)
        scale = 0.8

        x = tf.random.uniform((B, N, F, F), dtype=tf.half)
        dy = tf.random.uniform((B, N, F, F), dtype=tf.half)

        mask = tf.reshape(x[0, 0] > 0.3, shape=(1, 1, F, F))
        flipped_mask = x[0, 0] <= 0.3

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = tf.keras.layers.Softmax(axis=-1)(scale * x, flipped_mask)
            y = tf.cast(y, dtype=tf.half)
            loss = y * dy
        dx_ref = tape.gradient(loss, x)

        dx = tex.scaled_masked_softmax_backward(dy, y, scale, stream_id)

        self.assertAllClose(dx, dx_ref, rtol=0.001, atol=0.001)

    @test_util.run_gpu_only
    def testScaledSoftmaxFwd(self):
        stream_id = get_stream_id()
        B, N, F = (16, 4, 32)
        scale = 0.8

        x = tf.random.uniform((B, N, F, F), dtype=tf.half)

        y_ref = tf.keras.layers.Softmax(axis=-1)(scale * x)

        y = tex.scaled_softmax_forward(x, scale, stream_id)

        self.assertAllClose(y, y_ref, rtol=0.001, atol=0.001)

    @test_util.run_gpu_only
    def testScaledSoftmaxBwd(self):
        stream_id = get_stream_id()
        B, N, F = (16, 4, 32)
        scale = 0.8

        x = tf.random.uniform((B, N, F, F), dtype=tf.half)
        dy = tf.random.uniform((B, N, F, F), dtype=tf.half)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = tf.keras.layers.Softmax(axis=-1)(scale * x)
            y = tf.cast(y, tf.half)
            loss = y * dy
        dx_ref = tape.gradient(loss, x)

        dx = tex.scaled_softmax_backward(dy, y, scale, stream_id)

        self.assertAllClose(dx, dx_ref, rtol=0.001, atol=0.001)


if __name__ == '__main__':
    test.main()
