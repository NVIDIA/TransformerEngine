# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the Transformer layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import transformer_engine.tensorflow as te

from tensorflow.keras.layers import EinsumDense
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from transformer_engine.tensorflow import (
    DelayedScaling,
    Format,
    TransformerLayer,
)


def train_step(dy, x, x_mask, x_dec, x_dec_mask, model, use_fp8=False,
               fp8_recipe=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            y = model(
                hidden_states=x,
                attention_mask=x_mask,
                encoder_output=x_dec,
                enc_dec_attn_mask=x_dec_mask,
                training=True,
            )
        loss = y * tf.cast(dy, dtype=y.dtype)
    dx, dvars = tape.gradient(loss, [x, model.trainable_variables])
    return y, dx, dvars


class TransformerLayerTest(test.TestCase):
    def setUp(self):
        super().setUp()
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @test_util.run_gpu_only
    def testTransformerSanity(self):
        if len(context.context().list_physical_devices('GPU')) != 1:
            self.skipTest('Only supports a single GPU')

        use_fp8 = tf.test.is_gpu_available(True, (9, 0))
        # F=seq_len, B=batch, H=hidden_states, N=num_heads
        F, B, H, N = 8, 4, 32, 2
        # E=depth
        E = H // N
        # D=hidden_size
        D = N * E
        input_shape = (F, B, H)
        output_shape = (F, B, D)

        init = tf.keras.initializers.RandomUniform(minval=0., maxval=.1)
        x = tf.random.uniform(input_shape, minval=0., maxval=.1)
        x_dec = tf.random.uniform(input_shape, minval=0., maxval=10.)
        dy = tf.random.uniform(output_shape, minval=0., maxval=.1)

        transformer = TransformerLayer(
            hidden_size=D,
            ffn_hidden_size=D,
            num_attention_heads=N,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.01,
            attention_dropout=0.0,
            init_method=init,
            output_layer_init_method=init,
            layer_number=None,
            kv_channels=None,
            self_attn_mask_type="padding",
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="decoder",
            drop_path_rate=0.1,
            fuse_qkv_params=False,
        )

        fp8_recipe = DelayedScaling(
            margin=0, interval=1, fp8_format=Format.HYBRID,
            amax_compute_algo='max', amax_history_len=3)

        y_ref, dx_ref, dvars_ref = train_step(
            dy, x, None, x_dec, None, transformer, use_fp8=False)

        y, dx, dvars = train_step(dy, x, None, x_dec, None, transformer,
                                  use_fp8=use_fp8, fp8_recipe=fp8_recipe)

        self.assertAllClose(y, y_ref, rtol=0.1, atol=0.01, msg="fwd-y")
        self.assertAllClose(dx, dx_ref, rtol=0.5, atol=0.7, msg="bwd-dx")

        self.assertEqual(len(dvars), len(dvars_ref))

        dvs = []
        for v, dv, dv_ref in zip(
                transformer.trainable_variables, dvars, dvars_ref):
            dvs.append((v.name, dv, dv_ref))

        for v_name, dv, dv_ref in reversed(dvs):
            # The range of these two biases are relatively large. So, we choose
            # larger atols here.
            if v_name == 'multi_head_attention/dense/bias:0':
                self.assertAllClose(dv, dv_ref, rtol=.1,
                                    atol=4., msg="bwd-" + v_name)
                continue
            if v_name == 'multi_head_attention/qkv_bias:0':
                self.assertAllClose(dv, dv_ref, rtol=.1,
                                    atol=2., msg="bwd-" + v_name)
                continue

            atol, rtol = (0.5, 0.6) if tf.reduce_max(
                dv_ref) > 1. else (0.05, 0.05)
            self.assertAllClose(dv, dv_ref, rtol=rtol,
                                atol=atol, msg="bwd-" + v_name)


if __name__ == '__main__':
    test.main()
