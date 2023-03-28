# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the MHA layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import transformer_engine.tensorflow as te

from tensorflow.keras.layers import EinsumDense
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from transformer_engine.tensorflow import (
    DelayedScaling,
    Format,
    MultiHeadAttention,
)


def train_step(dy, x_q, x_kv, x_mask, model, attn_type, use_fp8=False,
               fp8_recipe=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_q)
        if attn_type == 'cross':
            tape.watch(x_kv)
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            # The MHA won't apply the bias addition for the last projection but
            # return the bias. So, we conduct the bias addition here at the end.
            y, b = model(x_q, x_mask, x_kv, training=True)
            y = y + tf.cast(b, y.dtype)
        loss = y * tf.cast(dy, dtype=y.dtype)
    xs = [x_q]
    if attn_type == 'cross':
        xs.append(x_kv)
    dxs, dvars = tape.gradient(loss, [xs, model.trainable_variables])
    return y, dxs, dvars


class MultiHeadAttentionKeras(tf.keras.Model):
    def __init__(self, hidden_size, num_heads, attention_type, init_method):
        super(MultiHeadAttentionKeras, self).__init__()

        assert hidden_size % num_heads == 0
        assert attention_type in ('self', 'cross')

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = hidden_size // self.num_heads
        self.attention_type = attention_type

        # Einsum symbols:
        # F=seq_q, T=seq_kv, B=batches, H=hidden_states, D=hidden_size,
        # N=num_heads, E=depth
        if attention_type == 'self':
            self.QKV = EinsumDense('FBH,HD->FBD',
                                   output_shape=(None, 3 * hidden_size),
                                   bias_axes='D',
                                   kernel_initializer=init_method)
        else:
            self.Q = EinsumDense('FBH,HD->FBD',
                                 output_shape=(None, hidden_size),
                                 bias_axes='D',
                                 kernel_initializer=init_method)
            self.KV = EinsumDense('TBH,HD->TBD',
                                  output_shape=(None, 2 * hidden_size),
                                  bias_axes='D',
                                  kernel_initializer=init_method)

        # The bias in the projection layer will be applied separately outside
        # the MHA. So, we disable the bias in the Einsum but handle the bias at
        # the end.
        self.dense = EinsumDense('FBNE,NED->FBD',
                                 output_shape=(None, hidden_size),
                                 bias_axes=None,
                                 kernel_initializer=init_method)
        b_init = tf.zeros_initializer()
        self.dense_bias = tf.Variable(
            initial_value=b_init(shape=(hidden_size,),
                                 dtype="float32"),
            trainable=True,
        )

    def __call__(self, q_input, mask=None, kv_input=None, training=None):
        if self.attention_type == 'self':
            # [F, B, 3 * D]
            qkv = self.QKV(q_input)
            # [F, B, N, 3 * E]
            qkv = tf.reshape(
                qkv, (*qkv.shape[: -1],
                      self.num_heads, 3 * self.depth))
            # 3 * [F, B, N, E]
            q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
        else:
            # [F, B, D]
            q = self.Q(q_input)
            # [F, B, N, E]
            q = tf.reshape(q, (*q.shape[:-1], self.num_heads, self.depth))
            # [F, B, 2 * D]
            kv = self.KV(kv_input)
            # [F, B, N, 2 * E]
            kv = tf.reshape(
                kv, (*kv.shape[: -1],
                     self.num_heads, 2 * self.depth))
            # 2 * [F, B, N, E]
            k, v = tf.split(kv, num_or_size_splits=2, axis=-1)

        dk = tf.cast(tf.shape(k)[-1], self._compute_dtype_object)
        matmul_qk = tf.einsum('FBNE,TBNE->BNFT', q, k)
        scaled_attn_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attn_logits = tf.where(mask, scaled_attn_logits, -10000.0)

        # [B, N, F, T]
        attention_weights = tf.nn.softmax(scaled_attn_logits, axis=-1)
        # [B, N, F, E]
        scaled_attention = tf.einsum('BNFT,TBNE->BNFE', attention_weights, v)
        # [F, B, N, E]
        scaled_attention = tf.transpose(scaled_attention, perm=(2, 0, 1, 3))

        # [F, B, D]
        output = self.dense(scaled_attention)
        return output, self.dense_bias


class MHATest(test.TestCase):
    @test_util.run_gpu_only
    def testMHAForward(self):
        use_fp8 = tf.test.is_gpu_available(True, (9, 0))
        batches, seq_q, seq_kv, hidden_states = 16, 32, 32, 64
        num_heads, depth = 4, 16
        hidden_size = num_heads * depth
        q_shape = (seq_q, batches, hidden_states)
        kv_shape = (seq_kv, batches, hidden_states)

        init = tf.keras.initializers.RandomUniform(minval=0., maxval=.1)
        x_q = tf.random.uniform(q_shape, minval=0., maxval=.1)
        x_kv = tf.random.uniform(kv_shape, minval=0., maxval=.1)

        for attn_type in ('self', 'cross'):
            for use_mask in (True, False):
                mha_einsum = MultiHeadAttentionKeras(
                    hidden_size, num_heads, attn_type, init)
                # The attention mask type needs to be `padding`, which will use
                # provided mask. Alternatively, the `causal` will ignore the
                # provided mask and use a upper triangular mask.
                mha = MultiHeadAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    kv_channels=depth,
                    attention_dropout=0.0,
                    attention_softmax_in_fp32=True,
                    init_method=init,
                    output_layer_init_method=init,
                    input_layernorm=False,
                    attention_type=attn_type,
                    attn_mask_type='padding',
                )

                x_mask = tf.random.uniform(
                    (seq_q, seq_kv)) > 0.5 if use_mask else None

                y_ref, y_b_ref = mha_einsum(x_q, x_mask, x_kv)

                fp8_recipe = DelayedScaling(
                    margin=0, interval=1, fp8_format=Format.HYBRID,
                    amax_compute_algo='max', amax_history_len=3)
                with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
                    y, y_b = mha(x_q, x_mask, x_kv)

                self.assertAllClose(y, y_ref, rtol=0.01, atol=0.01, msg='y')
                self.assertAllClose(y_b, y_b_ref, msg='y_bias')

    @test_util.run_gpu_only
    def testMHABackward(self):
        use_fp8 = tf.test.is_gpu_available(True, (9, 0))
        batches, seq_q, seq_kv, hidden_states = 4, 8, 8, 32
        num_heads, depth = 4, 8
        hidden_size = num_heads * depth
        q_shape = (seq_q, batches, hidden_states)
        kv_shape = (seq_kv, batches, hidden_states)
        out_shape = (seq_q, batches, hidden_size)

        init = tf.keras.initializers.RandomUniform(minval=0., maxval=.1)
        x_q = tf.random.uniform(q_shape, minval=0., maxval=.1)
        x_kv = tf.random.uniform(kv_shape, minval=0., maxval=.1)
        dy = tf.random.uniform(out_shape, minval=0., maxval=1.)

        for attn_type in ('self', 'cross'):
            for use_mask in (False, True):
                mha_einsum = MultiHeadAttentionKeras(
                    hidden_size, num_heads, attn_type, init)
                mha = MultiHeadAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    kv_channels=depth,
                    attention_dropout=0.0,
                    attention_softmax_in_fp32=True,
                    init_method=init,
                    output_layer_init_method=init,
                    input_layernorm=False,
                    attention_type=attn_type,
                    attn_mask_type='padding',
                )

                x_mask = tf.random.uniform(
                    (seq_q, seq_kv)) > 0.5 if use_mask else None

                y_ref, dxs_ref, dvars_ref = train_step(
                    dy, x_q, x_kv, x_mask, mha_einsum, attn_type)

                fp8_recipe = DelayedScaling(
                    margin=0, interval=1, fp8_format=Format.HYBRID,
                    amax_compute_algo='max', amax_history_len=3)
                y, dxs, dvars = train_step(
                    dy, x_q, x_kv, x_mask, mha, attn_type, use_fp8, fp8_recipe)

                for dx, dx_ref in zip(dxs, dxs_ref):
                    self.assertAllClose(
                        dx, dx_ref, rtol=0.1, atol=0.1, msg='dx')

                if attn_type == 'cross':
                    # The variable lists are:
                    #   [q_w, kv_w, q_b, kv_b, proj_w, proj_b] (target)
                    #   [q_w, q_b, kv_w, kv_b, proj_w, proj_b] (reference)
                    self.assertEqual(len(dvars), 6)
                    self.assertEqual(len(dvars), len(dvars_ref))
                    dws = [dvars[i] for i in [0, 1, 4]]
                    dws_ref = [dvars_ref[i] for i in [0, 2, 4]]
                    dbs = [dvars[i] for i in [2, 3, 5]]
                    dbs_ref = [dvars_ref[i] for i in [1, 3, 5]]
                else:
                    # The variable lists are:
                    #   [qkv_w, qkv_b, proj_w, proj_b] (target)
                    #   [qkv_w, qkv_b, proj_w, proj_b] (reference)
                    self.assertEqual(len(dvars), 4)
                    self.assertEqual(len(dvars), len(dvars_ref))
                    dws = [dvars[i] for i in [0, 2]]
                    dws_ref = [dvars_ref[i] for i in [0, 2]]
                    dbs = [dvars[i] for i in [1, 3]]
                    dbs_ref = [dvars_ref[i] for i in [1, 3]]

                for dv, dv_ref in zip(dws, dws_ref):
                    self.assertAllClose(
                        dv, tf.reshape(dv_ref, dv.shape),
                        rtol=0.1, atol=0.1, msg='dkernel')
                for dv, dv_ref in zip(dbs, dbs_ref):
                    self.assertAllClose(dv, dv_ref, rtol=0.2,
                                        atol=0.2, msg='dbias')


if __name__ == '__main__':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    test.main()
