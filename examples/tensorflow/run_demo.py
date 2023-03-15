# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.tensorflow as te
import tensorflow as tf

from transformer_engine.tensorflow import Dense, DelayedScaling, Format

input_shape = (16, 32)
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=12)

my_dense = Dense(16, kernel_initializer=initializer, use_bias=True,
                 bias_initializer=initializer)
my_dense.build(input_shape=input_shape)

fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID,
                            amax_compute_algo='max', amax_history_len=3)

def train_step(x, use_fp8):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    kernel = my_dense.kernel
    bias = my_dense.bias
    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
      y = my_dense(x, training=True)
    loss = tf.reduce_sum(y)
  dx, dweight, dbias = tape.gradient(loss, [x, kernel, bias])
  return y, dx, dweight, dbias

for i in range(4):
  x = tf.random.normal(input_shape)

  y, dx, dw, dbias = train_step(x, True)
  
  print("Iteration:", i)
  print("Amax history fwd:", my_dense.fp8_meta["scaling_fwd"]["amax_history"])
  print("Amax history bwd:", my_dense.fp8_meta["scaling_bwd"]["amax_history"])

  print("Results:", y[0])
  print("Bwd dx:", dx[0])
  print("Bwd dweight:", dw[0:2])
  print("Bwd dbias:", dbias)
