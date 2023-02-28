"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import transformer_engine.tensorflow as te

from itertools import product
from tensorflow.keras import initializers, layers
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from transformer_engine.tensorflow import (
    Dense,
    DelayedScaling,
    Format,
    LayerNorm,
    LayerNormDense,
    LayerNormMLP,
)

def get_fp8_recipe(override_wgrad=False):
  fp8_recipe = DelayedScaling(
      margin=0, interval=1, fp8_format=Format.HYBRID,
      amax_compute_algo='max', amax_history_len=3,
      override_linear_precision=(False, False, override_wgrad))
  return fp8_recipe

def compute_scale(amax, scale, fp8_max, margin):
  """Default function to convert amax to scaling factor."""
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(2., tf.math.abs(exp)))
  sf = tf.where(amax > 0.0, sf, scale)
  sf = tf.where(tf.math.is_finite(amax), sf, scale)
  sf = tf.where(exp < 0, 1.0 / sf, sf)
  return sf

def update_scale(amax_h, scale, fp8_meta, is_fwd):
  key = "fp8_max_fwd" if is_fwd else "fp8_max_bwd"
  amax = tf.reduce_max(amax_h, axis=0)
  fp8_max = fp8_meta[key]
  margin = fp8_meta["recipe"].margin
  scale = compute_scale(amax, scale, fp8_max, margin)
  scale_inv = 1. / scale
  return scale, scale_inv

def roll_and_update(amax_h, update):
  amax_h = tf.roll(amax_h, shift=-1, axis=0)
  amax_h = tf.tensor_scatter_nd_update(amax_h, [[0]], [update])
  return amax_h

# This function is to recompute the results of layernorm bprop.
def get_adjusted_layernorm_dx(x, ln_dy, init):
  assert x.shape == ln_dy.shape
  ln_layer = layers.LayerNormalization(
      gamma_initializer=init,
      beta_initializer=init,
  )
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = ln_layer(x)
    loss = y * ln_dy
  ln_dx, (ln_dgamma, ln_dbeta) = tape.gradient(loss, [x, ln_layer.variables])
  return ln_dx, ln_dgamma, ln_dbeta

class LayersTest(test.TestCase):
  @test_util.run_gpu_only
  def testDenseFwd(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=1.)
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    dense_ref = layers.Dense(**dense_kwargs)
    dense = Dense(**dense_kwargs)

    x = tf.random.uniform((B, M, K))
    fp8_recipe = get_fp8_recipe()

    for use_fp8 in [False, True]:
      y_ref = dense_ref(x)
      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        y = dense(x)

      # The TE higher precision calls use the bias fusion, so they are not
      # exactly same with the TF calls.
      atol, rtol = (0.01, 0.05) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(y, y_ref, rtol, atol, msg=f"use_fp8={use_fp8}")

  @test_util.run_gpu_only
  def testDenseBwd(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=1.)
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    dense_ref = layers.Dense(**dense_kwargs)
    dense = Dense(**dense_kwargs)

    dy = tf.random.uniform((B, M, N))
    def _train_step(x, model, use_fp8=False, fp8_recipe=None):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
          y = model(x, training=True)
        loss = y * tf.cast(dy, y.dtype)
      dx, (dw, db) = tape.gradient(loss, [x, model.trainable_variables])
      return dx, dw, db

    x = tf.random.uniform((B, M, K))

    for use_fp8, use_override in product([False, True], repeat=2):
      recipe = get_fp8_recipe(use_override)
      dx_ref, dw_ref, db_ref = _train_step(x, dense_ref)
      dx, dw, db = _train_step(x, dense, use_fp8=use_fp8, fp8_recipe=recipe)

      assert_msg=f"use_fp8={use_fp8},use_override={use_override}"

      atol, rtol = (0.01, 0.05) if use_fp8 else (1e-6, 1e-6)
      self.assertAllClose(dx, dx_ref, rtol, atol, msg="dx," + assert_msg)
      self.assertAllClose(db, db_ref, rtol, atol, msg="db," + assert_msg)
      atol, rtol = \
          (0.01, 0.05) if use_fp8 and not use_override else (1e-6, 1e-6)
      self.assertAllClose(dw, dw_ref, rtol, atol, msg="dw," + assert_msg)

  @test_util.run_gpu_only
  def testDenseSkipWeight(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=1.)
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    dense_ref = layers.Dense(**dense_kwargs)
    dense = Dense(**dense_kwargs, skip_weight_param_allocation=True)

    x = tf.random.uniform((B, M, K))
    fp8_recipe = get_fp8_recipe()

    for use_fp8 in [False, True]:
      y_ref = dense_ref(x)
      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        y = dense(x, kernel=dense_ref.kernel, bias=dense_ref.bias)

      atol, rtol = (0.01, 0.05) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(y, y_ref, rtol, atol, msg=f"use_fp8={use_fp8}")

  @test_util.run_gpu_only
  def testDenseBookkeeping(self):
    M, K, N = 16, 16, 32
    init = initializers.RandomNormal(mean=0., stddev=1.)

    dense = Dense(N, kernel_initializer=init)
    fp8_recipe = get_fp8_recipe()

    def _train_step(x, dy):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
          y = dense(x, training=True)
        loss = y * tf.cast(dy, y.dtype)
      dx, dw = tape.gradient(loss, [x, dense.kernel])
      return dx, dw

    scale_fwd_ref = tf.ones((2,))
    scale_bwd_ref = tf.ones((1,))
    scale_inv_fwd_ref = 1. / scale_fwd_ref
    scale_inv_bwd_ref = 1. / scale_bwd_ref
    amax_h_fwd_ref = tf.zeros((fp8_recipe.amax_history_len, 2))
    amax_h_bwd_ref = tf.zeros((fp8_recipe.amax_history_len, 1))
    atol, rtol = 0.001, 0.001
    for step in range(5):
      x = tf.random.normal((M, K))
      dy = tf.random.normal((M, N))
      dx, dw = _train_step(x, dy)

      amax_x = tf.math.reduce_max(tf.math.abs(x))
      amax_w = tf.math.reduce_max(tf.math.abs(dense.kernel))
      amax_dy = tf.math.reduce_max(tf.math.abs(dy))
      amax_h_fwd_ref = roll_and_update(amax_h_fwd_ref, [amax_x, amax_w])
      amax_h_bwd_ref = roll_and_update(amax_h_bwd_ref, [amax_dy])

      amax_h_fwd = dense.fp8_meta['scaling_fwd']['amax_history']
      amax_h_bwd = dense.fp8_meta['scaling_bwd']['amax_history']
      scale_fwd = dense.fp8_meta['scaling_fwd']['scale']
      scale_bwd = dense.fp8_meta['scaling_bwd']['scale']
      scale_inv_fwd = dense.fp8_meta['scaling_fwd']['scale_inv']
      scale_inv_bwd = dense.fp8_meta['scaling_bwd']['scale_inv']

      self.assertAllClose(
          amax_h_fwd, amax_h_fwd_ref, rtol, atol, msg="amax_history_fwd")
      self.assertAllClose(
          amax_h_bwd, amax_h_bwd_ref, rtol, atol, msg="amax_history_bwd")
      self.assertAllClose(scale_fwd, scale_fwd_ref, rtol, atol, msg="scale_fwd")
      self.assertAllClose(scale_bwd, scale_bwd_ref, rtol, atol, msg="scale_bwd")
      self.assertAllClose(
          scale_inv_fwd, scale_inv_fwd_ref, rtol, atol, msg="scale_inv_fwd")
      self.assertAllClose(
          scale_inv_bwd, scale_inv_bwd_ref, rtol, atol, msg="scale_inv_bwd")

      scale_fwd_ref, scale_inv_fwd_ref = update_scale(
          amax_h_fwd_ref, scale_fwd_ref, dense.fp8_meta, is_fwd=True)

      scale_bwd_ref, scale_inv_bwd_ref = update_scale(
          amax_h_bwd_ref, scale_bwd_ref, dense.fp8_meta, is_fwd=False)

      # Apply an update to the kernel to mimic the gradient descent.
      dense.kernel.assign_add(tf.cast(dw, tf.float32) * 0.1)

  @test_util.run_gpu_only
  def testLayerNormFwd(self):
    B, M, N = 4, 16, 32
    init = initializers.RandomNormal(mean=0., stddev=1.)
    # The keras layer norm actually uses fp32 computation in mixed precision
    # mode. So, for better comparison, we use fp32 in both reference and target
    # layers.
    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
        "dtype": 'float32',
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    ln = LayerNorm(**ln_kwargs)

    x = tf.random.normal((B, M, N))

    y_ref = ln_ref(x)
    y = ln(x)

    self.assertAllClose(y, y_ref, msg="fwd_layer_norm:y")

  @test_util.run_gpu_only
  def testLayerNormBwd(self):
    B, M, N = 4, 16, 32
    init = initializers.RandomNormal(mean=0., stddev=1.)
    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
        "dtype": 'float32',
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    ln = LayerNorm(**ln_kwargs)

    dy = tf.random.uniform((B, M, N))
    def _train_step(x, model):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x, training=True)
        loss = y * tf.cast(dy, y.dtype)
      dx, (dg, dB) = tape.gradient(loss, [x, model.trainable_variables])
      return dx, dg, dB

    x = tf.random.uniform((B, M, N))

    dx_ref, dg_ref, dB_ref = _train_step(x, ln_ref)
    dx, dg, dB = _train_step(x, ln)

    self.assertAllClose(dx, dx_ref, msg="bwd_layer_norm:dx")
    self.assertAllClose(dB, dB_ref, msg="bwd_layer_norm:dbeta")
    self.assertAllClose(dg, dg_ref, msg="bwd_layer_norm:dgamma")

  @test_util.run_gpu_only
  def testLayerNormDenseFwd(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=1.)
    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
    }
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    dense_ref = layers.Dense(**dense_kwargs)
    
    x = tf.random.uniform((B, M, K))
    fp8_recipe = get_fp8_recipe()

    for use_fp8, output_ln in product([False, True], repeat=2):
      ln_dense = LayerNormDense(
          **ln_kwargs,
          **dense_kwargs,
          return_layernorm_output=output_ln,
      )

      y_ln_ref = ln_ref(x)
      y_ref = dense_ref(y_ln_ref)
      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        ys = ln_dense(x)
      if output_ln:
        y, y_ln = ys
      else:
        y = ys

      assert_msg=f"use_fp8={use_fp8},output_ln={output_ln}"
      atol, rtol = (0.01, 0.1) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(y, y_ref, rtol, atol, msg="y," + assert_msg)
      if output_ln:
        self.assertAllClose(
            y_ln, y_ln_ref, rtol, atol, msg="y_ln," + assert_msg)

  @test_util.run_gpu_only
  def testLayerNormDenseBwd(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=.1)
    dy = tf.random.uniform((B, M, N), minval=0., maxval=1.)
    x = tf.random.uniform((B, M, K), minval=0., maxval=1.)

    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
    }
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    dense_ref = layers.Dense(**dense_kwargs)

    ln_dense = LayerNormDense(**ln_kwargs, **dense_kwargs)

    def _train_step(x, model, use_fp8=False, fp8_recipe=None):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
          y = model(x, training=True)
        loss = y * tf.cast(dy, y.dtype)
      dx, (dg, dB, dw, db) = tape.gradient(
          loss, [x, model.trainable_variables])
      return dx, dg, dB, dw, db

    def _train_step_ref(x):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        t = ln_ref(x)
        y = dense_ref(t)
        loss = y * tf.cast(dy, y.dtype)
      var_list = ln_ref.variables + dense_ref.variables
      dx, dt, (dg, dB, dw, db) = tape.gradient(loss, [x, t, var_list])
      return dx, dt, dg, dB, dw, db

    for use_fp8, use_override in product([False, True], repeat=2):
      recipe = get_fp8_recipe(use_override)

      dx_ref, ln_dy_ref, dg_ref, dB_ref, dw_ref, db_ref = _train_step_ref(x)

      dx, dg, dB, dw, db = _train_step(
          x, ln_dense, use_fp8=use_fp8, fp8_recipe=recipe)
      
      assert_msg=f"use_fp8={use_fp8},use_override={use_override}"

      atol, rtol = (0.01, 0.1) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(db, db_ref, rtol, atol, msg="dbias," + assert_msg)
      self.assertAllClose(dw, dw_ref, rtol, atol, msg="dkernel," + assert_msg)

      atol, rtol = (0.1, 0.1) if use_fp8 else (1e-2, 1e-2)
      self.assertAllClose(dx, dx_ref, rtol, atol, msg="ln_dx," + assert_msg)
      self.assertAllClose(dg, dg_ref, rtol, atol, msg="dgamma," + assert_msg)
      self.assertAllClose(dB, dB_ref, rtol, atol, msg="dbeta," + assert_msg)

  @test_util.run_gpu_only
  def testLayerNormDenseSkipWeight(self):
    B, M, K, N = 4, 8, 16, 32
    init = initializers.RandomUniform(minval=0., maxval=1.)
    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
    }
    dense_kwargs = {
        "units": N,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    dense_ref = layers.Dense(**dense_kwargs)

    ln_dense = LayerNormDense(
        **ln_kwargs,
        **dense_kwargs,
        skip_weight_param_allocation=True,
    )

    x = tf.random.uniform((B, M, K))
    fp8_recipe = get_fp8_recipe()

    for use_fp8 in [False, True]:
      y_ref = dense_ref(ln_ref(x))
      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        y = ln_dense(x, kernel=dense_ref.kernel, bias=dense_ref.bias)

      atol, rtol = (0.01, 0.1) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(y, y_ref, rtol, atol, msg=f"use_fp8={use_fp8}")

  @test_util.run_gpu_only
  def testLayerNormMLPFwd(self):
    B, M, K, N, O = 4, 8, 16, 32, 64
    init = initializers.RandomUniform(minval=0., maxval=1.)
    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
    }
    dense_common_kwargs = {
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    dense1_ref = layers.Dense(**dense_common_kwargs, units=N)
    dense2_ref = layers.Dense(**dense_common_kwargs, units=O)
    
    x = tf.random.uniform((B, M, K))
    fp8_recipe = get_fp8_recipe()

    for use_fp8, output_ln in product([False, True], repeat=2):
      ln_mlp = LayerNormMLP(
          **ln_kwargs,
          **dense_common_kwargs,
          units=N,
          ffn_units=O,
          ffn_kernel_initializer=init,
          return_layernorm_output=output_ln,
      )

      y_ln_ref = ln_ref(x)
      y_dense1_ref = dense1_ref(y_ln_ref)
      y_gelu_ref = tf.nn.gelu(y_dense1_ref, approximate=True)
      y_ref = dense2_ref(y_gelu_ref)

      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        ys = ln_mlp(x)
      if output_ln:
        y, y_ln = ys
      else:
        y = ys

      assert_msg=f"use_fp8={use_fp8},output_ln={output_ln}"
      atol, rtol = (0.01, 0.1) if use_fp8 else (1e-3, 2e-3)
      self.assertAllClose(y, y_ref, rtol, atol, msg="y," + assert_msg)
      if output_ln:
        self.assertAllClose(
            y_ln, y_ln_ref, rtol, atol, msg="y_ln," + assert_msg)

  @test_util.run_gpu_only
  def testLayerNormMLPBwd(self):
    B, M, K, N, O = 4, 8, 16, 32, 64
    init = initializers.RandomUniform(minval=0., maxval=.1)
    dy = tf.random.uniform((B, M, O), minval=0., maxval=1.)
    x = tf.random.uniform((B, M, K), minval=0., maxval=1.)

    ln_kwargs = {
        "gamma_initializer": init,
        "beta_initializer": init,
    }
    dense_common_kwargs = {
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    ln_ref = layers.LayerNormalization(**ln_kwargs)
    dense1_ref = layers.Dense(**dense_common_kwargs, units=N)
    dense2_ref = layers.Dense(**dense_common_kwargs, units=O)

    ln_mlp = LayerNormMLP(
        **ln_kwargs,
        **dense_common_kwargs,
        units=N,
        ffn_units=O,
        ffn_kernel_initializer=init,
    )

    def _train_step(x, model, use_fp8=False, fp8_recipe=None):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
          y = model(x, training=True)
        loss = y * tf.cast(dy, y.dtype)
      dx, (dg, dB, dw1, db1, dw2, db2) = tape.gradient(
          loss, [x, model.trainable_variables])
      return dx, dg, dB, dw1, db1, dw2, db2

    def _train_step_ref(x):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        t = ln_ref(x)
        y_gelu = tf.nn.gelu(dense1_ref(t), approximate=True)
        y = dense2_ref(y_gelu)
        loss = y * tf.cast(dy, y.dtype)
      var_list = ln_ref.variables + dense1_ref.variables + dense2_ref.variables
      dx, dt, (dg, dB, dw1, db1, dw2, db2) = tape.gradient(
          loss, [x, t, var_list])
      return dx, dt, dg, dB, dw1, db1, dw2, db2

    for use_fp8, use_override in product([False, True], repeat=2):
      recipe = get_fp8_recipe(use_override)

      dx_ref, ln_dy_ref, dg_ref, dB_ref, dw1_ref, db1_ref, dw2_ref, db2_ref = \
          _train_step_ref(x)

      dx, dg, dB, dw1, db1, dw2, db2 = _train_step(
          x, ln_mlp, use_fp8=use_fp8, fp8_recipe=recipe)

      assert_msg=f"use_fp8={use_fp8},use_override={use_override}"

      atol, rtol = (0.01, 0.1) if use_fp8 else (1e-3, 1e-3)
      self.assertAllClose(
          db2, db2_ref, rtol, atol, msg="fc2_dbias," + assert_msg)
      self.assertAllClose(
          dw2, dw2_ref, rtol, atol, msg="fc2_dw," + assert_msg)
      self.assertAllClose(
          db1, db1_ref, rtol, atol, msg="fc1_dbias," + assert_msg)
      self.assertAllClose(
          dw1, dw1_ref, rtol, atol, msg="fc1_dw," + assert_msg)

      atol, rtol = (0.1, 0.1) if use_fp8 else (1e-2, 1e-2)
      self.assertAllClose(dx, dx_ref, rtol, atol, msg="ln_dx," + assert_msg)
      self.assertAllClose(dg, dg_ref, rtol, atol, msg="dgamma," + assert_msg)
      self.assertAllClose(dB, dB_ref, rtol, atol, msg="dbeta," + assert_msg)

if __name__ == '__main__':
  tf.keras.mixed_precision.set_global_policy('mixed_float16')
  test.main()

      


