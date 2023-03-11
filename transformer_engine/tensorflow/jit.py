"""XLA functions and JIT utilities"""
from typing import Callable

import tensorflow as tf


@tf.function(jit_compile=True)
def _bgrad_dgelu_fused(grad_output, inp):
    """Bgrad-Dgelu fused"""
    x = inp
    tanh_out = tf.math.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    dgelu = ff * grad_output
    bgrad = tf.math.reduce_sum(dgelu, axis=0)
    return bgrad, dgelu


def bgrad_dgelu_fused(grad_output, inp):
    """Bgrad-Dgelu fused"""
    return _bgrad_dgelu_fused(grad_output, inp)


def bias_dropout_add(
    x: tf.Tensor,
    bias: tf.Variable,
    residual: tf.Tensor,
    prob: float,
    training: bool,
) -> tf.Tensor:
    """dropout(inp + bias) + residual"""
    # TODO(kaixih): Use stateless_dropout and specify the seed mainly for
    # debugging purpose. Should allow random seed.
    out = (
        tf.nn.experimental.stateless_dropout(
            x + bias,
            rate=prob,
            seed=[1, 0],
        )
        if training
        else x + bias
    )

    out = residual + out
    return out


def get_bias_dropout_add(training: bool) -> Callable:
    """bias_dropout_add based on training or not"""

    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@tf.function(jit_compile=True)
def bias_dropout_add_fused_train_(
    x: tf.Tensor,
    bias: tf.Variable,
    residual: tf.Tensor,
    prob: float,
) -> tf.Tensor:
    """Jit fused bias_dropout_add for training"""
    return bias_dropout_add(x, bias, residual, prob, True)


def bias_dropout_add_fused_train(
    x: tf.Tensor,
    bias: tf.Variable,
    residual: tf.Tensor,
    prob: float,
) -> tf.Tensor:
    """Jit fused bias_dropout_add for training"""
    return bias_dropout_add_fused_train_(x, bias, residual, prob)


@tf.function(jit_compile=True)
def bias_dropout_add_fused_inference_(
    x: tf.Tensor,
    bias: tf.Variable,
    residual: tf.Tensor,
    prob: float,
) -> tf.Tensor:
    """Jit fused bias_dropout_add for inference"""
    return bias_dropout_add(x, bias, residual, prob, False)


def bias_dropout_add_fused_inference(
    x: tf.Tensor,
    bias: tf.Variable,
    residual: tf.Tensor,
    prob: float,
) -> tf.Tensor:
    """Jit fused bias_dropout_add for inference"""
    return bias_dropout_add_fused_inference_(x, bias, residual, prob)
