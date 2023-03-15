# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for Transformer Engine modules"""
import tensorflow as tf


def attention_mask_func(
    attention_scores: tf.Tensor, attention_mask: tf.Tensor
) -> tf.Tensor:
    """Get attention mask"""
    return tf.where(attention_mask, -10000.0, attention_scores)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert (
        numerator % denominator == 0
    ), f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
