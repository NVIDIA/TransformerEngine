"""Enums for e2e transformer"""
import tensorflow as tf
import transformer_engine_tensorflow as tex


"""
This is a map: tf.dtype -> int
Used for passing dtypes into cuda
extension. Has one to one mapping
with enum in transformer_engine.h
"""
TE_DType = {
    tf.int8: tex.DType.kByte,
    tf.int32: tex.DType.kInt32,
    tf.float32: tex.DType.kFloat32,
    tf.half: tex.DType.kFloat16,
    tf.bfloat16: tex.DType.kBFloat16,
}

AttnMaskTypes = ("causal", "padding")

AttnTypes = ("self", "cross")

LayerTypes = ("encoder", "decoder")
