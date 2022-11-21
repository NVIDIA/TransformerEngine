# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

from functools import partial
import pytest

import flax

import jax
import jax.numpy as jnp

from utils import is_fp8_supported
from utils import EncoderLayer as RefEncoderLayer
from utils import DecoderLayer as RefDecoderLayer

from transformer_engine.jax.fp8 import FP8Helper
from transformer_engine.jax import TransformerLayer, TransformerLayerType
from transformer_engine.common.recipe import Format


def loss_fn(xs, params, others, model, rngs):
    output = model.apply({"params": params, **others}, *xs, rngs=rngs)
    return jnp.mean(output)


def generate_test_rngs():
    data_rng = jax.random.PRNGKey(0)
    init_rng = {
        'params': jax.random.PRNGKey(1),
        'dropout': jax.random.PRNGKey(2)
    }
    apply_rng = {'dropout': jax.random.PRNGKey(3)}
    return data_rng, init_rng, apply_rng


def generate_layer(layer_cls, init_rng, inputs):
    layer = layer_cls()
    variables = layer.init(init_rng, *inputs)
    others, params = variables.pop('params')
    del variables
    return layer, params, others


def compare_frozen_dict(ref_fd, test_fd, rtol=1e-05, atol=1e-08):
    for key in ref_fd:
        assert key in test_fd, \
            f"{key} not found in test FrozenDict {test_fd}"
        assert isinstance(test_fd[key], type(ref_fd[key])), \
            f"The data type is not match between ref and test " \
            f"FrozenDict on key: {key}"
        if isinstance(ref_fd[key], flax.core.frozen_dict.FrozenDict):
            compare_frozen_dict(ref_fd[key], test_fd[key], rtol, atol)
        else:
            assert jnp.allclose(ref_fd[key],
                                test_fd[key],
                                rtol=rtol,
                                atol=atol)


DATA_SHAPE = [(128, 32, 512), (512, 32, 512)]  # (seqlen, batch, emb_dim)
DTYPE = [jnp.float32, jnp.bfloat16]
FP8_FORMATS = [Format.E4M3, Format.HYBRID]

_KEY_OF_RESIDUAL_POST_LAYERNORM = "apply_residual_connection_post_layernorm"
_KEY_OF_OUTPUT_LAYERNORM = "output_layernorm"
_KEY_OF_DROP_PATH = "drop_path"
_KEY_OF_FUSE_QKV_PARAMS = "fuse_qkv_params"
_KEY_OF_DROPOUT_RATE = "dropout_rate"
_KEY_OF_MLP_ACTIVATIONS = "mlp_activations"
_KEY_OF_FUSE_MLP_WI = "fuse_mlp_wi"
ATTRS = [{}, {
    _KEY_OF_RESIDUAL_POST_LAYERNORM: True
}, {
    _KEY_OF_OUTPUT_LAYERNORM: True
}, {
    _KEY_OF_RESIDUAL_POST_LAYERNORM: True,
    _KEY_OF_OUTPUT_LAYERNORM: True
}, {
    _KEY_OF_DROP_PATH: 0.1
}, {
    _KEY_OF_FUSE_QKV_PARAMS: False
}, {
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_MLP_ACTIVATIONS: (('gelu', 'linear')),
    _KEY_OF_FUSE_MLP_WI: True
}]


class TestEncoderLayer():

    @staticmethod
    def sync_params(ref, target, attrs):
        fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

        unfreeze_target = target.unfreeze()
        if fuse_qkv:
            unfreeze_target['attention']['qkv']['kernel'] = \
                ref['attention']['qkv']['kernel']
        else:
            unfreeze_target['attention']['query']['kernel'] = \
                ref['attention']['query']['kernel']
            unfreeze_target['attention']['key']['kernel'] = \
                ref['attention']['key']['kernel']
            unfreeze_target['attention']['value']['kernel'] = \
                ref['attention']['value']['kernel']
        unfreeze_target['mlp']['wi_kernel'] = \
            ref['mlp']['wi']['kernel']
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, flax.core.frozen_dict.FrozenDict(unfreeze_target)

    def forward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype), )

        te_layer_attrs = {}
        for k, v in attrs.items():
            if k == 'dropout_rate':
                te_layer_attrs['attention_dropout'] = v
                te_layer_attrs['hidden_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefEncoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            layer_type=TransformerLayerType.ENCODER,
                            dtype=dtype,
                            **te_layer_attrs)

        ref_layer, ref_params, ref_others = generate_layer(
            ref_layer_cls, init_rng, inputs)
        test_layer, test_params, test_others = generate_layer(
            layer_cls, init_rng, inputs)

        ref_params, test_params = TestEncoderLayer.sync_params(
            ref_params, test_params, attrs)

        ref_out = loss_fn(inputs, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_params, test_others, test_layer,
                           apply_rng)

        assert jnp.allclose(ref_out, test_out, rtol=rtol, atol=atol)

        del data_rng, init_rng, apply_rng

    def forward_backward_runner(self,
                                data_shape,
                                dtype,
                                attrs,
                                rtol=1e-05,
                                atol=1e-08):
        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype), )

        te_layer_attrs = {}
        for k, v in attrs.items():
            if k == 'dropout_rate':
                te_layer_attrs['attention_dropout'] = v
                te_layer_attrs['hidden_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefEncoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            layer_type=TransformerLayerType.ENCODER,
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(
            ref_layer_cls, init_rng, inputs)
        test_layer, test_params, test_others = generate_layer(
            layer_cls, init_rng, inputs)

        ref_params, test_params = TestEncoderLayer.sync_params(
            ref_params, test_params, attrs)

        if FP8Helper.enable_fp8():
            _, tmp_grad = jax.value_and_grad(loss_fn,
                                             argnums=(2, ),
                                             has_aux=False)(inputs,
                                                            test_params,
                                                            test_others,
                                                            test_layer,
                                                            apply_rng)
            _, fp8_meta_grad = tmp_grad[0].pop(FP8Helper.FP8_COLLECTION_NAME)
            test_others = FP8Helper.update_collections(
                {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
            test_others = FP8Helper.update_fp8_metas(test_others)
            del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_params, test_others,
                                       test_layer, apply_rng)

        assert jnp.allclose(ref_out, test_out, rtol=rtol, atol=atol)
        assert jnp.allclose(ref_grads[0][0],
                            test_grads[0][0],
                            rtol=rtol,
                            atol=atol)  # dgrad

        def reorganize_test_wgrad(test_wgrad, attrs):
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

            unfreeze_test_wgrad = test_wgrad.unfreeze()
            if "output_layernorm" not in attrs:
                unfreeze_test_wgrad['pre_attention_layer_norm'] = {}
                pre_attn_layer_key = 'qkv' if fuse_qkv else 'query'
                unfreeze_test_wgrad['pre_attention_layer_norm']['scale'] = \
                    unfreeze_test_wgrad['attention'][pre_attn_layer_key]['scale']
                del unfreeze_test_wgrad['attention'][pre_attn_layer_key][
                    'scale']
            unfreeze_test_wgrad['pre_mlp_layer_norm'] = {}
            unfreeze_test_wgrad['pre_mlp_layer_norm']['scale'] = \
                unfreeze_test_wgrad['mlp']['scale']
            del unfreeze_test_wgrad['mlp']['scale']
            unfreeze_test_wgrad['mlp']['wi'] = {}
            unfreeze_test_wgrad['mlp']['wi']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wi_kernel']
            del unfreeze_test_wgrad['mlp']['wi_kernel']
            unfreeze_test_wgrad['mlp']['wo'] = {}
            unfreeze_test_wgrad['mlp']['wo']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wo_kernel']
            del unfreeze_test_wgrad['mlp']['wo_kernel']
            return flax.core.frozen_dict.FrozenDict(unfreeze_test_wgrad)

        compare_frozen_dict(ref_grads[1],
                            reorganize_test_wgrad(test_grads[1], attrs),
                            rtol=rtol,
                            atol=atol)  # wgrad

        del data_rng, init_rng, apply_rng

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()  # Ensure FP8 disabled.
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=1e-05)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_with_fp8(self, data_shape, dtype, fp8_format, attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-04, atol=1e-03)
        FP8Helper.finalize()

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()  # Ensure FP8 disabled.
        self.forward_backward_runner(data_shape,
                                     dtype,
                                     attrs,
                                     rtol=1e-05,
                                     atol=1e-05)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward_with_fp8(self, data_shape, dtype, fp8_format,
                                       attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_backward_runner(data_shape,
                                     dtype,
                                     attrs,
                                     rtol=1e-04,
                                     atol=1e-03)
        FP8Helper.finalize()


class TestDecoderLayer():

    @staticmethod
    def sync_params(ref, target, attrs):
        fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

        unfreeze_target = target.unfreeze()
        if fuse_qkv:
            unfreeze_target['self_attention']['qkv']['kernel'] = \
                ref['self_attention']['qkv']['kernel']
        else:
            unfreeze_target['self_attention']['query']['kernel'] = \
                ref['self_attention']['query']['kernel']
            unfreeze_target['self_attention']['key']['kernel'] = \
                ref['self_attention']['key']['kernel']
            unfreeze_target['self_attention']['value']['kernel'] = \
                ref['self_attention']['value']['kernel']
        unfreeze_target['encoder_decoder_attention']['query']['kernel'] = \
            ref['encoder_decoder_attention']['query']['kernel']
        unfreeze_target['mlp']['wi_kernel'] = \
            ref['mlp']['wi']['kernel']
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, flax.core.frozen_dict.FrozenDict(unfreeze_target)

    def forward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype),
                  jax.random.normal(data_rng, data_shape, dtype))

        te_layer_attrs = {}
        for k, v in attrs.items():
            if k == 'dropout_rate':
                te_layer_attrs['attention_dropout'] = v
                te_layer_attrs['hidden_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefDecoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            layer_type=TransformerLayerType.DECODER,
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(
            ref_layer_cls, init_rng, inputs)
        test_layer, test_params, test_others = generate_layer(
            layer_cls, init_rng, inputs)

        ref_params, test_params = TestDecoderLayer.sync_params(
            ref_params, test_params, attrs)

        ref_out = loss_fn(inputs, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_params, test_others, test_layer,
                           apply_rng)

        assert jnp.allclose(ref_out, test_out, rtol=rtol, atol=atol)

        del data_rng, init_rng, apply_rng

    def forward_backward_runner(self,
                                data_shape,
                                dtype,
                                attrs,
                                rtol=1e-05,
                                atol=1e-08):
        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype),
                  jax.random.normal(data_rng, data_shape, dtype))

        te_layer_attrs = {}
        for k, v in attrs.items():
            if k == 'dropout_rate':
                te_layer_attrs['attention_dropout'] = v
                te_layer_attrs['hidden_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefDecoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            layer_type=TransformerLayerType.DECODER,
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(
            ref_layer_cls, init_rng, inputs)
        test_layer, test_params, test_others = generate_layer(
            layer_cls, init_rng, inputs)

        ref_params, test_params = TestDecoderLayer.sync_params(
            ref_params, test_params, attrs)

        if FP8Helper.enable_fp8():
            _, tmp_grad = jax.value_and_grad(loss_fn,
                                             argnums=(2, ),
                                             has_aux=False)(inputs,
                                                            test_params,
                                                            test_others,
                                                            test_layer,
                                                            apply_rng)
            _, fp8_meta_grad = tmp_grad[0].pop(FP8Helper.FP8_COLLECTION_NAME)
            test_others = FP8Helper.update_collections(
                {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
            test_others = FP8Helper.update_fp8_metas(test_others)
            del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_params, test_others,
                                       test_layer, apply_rng)

        assert jnp.allclose(ref_out, test_out, rtol=rtol, atol=atol)
        assert jnp.allclose(ref_grads[0][0],
                            test_grads[0][0],
                            rtol=rtol,
                            atol=atol)  # dgrad

        def reorganize_test_wgrad(test_wgrad, attrs):
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

            unfreeze_test_wgrad = test_wgrad.unfreeze()
            if "output_layernorm" not in attrs:
                unfreeze_test_wgrad['pre_self_attention_layer_norm'] = {}
                pre_attn_layer_key = 'qkv' if fuse_qkv else 'query'
                unfreeze_test_wgrad['pre_self_attention_layer_norm']['scale'] = \
                    unfreeze_test_wgrad['self_attention'][pre_attn_layer_key]['scale']
                del unfreeze_test_wgrad['self_attention'][pre_attn_layer_key][
                    'scale']

            unfreeze_test_wgrad['pre_cross_attention_layer_norm'] = {}
            unfreeze_test_wgrad['pre_cross_attention_layer_norm']['scale'] = \
                unfreeze_test_wgrad['encoder_decoder_attention']['query']['scale']
            del unfreeze_test_wgrad['encoder_decoder_attention']['query'][
                'scale']
            unfreeze_test_wgrad['pre_mlp_layer_norm'] = {}
            unfreeze_test_wgrad['pre_mlp_layer_norm']['scale'] = \
                unfreeze_test_wgrad['mlp']['scale']
            del unfreeze_test_wgrad['mlp']['scale']
            unfreeze_test_wgrad['mlp']['wi'] = {}
            unfreeze_test_wgrad['mlp']['wi']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wi_kernel']
            del unfreeze_test_wgrad['mlp']['wi_kernel']
            unfreeze_test_wgrad['mlp']['wo'] = {}
            unfreeze_test_wgrad['mlp']['wo']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wo_kernel']
            del unfreeze_test_wgrad['mlp']['wo_kernel']
            return flax.core.frozen_dict.FrozenDict(unfreeze_test_wgrad)

        compare_frozen_dict(ref_grads[1],
                            reorganize_test_wgrad(test_grads[1], attrs),
                            rtol=rtol,
                            atol=atol)  # wgrad

        del data_rng, init_rng, apply_rng

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()  # Ensure FP8 disabled.
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=1e-05)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_with_fp8(self, data_shape, dtype, fp8_format, attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-04, atol=3e-02)
        FP8Helper.finalize()

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()  # Ensure FP8 disabled.
        self.forward_backward_runner(data_shape,
                                     dtype,
                                     attrs,
                                     rtol=1e-05,
                                     atol=1e-05)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward_with_fp8(self, data_shape, dtype, fp8_format,
                                       attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_backward_runner(data_shape,
                                     dtype,
                                     attrs,
                                     rtol=1e-04,
                                     atol=3e-02)
        FP8Helper.finalize()
