# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from functools import partial

import flax
import jax
import jax.numpy as jnp
import pytest

from transformer_engine.common.recipe import Format
from transformer_engine.jax.flax import TransformerLayer, TransformerLayerType
from transformer_engine.jax.fp8 import FP8Helper, is_fp8_available
from utils import assert_allclose
from utils import DecoderLayer as RefDecoderLayer
from utils import EncoderLayer as RefEncoderLayer

is_fp8_supported, reason = is_fp8_available()


def loss_fn(diff_xs, no_diff_xs, params, others, model, rngs):
    output = model.apply({"params": params, **others}, *diff_xs, *no_diff_xs, rngs=rngs)
    return jnp.mean(output)


def generate_test_rngs():
    data_rng = jax.random.PRNGKey(0)
    init_rng = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    apply_rng = {'dropout': jax.random.PRNGKey(3)}
    return data_rng, init_rng, apply_rng


def generate_layer(layer_cls, init_rng, diff_inputs, no_diff_inputs):
    layer = layer_cls()
    variables = layer.init(init_rng, *diff_inputs, *no_diff_inputs)
    others, params = variables.pop('params')
    del variables
    return layer, params, others


def compare_frozen_dict(ref_fd, test_fd, rtol=1e-05, atol=1e-08):
    for key in ref_fd:
        assert key in test_fd, \
            f"{key} not found in test FrozenDict {test_fd}"
        assert isinstance(test_fd[key], type(ref_fd[key])), \
            f"The data type is not match between ref and test " \
            f"FrozenDict on {key=}"
        if isinstance(ref_fd[key], flax.core.frozen_dict.FrozenDict):
            compare_frozen_dict(ref_fd[key], test_fd[key], rtol, atol)
        else:
            assert_allclose(ref_fd[key],
                            test_fd[key],
                            rtol=rtol,
                            atol=atol,
                            err_msg=f"{key=} is not close")


DATA_SHAPE = [(32, 128, 1024), (32, 512, 1024)]    # (batch, seqlen, emb_dim)
DTYPE = [jnp.float32, jnp.bfloat16]
FP8_FORMATS = [Format.E4M3, Format.HYBRID]

_KEY_OF_RESIDUAL_POST_LAYERNORM = "apply_residual_connection_post_layernorm"
_KEY_OF_OUTPUT_LAYERNORM = "output_layernorm"
_KEY_OF_DROP_PATH = "drop_path"
_KEY_OF_FUSE_QKV_PARAMS = "fuse_qkv_params"
_KEY_OF_DROPOUT_RATE = "dropout_rate"
_KEY_OF_MLP_ACTIVATIONS = "mlp_activations"
_KEY_OF_FUSE_MLP_WI = "fuse_mlp_wi"
_KEY_OF_LAYERNORM_TYPE = 'layernorm_type'
_KEY_OF_ZERO_CENTERED_GAMMA = 'zero_centered_gamma'
_KEY_OF_TRANSPOSE_BS = 'transpose_batch_sequence'
_KEY_OF_SCALE_ATTN_LOGITS = "scale_attn_logits"

BASE_ATTRS = {_KEY_OF_TRANSPOSE_BS: True}

ATTRS = [{
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
}, {
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
}, {
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_ZERO_CENTERED_GAMMA: True
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_RESIDUAL_POST_LAYERNORM: True
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_OUTPUT_LAYERNORM: True
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_RESIDUAL_POST_LAYERNORM: True,
    _KEY_OF_OUTPUT_LAYERNORM: True
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_DROP_PATH: 0.1
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_FUSE_QKV_PARAMS: False
}, {
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_MLP_ACTIVATIONS: (('gelu', 'linear')),
    _KEY_OF_FUSE_MLP_WI: True
}, {
    _KEY_OF_SCALE_ATTN_LOGITS: True,
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_DROPOUT_RATE: 0.8,
    _KEY_OF_MLP_ACTIVATIONS: (('gelu', 'linear')),
    _KEY_OF_FUSE_MLP_WI: True
}, {
    _KEY_OF_TRANSPOSE_BS: False,
    _KEY_OF_SCALE_ATTN_LOGITS: True,
    _KEY_OF_LAYERNORM_TYPE: 'rmsnorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_MLP_ACTIVATIONS: (('gelu', 'linear')),
    _KEY_OF_FUSE_MLP_WI: True
}]

ATTRS = [{**BASE_ATTRS, **attr} for attr in ATTRS]


class TestEncoderLayer:

    @staticmethod
    def sync_params(ref, target, attrs):
        fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

        unfreeze_target = target.unfreeze()
        if fuse_qkv:
            unfreeze_target['attention']['qkv']['kernel'] = \
                jnp.reshape(ref['attention']['qkv']['kernel'],
                unfreeze_target['attention']['qkv']['kernel'].shape)
        else:
            unfreeze_target['attention']['query']['kernel'] = \
                ref['attention']['query']['kernel']
            unfreeze_target['attention']['key']['kernel'] = \
                ref['attention']['key']['kernel']
            unfreeze_target['attention']['value']['kernel'] = \
                ref['attention']['value']['kernel']
        unfreeze_target['mlp']['wi_kernel'] = \
            jnp.reshape(ref['mlp']['wi']['kernel'], unfreeze_target['mlp']['wi_kernel'].shape)
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, flax.core.frozen_dict.FrozenDict(unfreeze_target)

    def forward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        transpose_batch_sequence = _KEY_OF_TRANSPOSE_BS in attrs and attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])
        sequence_dim = 0 if transpose_batch_sequence else 1

        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype),)

        padded_mask = jnp.zeros((batch, 1, seqlen, seqlen), dtype=jnp.uint8)
        ref_masks = (1 - padded_mask,)
        test_masks = (None, padded_mask)    # The second arg of Transformer is encoded tokens.

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
                            hidden_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.ENCODER,
                            self_attn_mask_type='padding',
                            dtype=dtype,
                            **te_layer_attrs)

        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestEncoderLayer.sync_params(ref_params, test_params, attrs)

        ref_out = loss_fn(inputs, ref_masks, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_masks, test_params, test_others, test_layer, apply_rng)

        assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)

        del data_rng, init_rng, apply_rng

    def forward_backward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        transpose_batch_sequence = _KEY_OF_TRANSPOSE_BS in attrs and attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])
        sequence_dim = 0 if transpose_batch_sequence else 1

        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape, dtype),)

        padded_mask = jnp.zeros((batch, 1, seqlen, seqlen), dtype=jnp.uint8)
        ref_masks = (1 - padded_mask,)
        test_masks = (None, padded_mask)    # The second arg of Transformer is encoded tokens.

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
                            hidden_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.ENCODER,
                            self_attn_mask_type='padding',
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestEncoderLayer.sync_params(ref_params, test_params, attrs)

        if FP8Helper.is_fp8_enabled():
            for _ in range(4):
                _, tmp_grad = jax.value_and_grad(loss_fn, argnums=(3,),
                                                 has_aux=False)(inputs, test_masks, test_params,
                                                                test_others, test_layer, apply_rng)
                _, fp8_meta_grad = tmp_grad[0].pop(FP8Helper.FP8_COLLECTION_NAME)
                test_others = FP8Helper.update_collections(
                    {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
                test_others = FP8Helper.update_fp8_metas(test_others)
                del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 2), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_masks, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_masks, test_params, test_others, test_layer,
                                       apply_rng)

        assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)
        assert_allclose(ref_grads[0][0], test_grads[0][0], rtol=rtol, atol=atol)    # dgrad

        def reorganize_test_wgrad(test_wgrad, attrs):
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

            attn_name = 'attention'
            unfreeze_test_wgrad = test_wgrad.unfreeze()
            if "output_layernorm" not in attrs:
                unfreeze_test_wgrad['pre_attention_layer_norm'] = {}
                pre_attn_layer_key = 'qkv' if fuse_qkv else 'query'
                unfreeze_test_wgrad['pre_attention_layer_norm']['scale'] = \
                    unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                if 'ln_bias' in unfreeze_test_wgrad[attn_name][pre_attn_layer_key]:
                    unfreeze_test_wgrad['pre_attention_layer_norm']['ln_bias'] = \
                        unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']
                    del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']
            if fuse_qkv:
                unfreeze_test_wgrad[attn_name]['qkv']['kernel'] = \
                    jnp.reshape(unfreeze_test_wgrad[attn_name]['qkv']['kernel'],
                        (unfreeze_test_wgrad[attn_name]['qkv']['kernel'].shape[0], -1))
            unfreeze_test_wgrad['pre_mlp_layer_norm'] = {}
            unfreeze_test_wgrad['pre_mlp_layer_norm']['scale'] = \
                unfreeze_test_wgrad['mlp']['scale']
            del unfreeze_test_wgrad['mlp']['scale']
            if 'ln_bias' in unfreeze_test_wgrad['mlp']:
                unfreeze_test_wgrad['pre_mlp_layer_norm']['ln_bias'] = \
                    unfreeze_test_wgrad['mlp']['ln_bias']
                del unfreeze_test_wgrad['mlp']['ln_bias']
            unfreeze_test_wgrad['mlp']['wi'] = {}
            unfreeze_test_wgrad['mlp']['wi']['kernel'] = \
                jnp.reshape(unfreeze_test_wgrad['mlp']['wi_kernel'],
                            (unfreeze_test_wgrad['mlp']['wi_kernel'].shape[0], -1))
            del unfreeze_test_wgrad['mlp']['wi_kernel']
            unfreeze_test_wgrad['mlp']['wo'] = {}
            unfreeze_test_wgrad['mlp']['wo']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wo_kernel']
            del unfreeze_test_wgrad['mlp']['wo_kernel']
            return flax.core.frozen_dict.FrozenDict(unfreeze_test_wgrad)

        compare_frozen_dict(ref_grads[1],
                            reorganize_test_wgrad(test_grads[1], attrs),
                            rtol=rtol,
                            atol=atol)    # wgrad

        del data_rng, init_rng, apply_rng

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()    # Ensure FP8 disabled.
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=2e-04)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
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
        FP8Helper.finalize()    # Ensure FP8 disabled.
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=2e-04)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward_with_fp8(self, data_shape, dtype, fp8_format, attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-04, atol=1e-03)
        FP8Helper.finalize()


class TestDecoderLayer:

    @staticmethod
    def sync_params(ref, target, attrs):
        fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)

        unfreeze_target = target.unfreeze()
        if fuse_qkv:
            unfreeze_target['self_attention']['qkv']['kernel'] = \
                jnp.reshape(ref['self_attention']['qkv']['kernel'],
                unfreeze_target['self_attention']['qkv']['kernel'].shape)
            unfreeze_target['encoder_decoder_attention']['kv']['kernel'] = \
                jnp.reshape(ref['encoder_decoder_attention']['kv']['kernel'],
                unfreeze_target['encoder_decoder_attention']['kv']['kernel'].shape)
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
            jnp.reshape(ref['mlp']['wi']['kernel'], unfreeze_target['mlp']['wi_kernel'].shape)
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, flax.core.frozen_dict.FrozenDict(unfreeze_target)

    def forward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        transpose_batch_sequence = _KEY_OF_TRANSPOSE_BS in attrs and attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])
        sequence_dim = 0 if transpose_batch_sequence else 1

        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape,
                                    dtype), jax.random.normal(data_rng, data_shape, dtype))

        padded_mask = jnp.zeros((batch, 1, seqlen, seqlen), dtype=jnp.uint8)
        causal_mask = jnp.triu(jnp.ones((batch, 1, seqlen, seqlen), dtype=jnp.uint8), k=1)
        ref_masks = (1 - causal_mask, 1 - padded_mask)
        test_masks = (causal_mask, padded_mask)

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
                            hidden_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.DECODER,
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestDecoderLayer.sync_params(ref_params, test_params, attrs)

        ref_out = loss_fn(inputs, ref_masks, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_masks, test_params, test_others, test_layer, apply_rng)

        assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)

        del data_rng, init_rng, apply_rng

    def forward_backward_runner(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        transpose_batch_sequence = _KEY_OF_TRANSPOSE_BS in attrs and attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])
        sequence_dim = 0 if transpose_batch_sequence else 1

        data_rng, init_rng, apply_rng = generate_test_rngs()
        inputs = (jax.random.normal(data_rng, data_shape,
                                    dtype), jax.random.normal(data_rng, data_shape, dtype))

        padded_mask = jnp.zeros((batch, 1, seqlen, seqlen), dtype=jnp.uint8)
        causal_mask = jnp.triu(jnp.ones((batch, 1, seqlen, seqlen), dtype=jnp.uint8), k=1)
        ref_masks = (1 - causal_mask, 1 - padded_mask)
        test_masks = (causal_mask, padded_mask)

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
                            hidden_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.DECODER,
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestDecoderLayer.sync_params(ref_params, test_params, attrs)

        if FP8Helper.is_fp8_enabled():
            for _ in range(4):
                _, tmp_grad = jax.value_and_grad(loss_fn, argnums=(3,),
                                                 has_aux=False)(inputs, test_masks, test_params,
                                                                test_others, test_layer, apply_rng)
                _, fp8_meta_grad = tmp_grad[0].pop(FP8Helper.FP8_COLLECTION_NAME)
                test_others = FP8Helper.update_collections(
                    {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
                test_others = FP8Helper.update_fp8_metas(test_others)
                del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 2), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_masks, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_masks, test_params, test_others, test_layer,
                                       apply_rng)

        assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)
        assert_allclose(ref_grads[0][0], test_grads[0][0], rtol=rtol, atol=atol)    # dgrad

        def reorganize_test_wgrad(test_wgrad, attrs):
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True)
            attn_name = 'self_attention'

            unfreeze_test_wgrad = test_wgrad.unfreeze()
            if "output_layernorm" not in attrs:
                unfreeze_test_wgrad['pre_self_attention_layer_norm'] = {}
                pre_attn_layer_key = 'qkv' if fuse_qkv else 'query'
                unfreeze_test_wgrad['pre_self_attention_layer_norm']['scale'] = \
                    unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                if 'ln_bias' in unfreeze_test_wgrad[attn_name][pre_attn_layer_key]:
                    unfreeze_test_wgrad['pre_self_attention_layer_norm']['ln_bias'] = \
                        unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']
                    del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']

            if fuse_qkv:
                unfreeze_test_wgrad[attn_name]['qkv']['kernel'] = \
                    jnp.reshape(unfreeze_test_wgrad[attn_name]['qkv']['kernel'],
                        (unfreeze_test_wgrad[attn_name]['qkv']['kernel'].shape[0], -1))
                attn_name = 'encoder_decoder_attention'
                unfreeze_test_wgrad[attn_name]['kv']['kernel'] = \
                    jnp.reshape(unfreeze_test_wgrad[attn_name]['kv']['kernel'],
                        (unfreeze_test_wgrad[attn_name]['kv']['kernel'].shape[0], -1))

            unfreeze_test_wgrad['pre_cross_attention_layer_norm'] = {}
            unfreeze_test_wgrad['pre_cross_attention_layer_norm']['scale'] = \
                unfreeze_test_wgrad['encoder_decoder_attention']['query']['scale']
            del unfreeze_test_wgrad['encoder_decoder_attention']['query']['scale']
            if 'ln_bias' in unfreeze_test_wgrad['encoder_decoder_attention']['query']:
                unfreeze_test_wgrad['pre_cross_attention_layer_norm']['ln_bias'] = \
                    unfreeze_test_wgrad['encoder_decoder_attention']['query']['ln_bias']
                del unfreeze_test_wgrad['encoder_decoder_attention']['query']['ln_bias']
            unfreeze_test_wgrad['pre_mlp_layer_norm'] = {}
            unfreeze_test_wgrad['pre_mlp_layer_norm']['scale'] = \
                unfreeze_test_wgrad['mlp']['scale']
            del unfreeze_test_wgrad['mlp']['scale']
            if 'ln_bias' in unfreeze_test_wgrad['mlp']:
                unfreeze_test_wgrad['pre_mlp_layer_norm']['ln_bias'] = \
                    unfreeze_test_wgrad['mlp']['ln_bias']
                del unfreeze_test_wgrad['mlp']['ln_bias']
            unfreeze_test_wgrad['mlp']['wi'] = {}
            unfreeze_test_wgrad['mlp']['wi']['kernel'] = \
                jnp.reshape(unfreeze_test_wgrad['mlp']['wi_kernel'],
                            (unfreeze_test_wgrad['mlp']['wi_kernel'].shape[0], -1))
            del unfreeze_test_wgrad['mlp']['wi_kernel']
            unfreeze_test_wgrad['mlp']['wo'] = {}
            unfreeze_test_wgrad['mlp']['wo']['kernel'] = \
                unfreeze_test_wgrad['mlp']['wo_kernel']
            del unfreeze_test_wgrad['mlp']['wo_kernel']
            return flax.core.frozen_dict.FrozenDict(unfreeze_test_wgrad)

        compare_frozen_dict(ref_grads[1],
                            reorganize_test_wgrad(test_grads[1], attrs),
                            rtol=rtol,
                            atol=atol)    # wgrad

        del data_rng, init_rng, apply_rng

    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward(self, data_shape, dtype, attrs):
        FP8Helper.finalize()    # Ensure FP8 disabled.
        self.forward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=2e-04)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
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
        FP8Helper.finalize()    # Ensure FP8 disabled.
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=2e-04)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward_with_fp8(self, data_shape, dtype, fp8_format, attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-04, atol=3e-02)
        FP8Helper.finalize()
