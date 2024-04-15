# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from functools import partial

import flax
import jax
import jax.numpy as jnp
import pytest

from utils import assert_allclose
from utils import DecoderLayer as RefDecoderLayer
from utils import EncoderLayer as RefEncoderLayer

from transformer_engine.common.recipe import Format
from transformer_engine.jax.flax import TransformerLayer, TransformerLayerType
from transformer_engine.jax.fp8 import FP8Helper, is_fp8_available

is_fp8_supported, reason = is_fp8_available()


@pytest.fixture(autouse=True, scope='module')
def enable_fused_attn():
    """
    Enable fused attention
    """
    os.environ["NVTE_FUSED_ATTN"] = "1"
    yield
    del os.environ["NVTE_FUSED_ATTN"]


@pytest.fixture(autouse=True, scope='function')
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    yield
    for arr in jax.live_arrays():
        arr.delete()


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
    others, params = flax.core.pop(variables, 'params')
    del variables
    return layer, params, others


def compare_dict(ref_fd, test_fd, rtol=1e-05, atol=1e-08):
    # To be compatible with both Flax>=0.7.1 or <0.7.1
    # since Flax 0.7.1 removed FrozenDict.
    ref_fd = flax.core.unfreeze(ref_fd)
    test_fd = flax.core.unfreeze(test_fd)
    for key in ref_fd:
        assert key in test_fd, \
            f"{key} not found in test dict {test_fd}"
        assert isinstance(test_fd[key], type(ref_fd[key])), \
            f"The data type is not match between ref and test " \
            f"dict on {key=}"
        if isinstance(ref_fd[key], dict):
            compare_dict(ref_fd[key], test_fd[key], rtol, atol)
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
_KEY_OF_NUM_HEADS = 'num_attention_heads'
_KEY_OF_NUM_GQA_GROUPS = 'num_gqa_groups'
_KEY_OF_ENABLE_ROPE = "enable_rotary_pos_emb"
_KEY_OF_ROPE_GROUP_METHOD = "rotary_pos_emb_group_method"

BASE_ATTRS = {
    _KEY_OF_TRANSPOSE_BS: True,
    _KEY_OF_NUM_HEADS: 8,
    _KEY_OF_DROPOUT_RATE: 0,
}

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
}, {
    _KEY_OF_NUM_HEADS: 8,
    _KEY_OF_NUM_GQA_GROUPS: 4,
    _KEY_OF_TRANSPOSE_BS: False,
    _KEY_OF_SCALE_ATTN_LOGITS: True,
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_MLP_ACTIVATIONS: (('gelu',)),
    _KEY_OF_FUSE_MLP_WI: True
}, {
    _KEY_OF_TRANSPOSE_BS: False,
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_FUSE_MLP_WI: True,
    _KEY_OF_ENABLE_ROPE: True,
    _KEY_OF_ROPE_GROUP_METHOD: "consecutive"
}, {
    _KEY_OF_TRANSPOSE_BS: True,
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_FUSE_MLP_WI: True,
    _KEY_OF_ENABLE_ROPE: True,
    _KEY_OF_ROPE_GROUP_METHOD: "consecutive"
}, {
    _KEY_OF_TRANSPOSE_BS: False,
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_FUSE_MLP_WI: True,
    _KEY_OF_ENABLE_ROPE: True,
    _KEY_OF_ROPE_GROUP_METHOD: "alternate"
}, {
    _KEY_OF_TRANSPOSE_BS: True,
    _KEY_OF_LAYERNORM_TYPE: 'layernorm',
    _KEY_OF_DROPOUT_RATE: 0.0,
    _KEY_OF_FUSE_MLP_WI: True,
    _KEY_OF_ENABLE_ROPE: True,
    _KEY_OF_ROPE_GROUP_METHOD: "alternate"
}]

ATTRS = [{**BASE_ATTRS, **attr} for attr in ATTRS]


class TestEncoderLayer:

    @staticmethod
    def sync_params(ref, target):
        unfreeze_target = flax.core.unfreeze(target)
        unfreeze_attn_scope = unfreeze_target['attention']
        ref_attn_scope = ref['attention']
        for key in ref_attn_scope.keys():
            unfreeze_attn_scope[key]['kernel'] = \
                ref_attn_scope[key]['kernel'].reshape(unfreeze_attn_scope[key]['kernel'].shape)
        unfreeze_target['mlp']['wi_kernel'] = \
            jnp.reshape(ref['mlp']['wi']['kernel'], unfreeze_target['mlp']['wi_kernel'].shape)
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, unfreeze_target

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
                te_layer_attrs['intermediate_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefEncoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            hidden_dropout_dims=(sequence_dim,),
                            intermediate_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.ENCODER,
                            self_attn_mask_type='padding',
                            dtype=dtype,
                            **te_layer_attrs)

        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestEncoderLayer.sync_params(ref_params, test_params)

        ref_out = loss_fn(inputs, ref_masks, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_masks, test_params, test_others, test_layer, apply_rng)

        if attrs[_KEY_OF_DROPOUT_RATE] == 0.:    # Skip elementwise checking for dropout
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
                te_layer_attrs['intermediate_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefEncoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            hidden_dropout_dims=(sequence_dim,),
                            intermediate_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.ENCODER,
                            self_attn_mask_type='padding',
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestEncoderLayer.sync_params(ref_params, test_params)

        if FP8Helper.is_fp8_enabled():
            for _ in range(4):
                _, tmp_grad = jax.value_and_grad(loss_fn, argnums=(3,),
                                                 has_aux=False)(inputs, test_masks, test_params,
                                                                test_others, test_layer, apply_rng)
                _, fp8_meta_grad = flax.core.pop(tmp_grad[0], FP8Helper.FP8_COLLECTION_NAME)
                test_others = FP8Helper.update_collections(
                    {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
                test_others = FP8Helper.update_fp8_metas(test_others)
                del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 2), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_masks, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_masks, test_params, test_others, test_layer,
                                       apply_rng)

        def reorganize_test_wgrad(test_wgrad, attrs):
            num_heads = attrs.get(_KEY_OF_NUM_HEADS)
            num_gqa_groups = attrs.get(_KEY_OF_NUM_GQA_GROUPS, num_heads)
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True) and \
                       num_heads == num_gqa_groups

            attn_name = 'attention'
            unfreeze_test_wgrad = flax.core.unfreeze(test_wgrad)
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

            for key in unfreeze_test_wgrad[attn_name].keys():
                unfreeze_test_wgrad[attn_name][key]['kernel'] = \
                    jnp.reshape(unfreeze_test_wgrad[attn_name][key]['kernel'],
                        (unfreeze_test_wgrad[attn_name][key]['kernel'].shape[0], -1))

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
            return unfreeze_test_wgrad

        if attrs[_KEY_OF_DROPOUT_RATE] == 0.:    # Skip elementwise checking for dropout
            assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)
            assert_allclose(ref_grads[0][0], test_grads[0][0], rtol=rtol, atol=atol)    # dgrad

            compare_dict(ref_grads[1],
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
    def sync_params(ref, target):
        unfreeze_target = flax.core.unfreeze(target)
        for scope in ['self_attention', 'encoder_decoder_attention']:
            unfreeze_scope = unfreeze_target[scope]
            ref_scope = ref[scope]
            for key in unfreeze_scope.keys():
                unfreeze_scope[key]['kernel'] = \
                    ref_scope[key]['kernel'].reshape(unfreeze_scope[key]['kernel'].shape)
        unfreeze_target['mlp']['wi_kernel'] = \
            jnp.reshape(ref['mlp']['wi']['kernel'], unfreeze_target['mlp']['wi_kernel'].shape)
        unfreeze_target['mlp']['wo_kernel'] = \
            ref['mlp']['wo']['kernel']
        return ref, unfreeze_target

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
                te_layer_attrs['intermediate_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefDecoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            hidden_dropout_dims=(sequence_dim,),
                            intermediate_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.DECODER,
                            self_attn_mask_type='padding_causal',
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestDecoderLayer.sync_params(ref_params, test_params)

        ref_out = loss_fn(inputs, ref_masks, ref_params, ref_others, ref_layer, apply_rng)
        test_out = loss_fn(inputs, test_masks, test_params, test_others, test_layer, apply_rng)

        if attrs[_KEY_OF_DROPOUT_RATE] == 0.:    # Skip elementwise checking for dropout
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
                te_layer_attrs['intermediate_dropout'] = v
            elif k == 'fuse_mlp_wi':
                continue
            else:
                te_layer_attrs[k] = v
        ref_layer_cls = partial(RefDecoderLayer, dtype=dtype, **attrs)
        layer_cls = partial(TransformerLayer,
                            hidden_dropout_dims=(sequence_dim,),
                            intermediate_dropout_dims=(sequence_dim,),
                            layer_type=TransformerLayerType.DECODER,
                            self_attn_mask_type='padding_causal',
                            dtype=dtype,
                            **te_layer_attrs)
        ref_layer, ref_params, ref_others = generate_layer(ref_layer_cls, init_rng, inputs,
                                                           ref_masks)
        test_layer, test_params, test_others = generate_layer(layer_cls, init_rng, inputs,
                                                              test_masks)

        ref_params, test_params = TestDecoderLayer.sync_params(ref_params, test_params)

        if FP8Helper.is_fp8_enabled():
            for _ in range(4):
                _, tmp_grad = jax.value_and_grad(loss_fn, argnums=(3,),
                                                 has_aux=False)(inputs, test_masks, test_params,
                                                                test_others, test_layer, apply_rng)
                _, fp8_meta_grad = flax.core.pop(tmp_grad[0], FP8Helper.FP8_COLLECTION_NAME)
                test_others = FP8Helper.update_collections(
                    {FP8Helper.FP8_COLLECTION_NAME: fp8_meta_grad}, test_others)
                test_others = FP8Helper.update_fp8_metas(test_others)
                del tmp_grad, fp8_meta_grad

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 2), has_aux=False)

        ref_out, ref_grads = grad_fn(inputs, ref_masks, ref_params, ref_others, ref_layer,
                                     apply_rng)
        test_out, test_grads = grad_fn(inputs, test_masks, test_params, test_others, test_layer,
                                       apply_rng)

        def reorganize_test_wgrad(test_wgrad, attrs):
            num_heads = attrs.get(_KEY_OF_NUM_HEADS)
            num_gqa_groups = attrs.get(_KEY_OF_NUM_GQA_GROUPS, num_heads)
            fuse_qkv = attrs.get(_KEY_OF_FUSE_QKV_PARAMS, True) and \
                       num_heads == num_gqa_groups

            unfreeze_test_wgrad = flax.core.unfreeze(test_wgrad)
            if "output_layernorm" not in attrs:
                attn_name = 'self_attention'
                unfreeze_test_wgrad['pre_self_attention_layer_norm'] = {}
                pre_attn_layer_key = 'qkv' if fuse_qkv else 'query'
                unfreeze_test_wgrad['pre_self_attention_layer_norm']['scale'] = \
                    unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['scale']
                if 'ln_bias' in unfreeze_test_wgrad[attn_name][pre_attn_layer_key]:
                    unfreeze_test_wgrad['pre_self_attention_layer_norm']['ln_bias'] = \
                        unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']
                    del unfreeze_test_wgrad[attn_name][pre_attn_layer_key]['ln_bias']

            for scope in ['self_attention', 'encoder_decoder_attention']:
                for key in unfreeze_test_wgrad[scope].keys():
                    unfreeze_test_wgrad[scope][key]['kernel'] = \
                        jnp.reshape(unfreeze_test_wgrad[scope][key]['kernel'],
                            (unfreeze_test_wgrad[scope][key]['kernel'].shape[0], -1))

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
            return unfreeze_test_wgrad

        if attrs[_KEY_OF_DROPOUT_RATE] == 0.:    # Skip elementwise checking for dropout
            assert_allclose(ref_out, test_out, rtol=rtol, atol=atol)
            assert_allclose(ref_grads[0][0], test_grads[0][0], rtol=rtol, atol=atol)    # dgrad
            compare_dict(ref_grads[1],
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
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-05, atol=3e-04)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('data_shape', DATA_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPE)
    @pytest.mark.parametrize('fp8_format', FP8_FORMATS)
    @pytest.mark.parametrize('attrs', ATTRS)
    def test_forward_backward_with_fp8(self, data_shape, dtype, fp8_format, attrs):
        FP8Helper.initialize(fp8_format=fp8_format)
        self.forward_backward_runner(data_shape, dtype, attrs, rtol=1e-04, atol=3e-02)
        FP8Helper.finalize()
