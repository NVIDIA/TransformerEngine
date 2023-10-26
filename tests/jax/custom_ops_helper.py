# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit, _UNSPECIFIED
from jax.sharding import PartitionSpec

import flax

from transformer_engine.jax.sharding import ShardingType
try:
    # try importing the new custom partitioning implementation
    from transformer_engine.jax.sharding import MeshResource
    ShardingResource = None
except ImportError:
    # must be using an older TE/JAX version so fall back on the xmap sharding implementation
    from transformer_engine.jax.sharding import ShardingResource
    MeshResource = None
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.softmax import SoftmaxType, softmax
from transformer_engine.jax.fused_attn import \
    AttnBiasType, AttnMaskType, is_fused_attn_kernel_available, self_fused_attn, cross_fused_attn


class FusedAttnBackend(Enum):
        Max512 = "0"
        Arbitrary = "1"


@pytest.fixture(name="backend", params=[FusedAttnBackend.Max512, FusedAttnBackend.Arbitrary])
def fixture_backend(request):
    backend = request.param
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = backend.value
    yield backend
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = ""


@dataclass
class CustomOpsTestHelper:
    qkv_shape: Tuple[int,int,int,int] = (32, 128, 16, 64)
    pad_ratio: float = 0.3
    dropout_prob: float = 0.1
    dtype: type = jnp.float16
    
    @staticmethod
    def use_custom_partitioning():
        return (MeshResource is not None)

    @staticmethod
    def get_sharding_spec(mesh_names, sharding_type):
        P = PartitionSpec
        if sharding_type is ShardingType.DP:
            return P(mesh_names[0], None), P(None), P(None)
        elif sharding_type is ShardingType.DP_TP_COL:
            return P(mesh_names[0], mesh_names[1]), P(None), P(None)
        else:
            raise NotImplementedError
    
    @staticmethod
    def get_sharding_resource(mesh_names, sharding_type):
        dp_r = None
        tp_r = None
        
        if sharding_type in (ShardingType.DP, ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
            dp_r = mesh_names[0]
        if sharding_type in (ShardingType.TP_COL, ShardingType.TP_ROW):
            tp_r = mesh_names[0]
        if sharding_type in (ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
            tp_r = mesh_names[1]
            
        if CustomOpsTestHelper.use_custom_partitioning():
            return MeshResource(dp_r, tp_r)
        else:
            return ShardingResource(dp_r, tp_r)

    @staticmethod
    def make_mask(q_tokens, kv_tokens, mask_type, dtype=jnp.uint8):
        if mask_type == AttnMaskType.CAUSAL_MASK:
            causal = flax.linen.make_causal_mask(q_tokens, dtype=dtype)
            padding = flax.linen.make_attention_mask(q_tokens > 0, kv_tokens > 0, dtype=dtype)
            return flax.linen.combine_masks(causal, padding)
        else:
            return flax.linen.make_attention_mask(q_tokens > 0, kv_tokens > 0, dtype=dtype)

    @staticmethod
    def count_collectives(hlo):
        tmp = hlo.splitlines()
        symb = "-start"
        result = {
            "all-reduce" : 0,
            "other" : 0
        }
        for line in tmp:
            txt = line.split()
            if len(txt) > 0 and symb in txt[0]:
                if "all-reduce" in txt[0]:
                    result["all-reduce"] += 1
                else:
                    result["other"] += 1
        return result

    def compare_ops(self, custom_func, ref_func, ref_count,
                    *args, grad_args=None,
                    in_shardings=_UNSPECIFIED, out_shardings=_UNSPECIFIED,
                    **kwargs):
        if isinstance(custom_func, partial):
            func_name = custom_func.func.__name__
        else:
            func_name = custom_func.__name__
        func_name = func_name.removeprefix('custom_')
        if grad_args is None:
            grad_args = tuple(range(len(args)))
        
        custom_gradded = jax.value_and_grad(custom_func, argnums=grad_args)
        test_fwd, test_grads = custom_gradded(*args, **kwargs)
        custom_pjitter = pjit(custom_gradded,
                              in_shardings=in_shardings,
                              out_shardings=out_shardings)
        custom_hlo = custom_pjitter.lower(*args, **kwargs).compile().as_text()
        custom_count = self.count_collectives(custom_hlo)
        if ref_count is not None:
            assert custom_count==ref_count, \
                f"`{func_name}`: Expected collective count is {ref_count}, but got {custom_count}."
        else:
            print(f"`{func_name}`: Output collective count is {custom_count}.")

        ref_gradded = jax.value_and_grad(ref_func, argnums=grad_args)
        ref_fwd, ref_grads = ref_gradded(*args, **kwargs)
        fwd_tol = max(np.finfo(jnp.float16).eps, np.spacing(jnp.float16(ref_fwd))) ** (2./3.)
        assert jnp.allclose(test_fwd, ref_fwd, rtol=0.0, atol=fwd_tol), \
            f"`{func_name}`: Output (fwd) error {jnp.max(jnp.abs(test_fwd - ref_fwd))}" + \
            f" exceeds tolerance ({fwd_tol})."

        if len(grad_args) == 1:
            ref_grads = (ref_grads, )
            test_grads = (test_grads, )
        failed_grads = {}
        for i, grads in enumerate(zip(test_grads, ref_grads)):
            test_grad, ref_grad = grads
            if test_grad is None and ref_grad is None:
                continue
            bwd_tol = max(np.finfo(jnp.float32).eps,
                        np.spacing(jnp.max(jnp.abs(ref_grad)).astype(jnp.float32))) ** (2./3.)
            if not jnp.allclose(test_grad, ref_grad, rtol=0.0, atol=bwd_tol):
                failed_grads[i] = jnp.max(jnp.abs(test_grad - ref_grad))
        assert len(failed_grads) == 0, \
            f"`{func_name}`: Gradient (bwd) max errors" + \
            f" [{', '.join([f'Arg{k}={v}' for k,v in failed_grads.items()])}]" + \
            f" exceed tolerance ({bwd_tol})."

    def check_fused_attn_inputs(self, q_seq, kv_seq, head_dim, pad_ratio, dropout_probability,
                                attn_bias_type, attn_mask_type, backend):
        if (q_seq > 512 or kv_seq > 512 or backend == FusedAttnBackend.Arbitrary) \
            and pad_ratio != 0:
            pytest.skip(
                "`fused_attention`: Arbitrary seqlen backend does not support padded input.")

        if not is_fused_attn_kernel_available(
            self.dtype, self.dtype, attn_bias_type, attn_mask_type,
            dropout_probability, q_seq, kv_seq, head_dim):
            pytest.skip(
                "`fused_attention`: Unsupported inputs combination or device compute capability.")

    def fused_attn_core(self, query, key, value, bias, mask, scale_factor,
                        attn_bias_type, attn_mask_type, dropout_rng, dropout_prob):
        # Q*K matmul
        query = jnp.squeeze(query)
        key = jnp.squeeze(key)
        value = jnp.squeeze(value)
        attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
        # scale and bias
        if attn_bias_type == AttnBiasType.PRE_SCALE_BIAS:
            attn_weights = scale_factor * (attn_weights + bias)
        elif attn_bias_type == AttnBiasType.POST_SCALE_BIAS:
            attn_weights = scale_factor * attn_weights + bias
        else:
            attn_weights = scale_factor * attn_weights
        # padding mask
        if attn_mask_type != AttnMaskType.NO_MASK and mask is not None:
            big_neg = jnp.finfo(self.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        # softmax
        attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)
        # dropout
        if dropout_prob == 1.0:
            attn_weights = jnp.zeros_like(attn_weights)
        elif dropout_prob > 0.0:
            keep_prob = 1.0 - dropout_prob
            keep = random.bernoulli(dropout_rng, p=keep_prob, shape=attn_weights.shape)
            multiplier = keep.astype(self.dtype) / jnp.asarray(keep_prob, dtype=self.dtype)
            attn_weights = attn_weights * multiplier
        # QK*V matmul
        result =  jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        return jnp.mean(result)

    @staticmethod
    def custom_layernorm(x, gamma, beta, zero_centered_gamma, epsilon, sharding_type):
        result = layernorm(x, gamma, beta,
                           layernorm_type='layernorm',
                           zero_centered_gamma=zero_centered_gamma,
                           epsilon=epsilon,
                           sharding_type=sharding_type,
                           dp_dim_index=0)
        return jnp.mean(result)

    def reference_layernorm(self, x, gamma, beta, zero_centered_gamma, epsilon):
        x_ = jnp.asarray(x, jnp.float32)
        mean = jnp.mean(x_, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
        normed_input = (x_ - mean) * jax.lax.rsqrt(var + epsilon)
        if zero_centered_gamma:
            result = jnp.asarray(normed_input * (gamma + 1) + beta).astype(self.dtype)
        else:
            result = jnp.asarray(normed_input * gamma + beta).astype(self.dtype)
        return jnp.mean(result)

    @staticmethod
    def custom_rmsnorm(x, gamma, epsilon, sharding_type):
        result = layernorm(x, gamma, None,
                           layernorm_type='rmsnorm',
                           zero_centered_gamma=False,
                           epsilon=epsilon,
                           sharding_type=sharding_type,
                           dp_dim_index=0)
        return jnp.mean(result)

    def reference_rmsnorm(self, x, gamma, epsilon):
        x = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), self.dtype)
        result = y * gamma
        return jnp.mean(result)

    @staticmethod
    def custom_softmax(x, mask, scale_factor, softmax_type, sharding_type):
        result = softmax(x, mask,
                         scale_factor=scale_factor, 
                         softmax_type=softmax_type,
                         sharding_type=sharding_type)
        return jnp.mean(result)

    def reference_softmax(self, x, mask, scale_factor, softmax_type):
        attn_weights = scale_factor * x
        if softmax_type != SoftmaxType.SCALED:
            big_neg = jnp.finfo(self.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        result = jax.nn.softmax(attn_weights).astype(self.dtype)
        return jnp.mean(result)

    @staticmethod
    def custom_self_fused_attn(qkv, bias, mask, rng_key, dropout_prob,
                               attn_bias_type, attn_mask_type,
                               scaling_factor, sharding_type):
        mask = (mask == 0)  # invert mask
        bias_ = None if attn_bias_type == AttnBiasType.NO_BIAS else bias
        result = self_fused_attn(qkv, bias_, mask,
                                 seed=rng_key,
                                 attn_bias_type=attn_bias_type,
                                 attn_mask_type=attn_mask_type,
                                 scaling_factor=scaling_factor,
                                 dropout_probability=dropout_prob,
                                 is_training=True,
                                 sharding_type=sharding_type)
        return jnp.mean(result)

    def reference_self_fused_attn(self, qkv, bias, mask, rng_key, dropout_prob,
                                  attn_bias_type, attn_mask_type,
                                  scaling_factor):
        # split interleaved QKV into separate matrices
        query, key, value = jnp.split(qkv, [1, 2], axis=-3)
        return self.fused_attn_core(
            query, key, value, bias, mask, scaling_factor,
            attn_bias_type, attn_mask_type,
            rng_key, dropout_prob)

    @staticmethod
    def custom_cross_fused_attn(query, key_value, mask, rng_key, dropout_prob,
                                attn_mask_type, scaling_factor, sharding_type):
        mask = (mask == 0)  # invert mask
        result = cross_fused_attn(query, key_value, mask,
                                  seed=rng_key,
                                  attn_bias_type=AttnBiasType.NO_BIAS,
                                  attn_mask_type=attn_mask_type,
                                  scaling_factor=scaling_factor,
                                  dropout_probability=dropout_prob,
                                  is_training=True,
                                  sharding_type=sharding_type)
        return jnp.mean(result)

    def reference_cross_fused_attn(self, query, key_value, mask, rng_key, dropout_prob,
                                   attn_mask_type, scaling_factor):
        key, value = jnp.split(key_value, [1], axis=-3)
        return self.fused_attn_core(
            query, key, value, None, mask, scaling_factor,
            AttnBiasType.NO_BIAS, attn_mask_type,
            rng_key, dropout_prob)
