from math import sqrt
import torch
from torch import nn
from .base import BaseModule
from ._common import ParameterInitMethod
from .linear import _default_weight_init_method
from .. import ops
from ..nvte import DType, make_nvte_tensor


class GroupedQuerySelfAttention(BaseModule):
    def __init__(
        self,
        token_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        causal_mask: bool = True,
        param_dtype: torch.dtype = torch.get_default_dtype(),
        weight_init_method: ParameterInitMethod = _default_weight_init_method,
        proj_init_method: ParameterInitMethod = _default_weight_init_method,
        attention_type: ops.Attention = ops.DotProductAttention,
    ):
        assert num_kv_heads <= num_query_heads
        assert num_query_heads % num_kv_heads == 0
        assert token_dim % num_query_heads == 0
        nn.Module.__init__(self)  # type: ignore

        kv_dim = token_dim // num_kv_heads
        norm_factor = sqrt(kv_dim)

        self.weight = nn.Parameter(
            weight_init_method(
                torch.empty(3 * token_dim, token_dim, dtype=param_dtype, device="cuda")
            )
        )
        self.proj = nn.Parameter(
            proj_init_method(
                torch.empty(token_dim, token_dim, dtype=param_dtype, device="cuda")
            )
        )

        return super().__init__(
            # TODO
        )


class MultiQuerySelfAttention(GroupedQuerySelfAttention):
    def __init__(
        self,
        token_dim: int,
        num_query_heads: int,
        causal_mask: bool = True,
        param_dtype: torch.dtype = torch.get_default_dtype(),
        weight_init_method: ParameterInitMethod = _default_weight_init_method,
        proj_init_method: ParameterInitMethod = _default_weight_init_method,
        attention_type: ops.Attention = ops.DotProductAttention,
    ):
        super().__init__(
            token_dim,
            num_query_heads,
            1,
            causal_mask,
            param_dtype,
            weight_init_method,
            proj_init_method,
            attention_type,
        )


class MultiHeadedSelfAttention(GroupedQuerySelfAttention):
    def __init__(
        self,
        token_dim: int,
        num_query_heads: int,
        causal_mask: bool = True,
        param_dtype: torch.dtype = torch.get_default_dtype(),
        weight_init_method: ParameterInitMethod = _default_weight_init_method,
        proj_init_method: ParameterInitMethod = _default_weight_init_method,
        attention_type: ops.Attention = ops.DotProductAttention,
    ):
        super().__init__(
            token_dim,
            num_query_heads,
            num_query_heads,
            causal_mask,
            param_dtype,
            weight_init_method,
            proj_init_method,
            attention_type,
        )
