from abc import abstractmethod, ABC
from math import sqrt
import torch
from torch import nn
from .base import BaseModule
from .. import ops
from ..nvte import DType, make_nvte_tensor

class Attention(ABC):
    @abstractmethod
    def make_op(self) -> ops.Op:
        ...

class DotProductAttention(Attention):
    def __init__(self, causal_mask: bool = True, pre_softmax_scale: float, dropout_p: float):
        self.causal_mask = causal_mask

    def make_op(self):
        return ops.DotProductAttention(causal_mask)

class GroupedQuerySelfAttention(BaseModule):
    def __init__(
        self,
        token_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        attention_mechanism: Attention,
    ):
        assert num_kv_heads <= num_query_heads
        assert num_query_heads % num_kv_heads == 0
        assert token_dim % num_query_heads == 0
        nn.Module.__init__(self)  # type: ignore

        return super().__init__(
            attention_type(),
        )


class MultiQuerySelfAttention(GroupedQuerySelfAttention):
    def __init__(
        self,
        token_dim: int,
        num_query_heads: int,
        attention_mechanism: Attention,
    ):
        super().__init__(
            token_dim,
            num_query_heads,
            1,
            attention_mechanism,
        )


class MultiHeadedSelfAttention(GroupedQuerySelfAttention):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        attention_mechanism: Attention,
    ):
        super().__init__(
            token_dim,
            num_heads,
            num_heads,
            attention_mechanism,
        )
