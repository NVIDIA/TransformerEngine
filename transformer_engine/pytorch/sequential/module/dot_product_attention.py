from abc import abstractmethod, ABC
from .base import BaseModule
from ..compute_pipeline import ops

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
        self.attention_mechanism = attention_mechanism
        super().__init__()

    def _ops(self) -> list[ops.Op | None]:
        return [self.attention_mechanism.make_op()]


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
