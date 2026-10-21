import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from linear import Linear
from einops import rearrange

from functions import softmax, scaled_dot_product_attention
from rope import RotaryPositionalEmbedding


class casual_multi_head_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.W_out = Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        l = x.shape[-2]
        Q = rearrange(self.Wq(x), "... queries (heads d_k) -> ... heads queries d_k", heads=self.num_heads)
        K = rearrange(self.Wk(x), "... keys (heads d_k) -> ... heads keys d_k", heads=self.num_heads)
        V = rearrange(self.Wv(x), "... keys (heads d_v) -> ... heads keys d_v", heads=self.num_heads)
        mask = torch.triu(torch.ones(l, l, device=Q.device), diagonal=1).to(torch.bool)
        out = self.W_out(rearrange(scaled_dot_product_attention(Q, K, V, ~mask), "... heads queries d_v -> ... queries (heads d_v)"))
        return out

class casual_multi_head_self_attention_with_rope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: int,
    ):
        super().__init__()
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.W_out = Linear(d_model, d_model)
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_k=d_k, max_seq_len=max_seq_len)


    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        l = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(l, device=x.device, dtype=torch.long)
        Q = self.rope(rearrange(self.Wq(x), "... queries (heads d_k) -> ... heads queries d_k", heads=self.num_heads), token_positions)
        K = self.rope(rearrange(self.Wk(x), "... keys (heads d_k) -> ... heads keys d_k", heads=self.num_heads), token_positions)
        V = rearrange(self.Wv(x), "... keys (heads d_v) -> ... heads keys d_v", heads=self.num_heads)
        mask = torch.triu(torch.ones(l, l, device=Q.device), diagonal=1).to(torch.bool)
        out = self.W_out(rearrange(scaled_dot_product_attention(Q, K, V, ~mask), "... heads queries d_v -> ... queries (heads d_v)"))
        return out