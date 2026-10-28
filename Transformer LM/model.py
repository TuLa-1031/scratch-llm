import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, einsum
from utils import *

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        freq = 1.0 / theta ** (torch.arange(0, d_k, 2, device=device)/d_k)
        pos = torch.arange(start=0, end=max_seq_len, device=device)
        angles = torch.outer(pos, freq)
        self.register_buffer("cos", torch.cos(angles), persistent=False) # persistent=False excludes buffers from state_dict.
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor 
    ) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2*sin
        out2 = x1*sin + x2*cos
        output = torch.stack((out1, out2), dim=-1).flatten(-2)
        return output



class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        nn.init.trunc_normal_(
            self.embedding,
            mean=0, std=1,
            a=-3, b=3
        )


    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        embed = self.embedding[token_ids]
        return embed


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        der = 2 / (in_features + out_features)
        deviation = math.sqrt(der)

        self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))

        nn.init.trunc_normal_(
            self.W, mean=0.0, std=der, a=-3*deviation, b=3*deviation
        )
        

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        y =  einx.dot(
                    "... d_in, d_out d_in"
                    "-> ... d_out",
                    x, self.W,
                )
        return y


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

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int, # d_ff = 8/3 * d_model
        devive: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff)
        self.W3 = Linear(d_model, d_ff)
        self.W2 = Linear(self.d_ff, d_model)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.W1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.W3(x)
        x = x1 * x2
        out = self.W2(x)
        return out


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        hi = (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))
        result = x / hi * self.g
        return result.to(in_dtype)


class trf_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: int,
    ):
        super().__init__()
        self.mha = casual_multi_head_self_attention_with_rope(
            d_model, num_heads, max_seq_len,theta
        )
        self.ff = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)
        x += shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x += shortcut
        
        return x


class transformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10000.0,

    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[trf_block(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)],
        )
        self.out_norm = RMSNorm(d_model)
        self.out_project = Linear(d_model, vocab_size)
        self.context_length = context_length


    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embed(in_indices)
        x = self.layers(x)
        x = self.out_norm(x)
        x = self.out_project(x)
        return x