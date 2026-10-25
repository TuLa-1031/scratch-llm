import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange

from transformer_block import trf_block
from embedding import Embedding
from rmsnorm import RMSNorm
from linear import Linear
from softmax import softmax

class transformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,

    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[trf_block(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)],
        )
        self.out_norm = RMSNorm(d_model)
        self.out_project = Linear(d_model, vocab_size)


    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embed(in_indices)
        x = self.layers(x)
        x = self.out_norm(x)
        x = self.out_project(x)
        return x