import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange

from multihead_self_attention import casual_multi_head_self_attention_with_rope
from positionwise_feedforward import SwiGLU
from rmsnorm import RMSNorm

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