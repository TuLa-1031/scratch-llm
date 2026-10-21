import torch
from einops import rearrange
import torch.nn as nn

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