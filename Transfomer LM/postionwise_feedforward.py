import torch
from einops import rearrange, einsum
import torch.nn as nn

from linear import Linear

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