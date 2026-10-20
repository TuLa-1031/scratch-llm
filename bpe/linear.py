import torch
from einops import rearrange, einsum
import einx
import torch.nn as nn
import math

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