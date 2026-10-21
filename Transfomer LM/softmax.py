import torch
from torch import Tensor
from jaxtyping import Float, Bool
import einx
import math


def softmax(
    x: torch.Tensor,
    i: int,
) -> torch.Tensor:
    e = torch.exp(x - torch.max(x, dim = i, keepdim=True).values)
    out = e / torch.sum(e, dim=i, keepdim=True)
    return out

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    score = einx.dot(
        "... queries [d_k], ... keys [d_k]"
        "-> ... queries keys",
        Q, K
    )
    score = score / math.sqrt(K.shape[-1])
    if mask is not None:
        score.masked_fill_(~mask, -torch.inf)
    score = softmax(score, -1)

    out = einx.dot(
        "... queries [keys], ... [keys] d_v"
        "-> ... queries d_v",
        score, V
    )
    return out