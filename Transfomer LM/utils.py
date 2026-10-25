import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einx
import math

def cross_entroy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> torch.Tensor:
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    loss = torch.mean(torch.log(torch.sum(torch.exp(inputs), dim=-1)) - inputs[torch.arange(inputs.shape[0]), targets])
    return loss


def lr_cosin_schedule(
    t: int,
    a_max: float,
    a_min: float, 
    Tw: int, 
    Tc: int
) -> float:
    if (t < Tw):
        alpha_t = t/Tw * a_max
    elif (Tw <= t and t <= Tc):
        alpha_t = a_min + 0.5 * (1 + math.cos( ((t - Tw) / (Tc - Tw)) * math.pi)) * (a_max - a_min)
    else: alpha_t = a_min
    return alpha_t


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

def grad_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norms = torch.sqrt(sum(torch.sum(g**2) for g in grads))
    if total_norms >= max_l2_norm:
        scale = max_l2_norm / (total_norms + eps)
        for g in grads:
            g.mul_(scale)
    return 