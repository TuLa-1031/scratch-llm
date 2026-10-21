import torch

def softmax(
    x: torch.Tensor,
    i: int,
) -> torch.Tensor:
    e = torch.exp(x - torch.max(x, dim = i, keepdim=True).values)
    out = e / torch.sum(e, dim=i, keepdim=True)
    return out