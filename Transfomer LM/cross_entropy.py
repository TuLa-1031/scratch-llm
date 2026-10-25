import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int


def cross_entroy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> torch.Tensor:
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    loss = torch.mean(torch.log(torch.sum(torch.exp(inputs), dim=-1)) - inputs[torch.arange(inputs.shape[0]), targets])
    return loss