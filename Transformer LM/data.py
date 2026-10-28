import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idx = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([
        torch.from_numpy((dataset[i:i+context_length]).astype(np.int64))
        for i in starting_idx
    ])
    y = torch.stack([
        torch.from_numpy((dataset[i + 1 : i + 1 +context_length]).astype(np.int64))
        for i in starting_idx
    ])
    x, y = x.to(device), y.to(device)
    return x, y