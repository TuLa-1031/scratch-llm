import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einx
import math
from collections.abc import Iterable
import os
from typing import BinaryIO, IO

def cross_entroy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> torch.Tensor:
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_partition = torch.logsumexp(inputs, dim=-1)

    target_logits = inputs[
        torch.arange(inputs.shape[0], device=inputs.device),
        targets
    ]

    return torch.mean(log_partition - target_logits)


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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint_dict={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint_dict, out)
    return

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    iteration = checkpoint_dict["iteration"]
    return iteration

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))