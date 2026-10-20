
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        nn.init.trunc_normal_(
            self.embedding,
            mean=0, std=1,
            a=-3, b=3
        )


    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        embed = self.embedding[token_ids]
        return embed
