import math

import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        pad_idx: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return x
