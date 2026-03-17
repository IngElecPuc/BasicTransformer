import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, T]
        returns:   [B, T, d_model]
        """
        return self.embedding(token_ids)
