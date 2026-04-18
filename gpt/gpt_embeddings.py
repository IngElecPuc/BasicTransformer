import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """
    Embeddings tipo GPT:
    - token embeddings
    - position embeddings aprendidas
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 1024,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_idx = pad_idx

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        position_ids: [B, T]

        returns:
            [B, T, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.dropout(x)
        return x
