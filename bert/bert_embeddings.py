import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    """
    Embeddings de entrada estilo BERT:
    - token embeddings
    - position embeddings aprendidas
    - token type embeddings
    - LayerNorm
    - Dropout
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_size,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_dim=hidden_size,
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids:      [B, T]
        token_type_ids: [B, T]
        position_ids:   [B, T]

        returns:
            [B, T, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
