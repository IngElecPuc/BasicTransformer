import math

import torch
import torch.nn as nn


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        self.head_dim = head_dim

        self.W_q = nn.Linear(d_model, head_dim, bias=bias)
        self.W_k = nn.Linear(d_model, head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, head_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False
    ):
        Q = self.W_q(x)                                  # [B, T, H]
        K = self.W_k(x)                                  # [B, T, H]
        V = self.W_v(x)                                  # [B, T, H]

        scores = Q @ K.transpose(-2, -1)                 # [B, T, T]
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)     # [B, T, T]
        attn_weights = self.attn_dropout(attn_weights)  #Dropout se agrega después de softmax

        out = attn_weights @ V                           # [B, T, H]

        if return_attention:
            return out, attn_weights

        return out

