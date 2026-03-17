import math

import torch
import torch.nn as nn


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.head_dim = head_dim

        self.W_q = nn.Linear(q_dim, head_dim, bias=bias) #Q viende del decoder
        self.W_k = nn.Linear(kv_dim, head_dim, bias=bias) #K y V vienen del encoder
        self.W_v = nn.Linear(kv_dim, head_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        x_q:  [B, T_dec, q_dim]
        x_kv: [B, T_enc, kv_dim]
        mask: broadcastable a [B, T_dec, T_enc]

        returns:
            out: [B, T_dec, head_dim]
        """
        Q = self.W_q(x_q)                                  # [B, T_dec, H]
        K = self.W_k(x_kv)                                 # [B, T_enc, H]
        V = self.W_v(x_kv)                                 # [B, T_enc, H]

        scores = Q @ K.transpose(-2, -1)                   # [B, T_dec, T_enc]
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)       # [B, T_dec, T_enc]
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ V                             # [B, T_dec, H]

        if return_attention:
            return out, attn_weights

        return out
