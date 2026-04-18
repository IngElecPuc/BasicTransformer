import math
import torch
import torch.nn as nn
from .self_attention import SelfAttentionHead

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model debe ser divisible por num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                d_model=d_model,
                head_dim=self.head_dim,
                dropout=dropout,
                bias=bias
            )
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False
    ):
        head_outputs = []
        all_attn_weights = []

        for head in self.heads:
            if return_attention:
                out, attn_weights = head(x, mask=mask, return_attention=True)
                head_outputs.append(out)
                all_attn_weights.append(attn_weights)
            else:
                out = head(x, mask=mask, return_attention=False)
                head_outputs.append(out)

        x_concat = torch.cat(head_outputs, dim=-1)       # [B, T, d_model]
        out = self.out_proj(x_concat)                    # [B, T, d_model]
        out = self.resid_dropout(out)

        if return_attention:
            return out, all_attn_weights

        return out