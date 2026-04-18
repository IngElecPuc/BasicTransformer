import torch
import torch.nn as nn
from .cross_attention import CrossAttentionHead

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        out_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if out_dim is None:
            out_dim = q_dim

        if out_dim % num_heads != 0:
            raise ValueError("out_dim debe ser divisible por num_heads")

        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.heads = nn.ModuleList([
            CrossAttentionHead(
                q_dim=q_dim,
                kv_dim=kv_dim,
                head_dim=self.head_dim,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(out_dim, out_dim, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        head_outputs = []
        all_attn_weights = []

        for head in self.heads:
            if return_attention:
                out, attn_weights = head(
                    x_q=x_q,
                    x_kv=x_kv,
                    mask=mask,
                    return_attention=True,
                )
                head_outputs.append(out)
                all_attn_weights.append(attn_weights)
            else:
                out = head(
                    x_q=x_q,
                    x_kv=x_kv,
                    mask=mask,
                    return_attention=False,
                )
                head_outputs.append(out)

        x_concat = torch.cat(head_outputs, dim=-1)         # [B, T_dec, out_dim]
        out = self.out_proj(x_concat)                      # [B, T_dec, out_dim]
        out = self.resid_dropout(out)

        if return_attention:
            return out, all_attn_weights

        return out
