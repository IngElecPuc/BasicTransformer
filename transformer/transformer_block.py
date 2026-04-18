import torch
import torch.nn as nn
from .multihead_attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} debe ser divisible por num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias
        )

        self.ln_2 = nn.LayerNorm(d_model)

        if activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "relu":
            act = nn.ReLU()
        else:
            raise ValueError("activation debe ser 'gelu' o 'relu'")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            act,
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False
    ):
        if return_attention:
            attn_out, attn_weights = self.attn(
                self.ln_1(x),
                mask=mask,
                return_attention=True
            )
            x = x + attn_out
            x = x + self.ffn(self.ln_2(x))
            return x, attn_weights

        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.ffn(self.ln_2(x))
        return x
