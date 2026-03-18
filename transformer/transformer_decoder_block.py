import torch
import torch.nn as nn
from multihead_attention import MultiHeadSelfAttention
from multihead_cross_attention import MultiHeadCrossAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        encoder_dim: int,
        cross_num_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} debe ser divisible por num_heads={num_heads}"
            )

        if cross_num_heads is None:
            cross_num_heads = num_heads

        if d_model % cross_num_heads != 0:
            raise ValueError(
                f"d_model={d_model} debe ser divisible por cross_num_heads={cross_num_heads}"
            )

        self.d_model = d_model
        self.encoder_dim = encoder_dim

        self.ln_1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.ln_2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(
            q_dim=d_model,
            kv_dim=encoder_dim,
            num_heads=cross_num_heads,
            out_dim=d_model,
            dropout=dropout,
            bias=bias,
        )

        self.ln_3 = nn.LayerNorm(d_model)

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
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ):
        if return_attentions:
            self_out, self_attn_weights = self.self_attn(
                self.ln_1(x),
                mask=self_mask,
                return_attention=True,
            )
            x = x + self_out

            cross_out, cross_attn_weights = self.cross_attn(
                x_q=self.ln_2(x),
                x_kv=encoder_out,
                mask=cross_mask,
                return_attention=True,
            )
            x = x + cross_out

            x = x + self.ffn(self.ln_3(x))

            return x, {
                "self_attn": self_attn_weights,
                "cross_attn": cross_attn_weights,
            }

        x = x + self.self_attn(self.ln_1(x), mask=self_mask)
        x = x + self.cross_attn(
            x_q=self.ln_2(x),
            x_kv=encoder_out,
            mask=cross_mask,
        )
        x = x + self.ffn(self.ln_3(x))

        return x
