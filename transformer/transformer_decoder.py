import math
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_configs: list[dict],
        encoder_dim: int,
        max_len: int = 5000,
        pad_idx: int = 0,
        input_dropout: float = 0.0,
        bias: bool = False,
        final_norm: bool = True,
        output_logits: bool = True,
    ):
        """
        block_configs:
            [
                {
                    "d_model": int,
                    "num_heads": int,
                    "d_ff": int,
                    "dropout": float,          # opcional
                    "activation": "gelu",      # opcional
                    "cross_num_heads": int,    # opcional
                },
                ...
            ]
        """
        super().__init__()

        if len(block_configs) == 0:
            raise ValueError("block_configs no puede estar vacío")

        self.block_configs = block_configs
        self.encoder_dim = encoder_dim
        self.output_logits = output_logits

        first_d_model = block_configs[0]["d_model"]

        self.token_embedding = nn.Embedding(
            vocab_size,
            first_d_model,
            padding_idx=pad_idx,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=first_d_model,
            max_len=max_len,
            dropout=input_dropout,
        )

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        prev_d_model = first_d_model

        for i, cfg in enumerate(block_configs):
            d_model = cfg["d_model"]
            num_heads = cfg["num_heads"]
            d_ff = cfg["d_ff"]
            dropout = cfg.get("dropout", 0.0)
            activation = cfg.get("activation", "gelu")
            cross_num_heads = cfg.get("cross_num_heads", num_heads)

            if i == 0:
                self.transitions.append(nn.Identity())
            else:
                if prev_d_model == d_model:
                    self.transitions.append(nn.Identity())
                else:
                    self.transitions.append(nn.Linear(prev_d_model, d_model, bias=bias))

            block = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                encoder_dim=encoder_dim,
                cross_num_heads=cross_num_heads,
                dropout=dropout,
                bias=bias,
                activation=activation,
            )

            self.blocks.append(block)
            prev_d_model = d_model

        self.final_norm = nn.LayerNorm(prev_d_model) if final_norm else nn.Identity()
        self.output_dim = prev_d_model

        if output_logits:
            self.lm_head = nn.Linear(self.output_dim, vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        token_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ):
        """
        token_ids:   [B, T_dec]
        encoder_out: [B, T_enc, encoder_dim]

        self_mask:  broadcastable a [B, T_dec, T_dec]
        cross_mask: broadcastable a [B, T_dec, T_enc]
        """
        x = self.token_embedding(token_ids) * math.sqrt(self.block_configs[0]["d_model"])
        x = self.positional_encoding(x)

        all_attentions = []

        for transition, block in zip(self.transitions, self.blocks):
            x = transition(x)

            if return_attentions:
                x, attn_dict = block(
                    x=x,
                    encoder_out=encoder_out,
                    self_mask=self_mask,
                    cross_mask=cross_mask,
                    return_attentions=True,
                )
                all_attentions.append(attn_dict)
            else:
                x = block(
                    x=x,
                    encoder_out=encoder_out,
                    self_mask=self_mask,
                    cross_mask=cross_mask,
                    return_attentions=False,
                )

        x = self.final_norm(x)

        if self.output_logits:
            logits = self.lm_head(x)                      # [B, T_dec, vocab_size]
            if return_attentions:
                return logits, all_attentions
            return logits

        if return_attentions:
            return x, all_attentions
        return x
