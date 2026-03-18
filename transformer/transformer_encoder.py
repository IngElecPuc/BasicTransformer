import torch
import torch.nn as nn
from input_embedding import InputEmbedding
from transformer_block import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_configs: list[dict],
        max_len: int = 5000,
        pad_idx: int = 0,
        input_dropout: float = 0.0,
        bias: bool = False,
        final_norm: bool = True
    ):
        """
        block_configs: lista de dicts, uno por bloque.
        Cada dict debe tener al menos:
            {
                "d_model": int,
                "num_heads": int,
                "d_ff": int,
                "dropout": float,        # opcional
                "activation": "gelu"     # opcional
            }
        """
        super().__init__()

        if len(block_configs) == 0:
            raise ValueError("block_configs no puede estar vacío")

        self.block_configs = block_configs

        first_d_model = block_configs[0]["d_model"]

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=first_d_model,
            max_len=max_len,
            pad_idx=pad_idx,
            dropout=input_dropout
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

            # transición entre bloque anterior y bloque actual
            if i == 0:
                self.transitions.append(nn.Identity())
            else:
                if prev_d_model == d_model:
                    self.transitions.append(nn.Identity())
                else:
                    self.transitions.append(nn.Linear(prev_d_model, d_model, bias=bias))

            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                bias=bias,
                activation=activation
            )

            self.blocks.append(block)
            prev_d_model = d_model

        self.final_norm = nn.LayerNorm(prev_d_model) if final_norm else nn.Identity()
        self.output_dim = prev_d_model

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attentions: bool = False
    ):
        """
        token_ids: [B, T]
        mask: broadcastable a [B, T, T] para cada bloque

        returns:
            x: [B, T, d_model_final]
            attentions: opcional, lista con las atenciones por bloque
        """
        x = self.input_embedding(token_ids)

        all_attentions = []

        for transition, block in zip(self.transitions, self.blocks):
            x = transition(x)

            if return_attentions:
                x, attn_weights = block(x, mask=mask, return_attention=True)
                all_attentions.append(attn_weights)
            else:
                x = block(x, mask=mask, return_attention=False)

        x = self.final_norm(x)

        if return_attentions:
            return x, all_attentions

        return x

"""Ejemplo de uso

import torch


block_configs = [
    {
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,
        "activation": "gelu",
    },
    {
        "d_model": 96,
        "num_heads": 6,
        "d_ff": 192,
        "dropout": 0.1,
        "activation": "relu",
    },
    {
        "d_model": 128,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.2,
        "activation": "gelu",
    },
]

model = TransformerEncoder(
    vocab_size=1000,
    block_configs=block_configs,
    max_len=256,
    pad_idx=0,
    input_dropout=0.1,
    bias=False,
    final_norm=True
)

token_ids = torch.randint(0, 1000, (2, 10))   # [B, T]
out = model(token_ids)

print("token_ids.shape:", token_ids.shape)
print("out.shape:", out.shape)
print("output_dim:", model.output_dim)
"""