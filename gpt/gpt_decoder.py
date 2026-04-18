import torch
import torch.nn as nn

from transformer.transformer_block import TransformerBlock
from transformer.utils import validate_block_configs


class GPTDecoder(nn.Module):
    """
    Stack decoder-only, autoregresivo.
    Permite:
    - block_configs explícito
    - patrón repetido
    - transiciones entre bloques si cambia d_model
    """

    def __init__(
        self,
        block_configs: list[dict],
        bias: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()

        validate_block_configs(block_configs)

        self.block_configs = block_configs
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        prev_d_model = block_configs[0]["d_model"]

        for i, cfg in enumerate(block_configs):
            d_model = cfg["d_model"]
            num_heads = cfg["num_heads"]
            d_ff = cfg["d_ff"]
            dropout = cfg.get("dropout", 0.0)
            activation = cfg.get("activation", "gelu")

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
                activation=activation,
            )

            self.blocks.append(block)
            prev_d_model = d_model

        self.final_norm = nn.LayerNorm(prev_d_model) if final_norm else nn.Identity()
        self.output_dim = prev_d_model

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        x: [B, T, H]
        attention_mask: broadcastable a [B, T, T]
        """
        all_hidden_states = []
        all_attentions = []

        if return_hidden_states:
            all_hidden_states.append(x)

        for transition, block in zip(self.transitions, self.blocks):
            x = transition(x)

            if return_attentions:
                x, attn_weights = block(
                    x=x,
                    mask=attention_mask,
                    return_attention=True,
                )
                all_attentions.append(attn_weights)
            else:
                x = block(
                    x=x,
                    mask=attention_mask,
                    return_attention=False,
                )

            if return_hidden_states:
                all_hidden_states.append(x)

        x = self.final_norm(x)

        outputs = {"last_hidden_state": x}

        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states

        if return_attentions:
            outputs["attentions"] = all_attentions

        return outputs
