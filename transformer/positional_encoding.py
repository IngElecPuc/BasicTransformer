import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # pe tendrá forma [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # posiciones: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # frecuencias para dimensiones pares: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # índices pares -> seno
        pe[:, 0::2] = torch.sin(position * div_term)

        # índices impares -> coseno
        pe[:, 1::2] = torch.cos(position * div_term)

        # agregamos dimensión batch: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # buffer no entrenable, pero guardado con el módulo
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]

        returns:
            [B, T, d_model]
        """
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)

        return x
