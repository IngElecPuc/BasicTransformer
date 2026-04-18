
import torch
import torch.nn as nn

class BertPooler(nn.Module):
    """
    Pooler estilo BERT:
    toma el hidden state del token [CLS] y lo proyecta.
    """

    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]

        returns:
            [B, H]
        """
        cls_token_state = hidden_states[:, 0]
        pooled_output = self.dense(cls_token_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output
