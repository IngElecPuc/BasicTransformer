
import torch
import torch.nn as nn

class BertPredictionHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size, bias=bias)

        if hidden_act.lower() == "gelu":
            self.transform_act_fn = nn.GELU()
        elif hidden_act.lower() == "relu":
            self.transform_act_fn = nn.ReLU()
        else:
            raise ValueError("hidden_act debe ser 'gelu' o 'relu'")

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Head de MLM.
    Puede atar pesos con word_embeddings del modelo.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.transform = BertPredictionHeadTransform(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.decoder.weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.predictions.tie_weights(embedding_weight)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)


class BertOnlyNSPHead(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2, bias=bias)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.seq_relationship(pooled_output)


class BertPreTrainingHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.predictions = BertLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )
        self.seq_relationship = nn.Linear(hidden_size, 2, bias=bias)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.predictions.tie_weights(embedding_weight)

    def forward(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores

