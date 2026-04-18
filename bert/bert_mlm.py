import torch
import torch.nn as nn
from .bert_model import BertModel
from .bert_heads import BertOnlyMLMHead

class BertForMaskedLM(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.bert = bert
        self.cls = BertOnlyMLMHead(
            hidden_size=bert.hidden_size,
            vocab_size=bert.vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        if tie_word_embeddings:
            self.cls.tie_weights(self.bert.embeddings.word_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_padding_mask: bool = True,
        labels: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            auto_padding_mask=auto_padding_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)

        result = {"logits": prediction_scores}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
            )
            result["loss"] = loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result
