import torch
import torch.nn as nn
from .bert_model import BertModel
from .bert_heads import BertOnlyNSPHead

class BertForNextSentencePrediction(nn.Module):
    def __init__(self, bert: BertModel, bias: bool = True):
        super().__init__()
        self.bert = bert

        if self.bert.pooler is None:
            raise ValueError("BertModel debe haberse creado con add_pooling_layer=True")

        self.cls = BertOnlyNSPHead(hidden_size=bert.hidden_size, bias=bias)

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

        pooled_output = outputs["pooler_output"]
        seq_relationship_scores = self.cls(pooled_output)

        result = {"logits": seq_relationship_scores}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                seq_relationship_scores.view(-1, 2),
                labels.view(-1),
            )
            result["loss"] = loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result
