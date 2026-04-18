import torch.nn as nn
from .bert_model import BertModel

#Plantilla: TODO

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels: int):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs["pooler_output"]
        logits = self.classifier(self.dropout(pooled))

        result = {"logits": logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss
        return result