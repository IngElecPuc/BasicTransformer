import torch.nn as nn
from .bert_model import BertModel

#Plantilla: TODO

class BertForTokenClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels: int):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs["last_hidden_state"]
        logits = self.classifier(self.dropout(sequence_output))

        result = {"logits": logits}
        return result