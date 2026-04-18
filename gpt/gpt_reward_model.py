import torch.nn as nn
from .gpt_model import GPTModel

#Plantilla: TODO

class GPTRewardModel(nn.Module):
    def __init__(self, gpt: GPTModel):
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(self.gpt.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs["last_hidden_state"]
        pooled = hidden[:, -1, :]
        reward = self.value_head(pooled)
        return {"reward": reward}