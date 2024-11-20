# model

import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

CLS_NAMES = ['business', 'entertainment', 'medicine', 'tech&science']


class NewsClsDistilBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc1 = nn.Linear(768, 128)
        self.dropout = nn.Dropout(.3)
        self.cls = nn.Linear(128, len(CLS_NAMES))

    def forward(self, input_ids, att_msk):
        out = self.dbert(input_ids=input_ids, attention_mask=att_msk)
        hidden_state = out[0]
        pooler = hidden_state[:, 0]
        x = F.relu(self.fc1(pooler))
        x = self.dropout(x)
        return self.cls(x)
