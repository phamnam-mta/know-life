import numpy as np
import torch
import torch.nn as nn



class SlotClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_slot_labels,
        attention_embedding_size=200,
        dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        self.num_slot_labels = num_slot_labels
        self.linear_slot = nn.Linear(input_dim, attention_embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(attention_embedding_size, num_slot_labels)

    def forward(self, x):
        x = self.linear_slot(x)
        x = self.dropout(x)
        return self.linear(x)
