import torch
import torch.nn as nn

from transformer.modules import LayerNormalization


class Residual(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, prev_layer):
        return x + self.dropout(prev_layer(self.norm(x)))
        