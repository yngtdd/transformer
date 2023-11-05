import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """Position-wise Feed-Forward block
        
        Args:
            d_model: dimension of the transformer model
            d_ff: hidden layer size in the feed forward block
            dropout: the percent dropout

        Note:
            See section 3.3 Position-wise Feed-Forward Networks of 
            "Attention is All You Need"
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for the feed forward block
        
        Note: The tensor shapes are as follows:
              1. x: `(batch, seq_len, d_model)` ->
              2. Linear_1: `(batch, seq_len, d_model)` -> `(batch, seq_len, d_ff)`
              3. Linear_2: `(batch, seq_len, d_ff)` ->  `(batch, seq_len, d_model)`
        """
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

