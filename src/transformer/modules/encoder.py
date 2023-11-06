import torch
import torch.nn as nn

from transformer.modules import (
    FeedForward,
    MultiHeadAttention,
    Residual
)


class EncoderBlock(nn.Module):

    def __init__(
        self, 
        self_attention: MultiHeadAttention, 
        feed_forward: FeedForward, 
        dropout: float
    ):
        """Transformer Encoder Block
        
        Args:
            self_attention: Multi-head attention
            feed_forward: feed forward network
            dropout: Percent dropout
        """
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList([Residual(dropout) for _ in range(2)])

     def forward(self, x, mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer is self.layers:
            x = layer(x, mask)
        return self.norm(x)

