import torch
import torch.nn as nn

from transformer.modules.attention import MultiHeadAttention
from transformer.modules.feed_forward import FeedForward
from transformer.modules.layer_norm import LayerNormalization
from transformer.modules.residual import Residual


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

    def forward(self, x, src_mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

