import torch
import torch.nn as nn

from transformer.modules import (
    MultiHeadAttention, 
    FeedForward, 
    Residual, 
    LayerNormalization
)


class DecoderBlock(nn.Module):

    def __init__(
        self, 
        self_attention: MultiHeadAttention, 
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residuals = nn.Module([Residual(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        """"""
        x = residuals[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
        

        
