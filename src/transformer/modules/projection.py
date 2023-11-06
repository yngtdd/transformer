import torch
import torch.nn as nn


class LinearProjection(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Forward pass
        
        Note:
            We want to go from our Decoder's output shape of 
            `(batch, seq_len, d_model)` -> `(batch, seq_len, vocab_size)`,
            a linear projection from the dimension of the model to the size
            of our vocabulary.
        """
        return torch.log_softmax(self.linear(x), dim = -1)


        
