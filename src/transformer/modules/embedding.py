import math
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """Transformer Input Embedding

        Args:
            d_model: the Transformer model dimension
            vocab_size: number of terms in our vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """Embed our tokenized inputs
        
        Note:
            Following section 3.4 in "Attention is All You Need",
            we multiply the embeddings by the square root of the 
            model's dimension
        """
        return self.embedding(x) * math.sqrt(self.d_model)


