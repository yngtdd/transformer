import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Transformer Input Embedding"""

    def __init__(self, d_model: int, vocab_size: int):
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



class PositionalEncoding(nn.Module):
    """Trigonometric Positional Encoding
    
    Represents the position of a word within a sequence
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self._position_encoding(d_model, seq_len)

    def _positional_encoding(self, d_model: int, seq_len: int):
        """Create a positional encoding

        Using trigonometric functions, we create a positional encoding
        for every token in the sequence. For tokens in even positions of the
        sequence, we use `sin()`, and for odd positions we use `cos()`.
        
        Args:
            d_model: the model embedding dimension
            seq_len: the sequence length of our data
        """
        position_encoding = torch.zeros(seq_len, d_model)
        # Position vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000) / d_model))
        # For even terms in the position encoding, we use the sin()
        position_encoding[:, 0::2] = torch.sin(position * denominator)
        # For odd terms, we use the cos()
        position_encoding[:, 1::2] = torch.cos(position * denominator)
        # Reshape the position encoding to be of shape (1, seq_len, d_model)
        position_encoding = position_encoding.unsqueeze(0)
        # Store the positional encoding in the module, not as a parameter
        self.register_buffer('positional_encoding', position_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

