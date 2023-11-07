import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        """Multihead Attention
        
        Args:
            d_model: Transformer model dimension
            num_heads: Number of attentions heads (`h` in "Attention is All You Need")
            dropout: Percent dropout

        Raises:
            ValueError: if `d_model` is not divisble by `num_heads`
        """
        super().__init__()
        self._check_d_model_divisible(d_model, num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        # Our Query (Wq), Key (Wk), and Value (Wv) layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _check_d_model_divisible(self, d_model: int, num_heads: int):
        """Check that `d_model` is divisble by `num_heads`
        
        Args:
            d_model: Transformer model dimension
            num_heads: Number of attention heads

        Raises:
            ValueError: if `d_model` is not divisble by `num_heads`
        """        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model [{d_model}] is not divisible by num_heads [{num_heads}]")

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        """Self attention

        Args:
            query
            key:
            value:
            mask:
            dropout
        """
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """Forward pass

        Args:
            q: query tensor
            k: key tensor
            v: value tensor,
            mask: mask

        Note:
            For the query, key, and value tensors we move from 
            `(batch, seq_len, d_model) -> (batch, seq_len, d_model)`.
            We then split the tensors by the number of heads to be of shape
            `(batch, seq_len, num_heads, d_k)` and finally transpose two of the
            dimensions such that we finally have `(batch, num_heads, seq_len, d_k)`.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split the tensors from `(batch, seq_len, num_heads, d_k)` -> `(batch, num_heads, seq_len, d_k)`
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_len, d_k) -> (batch seq_len, num_heads, d_k) -> (batch seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

        
        
        
        
