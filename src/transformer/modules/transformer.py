import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from transformer.modules.attention import MultiHeadAttention
from transformer.modules.encoder import Encoder, EncoderBlock
from transformer.modules.decoder import Decoder, DecoderBlock
from transformer.modules.embedding import Embedding
from transformer.modules.positional_encoding import PositionalEncoding
from transformer.modules.projection import LinearProjection
from transformer.modules.feed_forward import FeedForward


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embedding,
        target_embed: Embedding,
        src_pos: PositionalEncoding,
        target_pos: PositionalEncoding,
        projection: LinearProjection
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection(x)


class LitTransformer(pl.LightningModule):

    def __init__(
        self, 
        transformer: Transformer, 
        pad_token_id: int, 
        target_vocab_size: int
    ):
        """Pytorch Lightning Transformer
        
        Args:
            transformer: our Transformer model
            pad_token_id: padding token ID from tokenizer. This 
                is ignored by our loss function.
            target_vocab_size: size of the target language vocabulary

        Note:
            A couple
        """
        super().__init__()
        self.transformer = transformer
        self.pad_token_id = pad_token_id
        self.target_vocab_size = target_vocab_size

    def training_step(self, batch, batch_idx):
        """Training step"""
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        target = batch["target"]

        encoder_output = self.transformer.encode(
            encoder_input, 
            encoder_mask
        )
        decoder_output = self.transformer.decode(
            encoder_output, 
            encoder_mask, 
            decoder_input, 
            decoder_mask
        )
        pred = self.transformer.project(decoder_output)
        # Reshape the model predictions to be of size (batch, seq_len, target_vocab_size)
        pred = pred.view(-1 self.target_vocab_size)
        loss = F.cross_entropy(pred, target.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        target = batch["target"]

        encoder_output = self.transformer.encode(
            encoder_input, 
            encoder_mask
        )
        decoder_output = self.transformer.decode(
            encoder_output, 
            encoder_mask, 
            decoder_input, 
            decoder_mask
        )
        pred = self.transformer.project(decoder_output)
        pred = pred.view(-1 self.target_vocab_size)
        loss = F.cross_entropy(pred, target.view(-1))
        self.log("valid_loss", loss)
        return loss


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    num_encoder_decoder_blocks: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    """Build a `Transformer` model
    
    This gives us a way to handle the various hyperparameters 
    of the network without packing everything in the Transformer's
    constructor.

    Args:
        src_vocab_size: source language vocab size
        target_vocab_size: target language vocab size
        src_seq_len: source language sequence length
        target_seq_len: target language sequence length
        d_model: Transformer model dimension
        num_encoder_decoder_blocks: the number of encoder decoder block pairs
        num_heads: the number of attention heads
        dropout: the percent dropout
        d_ff: the hidden dimension of the fully connected layer within the feed forward nets

    Returns:
        transformer: a transformer model with weights initialized by the xavier uniform
    """
    src_embed = Embedding(d_model, src_vocab_size)
    target_embed = Embedding(d_model, target_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(num_encoder_decoder_blocks):
        encoder_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(num_encoder_decoder_blocks):
        decoder_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Encoder(nn.ModuleList(decoder_blocks))
    projection = LinearProjection(d_model, target_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_pos,
        target_pos,
        projection
    )

    # Initialize parameters
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    return transformer
