import torch
import torch.nn as nn

from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer_src,
        tokenizer_target,
        src_lang,
        target_lang,
        seq_len
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_target.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_target.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_target.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        target_text = src_target_pair["translation"][self.target_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # We do not need an EOS token here

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        target = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                    
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert target.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len,)
            "decoder_input": decoder_input,
            "encoder_mask": create_mask(encoder_input, self.pad_token), # (1, 1, seq_len)
            "decoder_mask": create_decoder_mask(decoder_input, self.pad_token, decoder_input.size(0)),
            "target": target, # (seq__len,)
            "src_text": src_text,
            "target_text": target_text
        }


def create_mask(input_tokens, pad_token):
    """Create an input token mask

    Mask all tokens where the token is not the padding token.
    We do not want our model attending to padding.

    Args:
        input_tokens: padded token sequence
        pad_token: the padding token to creata a mask over

    Returns:
        input_mask: the input token mask
    """
    # Add 
    input_mask = (input_tokens != pad_token).unsqueeze(0).unsqueeze(0).int()
    return input_mask


def create_decoder_mask(input_tokens, pad_token, decoder_input_size):
    """Create the decder input token mask

    The decoder token mask accounts for both the padding token
    as well as future tokens. We do not want the model's decoder 
    to attend to future tokens. Since this is a sequence model, we do
    not want the future to inform the past.
    """
    decoder_mask = create_mask(input_tokens, pad_token) & causal_mask(decoder_input_size)
    return decoder_mask
       

def causal_mask(size: int):
    """Create a causal mask for attention

    We do not want our decoder attention to attend 
    to future words. In order to prevent this, we create
    what we call a causal mask, the upper triangular matrix 
    of the decoder input. This sets that upper triangle to 
    all zeros.

    Args:
        size: the size of the decoder input matrix

    Returns:
        mask: a mask for the decoder matrix
    """
    mask = torch.triu(
        torch.ones(1, size, size), 
        diagonal = 1
    ).type(torch.int)

    return mask == 0
