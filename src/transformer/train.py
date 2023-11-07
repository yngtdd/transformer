import lightning.pytorch as pl

from transformer.config import load_config
from transformer.data import opus_bilingual_dataloaders
from transformer.modules import build_transformer, LitTransformer
from lightning.pytorch import Trainer


def main():
    config = load_config()
    dataloaders_tokenizers = opus_bilingual_dataloaders(config)

    train_dataloader = dataloaders_tokenizers["train_dataloader"]
    valid_dataloader = dataloaders_tokenizers["valid_dataloader"]

    tokenizer_src = dataloaders_tokenizers["tokenizer_src"]
    tokenizer_target = dataloaders_tokenizers["tokenizer_target"]

    pad_token_id = tokenizer_src.token_to_id("[PAD]")
    target_vocab_size = tokenizer_target.get_vocab_size()

    transformer = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_target.get_vocab_size(),
        src_seq_len = 350,
        target_seq_len = 350
    )

    model = LitTransformer(transformer, pad_token_id, target_vocab_size)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(
        model, 
        train_dataloaders = train_dataloader,
        val_dataloaders = valid_dataloader,
    )


if __name__=="__main__":
    main()

