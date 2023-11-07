from transformer.data import opus_bilingual_dataloaders
from transformer.modules import build_transformer, LitTransformer


def main():
    config = load_config()
    dataloaders_tokenizers = opus_bilingual_dataloaders(config)

