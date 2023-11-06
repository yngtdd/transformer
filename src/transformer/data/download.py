from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path


def get_all_sentences(dataset, lang):
    """Get all translation sententes for a `lang`
    
    Args:
        dataset:
        lang: language to use for translation
    """
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, dataset, lang) -> Tokenizer:
    """Get a tokenizer
    
    Args:
        config:
        dataset:
        lang:

    Returns:
        tokenizer: a world level tokenizer
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens = ["[Unk]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency = 2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    """"""
    dataset_raw = load_dataset(
        "opus_books",
        f"{config['language_src']}-{config['lang_target']}",
        split = "train"
    )

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(config, dataset_raw, config["lang_target"])
    train_dataset_raw, valid_dataset_raw = train_valid_split(dataset_raw, train_percent = 0.9)


def train_valid_split(dataset, train_percent: float = 0.9)
    """Split dataset for training and validation

    Args:
        dataset: the dataset to be split
        train_percent: the percentage of the data to reserve for training

    Returns:
        train: training dataset split
        valid: validation dataset split
    """
    valid_percent = 1.0 - train_percent
    train_dataset_size = int(train_percent * len(dataset_raw))
    valid_dataset_size = int(valid_percent * len(dataset_raw))
    train_dataset_raw, valid_dataset_raw = random_split(
        dataset, 
        [train_dataset_size, valid_dataset_size]
    )

    return train_dataset_raw, valid_dataset_raw

