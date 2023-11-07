from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from torch.utils.data import random_split
from transformer.data.bilingual_dataset import BilingualDataset


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

        tokenizer.train_from_iterator(
            get_all_sentences(dataset, lang), trainer = trainer
        )

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def max_seq_lengths(dataset, tokenizer, config):
    """Check the maximum sequence length of a dataset's src and target
    
    Args:
        dataset: a raw HuggingFace dataset
        tokenizer: a HuggingFace tokenizer to get sequences
        config: config containing information about the src and target languages

    Returns:
        max_seq_src: the maximum sequence length of a src sentence
        max_seq_target: the maximum sequence length of a target sentence
    """
    max_len_src = 0
    max_len_target = 0
    for data in dataset:
        src_ids = tokenizer.encode(data["translation"][config["lang_src"]])
        target_ids = tokenizer.encode(data["translation"][config["lang_target"]])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    return max_len_src, max_len_target


def train_valid_split(dataset, train_percent: float = 0.9):
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


def load_opus_dataset(config):
    """Load the Opus Books bilingual dataset

    Args:
        config: our configuration for the dataset

    Returns:
        train_dataset:
        valid_dataset:
        tokenizer_src:
        tokenizer_target:
    """
    dataset_raw = load_dataset(
        "opus_books",
        f"{config['language_src']}-{config['lang_target']}",
        split = "train"
    )

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(config, dataset_raw, config["lang_target"])
    train_dataset_raw, valid_dataset_raw = train_valid_split(dataset_raw, train_percent = 0.9)
    max_len_src, max_len_target = max_seq_lengths(dataset_raw, tokenizer, config)

    if max_len_src > config["seq_len"]:
        raise ValueError(
            f"The source sequence length [{config["seq_len"]}] is shorter "
            "than the dataset max sequence length [{max_len_seq}]"
        )

    if max_len_target > config["seq_len"]:
        raise ValueError(
            f"The target sequence length [{config["seq_len"]}] is shorter "
            "than the dataset max sequence length [{max_len_target}]"
        )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_target,
        config["lang_src"],
        config["lang_target"],
        config["seq_len"]
    )

    train_dataset = BilingualDataset(
        valid_dataset_raw,
        tokenizer_src,
        tokenizer_target,
        config["lang_src"],
        config["lang_target"],
        config["seq_len"]
    )

    return train_dataset, valid_dataset, tokenizer_src, tokenizer_target


def opus_bilingual_dataloaders(config):
    """Create dataloaders for the Opus Books bilingual dataset

    Args:
        config: hyperparameters for our data

    Returns:
        train_dataloader:
        valid_dataloader:
        tokenizer_src:
        tokenizer_target:
    """
    train_dataset, valid_dataset, tokenizer_src, tokenizer_target = load_opus_dataset(config)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = config["batch_size"], 
        shuffle = True
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size = config["batch_size"]
    )

    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_target
