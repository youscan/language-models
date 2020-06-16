from tokenizers import ByteLevelBPETokenizer

from src.configs import TrainTokenizerConfig

config = TrainTokenizerConfig(
    source_folder_path="data/version_1/ukr/prepare_one_line_text_format/wiki_oscar_ys/",
    tokenizer=ByteLevelBPETokenizer(),
    vocab_size=52000,
    min_frequency=5,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
