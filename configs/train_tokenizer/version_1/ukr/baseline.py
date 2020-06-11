from tokenizers import ByteLevelBPETokenizer

from src.configs import TrainTokenizerConfig

config = TrainTokenizerConfig(
    source_folder_path="data/version_1/ukr/prepare_one_line_text_format/sampled_mentions/",
    tokenizer=ByteLevelBPETokenizer(),
    vocab_size=32000,
    min_frequency=5,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
