from language_model.configs import TrainTokenizerConfig
from tokenizers import ByteLevelBPETokenizer

config = TrainTokenizerConfig(
    source_folder_path="data/version_1/ukr/data/wiki_oscar_data/",
    tokenizer=ByteLevelBPETokenizer(),
    vocab_size=52000,
    min_frequency=5,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
