from tokenizers.implementations import ByteLevelBPETokenizer

from language_model.tokenization.trainer import ByteLevelBPETokenizerTrainer

task = ByteLevelBPETokenizerTrainer(
    source_folder_path="data/ukr/data/wiki_oscar_data/",
    tokenizer=ByteLevelBPETokenizer(),
    vocab_size=52000,
    min_frequency=5,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
