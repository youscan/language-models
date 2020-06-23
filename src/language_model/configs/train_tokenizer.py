from typing import List

from tokenizers import ByteLevelBPETokenizer

from .default_with_saving import ConfigWithSaving


class TrainTokenizerConfig(ConfigWithSaving):
    def __init__(
        self,
        source_folder_path: str,
        tokenizer: ByteLevelBPETokenizer,
        vocab_size: int,
        min_frequency: int,
        special_tokens: List[str],
        saving_folder_prefix: str = "results",
        logs: str = "logs",
    ) -> None:
        super().__init__(saving_folder_prefix=saving_folder_prefix, logs_dir=logs)
        self.source_folder_path = source_folder_path
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.min_frequency = min_frequency
        self.vocab_size = vocab_size
