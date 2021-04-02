import os
from typing import List

from tokenizers import ByteLevelBPETokenizer

from ..pipeline import SandboxTask


class ByteLevelBPETokenizerTrainer(SandboxTask):
    def __init__(
        self,
        source_folder_path: str,
        tokenizer: ByteLevelBPETokenizer,
        vocab_size: int,
        min_frequency: int,
        special_tokens: List[str],
    ) -> None:
        super().__init__()
        self.source_folder_path = source_folder_path
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.min_frequency = min_frequency
        self.vocab_size = vocab_size

    def execute(self, sandbox_folder_path: str) -> None:
        files = self.get_all_files_in_folder(self.source_folder_path)

        self.tokenizer.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )

        self.tokenizer.save(os.path.join(sandbox_folder_path, "tokenizer"))

    @staticmethod
    def get_all_files_in_folder(data_folder_path: str) -> List[str]:
        data_files_paths = []
        for (dir_path, _, filenames) in os.walk(data_folder_path):
            data_files_paths.extend([os.path.join(dir_path, file_name) for file_name in filenames])
        return data_files_paths
