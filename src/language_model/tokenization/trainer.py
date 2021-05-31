import os
from typing import Iterator, List, Union

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.trainers import Trainer

from ..pipeline import ITask
from .factory import FAST_TOKENIZER_DEFAULT_FILE_NAME


class ByteLevelBPETokenizerTrainer(ITask):
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

    def execute(self, environment_path: str) -> None:
        files = self.get_all_files_in_folder(self.source_folder_path)

        self.tokenizer.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )

        self.tokenizer.save(os.path.join(environment_path, "tokenizer"))

    @staticmethod
    def get_all_files_in_folder(data_folder_path: str) -> List[str]:
        data_files_paths = []
        for (dir_path, _, filenames) in os.walk(data_folder_path):
            data_files_paths.extend([os.path.join(dir_path, file_name) for file_name in filenames])
        return data_files_paths


class TrainTokenizerTask(ITask):
    def __init__(
        self,
        tokenizer: Tokenizer,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: Trainer,
        tokenizer_file_name: str = FAST_TOKENIZER_DEFAULT_FILE_NAME,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.iterator = iterator
        self.trainer = trainer
        self.tokenizer_file_name = tokenizer_file_name

    def execute(self, environment_path: str) -> None:
        self.tokenizer.train_from_iterator(self.iterator, trainer=self.trainer)
        self.tokenizer.save(path=os.path.join(environment_path, self.tokenizer_file_name), pretty=True)
