import os

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from ..pipeline import ITask
from .factory import FAST_TOKENIZER_DEFAULT_FILE_NAME


class FastTokenizerSavingTask(ITask):
    def __init__(self, fast_tokenizer: Tokenizer) -> None:
        super().__init__()
        self.fast_tokenizer = fast_tokenizer

    def execute(self, environment_path: str) -> None:
        self.fast_tokenizer.save(os.path.join(environment_path, FAST_TOKENIZER_DEFAULT_FILE_NAME), pretty=True)


class PreTrainedTokenizerFastSavingTask(ITask):
    def __init__(
        self, pretrained_fast_tokenizer: PreTrainedTokenizerFast, tokenizer_folder_name: str = "tokenizer"
    ) -> None:
        super().__init__()
        self.pretrained_fast_tokenizer = pretrained_fast_tokenizer
        self.tokenizer_folder_name = tokenizer_folder_name

    def execute(self, environment_path: str) -> None:
        self.pretrained_fast_tokenizer.save_pretrained(
            os.path.join(environment_path, self.tokenizer_folder_name), legacy_format=False
        )
