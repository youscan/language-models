import os

from tokenizers import Tokenizer

from ..pipeline import ITask
from .factory import FAST_TOKENIZER_DEFAULT_FILE_NAME


class FastTokenizerSavingTask(ITask):
    def __init__(self, fast_tokenizer: Tokenizer) -> None:
        super().__init__()
        self.fast_tokenizer = fast_tokenizer

    def execute(self, environment_path: str) -> None:
        self.fast_tokenizer.save(os.path.join(environment_path, FAST_TOKENIZER_DEFAULT_FILE_NAME), pretty=True)
