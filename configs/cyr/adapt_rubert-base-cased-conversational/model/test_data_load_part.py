import logging

from transformers import BertTokenizerFast

from language_model.data.dataset import LineByLineTextDataset
from language_model.pipeline import SandboxTask
from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME
from src.language_model.data.dataset import LazyDataset, split_lazy_dataset

LOGGING_FORMAT: str = "%(asctime)s : %(levelname)s : %(module)s : %(message)s"

TOKENIZER_PATH = (
    f"outputs/cyr/adapt_rubert-base-cased-conversational/tokenizer/extend_base_vocab=96k"
    f"/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"
)
TEXT_FILE_PATHS = [
    # "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/rus/texts.txt",
    "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/unc/texts.txt",
    # "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/ukr/texts.txt",
]
# TEXT_FILE_PATHS = [
#     "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test.txt",
#     "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test2.txt",
#     "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test3.txt",
# ]
TRAIN_TEST_PORTIONS = [0.95, 0.05]

tokenizer = BertTokenizerFast(
    vocab_file=None,
    tokenizer_file=TOKENIZER_PATH,
    from_slow=False,
    do_lower_case=False,
    strip_accents=False,
    tokenize_chinese_chars=False,
)

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths=TEXT_FILE_PATHS, block_size=128)
train_set, eval_set = split_lazy_dataset(dataset, portions=TRAIN_TEST_PORTIONS)


class TestTask(SandboxTask):
    def __init__(self, train_set: LazyDataset, test_set: LazyDataset, sandbox_folder_path: str = "outputs") -> None:
        super().__init__(sandbox_folder_path)
        self.train_set = train_set
        self.test_set = test_set

    def execute(self, environment_path: str) -> None:
        logging.info(f"Train set size: {len(self.train_set)}")
        logging.info(f"Train set size: {len(self.test_set)}")


task = TestTask(train_set=train_set, test_set=eval_set)
