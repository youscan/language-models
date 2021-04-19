from tokenizers.normalizers import NFC

from language_model.tokenization.factory import BertTokenizerBuilder
from src.language_model.tokenization.tasks import FastTokenizerSavingTask

VOCAB_FILE_PATH = "data/DeepPavlov/rubert-base-cased-conversational/vocab.txt"

tokenizer = BertTokenizerBuilder(
    lowercase=False,
    unicode_normalizer=NFC(),
    strip_hanging_accents=True,
    strip_accents=False,
    handle_chinese_chars=False,
).from_vocab_file(vocab_file=VOCAB_FILE_PATH, additional_tokens=["<b>", "</b>", "<url>", "</url>"])

task = FastTokenizerSavingTask(fast_tokenizer=tokenizer)
