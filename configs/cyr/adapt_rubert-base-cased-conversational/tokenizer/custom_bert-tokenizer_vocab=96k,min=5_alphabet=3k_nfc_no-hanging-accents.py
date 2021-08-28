from pynlple.data.corpus import FileLineSource, StackingSource
from tokenizers.normalizers import NFC
from tokenizers.trainers import WordPieceTrainer

from language_model.tokenization.factory import BertTokenizerBuilder
from language_model.tokenization.trainer import TrainTokenizerTask

filepaths = [
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/rus/texts.txt",
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/ukr/texts.txt",
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/unc/texts.txt",
]

text_source = iter(StackingSource([FileLineSource(path, encoding="utf-8") for path in filepaths]))


task = TrainTokenizerTask(
    tokenizer=BertTokenizerBuilder(
        lowercase=False,
        unicode_normalizer=NFC(),
        strip_hanging_accents=True,
        strip_accents=False,
        handle_chinese_chars=False,
    ).from_vocab(additional_tokens=["<b>", "</b>", "<url>", "</url>"]),
    iterator=text_source,
    trainer=WordPieceTrainer(
        vocab_size=96000,
        min_frequency=5,
        limit_alphabet=3000,
        initial_alphabet=[],
        show_progress=True,
        continuing_subword_prefix="##",
    ),
)
