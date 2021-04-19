from tokenizers.normalizers import NFC
from tokenizers.tokenizers import Tokenizer

from language_model.tokenization.extend import ExtendTokenizer
from language_model.tokenization.factory import BertTokenizerBuilder

task = ExtendTokenizer(
    base_tokenizer=Tokenizer.from_file(
        path="outputs/cyr/adapt_rubert-base-cased-conversational/tokenizer/base_bert-tokenizer/fast_tokenizer.tks"
    ),
    extension_tokenizer=Tokenizer.from_file(
        path="outputs/cyr/adapt_rubert-base-cased-conversational/tokenizer/custom_bert-tokenizer_vocab=96k,"
        "min=5_alphabet=3k_nfc_no-hanging-accents/fast_tokenizer.tks"
    ),
    tokenizer_builder=BertTokenizerBuilder(
        lowercase=False,
        unicode_normalizer=NFC(),
        strip_hanging_accents=True,
        strip_accents=False,
        handle_chinese_chars=False,
    ),
)
