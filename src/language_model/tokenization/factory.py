from typing import Any, Dict, Iterable, Optional, Union

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer, Normalizer
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.normalizers import StripAccents
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
from tokenizers.processors import BertProcessing

FAST_TOKENIZER_DEFAULT_FILE_NAME = "fast_tokenizer.tks"


class TokenizerBuilder(object):
    def from_vocab(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        additional_special_tokens: Iterable[str] = (),
        additional_tokens: Iterable[str] = (),
    ) -> Tokenizer:
        raise NotImplementedError()


class BertTokenizerBuilder(TokenizerBuilder):
    def __init__(
        self,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        unicode_normalizer: Optional[Normalizer] = None,
        strip_hanging_accents: bool = False,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        max_input_chars_per_word: int = 100,
    ):
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.clean_text = clean_text
        self.handle_chinese_chars = handle_chinese_chars
        self.unicode_normalizer = unicode_normalizer
        self.strip_hanging_accents = strip_hanging_accents
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.wordpieces_prefix = wordpieces_prefix
        self.max_input_chars_per_word = max_input_chars_per_word

    def from_vocab(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        additional_special_tokens: Iterable[str] = (),
        additional_tokens: Iterable[str] = (),
    ) -> Tokenizer:

        if vocab is not None:
            tokenizer = Tokenizer(
                WordPiece(vocab, unk_token=str(self.unk_token), max_input_chars_per_word=self.max_input_chars_per_word)
            )
        else:
            tokenizer = Tokenizer(
                WordPiece(unk_token=str(self.unk_token), max_input_chars_per_word=self.max_input_chars_per_word)
            )

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(self.unk_token)) is not None:
            tokenizer.add_special_tokens([str(self.unk_token)])
        if tokenizer.token_to_id(str(self.sep_token)) is not None:
            tokenizer.add_special_tokens([str(self.sep_token)])
        if tokenizer.token_to_id(str(self.cls_token)) is not None:
            tokenizer.add_special_tokens([str(self.cls_token)])
        if tokenizer.token_to_id(str(self.pad_token)) is not None:
            tokenizer.add_special_tokens([str(self.pad_token)])
        if tokenizer.token_to_id(str(self.mask_token)) is not None:
            tokenizer.add_special_tokens([str(self.mask_token)])

        tokenizer.add_special_tokens(list(additional_special_tokens))

        tokenizer.add_tokens(list(additional_tokens))

        normalizers = []
        if self.unicode_normalizer:
            normalizers.append(self.unicode_normalizer)
        if self.strip_hanging_accents:
            normalizers.append(StripAccents())

        normalizers.append(
            BertNormalizer(
                clean_text=self.clean_text,
                handle_chinese_chars=self.handle_chinese_chars,
                strip_accents=self.strip_accents,
                lowercase=self.lowercase,
            )
        )

        tokenizer.normalizer = NormalizerSequence(normalizers)

        tokenizer.pre_tokenizer = Sequence(
            [
                Split(Regex(r"\p{Nd}"), behavior="contiguous"),
                Split(Regex(r"\p{Nl}"), behavior="contiguous"),
                Split(Regex(r"\p{P}|\p{Sc}|\p{Sm}|\p{So}|\p{No}"), behavior="isolated"),
                WhitespaceSplit(),
            ]
        )
        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(self.sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(self.cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = BertProcessing(
                (str(self.sep_token), sep_token_id), (str(self.cls_token), cls_token_id)
            )
        tokenizer.decoder = WordPieceDecoder(prefix=self.wordpieces_prefix)

        return tokenizer

    def from_vocab_file(self, vocab_file: str, **kwargs: Any) -> Tokenizer:
        vocab = WordPiece.read_file(vocab_file)
        return self.from_vocab(vocab, **kwargs)
