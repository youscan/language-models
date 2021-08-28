import logging
import os

from tokenizers.tokenizers import Tokenizer

from ..pipeline import ITask
from .factory import FAST_TOKENIZER_DEFAULT_FILE_NAME, TokenizerBuilder

# from json import dump, loads


class ExtendTokenizer(ITask):
    def __init__(
        self, base_tokenizer: Tokenizer, extension_tokenizer: Tokenizer, tokenizer_builder: TokenizerBuilder
    ) -> None:
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.extension_tokenizer = extension_tokenizer
        self.tokenizer_builder = tokenizer_builder

    def execute(self, environment_path: str) -> None:
        base_vocabulary = dict(self.base_tokenizer.get_vocab(with_added_tokens=False))
        base_vocabulary_full = dict(self.base_tokenizer.get_vocab(with_added_tokens=True))
        base_added_tokens = dict(
            (token, index) for token, index in base_vocabulary_full.items() if token not in base_vocabulary
        )
        last_id = max(base_vocabulary_full.values())
        logging.info(
            f"Base vocabulary with: {len(base_vocabulary)} base tokens, {len(base_vocabulary_full)} all tokens,"
            f" and particular added tokens: {list(sorted(base_added_tokens.keys()))}"
        )

        extension_vocabulary = dict(self.extension_tokenizer.get_vocab(with_added_tokens=False))
        extension_vocabulary_full = dict(self.extension_tokenizer.get_vocab(with_added_tokens=True))
        extension_added_tokens = dict(
            (token, index) for token, index in extension_vocabulary_full.items() if token not in extension_vocabulary
        )

        logging.info(
            f"Extension vocabulary with: {len(extension_vocabulary)} base tokens, "
            f"{len(extension_vocabulary_full)} all tokens,"
            f" and particular added tokens: {list(sorted(extension_added_tokens.keys()))}"
        )

        extension_tokens_added = 0
        extension_tokens_skipped = 0
        for extension_token, _ in sorted(extension_vocabulary.items()):
            if extension_token not in base_vocabulary_full:
                extension_tokens_added += 1
                extension_token_id = last_id + extension_tokens_added
                base_vocabulary_full[extension_token] = extension_token_id
            else:
                extension_tokens_skipped += 1
        logging.info(
            f"Extended vocabulary with: {extension_tokens_added} tokens, skipped {extension_tokens_skipped} "
            f"tokens as already present."
        )

        added_tokens_to_add = list(token for token, item in sorted(base_added_tokens.items(), key=lambda t: t[1]))
        extension_added_tokens_to_add = []
        for extension_added_token, _ in sorted(extension_added_tokens.items(), key=lambda t: t[1]):
            if extension_added_token in base_added_tokens:
                logging.info(
                    f"Skipping extension added token: {extension_added_token} as already present in base "
                    f"added vocabulary."
                )
            else:
                if extension_added_token in base_vocabulary:
                    logging.warning(
                        f"Extension added token: {extension_added_token} is already present in base "
                        f"vocabulary (not added). This token will be converted to added token."
                    )
                extension_added_tokens_to_add.append(extension_added_token)
        extended_tokenizer = self.tokenizer_builder.from_vocab(
            base_vocabulary_full, additional_special_tokens=(added_tokens_to_add + extension_added_tokens_to_add)
        )
        extended_tokenizer.save(path=os.path.join(environment_path, FAST_TOKENIZER_DEFAULT_FILE_NAME), pretty=True)
        # We need to drop from general vocab the base added tokens (as we added them with `base_vocabulary_full` to
        # preserve indices
        # NOTE: This does not work if we need to serialize such a tokenizer once more.
        # tokenizer_string = extended_tokenizer.to_str(pretty=True)
        # tokenizer_json = loads(tokenizer_string)
        # logging.info("[HOTFIX] Dropping base added tokens from the extended vocabulary")
        # for token in added_tokens_to_add:
        #     token_id = tokenizer_json["model"]["vocab"].pop(token, None)
        #     if token_id is None:
        #         logging.warning(f"Tried to drop `{token}` base added token, but failed. Please review your "
        #                         f"vocabularies.")
        #     else:
        #         logging.info(f"Successfully dropped added token from general vocabulary : {token, token_id}")
        #
        # with open(os.path.join(environment_path, FAST_TOKENIZER_DEFAULT_FILE_NAME), "wt", encoding="utf-8") as \
        #         output_stream:
        #     dump(obj=tokenizer_json, fp=output_stream, ensure_ascii=False, indent=1, sort_keys=False)
