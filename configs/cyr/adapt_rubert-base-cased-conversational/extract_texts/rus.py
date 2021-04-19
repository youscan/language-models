from pynlple.processing.preprocessor import (
    MultiLetterReplacer,
    MultiNonLetterReplacer,
    StackingPreprocessor,
    URLReplacerPreservingHighlightedEntity,
)

from language_model.data.extract import ExtractTextsFromData

task = ExtractTextsFromData(
    source_folder_paths=[
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=283676/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=14515764/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=12853815/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=19273117/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=20634745/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=2869642/",
        "data/cyr/adapt_rubert-base-cased-conversational/data/rus_seq=1570560/",
    ],
    preprocessor=StackingPreprocessor(
        [
            MultiNonLetterReplacer(include_digits=False),
            MultiLetterReplacer(),
            URLReplacerPreservingHighlightedEntity(),
            # EmailReplacerPreservingHighlightedEntity(),
        ]
    ),
)
