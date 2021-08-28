from pynlple.processing.preprocessor import (
    MultiLetterReplacer,
    MultiNonLetterReplacer,
    StackingPreprocessor,
    URLReplacerPreservingHighlightedEntity,
)

from language_model.data.extract import ExtractTextsFromData

task = ExtractTextsFromData(
    source_folder_paths=["/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/data/unc/"],
    preprocessor=StackingPreprocessor(
        [
            MultiNonLetterReplacer(include_digits=False),
            MultiLetterReplacer(),
            URLReplacerPreservingHighlightedEntity(),
            # EmailReplacerPreservingHighlightedEntity(),
        ]
    ),
)
