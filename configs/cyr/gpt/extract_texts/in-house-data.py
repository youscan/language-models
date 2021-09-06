from pynlple.processing.preprocessor import (
    HtmlTagReplacer,
    MultiLetterReplacer,
    MultiNonLetterReplacer,
    StackingPreprocessor,
    URLReplacer,
)

from language_model.data.extract import ExtractTextsFromData, FromLoadedYsDataSource

YS_FOLDER_PATHS = ["outputs/cyr/gpt/load_data/in-house"]


preprocessor = StackingPreprocessor(
    [HtmlTagReplacer(), URLReplacer(), MultiNonLetterReplacer(include_digits=False), MultiLetterReplacer()]
)

ys_train = FromLoadedYsDataSource(source_folder_paths=YS_FOLDER_PATHS)

task = ExtractTextsFromData(
    text_source=ys_train, preprocessor=preprocessor, seeds=100, char_ngram=20, bands=20, min_jaccard=0.9
)
