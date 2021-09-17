from itertools import chain

from datasets import load_dataset
from pynlple.processing.preprocessor import (
    HtmlTagReplacer,
    MultiLetterReplacer,
    MultiNonLetterReplacer,
    StackingPreprocessor,
    URLReplacer,
)

from language_model.data.extract import PostWikiExtractorDataSource, RandomSplitTextsFromData

WIKI_EXTRACTED_PATH = "outputs/cyr/gpt/load_data/wiki/ukwiki-latest-pages-articles"


preprocessor = StackingPreprocessor(
    [HtmlTagReplacer(), URLReplacer(), MultiNonLetterReplacer(include_digits=False), MultiLetterReplacer()]
)

oscar_train = (item["text"] for item in load_dataset("oscar", "unshuffled_deduplicated_uk", split="train"))
cc100_train = (item["text"] for item in load_dataset("cc100", lang="uk", split="train"))
wiki_train = (item["text"] for item in PostWikiExtractorDataSource(WIKI_EXTRACTED_PATH))


task = RandomSplitTextsFromData(
    text_source=chain(oscar_train, cc100_train, wiki_train), preprocessor=preprocessor, test_size=5_000
)
