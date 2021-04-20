import io
import logging
import os
from collections import Hashable as HashableType
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, Optional

from ds_shared.loading import load_pickle
from pynlple.data.corpus import FilteringSource, JsonFieldSource, MappingSource, SplittingSource, StackingSource
from pynlple.data.filesource import FilePathSource
from pynlple.data.source import Source
from pynlple.processing.preprocessor import BoldTagReplacer, IPreprocessor, StackingPreprocessor

from language_model.pipeline import SandboxTask

MIN_TEXT_LEN = 10
MIN_TEXT_TOKEN_LENGTH = 2


def extractor_stub(input_: Any) -> Hashable:
    if isinstance(input_, HashableType):
        return input_
    raise ValueError("Object is not of `Hashable` type")


def pick_text_hash(input_: Dict[str, Any]) -> Hashable:
    value = input_["textHash"]
    if isinstance(value, HashableType):
        return value
    raise ValueError("Object value of `textHash` field is not of `Hashable` type")


class CacheDeduplicatingSource(Source):
    def __init__(
        self,
        source: Source,
        cache_size: int = 10000,
        refresh: bool = False,
        feature_extractor: Callable[[Any], Hashable] = extractor_stub,
        log: int = 100000,
        log_per_feature: int = 50,
    ) -> None:
        self.source = source
        self.cache_size = cache_size
        self.refresh = refresh
        self.feature_extractor = feature_extractor
        self.log = log
        self.log_per_feature = log_per_feature
        self.__cache: OrderedDict[Hashable, int] = OrderedDict()
        logging.info(f"f={id(self.feature_extractor)}")

    def __iter__(self) -> Iterator[Any]:
        skipped = 0
        total = 0
        for i, entry in enumerate(self.source):
            total += 1
            if self.log and i % self.log == 0:
                logging.info(f"f={id(self.feature_extractor)} skipped {skipped}/{i}")
            f_entry = self.feature_extractor(entry)
            if f_entry in self.__cache:
                skipped += 1
                self.__cache[f_entry] = self.__cache[f_entry] + 1
                if self.refresh:
                    self.__cache.move_to_end(f_entry)
                if self.log and self.__cache[f_entry] % self.log_per_feature == 0:
                    logging.info(
                        f"f={id(self.feature_extractor)} skipped for {self.__cache[f_entry]}th time: " f"{f_entry}"
                    )
            else:
                self.__cache.__setitem__(f_entry, 0)
                while len(self.__cache) > self.cache_size:
                    self.__cache.popitem(last=False)
                yield entry
        logging.info(f"f={id(self.feature_extractor)} skipped {skipped}/{total}")


class PickleDataSource(Source):
    def __init__(self, pickle_filepath: str) -> None:
        super().__init__()
        self.pickle_filepath = pickle_filepath

    def __iter__(self) -> Iterator[Any]:
        return iter(load_pickle(self.pickle_filepath))


class ExtractTextsFromData(SandboxTask):
    def __init__(
        self,
        source_folder_paths: Iterable[str],
        preprocessor: Optional[IPreprocessor] = None,
        min_text_length: int = MIN_TEXT_LEN,
        min_text_token_length: int = MIN_TEXT_TOKEN_LENGTH,
        sandbox_folder_path: str = "data",
    ) -> None:
        super().__init__(sandbox_folder_path)
        preprocessors = [BoldTagReplacer()]
        if preprocessor is not None:
            preprocessors.append(preprocessor)
        self.preprocessor = StackingPreprocessor(preprocessor_list=preprocessors)
        self.min_text_length = min_text_length
        self.min_text_token_length = min_text_token_length
        self.source_folder_paths = source_folder_paths

    def execute(self, environment_path: str) -> None:
        filepath_source = FilePathSource(paths=self.source_folder_paths, extension_suffix=".p")
        json_data_source = StackingSource([PickleDataSource(path) for path in filepath_source])
        text_hash_filtered_source = CacheDeduplicatingSource(
            json_data_source, cache_size=100000, refresh=False, feature_extractor=pick_text_hash, log=20000
        )
        text_source = JsonFieldSource(text_hash_filtered_source, key="text", default="")
        line_text_source = SplittingSource(text_source, splitting_function=str.splitlines)
        processed_text_source = MappingSource(line_text_source, function=self.preprocessor.preprocess)
        short_text_filtered_source = FilteringSource(
            processed_text_source,
            condition=lambda x: len(x) >= self.min_text_length and len(x.split()) >= self.min_text_token_length,
        )
        left_bound_duplicate_filtered_source = CacheDeduplicatingSource(
            short_text_filtered_source,
            cache_size=100000,
            refresh=False,
            feature_extractor=lambda text: str(text[:50]),
            log=10000,
        )
        right_bound_duplicate_filtered_source = CacheDeduplicatingSource(
            left_bound_duplicate_filtered_source,
            cache_size=100000,
            refresh=False,
            feature_extractor=lambda text: str(text[-50:]),
            log=10000,
        )
        output_file_path = os.path.join(environment_path, "texts.txt")
        lines = 0
        with io.open(output_file_path, mode="wt", encoding="utf-8") as output_stream:
            for line in right_bound_duplicate_filtered_source:
                output_stream.write(line)
                output_stream.write("\n")
                lines += 1
        logging.info(f"Completed extraction of texts: {lines} lines written to file.")
