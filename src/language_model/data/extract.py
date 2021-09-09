import json
import logging
import multiprocessing
import os
import random
from collections import Hashable as HashableType
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Hashable, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from bs4 import BeautifulSoup
from ds_shared.loading import load_pickle
from math import ceil
from more_itertools import chunked
from pynlple.data.corpus import FilteringSource, JsonFieldSource, MappingSource, StackingSource
from pynlple.data.filesource import FilePathSource
from pynlple.data.source import Source
from pynlple.processing.preprocessor import IPreprocessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .utils import write_to_texts_file, write_to_train_val_files
from ..pipeline import ITask

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


class LineByLineSource(Source):
    def __init__(self, text_filepath: str) -> None:
        super().__init__()
        self.text_filepath = text_filepath

    def __iter__(self) -> Iterator[str]:
        with open(self.text_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


class ShuffledSources(Source):
    def __init__(self, *sources: Generator[str, None, None]) -> None:
        self.sources = list(sources)

    def __iter__(self) -> Iterable[str]:
        return self

    def __next__(self) -> str:
        if not self.sources:
            raise StopIteration
        source_id = random.choice(range(len(self.sources)))
        try:
            return next(iter(self.sources[source_id]))
        except StopIteration:
            self.sources.pop(source_id)
            return next(self)


class PickleDataSource(Source):
    def __init__(self, pickle_filepath: str) -> None:
        super().__init__()
        self.pickle_filepath = pickle_filepath

    def __iter__(self) -> Iterator[Any]:
        return iter(load_pickle(self.pickle_filepath))


class PostWikiExtractorDataSource(Source):
    """Provides wiki articles preprocessed by `python -m wikiextractor.WikiExtractor dump.xml.bz2 ...`"""

    def __init__(self, article_dir: str) -> None:
        super().__init__()
        self.article_dir = Path(article_dir)

    def __iter__(self) -> Iterator[Any]:
        for subdir in self.article_dir.glob("*"):
            for file in subdir.glob("*"):
                with open(file, "r") as f:
                    data = f.read()
                soup = BeautifulSoup(data, "lxml")
                for doc in soup.find_all("doc"):
                    yield {"id": doc["id"], "title": doc["title"], "url": doc["url"], "text": doc.text}


class FromLoadedYsDataSource(Source):
    def __init__(self, source_folder_paths: Iterable[str], preprocessor: Optional[IPreprocessor] = None):
        self.source_folder_paths = source_folder_paths
        self.preprocessor = preprocessor

    def __iter__(self) -> Iterator[str]:
        filepath_source = FilePathSource(paths=self.source_folder_paths, extension_suffix=".p")
        json_data_source = StackingSource([PickleDataSource(path) for path in filepath_source])
        subtitles_filtering_source = FilteringSource(
            json_data_source, condition=lambda mention: "subtitles" not in mention.get("contentTypes", set())
        )
        text_hash_filtered_source = CacheDeduplicatingSource(
            subtitles_filtering_source, cache_size=100000, refresh=False, feature_extractor=pick_text_hash, log=20000
        )
        text_source = JsonFieldSource(text_hash_filtered_source, key="text", default="")
        if self.preprocessor is not None:
            yield from MappingSource(text_source, function=self.preprocessor.preprocess)
        else:
            yield from text_source


class ExtractTextsFromData(ITask):
    def __init__(
        self,
        text_source: Iterable[str],
        preprocessor: Optional[IPreprocessor] = None,
        min_text_length: int = MIN_TEXT_LEN,
        min_text_token_length: int = MIN_TEXT_TOKEN_LENGTH,
        cache_size: int = 100_000,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.min_text_length = min_text_length
        self.min_text_token_length = min_text_token_length
        self.text_source = text_source
        self.cache_size = cache_size

    def execute(self, environment_path: str) -> None:
        self._write_to_file(self._deduplicate(self._filter(self._preprocess())), environment_path=environment_path)

    def _preprocess(self) -> Union[Source, Iterable[str]]:
        if self.preprocessor is not None:
            return MappingSource(self.text_source, function=self.preprocessor.preprocess)
        return self.text_source

    def _filter(self, preprocessed_source: Union[Source, Iterable[str]]) -> Source:
        return FilteringSource(
            preprocessed_source,
            condition=lambda x: len(x) >= self.min_text_length and len(x.split()) >= self.min_text_token_length,
        )

    def _deduplicate(self, filtered_source: Source) -> Iterable[str]:
        left_bound_duplicate_filtered_source = CacheDeduplicatingSource(
            filtered_source,
            cache_size=self.cache_size,
            refresh=False,
            feature_extractor=lambda text: str(text[:50]),
            log=10000,
        )
        right_bound_duplicate_filtered_source = CacheDeduplicatingSource(
            left_bound_duplicate_filtered_source,
            cache_size=self.cache_size,
            refresh=False,
            feature_extractor=lambda text: str(text[-50:]),
            log=10000,
        )
        return right_bound_duplicate_filtered_source

    def _write_to_file(self, texts: Iterable[str], environment_path: str) -> None:
        return write_to_texts_file(texts, environment_path)


class RandomSplitTextsFromData(ExtractTextsFromData):
    def __init__(
        self,
        text_source: Iterable[str],
        preprocessor: Optional[IPreprocessor] = None,
        min_text_length: int = MIN_TEXT_LEN,
        min_text_token_length: int = MIN_TEXT_TOKEN_LENGTH,
        seeds: Union[int, np.ndarray] = 100,
        test_size: Union[float, int] = 0.1,
    ) -> None:
        super().__init__(text_source, preprocessor, min_text_length, min_text_token_length, seeds)
        # if test_size > 1 (absolute number) there is a probability that we won't reach this number
        # if size of full dataset is twice less or approximately equal than test_size
        self.test_ratio = 0.5 if test_size > 1 else test_size
        self.test_size = test_size if test_size > 1 else float("inf")

    def _write_to_file(self, texts: Iterable[str], environment_path: str) -> None:
        return write_to_train_val_files(
            texts, environment_path, test_ratio=self.test_ratio, test_size=self.test_size  # type: ignore
        )


class ExtractVectorsFromTexts(ITask):
    def __init__(
            self,
            data_source: Iterable[str],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            block_size: int,
            workers: int = -1,
            process_batch_size: int = 8192,
    ):
        self.workers = multiprocessing.cpu_count() if workers == -1 else workers
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_source = data_source
        self.process_batch_size = process_batch_size

    def execute(self, environment_path: str) -> None:
        input_ids_file = os.path.join(environment_path, f"processed_batch.jsonl")
        if os.path.exists(input_ids_file):
            raise FileExistsError(f"{input_ids_file} already exists")

        counter = 1
        for lines in self._read_chunk():
            batch_size = ceil(len(lines) / self.workers)
            batched_lines = chunked(lines, batch_size)
            extract_batch_args = ((batch_lines, self.tokenizer, self.block_size) for batch_lines in batched_lines)

            with open(input_ids_file, "a") as fp:
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    for batch in executor.map(self._extract_batch, extract_batch_args):
                        for input_ids in batch:
                            fp.write(json.dumps(input_ids))
                            fp.write("\n")
            logging.info(f"Currently extracted {counter} batches of size {self.process_batch_size}")
            counter += 1

    def _read_chunk(self) -> Iterator[List[str]]:
        lines: List[str] = []
        for line in self.data_source:
            if len(line) > 0 and not line.isspace():
                lines.append(line)

            if len(lines) == self.process_batch_size:
                yield lines
                lines = []
        if len(lines) > 0:
            yield lines

    @staticmethod
    def _extract_batch(
            args: Tuple[List[str], Union[PreTrainedTokenizer, PreTrainedTokenizerFast], int]
    ) -> List[List[int]]:
        lines, tokenizer, block_size = args
        batch_encoding: List[List[int]] = []
        current_line = [tokenizer.bos_token]
        for line in lines:
            tokens = tokenizer.tokenize(line)
            if len(current_line) + len(tokens) + 1 <= block_size:
                current_line.append(tokenizer.bos_token)
                current_line.extend(tokens)
            elif len(current_line) == block_size:
                input_ids = tokenizer.convert_tokens_to_ids(current_line)
                batch_encoding.append(input_ids)
                current_line = [tokenizer.bos_token] + tokens
            else:
                current_line.append(tokenizer.bos_token)
                n_tokens_to_add = block_size - len(current_line)
                current_line.extend(tokens[:n_tokens_to_add])
                input_ids = tokenizer.convert_tokens_to_ids(current_line)
                batch_encoding.append(input_ids)

                tokens = tokens[n_tokens_to_add:]
                while len(tokens) >= block_size:
                    input_ids = tokenizer.convert_tokens_to_ids(tokens[: block_size])
                    batch_encoding.append(input_ids)
                    tokens = tokens[block_size:]

                current_line = tokens
        return batch_encoding
