import gc
import logging
import multiprocessing
import random
from collections import Hashable as HashableType
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, Optional, Union, List, Set, Tuple, Generator

import numpy as np
from bs4 import BeautifulSoup
from ds_shared.loading import load_pickle
from lsh import minhash, cache
from pynlple.data.corpus import FilteringSource, JsonFieldSource, MappingSource, StackingSource
from pynlple.data.filesource import FilePathSource
from pynlple.data.source import Source
from pynlple.processing.preprocessor import IPreprocessor

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


class MinHashLSHDeduplicator:
    def __init__(self, seeds: Union[int, np.ndarray], char_ngram: int, bands: int, workers: int = -1):
        self.workers = multiprocessing.cpu_count() if workers == -1 else workers
        hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, random_state=42)
        self.lsh_cache = cache.Cache(num_bands=bands, hasher=hasher)

    def deduplicate(self, docs: List[str], min_jaccard: float, clear: bool = True) -> List[str]:
        if clear:
            self.lsh_cache.clear()

        duplicate_ids = set()
        keep_form_duplicate_ids = set()
        for i, j in self.get_all_duplicates(docs, min_jaccard):
            if i not in duplicate_ids and j not in duplicate_ids:
                keep_form_duplicate_ids.add(i)
            elif i in keep_form_duplicate_ids and j in keep_form_duplicate_ids:
                keep_form_duplicate_ids.remove(j)
            duplicate_ids.add(i)
            duplicate_ids.add(j)

        keep = set(range(len(docs))) - duplicate_ids | keep_form_duplicate_ids

        return [docs[i] for i in keep]

    def in_memory_batch_deduplicate(self, docs: Iterable[str], min_jaccard: float, batch_size: int) -> Iterable[str]:
        batch_docs = list(islice(docs, batch_size))
        while batch_docs:
            batch_docs = self.deduplicate(batch_docs, min_jaccard=min_jaccard, clear=True)
            gc.collect()
            add_into_batch = batch_size - len(batch_docs)
            if add_into_batch > 0:
                new_docs = list(islice(docs, add_into_batch))
                if not new_docs:
                    break
                batch_docs.extend(new_docs)
            else:
                yield from batch_docs
                batch_docs = list(islice(docs, batch_size))
        yield from batch_docs

    def lsh_batch_deduplicate(
            self, docs: Iterable[str], min_jaccard: float, batch_size: int, clear: bool = True
    ) -> Iterable[str]:
        """
        batch_size: max size of docs will be deduplicated at time, while `batch_docs` list will be extended
            until `batch_size` unique docs will be collected into it
        """
        if clear:
            self.lsh_cache.clear()

        start_id, end_id = 0, batch_size
        duplicate_ids, keep_form_duplicate_ids = set(), set()

        batch_docs = list(islice(docs, batch_size))

        while batch_docs:

            new_duplicate_ids = set()
            # batch_docs[start_id:] to process just recently appended docs,
            # start_id = start_id to set new ids for recently appended docs
            for i, j in self.get_all_duplicates(batch_docs[start_id:], min_jaccard, start_id=start_id):
                if i not in duplicate_ids and j not in duplicate_ids:
                    keep_form_duplicate_ids.add(i)
                elif i in keep_form_duplicate_ids and j in keep_form_duplicate_ids:
                    keep_form_duplicate_ids.remove(j)
                duplicate_ids.add(i)
                duplicate_ids.add(j)
                new_duplicate_ids.add(i)
                new_duplicate_ids.add(j)

            # new_duplicate_ids - to clear duplicates from recently appended
            # docs, as previous duplicate ids have already cleared
            drop_ids = new_duplicate_ids - keep_form_duplicate_ids

            if drop_ids:
                for i in drop_ids:
                    self.lsh_cache.remove_id(i)

                start_id = end_id
                end_id = start_id + len(drop_ids)
            else:
                # all non duplicated + keep_form_duplicate_ids
                for i in set(range(len(batch_docs))) - duplicate_ids | keep_form_duplicate_ids:
                    yield batch_docs[i]

                # new batch
                self.lsh_cache.clear()
                start_id, end_id = 0, batch_size
                duplicate_ids, keep_form_duplicate_ids = set(), set()
                batch_docs = []

            # if drop_ids: we append next len(drop_ids) examples
            batch_docs.extend(islice(docs, end_id - start_id))

        if batch_docs:
            # all non duplicated + keep_form_duplicate_ids
            for i in set(range(len(batch_docs))) - duplicate_ids | keep_form_duplicate_ids:
                yield batch_docs[i]

    def get_all_duplicates(self, docs: Iterable[str], min_jaccard: float, start_id: int = 0) -> Set[Tuple[int, int]]:
        self._cache_texts_parallel(docs, start_id=start_id)
        import pdb; pdb.set_trace()
        return self.lsh_cache.get_all_duplicates(min_jaccard)

    def _cache_texts(self, docs: Iterable[str], start_id: int = 0) -> None:
        for i, doc in enumerate(docs, start_id):
            self.lsh_cache.add_doc(doc, i)

    def _cache_texts_parallel(self, docs: Iterable[str], start_id: int = 0) -> None:
        encoded_docs = (doc.encode("utf8") for doc in docs)
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            for i, fingerprint in enumerate(executor.map(self.lsh_cache.hasher.fingerprint, encoded_docs), start_id):
                self.lsh_cache.add_fingerprint(fingerprint, i)


class ExtractTextsFromData(ITask):
    def __init__(
        self,
        text_source: Iterable[str],
        preprocessor: Optional[IPreprocessor] = None,
        min_text_length: int = MIN_TEXT_LEN,
        min_text_token_length: int = MIN_TEXT_TOKEN_LENGTH,
        cache_size: int = 100_000
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.min_text_length = min_text_length
        self.min_text_token_length = min_text_token_length
        self.text_source = text_source
        self.cache_size = cache_size

    def execute(self, environment_path: str) -> None:
        self._write_to_file(self._deduplicate(self._filter(self._preprocess())), environment_path=environment_path)

    def _preprocess(self) -> Source:
        return MappingSource(self.text_source, function=self.preprocessor.preprocess)

    def _filter(self, preprocessed_source: Source) -> Source:
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
            test_size: Union[float, int] = 0.1
    ) -> None:
        super().__init__(text_source, preprocessor, min_text_length, min_text_token_length, seeds,)
        # if test_size > 1 (absolute number) there is a probability that we won't reach this number
        # if size of full dataset is twice less or approximately equal than test_size
        self.test_ratio = 0.5 if test_size > 1 else test_size
        self.test_size = test_size if test_size > 1 else float("inf")

    def _write_to_file(self, texts: Iterable[str], environment_path: str) -> None:
        return write_to_train_val_files(texts, environment_path, test_ratio=self.test_ratio, test_size=self.test_size)
