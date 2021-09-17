import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Iterable, List, Set, Tuple, Union

import numpy as np

try:
    from lsh import cache, minhash
except ImportError:
    cache, minhash = None, None


class MinHashLSHDeduplicator:
    def __init__(self, seeds: Union[int, np.ndarray], char_ngram: int, bands: int, workers: int = -1):
        if cache is None or minhash is None:
            raise ImportError(
                "It seems like you do not have lsh package. To use 'MinHashLSHDeduplicator' you need install it: "
                "$git clone https://github.com/mattilyra/LSH "
                "$cd LSH && python setup.py install"
            )
        hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, random_state=42)
        self.lsh_cache = cache.Cache(num_bands=bands, hasher=hasher)
        self.workers = multiprocessing.cpu_count() if workers == -1 else workers

    def deduplicate(self, docs: List[str], min_jaccard: float, clear: bool = True) -> List[str]:
        if clear:
            self.lsh_cache.clear()

        duplicate_ids: Set[int] = set()
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
        keep_form_duplicate_ids: Set[int] = set()
        duplicate_ids: Set[int] = set()

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
        return self.lsh_cache.get_all_duplicates(min_jaccard)  # type: ignore

    def _cache_texts(self, docs: Iterable[str], start_id: int = 0) -> None:
        for i, doc in enumerate(docs, start_id):
            self.lsh_cache.add_doc(doc, i)

    def _cache_texts_parallel(self, docs: Iterable[str], start_id: int = 0) -> None:
        encoded_docs = (doc.encode("utf8") for doc in docs)
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            for i, fingerprint in enumerate(executor.map(self.lsh_cache.hasher.fingerprint, encoded_docs), start_id):
                self.lsh_cache.add_fingerprint(fingerprint, i)
