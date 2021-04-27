import itertools
import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch._utils import _accumulate
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer


class LazyDataset(Dataset):
    def __init__(self) -> None:
        self._entries: Optional[Sequence[T_co]] = None

    def __getitem__(self, index: int) -> T_co:
        return self.entries[index]

    def __len__(self) -> int:
        return len(self.entries)

    def __linit_entries__(self) -> Sequence[T_co]:
        raise NotImplementedError()

    @property
    def entries(self) -> Sequence[T_co]:
        if self._entries is None:
            self._entries = self.__linit_entries__()
        return self._entries


class LineByLineTextDataset(LazyDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_paths: Iterable[str],
        block_size: int,
        return_overflowing_tokens: bool = True,
        process_batch_size: int = 8192,
    ) -> None:
        super().__init__()
        self.return_overflowing_tokens = return_overflowing_tokens
        self.tokenizer = tokenizer
        self.file_paths = file_paths
        self.block_size = block_size
        self.process_batch_size = process_batch_size

    def __linit_entries__(self) -> Sequence[T_co]:
        logging.info(f"Creating features from dataset files: {self.file_paths}")
        _entries: List[Dict[str, torch.tensor]] = []
        lines: List[str] = []
        for file_path in self.file_paths:
            with open(file_path, encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if len(line) > 0 and not line.isspace():
                        lines.append(line)

                    if len(lines) == self.process_batch_size:
                        batch_encoding = self.tokenizer(
                            lines,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=self.block_size,
                            return_token_type_ids=False,
                            return_attention_mask=False,
                            return_overflowing_tokens=self.return_overflowing_tokens,
                        )
                        _entries.extend(
                            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]
                        )
                        lines = []
            logging.info(f"Currently read total {len(_entries)} after file name: {file_path}")
        logging.info("Extracted and converted training data to `input_ids`.")
        return _entries


class Portions(LazyDataset):
    def __init__(
        self, dataset: LazyDataset, portions: Sequence[float], generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.portions = portions
        if generator is None:
            generator = torch.Generator().manual_seed(42)
        self.generator = generator

    def __linit_entries__(self) -> Sequence[T_co]:
        if sum(self.portions) > 1.0 or sum(self.portions) < 1.0:  # type: ignore
            raise ValueError("Sum of input portions does not equal 1.0!")
        length = len(self.dataset)
        lengths = [math.ceil(portion * length) for portion in self.portions]
        lengths_ids_iter = itertools.cycle(range(len(lengths)))
        while sum(lengths) > length:
            id_ = next(lengths_ids_iter)
            logging.info(f"Rounding down {id_} portion size:{lengths[id_]} by 1 to sum up to 1.0 totals.")
            lengths[id_] -= 1
        indices = torch.randperm(length, generator=self.generator).tolist()
        return [indices[offset - length : offset] for offset, length in zip(_accumulate(lengths), lengths)]


class LazySubset(Dataset):
    def __init__(self, dataset: Dataset[T_co], portions_provider: Portions, portion_id: int) -> None:
        self.dataset = dataset
        self.portions = portions_provider
        self.portion_id = portion_id

    def __getitem__(self, idx: int) -> T_co:
        return self.dataset[self.portions[self.portion_id][idx]]

    def __len__(self) -> int:
        return len(self.portions[self.portion_id])


def split_lazy_dataset(dataset: LazyDataset, portions: Sequence[float]) -> List[LazySubset]:
    portions_provider = Portions(dataset=dataset, portions=portions)
    return [LazySubset(dataset, portions_provider=portions_provider, portion_id=i) for i in range(len(portions))]
