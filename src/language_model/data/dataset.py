import logging
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_paths: Iterable[str], block_size: int) -> None:
        logging.info(f"Creating features from dataset files: {file_paths}")
        lines: List[str] = []
        for file_path in file_paths:
            with open(file_path, encoding="utf-8") as f:
                lines.extend(line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()))

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples: List[Dict[str, torch.tensor]] = [
            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.tensor]:
        return self.examples[i]
