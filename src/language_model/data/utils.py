import io
import logging
import os
import random
from typing import Iterable


def write_to_texts_file(texts: Iterable[str], environment_path: str) -> None:
    output_file_path = os.path.join(environment_path, "texts.txt")
    lines = 0
    with io.open(output_file_path, mode="wt", encoding="utf-8") as output_stream:
        for line in texts:
            output_stream.write(line)
            output_stream.write("\n")
            lines += 1
    logging.info(f"Completed extraction of texts: {lines} lines written to file.")


def write_to_train_val_files(texts: Iterable[str], environment_path: str, test_ratio: float, test_size: int):
    train_file_path = os.path.join(environment_path, "train.txt")
    validation_file_path = os.path.join(environment_path, "validation.txt")
    train_lines = 0
    test_lines = 0
    train_stream = io.open(train_file_path, mode="wt", encoding="utf-8")
    validation_stream = io.open(validation_file_path, mode="wt", encoding="utf-8")
    for line in texts:
        if test_ratio > random.random() and test_lines <= test_size:
            validation_stream.write(line)
            validation_stream.write("\n")
            test_lines += 1
        else:
            train_stream.write(line)
            train_stream.write("\n")
            train_lines += 1

    validation_stream.close()
    train_stream.close()
    logging.info(
        f"Completed extraction of texts: {test_lines} lines written to test file "
        f"and {train_lines} lines written to train file"
    )
