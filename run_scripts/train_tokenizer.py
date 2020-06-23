import argparse
import os
from typing import List

from language_model.configs import TrainTokenizerConfig


def get_all_files_in_folder(data_folder_path: str) -> List[str]:
    data_files_paths = []
    for (dir_path, _, filenames) in os.walk(data_folder_path):
        data_files_paths.extend([os.path.join(dir_path, file_name) for file_name in filenames])
    return data_files_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config_file = TrainTokenizerConfig.load(args.config_file)
    logger = config_file.logger()  # noqa: F841

    files = get_all_files_in_folder(config_file.source_folder_path)

    config_file.tokenizer.train(
        files=files,
        vocab_size=config_file.vocab_size,
        min_frequency=config_file.min_frequency,
        special_tokens=config_file.special_tokens,
    )

    config_file.tokenizer.save(config_file.saving_folder)


if __name__ == "__main__":
    main()
