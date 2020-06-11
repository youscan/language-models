import argparse
import os

from ds_shared.loading import get_all_files_in_folder, load_pickle

from src.configs import PrepareOneLineTextFormatConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config_file = PrepareOneLineTextFormatConfig.load(args.config_file)
    logger = config_file.logger()  # noqa: F841

    files = get_all_files_in_folder(config_file.source_folder_path)

    with open(os.path.join(config_file.saving_folder, "data.txt"), "w", encoding="utf-8") as f:
        for file in files:
            data = load_pickle(file)
            for sample in data:
                if config_file.keep_title:
                    f.write(sample["title"])
                    f.write("\n")
                f.write(sample["text"])
                f.write("\n")


if __name__ == '__main__':
    main()
