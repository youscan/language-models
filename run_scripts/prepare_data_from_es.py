import argparse

from ds_shared.preprocessing import Deduplicator, EsPreprocessor
from ds_shared.preprocessing.feature_extractor import letters_from_text

from src.configs import PrepareDataFromEsConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config_file = PrepareDataFromEsConfig.load(args.config_file)
    logger = config_file.logger()  # noqa: F841

    deduplicator = Deduplicator(
        [],
        cache_size=config_file.cache_size,
        feature_extractors=[
            lambda x: letters_from_text(x["text"].lower()),
            lambda x: letters_from_text(x["text"].lower(), is_start=False),
        ],
    )

    es_preprocessor = EsPreprocessor(
        source_data_folder=config_file.source_folder_path,
        save_data_folder=config_file.saving_folder,
        deduplicator=deduplicator,
        cache_size=config_file.cache_size,
        logging_step=config_file.logging_step,
        checkpoint_step=config_file.checkpoint_step,
        proper_languages=config_file.proper_languages,
    )
    es_preprocessor.preprocess_and_save()


if __name__ == "__main__":
    main()
